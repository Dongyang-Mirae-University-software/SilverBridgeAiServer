#!/usr/bin/env python3
"""챗봇 오프라인 재생 도구 - API 를 적게 쓰는 판정 재생(2026-11-05).

`ChatService.process_message` 를 SQLite 세션으로 직접 부른다(턴 단계·후처리 전부 포함, MedGemma 는 이 머신의 모델).
라우터(Claude) 호출이 비용의 대부분이라 라우터만 세 가지 방식으로 바꿔 끼운다.

  stub  - 시험지 expect 값으로 라우터 결과를 만든다(API 0회). 서버 쪽 판정·답 모양 회귀 확인용.
  cache - 기록해 둔 라우터 응답만 쓴다(API 0회). 없는 문장은 라우터 실패로 처리되고 요약에 '누락'으로 센다.
  live  - 실제 라우터를 부르고 응답을 캐시 파일에 기록한다. 키 필수, 호출 상한(--max-router-calls, 기본 100) 필수.
          시작 전에 예상 호출 수를 출력하고, 예상이 상한을 넘으면 시작하지 않는다.

기본값: 회차 1(MedGemma 는 같은 입력이면 같은 답이라 회차를 늘려도 라우터 흔들림만 잰다), 이어서 물어볼 질문 끔(Haiku 호출 0).
라우터 흔들림은 `router-variance` 명령으로 소수 문항만 따로 잰다(MedGemma 없이, live 상한 필수).

출력: 답 본문이 든 결과(--out)와 캐시 파일은 저장소 밖 경로만 허용한다. --summary 는 본문 없이 문항별 구조·개수만.
사용 예는 docs/tools/replay-offline.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
import uuid
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MAX_ROUTER_CALLS = 100


class BudgetExceeded(RuntimeError):
    """라우터 호출 상한을 넘었다."""


class RouterCacheMiss(RuntimeError):
    """cache 모드에서 기록되지 않은 라우터 입력."""


def outside_repo(path: str | os.PathLike[str]) -> pathlib.Path:
    """본문·캐시 파일은 저장소 밖에만 쓴다(상담 본문이 커밋되지 않게)."""
    p = pathlib.Path(path).expanduser().resolve()
    if p == REPO_ROOT or REPO_ROOT in p.parents:
        raise SystemExit(f"저장소 안 경로는 쓸 수 없습니다: {p}")
    return p


def router_cache_key(model: str, system_prompt: str, schema: dict, message: str, history: list, prev_reservation: str | None) -> str:
    """같은 모델·지시문·스키마·문장·이력이면 같은 키(정렬된 JSON 의 sha256). 지시문이 바뀌면 옛 캐시는 쓰이지 않는다."""
    payload = {
        "model": model,
        "system": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "schema": hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "message": message,
        "history": history,
        "prev": prev_reservation,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class CallBudget:
    def __init__(self, limit: int) -> None:
        self.limit, self.used = limit, 0

    def take(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(f"라우터 호출 상한 {self.limit}회를 넘었습니다")
        self.used += 1


def stub_route(item: dict, message: str) -> dict[str, Any]:
    """시험지 기대값으로 만든 라우터 결과(API 없음)."""
    ex = item.get("expect") or {}
    urgent = ex.get("urgent_condition")
    return {
        "route": ex.get("route") or "symptom",
        "engine": "stub",
        "normalized_message": message,
        "symptom_summary": message,
        "emergency": bool(ex.get("emergency")) or bool(ex.get("cpr_aed")),
        "self_harm": bool(ex.get("crisis")),
        "cardiac_arrest": bool(ex.get("cpr_aed")),
        "urgent_condition": urgent if isinstance(urgent, str) else None,
        "reason": "stub",
    }


def require_key(key_file: str | None) -> None:
    if key_file:
        os.environ["ANTHROPIC_API_KEY"] = pathlib.Path(key_file).expanduser().read_text().strip()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("live 모드에는 ANTHROPIC_API_KEY(또는 --key-file)가 필요합니다")


def planned_router_calls(items: list[dict], runs: int) -> int:
    """예상 라우터 호출 수 상한(앞 턴 재생 포함) - 실제는 키워드 위기·예약 UI 턴 등으로 더 적을 수 있다."""
    return sum(len(it.get("history_user") or []) + 1 for it in items) * runs


def planned_claude_calls(items: list[dict], runs: int, follow_up: bool) -> int:
    """예상 Claude 호출 수 상한 - 라우터 + 예약 경로 문항의 예약 추출 + (켰으면) 이어서 물어볼 질문, 턴마다."""
    total = 0
    for it in items:
        turns = len(it.get("history_user") or []) + 1
        per_turn = 1 + (1 if follow_up else 0)
        if ((it.get("expect") or {}).get("route")) == "reservation":
            per_turn += 1
        total += turns * per_turn
    return total * runs


def load_items(set_path: str, ids: str | None) -> list[dict]:
    data = json.loads(pathlib.Path(set_path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    out = []
    want = set(ids.split(",")) if ids else None
    for it in items:
        if want and it["id"] not in want:
            continue
        it = dict(it)
        if "history_user" not in it:
            it["history_user"] = [h["content"] for h in it.get("history") or [] if h.get("role") == "user"]
        out.append(it)
    return out


class _CachedRouter:
    """실제 라우터를 감싼다 - cache 모드는 기록만 읽고, live 모드는 없으면 불러서 기록한다(상한 안에서)."""

    def __init__(self, real: Any, cache: dict, mode: str, budget: CallBudget, stats: dict) -> None:
        self.real, self.cache, self.mode, self.budget, self.stats = real, cache, mode, budget, stats

    def route(self, message, history, user_context, prev_reservation=None):
        from app.services import chat_router_service as crs

        key = router_cache_key(self.real._loader.state.model_name, crs.SYSTEM_PROMPT, crs.ROUTE_SCHEMA, message,
                               list(history or []), prev_reservation)
        if key in self.cache:
            self.stats["hit"] += 1
            return dict(self.cache[key])
        if self.mode != "live":
            self.stats["miss"] += 1
            raise RouterCacheMiss(key[:12])
        out = self.real.route(message, history, user_context, prev_reservation=prev_reservation)  # 상한은 guard_claude_calls 가 센다
        self.stats["live"] += 1
        self.cache[key] = out
        return out


def guard_claude_calls(mode: str, budget: "CallBudget", stats: dict) -> None:
    """ClaudeStructuredClient 의 모든 호출(라우터·예약 추출·이어서 물어볼 질문)을 지킨다.
    stub·cache: 호출을 막고 시도를 센다(요약 blockedClaudeCalls - 예약 추출처럼 라우터 밖 호출은 실패한 채로 이어진다).
    live: 단계와 상관없이 호출마다 상한을 쓴다(2026-11-05 점검 M-1 - 라우터만 세던 것)."""
    from app.services import claude_structured_client as csc

    original = csc.ClaudeStructuredClient.generate_structured_text

    def guarded(self, system_prompt, user_prompt, schema=None, *, stage="claude", reserve_sec=0.0):
        if mode != "live":
            stats["blocked"] += 1
            raise csc.StructuredCallError("replay_blocked")
        budget.take()
        stats["claude"] += 1
        return original(self, system_prompt, user_prompt, schema, stage=stage, reserve_sec=reserve_sec)

    csc.ClaudeStructuredClient.generate_structured_text = guarded


def _prepare_env(args: argparse.Namespace) -> None:
    os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='replay_offline_')}/r.db"
    os.environ.setdefault("STREAM_STATE_BACKEND", "memory")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not args.follow_up:
        os.environ["FOLLOW_UP_ENABLED"] = "false"  # 이어서 물어볼 질문(Haiku) 호출 0
    if args.router != "live":
        # stub·cache 는 API 를 부를 수 없게 - 지우지 않고 빈 값으로 넣는다. app.core.config 의 load_dotenv() 는 이미 있는
        # 환경변수를 덮지 않으므로 .env 가 있는 저장소(gosky)에서도 키가 다시 채워지지 않는다(2026-11-05 점검 H-1).
        os.environ["ANTHROPIC_API_KEY"] = ""
        os.environ["CLAUDE_API_KEY"] = ""


def _summary_row(item: dict, run: int, res: dict | None, err: str | None, miss: bool) -> dict:
    reply = (res or {}).get("reply") or ""
    return {
        "id": item["id"], "run": run, "err": err, "routerCacheMiss": miss,
        "riskLevel": (res or {}).get("riskLevel"), "engine": (res or {}).get("engine"),
        "has119": "119" in reply, "has109": "109" in reply, "warning": bool((res or {}).get("emergencyWarning")),
        "cpr": "가슴압박" in reply, "followup": "여쭤볼게요" in reply, "lines": reply.count("\n") + 1 if reply else 0,
    }


def cmd_run(args: argparse.Namespace) -> int:
    out_path = outside_repo(args.out)
    cache_path = outside_repo(args.cache_file) if args.cache_file else None
    summary_path = outside_repo(args.summary) if args.summary else None  # 본문은 없지만 저장소에 쌓이지 않게(L-1)
    items = load_items(args.set, args.ids)
    planned = planned_claude_calls(items, args.runs, args.follow_up)
    print(f"문항 {len(items)} × 회차 {args.runs}, 라우터 모드 {args.router}, 예상 Claude 호출 최대 {planned}회"
          f"(라우터 {planned_router_calls(items, args.runs)}회 포함)", flush=True)
    if args.router == "live":
        require_key(args.key_file)
        if planned > args.max_router_calls:
            print(f"예상 Claude 호출이 상한 {args.max_router_calls}회를 넘어 시작하지 않습니다(--max-router-calls 로 올리거나 문항을 줄이세요)")
            return 2
    if args.router in ("cache", "live") and cache_path is None:
        raise SystemExit("cache·live 모드에는 --cache-file 이 필요합니다(저장소 밖)")
    _prepare_env(args)

    from app.database.base import Base
    from app.database.session import SessionLocal, engine
    import app.models.chat_log  # noqa: F401
    from app.schemas.chat_schema import ChatContext, ChatRequest
    import app.services.chat_service as cs

    try:
        import app.models.reservation_token_claim  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    Base.metadata.create_all(bind=engine)

    budget = CallBudget(args.max_router_calls)
    stats = {"hit": 0, "miss": 0, "live": 0, "blocked": 0, "claude": 0}
    guard_claude_calls(args.router, budget, stats)
    current: dict = {}
    cache: dict = {}
    if args.router == "stub":
        class _Stub:
            def route(self, message, *a, **k):
                return stub_route(current, message)
        cs.get_chat_router_service = lambda: _Stub()
    else:
        if cache_path and cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        real = cs.get_chat_router_service() if args.router == "live" else _KeylessRouter()
        wrapped = _CachedRouter(real, cache, args.router, budget, stats)
        cs.get_chat_router_service = lambda: wrapped

    svc = cs.ChatService()
    bodies, summary = [], []
    try:
        for run in range(1, args.runs + 1):
            for n, it in enumerate(items):
                current.clear()
                current.update(it)
                ctx = ChatContext(age=it.get("age", 76), gender=it.get("gender"), wardId=it.get("wardId"), role=it.get("role"))
                sid, uid = uuid.uuid4().hex[:16], f"rp{run}{n:03d}"
                res, err, miss_before = None, None, stats["miss"]
                for msg in list(it.get("history_user") or []) + [it["q"]]:
                    db = SessionLocal()
                    try:
                        res = svc.process_message(db, ChatRequest(userId=uid, message=msg, sessionId=sid, context=ctx))
                    except BudgetExceeded:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        err = type(exc).__name__
                    finally:
                        db.close()
                miss = stats["miss"] > miss_before
                bodies.append({"id": it["id"], "run": run, "reply": (res or {}).get("reply"), "riskLevel": (res or {}).get("riskLevel"),
                               "emergencyWarning": (res or {}).get("emergencyWarning")})
                summary.append(_summary_row(it, run, res, err, miss))
    except BudgetExceeded as exc:
        print(f"멈춤: {exc}", flush=True)
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bodies, ensure_ascii=False), encoding="utf-8")
        if cache_path and args.router == "live":
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        if summary_path:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps({"stats": stats, "claudeCallsUsed": budget.used, "blockedClaudeCalls": stats["blocked"],
                                                "rows": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"끝: 결과 {len(summary)}건, 라우터 캐시 적중 {stats['hit']}·누락 {stats['miss']}·실제 라우터 {stats['live']}회, "
          f"Claude 호출 {stats['claude']}회·차단 {stats['blocked']}회", flush=True)
    if stats["blocked"]:
        print("주의: stub·cache 모드에서 Claude 호출 시도가 있었고 막았다(예약 추출 등) - 해당 문항 결과는 라우터 밖 판정이 실패한 상태다")
    return 1 if stats["miss"] and not args.allow_miss else 0


class _KeylessRouter:
    """cache 모드용 - 모델 이름만 알면 된다(호출하지 않는다)."""

    class _State:
        def __init__(self) -> None:
            from app.core.config import get_settings
            self.model_name = get_settings().chat_router_model

    def __init__(self) -> None:
        self._loader = type("L", (), {"state": self._State()})()

    def route(self, *a, **k):  # pragma: no cover - _CachedRouter 가 먼저 막는다
        raise RouterCacheMiss("keyless")


def cmd_router_variance(args: argparse.Namespace) -> int:
    """라우터만 N회 불러 판정 칸이 흔들리는지 센다(MedGemma 없음, 본문 없음)."""
    items = load_items(args.set, args.ids)
    planned = len(items) * args.times
    print(f"문항 {len(items)} × {args.times}회 = 라우터 호출 {planned}회(상한 {args.max_router_calls})", flush=True)
    if planned > args.max_router_calls:
        print("예상 호출이 상한을 넘어 시작하지 않습니다")
        return 2
    require_key(args.key_file)
    os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='replay_variance_')}/r.db"
    from app.services.chat_router_service import get_chat_router_service

    router = get_chat_router_service()
    budget = CallBudget(args.max_router_calls)
    out = {}
    for it in items:
        counts = {"n": 0, "failed": 0, "emergency": 0, "self_harm": 0, "cardiac_arrest": 0, "urgent": 0, "routes": {}}
        for _ in range(args.times):
            budget.take()
            counts["n"] += 1
            try:
                r = router.route(it["q"], [], None)
            except Exception:  # noqa: BLE001
                counts["failed"] += 1
                continue
            counts["emergency"] += bool(r.get("emergency"))
            counts["self_harm"] += r.get("self_harm") is True
            counts["cardiac_arrest"] += r.get("cardiac_arrest") is True
            counts["urgent"] += bool(r.get("urgent_condition"))
            counts["routes"][r.get("route")] = counts["routes"].get(r.get("route"), 0) + 1
            time.sleep(args.interval)
        out[it["id"]] = counts
        print(it["id"], counts, flush=True)
    if args.summary:
        pathlib.Path(args.summary).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="시험지를 재생한다")
    run.add_argument("--set", required=True)
    run.add_argument("--ids", default="")
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--router", choices=("stub", "cache", "live"), default="stub")
    run.add_argument("--cache-file", default="")
    run.add_argument("--key-file", default="")
    run.add_argument("--max-router-calls", type=int, default=DEFAULT_MAX_ROUTER_CALLS)
    run.add_argument("--follow-up", action="store_true", help="이어서 물어볼 질문(Haiku)도 부른다 - 기본은 끔")
    run.add_argument("--allow-miss", action="store_true", help="cache 모드 누락이 있어도 0 으로 끝낸다")
    run.add_argument("--out", required=True, help="답 본문 결과(저장소 밖)")
    run.add_argument("--summary", default="", help="본문 없는 요약 JSON")
    var = sub.add_parser("router-variance", help="라우터만 N회 불러 흔들림을 센다(live, 상한 필수)")
    var.add_argument("--set", required=True)
    var.add_argument("--ids", required=True, help="쉼표로 구분한 소수 문항")
    var.add_argument("--times", type=int, default=5)
    var.add_argument("--interval", type=float, default=0.0)
    var.add_argument("--key-file", default="")
    var.add_argument("--max-router-calls", type=int, default=DEFAULT_MAX_ROUTER_CALLS)
    var.add_argument("--summary", default="")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_router_variance(args)


if __name__ == "__main__":
    sys.exit(main())
