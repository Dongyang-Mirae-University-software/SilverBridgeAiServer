#!/usr/bin/env python3
"""챗봇 오프라인 재생 도구 - API 를 적게 쓰는 판정 재생(2026-11-05, v8 확장 2026-11-20).

`ChatService.process_message` 를 SQLite 세션으로 직접 부른다(턴 단계·후처리 전부 포함, MedGemma 는 이 머신의 모델).
라우터(Claude) 호출이 비용의 대부분이라 라우터만 바꿔 끼운다.

  stub  - 시험지 expect 값(v8 은 턴별 route)으로 라우터 결과를 만든다(API 0회). 서버 쪽 판정·답 모양 회귀 확인용.
  fail  - 라우터를 매 턴 실패시킨다(API 0회). 키워드 폴백(영역 E) 시험용. v8 턴의 route="fail" 은 stub 에서도 그 턴만 실패.
  cache - 기록해 둔 라우터 응답만 쓴다(API 0회). 없는 문장은 라우터 실패로 처리되고 요약에 '누락'으로 센다.
  live  - 실제 라우터를 부르고 응답을 캐시 파일에 기록한다. 키 필수, 호출 상한(--max-router-calls, 기본 100) 필수.
          시작 전에 예상 호출 수를 출력하고, 예상이 상한을 넘으면 시작하지 않는다.

기본값: 회차 1(MedGemma 는 같은 입력이면 같은 답이라 회차를 늘려도 라우터 흔들림만 잰다), 이어서 물어볼 질문 끔(Haiku 호출 0).
라우터 흔들림은 `router-variance` 명령으로 소수 문항만 따로 잰다(MedGemma 없이, live 상한 필수).

예약은 언제나 가짜 예약 서비스(tools/eval/v8/fake_reservation.py, 가상 병원 46곳)로 간다 - 운영 예약 서비스·네트워크를 부르지 않는다.
v8 문항(turns·context·uiSelection 턴)과 옛 시험지(q·history·expect)를 모두 읽는다(.json 또는 .jsonl).

출력: 답 본문이 든 결과(--out)와 캐시 파일은 저장소 밖 경로만 허용한다. --summary 는 본문 없이 문항별 구조·턴별 특징만.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MAX_ROUTER_CALLS = 100
FAKE_RESERVATION_BASE_URL = "http://fake-reservation.invalid/api/v1"
FAKE_RESERVATION_SECRET = "replay-fake-account-secret"  # 예약 기능 스위치용 가짜 값(실제 비밀값 아님)
CONTEXT_KEYS = ("age", "gender", "wardId", "role", "name", "phone", "address", "location", "birthDate", "wardName", "wardPhone")


class BudgetExceeded(RuntimeError):
    """라우터 호출 상한을 넘었다."""


class RouterCacheMiss(RuntimeError):
    """cache 모드에서 기록되지 않은 라우터 입력."""


class RouterForcedFailure(RuntimeError):
    """--router fail 또는 턴의 route="fail" - 라우터 실패를 일부러 낸다(영역 E, 키워드 폴백 시험)."""


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
        # v7·v4·v5 는 위기 기대값을 crisis_reply 로 적었다 - 같은 뜻으로 읽는다(2026-11-20, 그전에는 v7 위기 문항의 라우터 칸이 늘 false 였다).
        "self_harm": bool(ex.get("crisis")) or ex.get("crisis_reply") is True,
        "cardiac_arrest": bool(ex.get("cpr_aed")),
        "urgent_condition": urgent if isinstance(urgent, str) else None,
        "reason": "stub",
    }


def turn_route(item: dict, turn: dict | None, message: str, mode: str = "stub") -> dict[str, Any]:
    """v8 턴별 라우터 고정: 턴 route 가 "fail" 이거나 --router fail 이면 실패를 내고, dict 면 그 값으로 stub 결과를 만든다.
    턴에 route 가 없으면 예전처럼 문항 expect 로(옛 시험지 동작 그대로)."""
    route = (turn or {}).get("route")
    if mode == "fail" or route == "fail":
        raise RouterForcedFailure("forced")
    if isinstance(route, dict):
        return stub_route({"expect": route}, message)
    return stub_route(item, message)


def resolve_ui(spec: dict, prev: dict | None, today: Any = None) -> dict[str, str]:
    """uiSelection 턴 - value 가 있으면 그대로, pick 이면 직전 응답에서 고른다.
    pick: first | prefix:<값 앞부분> | label:<표시 글자 포함> | hospital:<병원 이름> | date+<N>(오늘부터 N일 뒤 ISO 날짜)."""
    import datetime as _dt

    field = spec["field"]
    if "value" in spec:
        return {"field": field, "value": str(spec["value"])}
    pick = str(spec.get("pick") or "first")
    if pick.startswith("date+"):
        base = today or _dt.date.today()
        return {"field": field, "value": (base + _dt.timedelta(days=int(pick[5:]))).isoformat()}
    if pick.startswith("hospital:"):
        return {"field": field, "value": pick.split(":", 1)[1]}
    ui = (prev or {}).get("ui") or {}
    options = ui.get("options") if isinstance(ui, dict) else getattr(ui, "options", None)
    values = [((o.get("value"), o.get("label")) if isinstance(o, dict) else (o.value, o.label)) for o in options or []]
    if not values:
        raise ValueError(f"no_options:{field}")
    if pick == "first":
        return {"field": field, "value": values[0][0]}
    kind, _, needle = pick.partition(":")
    for value, label in values:
        if (kind == "prefix" and str(value).startswith(needle)) or (kind == "label" and needle in str(label)):
            return {"field": field, "value": value}
    raise ValueError(f"no_match:{field}")


def auto_step(auto: dict, prev: dict | None) -> dict | None:
    """auto 턴 하나 - 직전 응답이 고르게 한 칸(병원 카드면 hospital_confirm)이 plan 에 있으면 그 uiSelection 턴, 없으면 None(끝).
    plan 값은 resolve_ui 의 pick 과 같고, hospital_confirm 의 first 는 카드 첫 병원이다."""
    if prev is None or int(auto.get("max_steps", 12)) <= 0:
        return None
    plan = auto.get("plan") or {}
    ui = prev.get("ui")
    field = (ui.get("field") if isinstance(ui, dict) else getattr(ui, "field", None)) if ui else None
    if prev.get("tool") == "search_hospital" and "hospital_confirm" in plan:
        rows = prev.get("toolData") if isinstance(prev.get("toolData"), list) else []
        pick = plan["hospital_confirm"]
        if pick == "first" and rows:
            pick = f"hospital:{rows[0].get('hospital') or rows[0].get('name')}"
        return {"ui": {"field": "hospital_confirm", "pick": pick}}
    if field and field in plan:
        return {"ui": {"field": field, "pick": plan[field]}}
    return None


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


def normalize_turns(it: dict) -> list[dict]:
    """재생할 턴 목록. v8 은 turns 그대로, 옛 시험지는 history_user + q (턴 route 없음 = 문항 expect)."""
    if it.get("turns"):
        return [dict(t) for t in it["turns"]]
    return [{"message": m} for m in it.get("history_user") or []] + [{"message": it["q"]}]


def load_items(set_path: str, ids: str | None) -> list[dict]:
    p = pathlib.Path(set_path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        items = data["items"] if isinstance(data, dict) else data
    out = []
    want = set(ids.split(",")) if ids else None
    for it in items:
        if want and it["id"] not in want:
            continue
        it = dict(it)
        if "history_user" not in it:
            if it.get("turns"):
                it["history_user"] = [t["message"] for t in it["turns"][:-1] if t.get("message")]
            else:
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
    stub·fail·cache: 호출을 막고 시도를 센다(요약 blockedClaudeCalls - 예약 추출처럼 라우터 밖 호출은 실패한 채로 이어진다).
    live: 단계와 상관없이 호출마다 상한을 쓴다(2026-11-05 점검 M-1 - 라우터만 세던 것)."""
    from app.services import claude_structured_client as csc

    original = csc.ClaudeStructuredClient.generate_structured_text

    def guarded(self, system_prompt, user_prompt, schema=None, *, stage="claude", reserve_sec=0.0, **kwargs):
        if mode != "live":
            stats["blocked"] += 1
            raise csc.StructuredCallError("replay_blocked")
        budget.take()
        stats["claude"] += 1
        return original(self, system_prompt, user_prompt, schema, stage=stage, reserve_sec=reserve_sec, **kwargs)

    csc.ClaudeStructuredClient.generate_structured_text = guarded


def _prepare_env(args: argparse.Namespace) -> None:
    os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='replay_offline_')}/r.db"
    os.environ.setdefault("STREAM_STATE_BACKEND", "memory")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not args.follow_up:
        os.environ["FOLLOW_UP_ENABLED"] = "false"  # 이어서 물어볼 질문(Haiku) 호출 0
    if args.router != "live":
        # stub·fail·cache 는 API 를 부를 수 없게 - 지우지 않고 빈 값으로 넣는다. app.core.config 의 load_dotenv() 는 이미 있는
        # 환경변수를 덮지 않으므로 .env 가 있는 저장소(gosky)에서도 키가 다시 채워지지 않는다(2026-11-05 점검 H-1).
        os.environ["ANTHROPIC_API_KEY"] = ""
        os.environ["CLAUDE_API_KEY"] = ""
    # 예약 서비스 주소·계정 비밀값도 가짜로 고정한다(.env 의 운영 값이 다시 채워지지 않게 - 같은 이유).
    os.environ["RESERVATION_API_BASE_URL"] = FAKE_RESERVATION_BASE_URL
    os.environ["RESERVATION_ACCOUNT_SECRET"] = FAKE_RESERVATION_SECRET


def _summary_row(item: dict, run: int, res: dict | None, err: str | None, miss: bool) -> dict:
    reply = (res or {}).get("reply") or ""
    return {
        "id": item["id"], "run": run, "err": err, "routerCacheMiss": miss,
        "riskLevel": (res or {}).get("riskLevel"), "engine": (res or {}).get("engine"),
        "has119": "119" in reply, "has109": "109" in reply, "warning": bool((res or {}).get("emergencyWarning")),
        "cpr": "가슴압박" in reply, "followup": "여쭤볼게요" in reply, "lines": reply.count("\n") + 1 if reply else 0,
    }


def _upstream_meta(db_factory, log_id: Any) -> dict:
    """챗 기록 upstreamMeta(자해 고정 응답·라우터 실패·되묻기 표시) - 응답 필드에는 없다."""
    if not log_id:
        return {}
    from app.models.chat_log import ChatLog

    db = db_factory()
    try:
        row = db.get(ChatLog, log_id)
        ctx = json.loads(row.context_json or "{}") if row is not None else {}
    except Exception:  # noqa: BLE001
        ctx = {}
    finally:
        db.close()
    meta = ctx.get("upstreamMeta") if isinstance(ctx, dict) else None
    return meta if isinstance(meta, dict) else {}


def turn_features(message: str, res: dict | None, meta: dict, creates: int) -> dict:
    """턴 하나의 본문 없는 특징(rule_grader 가 채점한다)."""
    from app.services.medical_llm_service import MedicalLlmService
    from tools.eval.v8.rule_grader import reply_features

    res = res or {}
    reply = res.get("reply") or ""
    ui = res.get("ui")
    ui_field = (ui.get("field") if isinstance(ui, dict) else getattr(ui, "field", None)) if ui else None
    rules = MedicalLlmService.drug_alcohol_rules(message) if message else []
    sentences = []
    for r in rules:
        texts = [str(r.sentence), getattr(r.sentence, "proxy", None) or ""]
        if any(t and t in reply for t in texts):
            sentences.append(r.name)
    data = res.get("toolData")
    rows = data if isinstance(data, list) else (data or {}).get("rows") if isinstance(data, dict) else None
    return {
        "warning": bool(res.get("emergencyWarning")), "riskLevel": res.get("riskLevel"), "engine": res.get("engine"),
        "selfHarm": bool(meta.get("selfHarm")), "crisisFollowup": bool(meta.get("crisisFollowup")),
        "routerFailed": bool(meta.get("routerFailed")), "followupAsked": bool(meta.get("followupAsked")),
        "generationFailed": bool(meta.get("generationFailed")),
        "uiField": ui_field, "tool": res.get("tool"),
        "cardRegions": [str(r.get("location") or "") for r in rows or [] if isinstance(r, dict)][:6],
        "drugAlcoholRules": [r.name for r in rules], "drugAlcoholSentences": sentences,
        "creates": creates, "features": reply_features(reply),
    }


def reservation_outcome(backend: Any, turns: list[dict], today: Any = None) -> dict:
    """가짜 예약 서비스의 최종 상태 - 생성 수, 확인 버튼 전 생성 수, 최종 예약(병원·과·날짜 차이·시간), 첫 병원 카드 지역."""
    import datetime as _dt

    base = today or _dt.date.today()
    before = sum(t["creates"] for t in turns if not t.get("confirmClick"))
    booked = [r for r in backend.appointments.values() if r["status"] == "BOOKED"]
    final = None
    if booked:
        r = booked[-1]
        try:
            offset = (_dt.date.fromisoformat(r["reservationDate"]) - base).days
        except ValueError:
            offset = None
        final = {"hospital": r["hospitalName"], "department": r["department"], "dateOffset": offset, "time": r["reservationTime"]}
    first_card = next((t["cardRegions"] for t in turns if t.get("tool") == "search_hospital" and t.get("cardRegions")), None)
    return {"creates": backend.creates(), "createsBeforeConfirm": before, "finalRecord": final,
            "firstCardRegion": first_card[0] if first_card else None}


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
    from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection
    import app.services.chat_service as cs
    from tools.eval.v8 import fake_reservation

    try:
        import app.models.reservation_token_claim  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    Base.metadata.create_all(bind=engine)

    hospitals = fake_reservation.load_hospitals(args.fixture or None)
    box = {"backend": fake_reservation.FakeReservationBackend(hospitals)}
    fake_reservation.install(lambda: box["backend"])

    budget = CallBudget(args.max_router_calls)
    stats = {"hit": 0, "miss": 0, "live": 0, "blocked": 0, "claude": 0, "forcedFail": 0}
    guard_claude_calls(args.router, budget, stats)
    current: dict = {}
    current_turn: dict = {}
    cache: dict = {}
    if args.router in ("stub", "fail"):
        class _Stub:
            def route(self, message, *a, **k):
                try:
                    return turn_route(current, current_turn, message, args.router)
                except RouterForcedFailure:
                    stats["forcedFail"] += 1
                    raise
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
                box["backend"] = fake_reservation.FakeReservationBackend(hospitals)
                raw_ctx = {k: it.get(k) for k in ("age", "gender", "wardId", "role") if it.get(k) is not None}
                raw_ctx.setdefault("age", 76)
                raw_ctx.update({k: v for k, v in (it.get("context") or {}).items() if k in CONTEXT_KEYS})
                ctx = ChatContext(**raw_ctx)
                sid, uid = uuid.uuid4().hex[:16], f"rp{run}{n:03d}"
                res, err, miss_before = None, None, stats["miss"]
                turn_rows, turn_bodies = [], []
                queue = normalize_turns(it)
                while queue:
                    turn = queue.pop(0)
                    if turn.get("auto"):
                        # 예약처럼 버튼 순서가 서버 상태에 따라 바뀌는 흐름: 직전 응답이 묻는 칸이 plan 에 있으면 그 값을 고른다.
                        nxt = auto_step(turn["auto"], res)
                        if nxt is None:
                            continue
                        queue.insert(0, dict(turn, auto=dict(turn["auto"], max_steps=int(turn["auto"].get("max_steps", 12)) - 1)))
                        turn = nxt
                    current_turn.clear()
                    current_turn.update(turn)
                    msg = turn.get("message") or ""
                    ui_sel = None
                    if turn.get("ui"):
                        try:
                            ui_sel = resolve_ui(turn["ui"], res)
                        except ValueError as exc:
                            err = f"ui:{exc}"
                            break
                    creates_before = box["backend"].creates()
                    db = SessionLocal()
                    try:
                        res = svc.process_message(db, ChatRequest(
                            userId=uid, message=msg, sessionId=sid, context=ctx,
                            uiSelection=UiSelection(**ui_sel) if ui_sel else None,
                        ))
                    except BudgetExceeded:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        err = type(exc).__name__
                        res = None
                    finally:
                        db.close()
                    meta = _upstream_meta(SessionLocal, (res or {}).get("chatLogId"))
                    row = turn_features(msg, res, meta, box["backend"].creates() - creates_before)
                    row["confirmClick"] = bool(ui_sel and ui_sel["field"] == "reservation_confirm" and str(ui_sel["value"]).startswith("confirm:"))
                    turn_rows.append(row)
                    turn_bodies.append({"reply": (res or {}).get("reply"), "riskLevel": (res or {}).get("riskLevel"),
                                        "emergencyWarning": (res or {}).get("emergencyWarning")})
                    if res is None:
                        break
                miss = stats["miss"] > miss_before
                bodies.append({"id": it["id"], "run": run, "reply": (res or {}).get("reply"), "riskLevel": (res or {}).get("riskLevel"),
                               "emergencyWarning": (res or {}).get("emergencyWarning"), "turns": turn_bodies})
                srow = _summary_row(it, run, res, err, miss)
                srow["turns"] = turn_rows
                srow["reservation"] = reservation_outcome(box["backend"], turn_rows)
                summary.append(srow)
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
            summary_path.write_text(json.dumps({"stats": stats, "router": args.router, "claudeCallsUsed": budget.used,
                                                "blockedClaudeCalls": stats["blocked"], "rows": summary},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"끝: 결과 {len(summary)}건, 라우터 캐시 적중 {stats['hit']}·누락 {stats['miss']}·실제 라우터 {stats['live']}회, "
          f"강제 실패 {stats['forcedFail']}회, Claude 호출 {stats['claude']}회·차단 {stats['blocked']}회", flush=True)
    if stats["blocked"]:
        print("주의: stub·fail·cache 모드에서 Claude 호출 시도가 있었고 막았다(예약 추출 등) - 해당 문항 결과는 라우터 밖 판정이 실패한 상태다")
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
    run.add_argument("--router", choices=("stub", "fail", "cache", "live"), default="stub")
    run.add_argument("--cache-file", default="")
    run.add_argument("--key-file", default="")
    run.add_argument("--max-router-calls", type=int, default=DEFAULT_MAX_ROUTER_CALLS)
    run.add_argument("--follow-up", action="store_true", help="이어서 물어볼 질문(Haiku)도 부른다 - 기본은 끔")
    run.add_argument("--allow-miss", action="store_true", help="cache 모드 누락이 있어도 0 으로 끝낸다")
    run.add_argument("--fixture", default="", help="가짜 예약 서비스 병원 목록(기본 tools/eval/v8/fake_hospitals.json)")
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
