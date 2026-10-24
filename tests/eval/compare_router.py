#!/usr/bin/env python3
"""라우터·예약 추출 Claude 모델 오프라인 비교 (2026-10-24 O-2).

AI 서버 HTTP를 거치지 않고 라우터 함수(ChatRouterService.route)와 예약 추출 호출에 시험 문장을 직접 넣는다.
예약 API는 부르지 않는다(추출 JSON만 비교) - 예약이 생기지 않는다.

- 라우터: 시험지(--set, 기본 eval_set_v3.json 90문항) × 모델 × 회차.
  route·emergency 의 Opus 대비 일치율(Opus 다수결 값 기준), expect.route 합격률(라우터 원값 / 운영과 같은 예약 고정·키워드 응급 강제 적용값),
  응급 문항(expect.emergency=true, 자해 포함) emergency 누락(라우터 원값 / 키워드 또는 라우터 최종 판정),
  자해 문항(sh.*)의 위기 응답 근거(ChatService._self_harm_trigger: keyword·router·history·없음), 호출 시간(평균·p95), 스키마·파싱 실패 수.
- 예약 추출: 아래 가상 예약 문장 20개 × 모델 × 회차, 칸별 Opus 다수결 값 대비 일치율.

결과 파일에는 문항 id·수치·예외 클래스명만 남긴다(사용자 문장·모델 출력 금지).
키는 컨테이너 환경변수 ANTHROPIC_API_KEY 로만 받고 출력하지 않는다.

사용 예 (gosky, 운영 이미지 임시 컨테이너 - AI 서버 재시작 없음):
  docker run --rm -e PYTHONPATH=/w -e ANTHROPIC_API_KEY="$(docker exec silverbridge-ai-server printenv ANTHROPIC_API_KEY)" \\
    -v ~/tmp-router-wt:/w -w /w --entrypoint python silverbridgeaiserver-ai-server \\
    tests/eval/compare_router.py --out /w/out/compare.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.reservation_prompts import EXTRACTION_SCHEMA, SYSTEM_PROMPT as EXTRACT_PROMPT, build_user_prompt  # noqa: E402
from app.services.chat_router_service import ChatRouterService  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.claude_structured_client import ClaudeStructuredClient, failure_cause  # noqa: E402
from app.services.medical_llm_service import MedicalLlmService  # noqa: E402
from app.utils.json_extract import extract_json_object  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
BASELINE = "claude-opus-5-5"
DEFAULT_MODELS = (BASELINE, "claude-sonnet-5-5", "claude-haiku-5-5")
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_SET = "eval_set_v3.json"
# 날짜 해석이 실행일마다 달라지지 않게 고정한다.
FIXED_TODAY = "2026-10-24 (토요일)"
EXTRACT_FIELDS = ("intent", "hospital_name", "department", "location", "reservation_date", "reservation_time", "needs_search")

# 가상 예약 문장(실제 사용자 기록 아님). 날짜·시간·진료과·병원명·지역 조합.
RESERVATION_CASES = [
    ("rx.01", "내일 오전 10시에 서울 강남구 이비인후과 예약해 줘"),
    ("rx.02", "다음 주 월요일 오후 2시 반에 송파구 내과로 예약하고 싶어요"),
    ("rx.03", "서울 마포구 튼튼정형외과에 모레 3시 예약 부탁해요"),
    ("rx.04", "판교 근처 피부과 찾아 줘"),
    ("rx.05", "10월 30일 오전 9시 안과 예약"),
    ("rx.06", "어머니 무릎이 아파서 정형외과 예약하려고 해요. 구로구로 금요일 오후에"),
    ("rx.07", "인천 연수구 바른내과 다음 주 화요일 11시"),
    ("rx.08", "치과 말고 이비인후과요, 강서구에서 내일 오후 4시"),
    ("rx.09", "부산 해운대구 가정의학과 이번 주 토요일 오전 10시 예약해 줘"),
    ("rx.10", "11월 3일 오후 1시에 서울대병원 신경과 예약할 수 있어?"),
    ("rx.11", "목이 아프고 열이 나는데 내일 아침 일찍 동작구 병원 예약해 줘"),
    ("rx.12", "성남시 분당구 맑은안과에 다음 주 수요일 오후 3시"),
    ("rx.13", "대구 수성구 산부인과 찾고 싶어요"),
    ("rx.14", "모레 오전 11시 반 노원구 행복정신건강의학과 예약"),
    ("rx.15", "경기 수원시 영통구 소아청소년과 오늘 오후 5시 가능해?"),
    ("rx.16", "강남역 근처 내과 아무 데나 내일 9시 반"),
    ("rx.17", "광주 북구 우리피부과 다음 달 2일 오후 2시"),
    ("rx.18", "서울 종로구 이비인후과 금요일 오전으로 예약 잡아 줘"),
    ("rx.19", "용산구 한빛정형외과 11월 5일 10시 예약하고 싶어"),
    ("rx.20", "대전 유성구 외과 다음 주 목요일 오후 4시 반"),
]


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def _norm(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return "".join(str(value or "").split()).lower()


def _majority(values: list) -> object:
    present = [v for v in values if v is not None]
    return Counter(present).most_common(1)[0][0] if present else None


def run_router(model: str, items: list[dict], reps: int, workers: int) -> list[dict]:
    svc = ChatRouterService(ClaudeStructuredClient(get_settings(), model), get_settings().chat_history_keep_turns)

    def one(job: tuple[dict, int]) -> dict:
        item, rep = job
        history = [{"role": h["role"], "content": h["content"]} for h in (item.get("history") or [])]
        started = time.perf_counter()
        rec: dict = {"id": item["id"], "rep": rep}
        try:
            out = svc.route(item["q"], history, {})
            rec.update(ok=True, route=out["route"], emergency=bool(out["emergency"]), self_harm=bool(out.get("self_harm")))
            # 운영(chat_service)과 같은 순서: 라우터 뒤 예약 고정 → 키워드 응급 강제 → 자해 근거 판정. 요약 문구는 저장하지 않는다.
            # (위기 뒤 예약 재개는 서버 챗 기록이 필요해 여기서는 재지 않는다.)
            final = ChatService._force_emergency(ChatService._apply_router_overrides(out, item["q"]), item["q"])
            keyword = bool(MedicalLlmService.self_harm_signals(item["q"]))
            rec["self_harm_trigger"] = ChatService._self_harm_trigger(keyword, final, history)
            rec["eff_emergency"] = bool(final["emergency"])
            rec["eff_route"] = final["route"]
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, error=failure_cause(exc))
        rec["ms"] = round((time.perf_counter() - started) * 1000)
        rec["kw_emergency"] = bool(MedicalLlmService.emergency_signals(item["q"]) or MedicalLlmService.self_harm_signals(item["q"]))
        return rec

    jobs = [(item, rep) for rep in range(1, reps + 1) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def run_extract(model: str, reps: int, workers: int) -> list[dict]:
    client = ClaudeStructuredClient(get_settings(), model)

    def one(job: tuple[tuple[str, str], int]) -> dict:
        (case_id, message), rep = job
        prompt = build_user_prompt(
            message=message,
            history_text="(없음)",
            state_json='{"facts":{},"intent":null,"history":[],"resolved":[]}',
            location="",
            profile_text="(없음)",
            today=FIXED_TODAY,
        )
        started = time.perf_counter()
        rec: dict = {"id": case_id, "rep": rep}
        try:
            parsed = extract_json_object(client.generate_structured_text(EXTRACT_PROMPT, prompt, schema=EXTRACTION_SCHEMA))
            if not isinstance(parsed, dict):
                raise ValueError("parse")
            rec.update(ok=True, fields={f: _norm(parsed.get(f)) for f in EXTRACT_FIELDS})
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, error=type(exc).__name__)
        rec["ms"] = round((time.perf_counter() - started) * 1000)
        return rec

    jobs = [(case, rep) for rep in range(1, reps + 1) for case in RESERVATION_CASES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def summarize_router(rows: list[dict], items: list[dict], base_major: dict) -> dict:
    by_id = {i["id"]: i for i in items}
    ok = [r for r in rows if r["ok"]]
    ms = [r["ms"] for r in ok]
    route_cmp = [r for r in ok if base_major.get(r["id"], {}).get("route") is not None]
    exp = [r for r in ok if by_id[r["id"]]["expect"].get("route")]
    emerg_items = {i["id"] for i in items if i["expect"].get("emergency") is True}
    self_harm = {i["id"] for i in items if i["id"].startswith("sh.")}
    emerg_rows = [r for r in rows if r["id"] in emerg_items]
    miss_raw = sorted({f'{r["id"]}#{r["rep"]}' for r in emerg_rows if not (r["ok"] and r["emergency"])})
    miss_eff = sorted({f'{r["id"]}#{r["rep"]}' for r in emerg_rows if not (r["ok"] and r["eff_emergency"]) and not r["kw_emergency"]})
    route_fail = sorted({f'{r["id"]}#{r["rep"]}' for r in exp if r["route"] != by_id[r["id"]]["expect"]["route"]})
    return {
        "calls": len(rows),
        "failures": len(rows) - len(ok),
        "failureTypes": dict(Counter(r["error"] for r in rows if not r["ok"])),
        "msAvg": round(statistics.mean(ms)) if ms else None,
        "msMedian": round(statistics.median(ms)) if ms else None,
        "msP95": _p95(ms),
        "routeAgreeVsOpus": round(sum(r["route"] == base_major[r["id"]]["route"] for r in route_cmp) / len(route_cmp), 4) if route_cmp else None,
        "emergencyAgreeVsOpus": round(sum(r["emergency"] == base_major[r["id"]]["emergency"] for r in route_cmp) / len(route_cmp), 4) if route_cmp else None,
        "expectRoutePass": f'{sum(r["route"] == by_id[r["id"]]["expect"]["route"] for r in exp)}/{len(exp)}',
        "expectRoutePassEffective": f'{sum(r["eff_route"] == by_id[r["id"]]["expect"]["route"] for r in exp)}/{len(exp)}',
        "expectRouteFailIds": route_fail,
        "emergencyRows": len(emerg_rows),
        "emergencyMissRouter": len(miss_raw),
        "emergencyMissRouterIds": miss_raw,
        # 키워드 강제로도 못 받쳐 주는(=실제로 응급 안내가 빠지는) 누락
        "emergencyMissAfterKeyword": len(miss_eff),
        "emergencyMissAfterKeywordIds": miss_eff,
        "selfHarmRouterEmergency": f'{sum(1 for r in emerg_rows if r["id"] in self_harm and r["ok"] and r["emergency"])}/{sum(1 for r in emerg_rows if r["id"] in self_harm)}',
        # 위기 고정 응답 근거(None = 위기 응답이 나가지 않음). 문항별 회차 순서.
        "selfHarmTrigger": {
            sid: [r.get("self_harm_trigger") if r["ok"] else "error" for r in sorted((x for x in rows if x["id"] == sid), key=lambda x: x["rep"])]
            for sid in sorted(self_harm & emerg_items)
        },
        "selfHarmNoCrisis": sorted({f'{r["id"]}#{r["rep"]}' for r in emerg_rows if r["id"] in self_harm and not (r["ok"] and r.get("self_harm_trigger"))}),
        # 라우터 self_harm 칸 원값(자해 문항, 회차 순서)
        "selfHarmFlag": {
            sid: [r.get("self_harm") if r["ok"] else "error" for r in sorted((x for x in rows if x["id"] == sid), key=lambda x: x["rep"])]
            for sid in sorted(self_harm & emerg_items)
        },
        # 응급이 아니어야 하는 문항(expect.emergency=false)에 위기 고정 응답이 나가는 경우 = 오탐
        "falseCrisis": sorted({f'{r["id"]}#{r["rep"]}' for r in rows if r["ok"] and r.get("self_harm_trigger") and by_id[r["id"]]["expect"].get("emergency") is False}),
        "falseSelfHarmFlag": sorted({f'{r["id"]}#{r["rep"]}' for r in rows if r["ok"] and r.get("self_harm") and by_id[r["id"]]["expect"].get("emergency") is False}),
        # 문항별 원값(문장·요약 없음) - 회차별 흔들림을 나중에 다시 보려고 남긴다.
        "rows": [
            {k: r.get(k) for k in ("id", "rep", "ok", "error", "route", "emergency", "self_harm", "eff_route", "eff_emergency", "self_harm_trigger")}
            for r in sorted(rows, key=lambda x: (x["id"], x["rep"]))
        ],
    }


def summarize_extract(rows: list[dict], base_major: dict) -> dict:
    ok = [r for r in rows if r["ok"]]
    ms = [r["ms"] for r in ok]
    per_field = {}
    mismatch: dict[str, list[str]] = {}
    for f in EXTRACT_FIELDS:
        cmp = [r for r in ok if base_major.get(r["id"], {}).get(f) is not None]
        same = [r for r in cmp if r["fields"][f] == base_major[r["id"]][f]]
        per_field[f] = round(len(same) / len(cmp), 4) if cmp else None
        mismatch[f] = sorted({f'{r["id"]}#{r["rep"]}' for r in cmp if r not in same})
    cells = [(r, f) for r in ok for f in EXTRACT_FIELDS if base_major.get(r["id"], {}).get(f) is not None]
    overall = sum(r["fields"][f] == base_major[r["id"]][f] for r, f in cells) / len(cells) if cells else None
    return {
        "calls": len(rows),
        "failures": len(rows) - len(ok),
        "failureTypes": dict(Counter(r["error"] for r in rows if not r["ok"])),
        "msAvg": round(statistics.mean(ms)) if ms else None,
        "msP95": _p95(ms),
        "fieldAgreeVsOpus": per_field,
        "overallAgreeVsOpus": round(overall, 4) if overall is not None else None,
        "mismatchIds": {k: v for k, v in mismatch.items() if v},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--set", default=DEFAULT_SET, help="tests/eval 아래 시험지 파일 이름")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not get_settings().anthropic_api_key.strip():
        print("ANTHROPIC_API_KEY 가 없습니다.", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if models[0] != BASELINE:
        models = [BASELINE] + [m for m in models if m != BASELINE]
    set_path = EVAL_DIR / args.set
    items = json.loads(set_path.read_text(encoding="utf-8"))["items"]

    router_rows: dict[str, list[dict]] = {}
    extract_rows: dict[str, list[dict]] = {}
    for model in models:
        print(f"[router] {model} ...", flush=True)
        router_rows[model] = run_router(model, items, args.reps, args.workers)
        if not args.skip_extract:
            print(f"[extract] {model} ...", flush=True)
            extract_rows[model] = run_extract(model, args.reps, args.workers)

    base_router = {
        i["id"]: {
            "route": _majority([r["route"] for r in router_rows[BASELINE] if r["id"] == i["id"] and r["ok"]]),
            "emergency": _majority([r["emergency"] for r in router_rows[BASELINE] if r["id"] == i["id"] and r["ok"]]),
        }
        for i in items
    }
    base_extract = {
        cid: {f: _majority([r["fields"][f] for r in extract_rows.get(BASELINE, []) if r["id"] == cid and r["ok"]]) for f in EXTRACT_FIELDS}
        for cid, _ in RESERVATION_CASES
    }
    result = {
        "createdAt": datetime.now(KST).isoformat(timespec="seconds"),
        "set": set_path.name,
        "reps": args.reps,
        "workers": args.workers,
        "baseline": BASELINE,
        "note": "Opus 대비 일치율은 Opus 3회 다수결 값 기준(Opus 자기 일치율 = 회차 간 흔들림). 문장·모델 출력은 저장하지 않음.",
        "router": {m: summarize_router(rows, items, base_router) for m, rows in router_rows.items()},
        "extract": {m: summarize_extract(rows, base_extract) for m, rows in extract_rows.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for m in models:
        r = result["router"][m]
        print(m, "router", {k: r[k] for k in ("msAvg", "msP95", "failures", "routeAgreeVsOpus", "emergencyAgreeVsOpus", "expectRoutePass", "emergencyMissRouter", "emergencyMissAfterKeyword", "selfHarmRouterEmergency", "selfHarmNoCrisis", "falseCrisis")})
        if m in result["extract"]:
            e = result["extract"][m]
            print(m, "extract", {k: e[k] for k in ("msAvg", "msP95", "failures", "overallAgreeVsOpus")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
