"""말벗 품질 평가 세트(companion_quality_v1, 37대화·142턴) 재생기 - DGX 전용, 실제 모델 (2026-11-20).

운영 코드(app/)의 글 대답 경로를 그대로 탄다: reply_text 엔드포인트와 같은 `_run_turn` → `_turn_response`.
표정 없음(글 대답), 음성 합성은 끈다. 결과 run.json 은 tools/eval/score_companion_quality.py 입력 형식이다.
외부 API(Anthropic 등)를 부르지 않는다.

실행 (DGX, 저장소 사본 루트에서):
  MODEL_BASE_PATH=<companion/llm/... 가 있는 폴더> COMPANION_REPLY_HISTORY=summary \
    python tools/eval/companion_quality_dgx_run.py <run.json> --label summary
  python tools/eval/score_companion_quality.py <run.json> <out_dir>

- 측정에서만 시간 제한을 넉넉히 둔다 (DGX 를 다른 작업과 같이 쓰면 시간 초과로 대체 문장이 섞여 비교가 흐려진다).
  대체 문장 수는 run.json 의 fallbackReplies 로 남긴다.
- run.json 에는 대답 본문이 들어간다. 저장소 밖에 두고, 확인이 끝나면 지운다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for k, v in {"COMPANION_LLM_TIMEOUT_SEC": "180", "COMPANION_TEXT_TIMEOUT_SEC": "180", "COMPANION_TURN_CAP_SEC": "600",
             "COMPANION_LLM_WARMUP": "false",
             # 검수 목록(review_store)은 측정마다 버리는 임시 SQLite 에 쓴다 (저장소 data/ 에 남기지 않는다)
             "COMPANION_REVIEW_DATABASE_URL": f"sqlite:///{tempfile.mkdtemp(prefix='cq-review-')}/review.db"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.routers import companion_router as cr  # noqa: E402
from app.services.companion.tts_service import tts_service  # noqa: E402
from app.services.companion.dialogue_store import store  # noqa: E402

DEFAULT_SET = ROOT / "tests" / "companion_eval" / "companion_quality_v1.json"


def _mood(text_mood: dict | None) -> str:
    """보고되는 글 기분(textMood.label, 회피 보정 뒤). 쓸 수 없는 결과면 UNKNOWN."""
    if text_mood and text_mood.get("usable") and text_mood.get("label"):
        return text_mood["label"]
    return "UNKNOWN"


def run(set_path: Path, label: str) -> dict:
    s = get_settings()
    tts_service.start = lambda *a, **k: False  # 음성 합성은 측정 대상이 아니다
    classify_text, llm = cr.get_text_classifier(), cr.get_llm()
    data = json.loads(set_path.read_text(encoding="utf-8"))
    runs, sources, t0 = [], Counter(), time.time()
    for n, sc in enumerate(data["scenarios"]):
        d = store.create(f"eval_{sc['id']}", f"eval_{n}", sc["slot"], sc["opening"], s.companion_max_turns)
        turns = []
        for gold in sc["turns"]:
            if store.get(d.dialogueId).ended:
                break
            d = store.get(d.dialogueId)
            result = cr._run_turn(d, gold["elder"], cr._no_face(), classify_text, llm, input_kind="text")
            out = cr._turn_response(d, result)
            sources[result.get("replySource") or "?"] += 1
            reply = out.get("reply") or {}
            turns.append({"elder": gold["elder"], "reply": reply.get("text") or "",
                          "isFollowUp": bool(reply.get("isFollowUp")), "riskFlag": bool(out.get("riskFlag")),
                          "riskCarry": bool(out.get("riskCarry")), "mood": _mood(out.get("textMood")), "moodMixed": None,
                          "ended": bool(out.get("ended")), "endReason": out.get("endReason"),
                          "replySource": result.get("replySource")})
        runs.append({"id": sc["id"], "turns": turns})
        print(f"{sc['id']:5s} {len(turns)}/{len(sc['turns'])}턴 {time.time() - t0:6.0f}s", flush=True)
    return {"set": data["version"], "label": label, "replyHistory": s.companion_reply_history,
            "replySources": dict(sources), "fallbackReplies": sources.get("FALLBACK", 0),
            "seconds": round(time.time() - t0), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    result = run(a.set, a.label or get_settings().companion_reply_history)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("replySources", result["replySources"], "seconds", result["seconds"])


if __name__ == "__main__":
    main()
