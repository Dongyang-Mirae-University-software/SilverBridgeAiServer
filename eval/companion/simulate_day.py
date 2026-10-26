"""하루 시뮬레이션 + 실패 처리 확인 (계획서 Phase 7, 7.4 / 7.완료a·c).

1) 가상의 피보호자(SIM-TEST) 하루 5개 시각(09·12·15·18·21시)에 대답을 넣고 하루 결과가 나오는지 본다.
2) 모델이 오류를 내는 상황을 흉내 내서 '분석 실패'로 기록되고, 다음 시각에 영향이 없는지 본다.
3) 테스트로 넣은 기록은 끝나면 지운다 (다른 피보호자 데이터는 건드리지 않음).

실행 (저장소 루트에서):
    venv\\Scripts\\python.exe eval\\simulate_day.py
결과: logs\\day_simulation.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
sys.path.insert(0, str(ROOT / "app" / "companion"))
import api_mock as api  # noqa: E402

WARD = "SIM-TEST"
DAY_ANSWERS = {
    "09:00": "잘 잤어요, 기분이 좋아요",
    "12:00": "손주가 와서 같이 점심 먹었어요",
    "15:00": "글쎄요 잘 모르겠어요",
    "18:00": "영감이 떠난 뒤로 집이 너무 조용해요",
    "21:00": "오늘 하루 편안하게 보냈어요",
}


def main() -> None:
    results = {"day": [], "failure_check": {}}
    # 1) 하루 시뮬레이션
    for slot, text in DAY_ANSWERS.items():
        out = api._store_answer(WARD, f"sim-{slot}", slot, text, None)
        results["day"].append({"slot": slot, "label": out["data"]["label"], "reason": out["data"]["reason"],
                               "followUp": out["data"].get("followUp")})
    today = datetime.now().date().isoformat()
    day = api.get_emotions(wardId=WARD, date=today)["data"]
    results["day_result"] = day["dayResult"]

    # 2) 실패 처리: 모델이 오류를 내는 상황
    original = api.text_label

    def broken(_text):
        raise RuntimeError("simulated model failure")

    api.text_label = broken
    try:
        fail = api._store_answer(WARD, "sim-fail", "15:00", "오늘 기분이 좋아요", None)
    finally:
        api.text_label = original
    results["failure_check"]["failed_slot_label"] = fail["data"]["label"]
    results["failure_check"]["failed_slot_reason"] = fail["data"]["reason"]

    # 실패 뒤 다음 시각은 정상 동작해야 한다
    nxt = api._store_answer(WARD, "sim-next", "18:00", "저녁 잘 먹었어요", None)
    results["failure_check"]["next_slot_label"] = nxt["data"]["label"]
    day2 = api.get_emotions(wardId=WARD, date=today)["data"]
    results["failure_check"]["day_result_after_failure"] = day2["dayResult"]

    # 정리: 테스트 기록 삭제 (다른 피보호자 기록은 그대로)
    rows = api._load(api.ANSWERS_FILE, [])
    api._save(api.ANSWERS_FILE, [r for r in rows if r.get("wardId") != WARD])

    LOGS = ROOT / "logs"
    LOGS.mkdir(exist_ok=True)
    (LOGS / "day_simulation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
