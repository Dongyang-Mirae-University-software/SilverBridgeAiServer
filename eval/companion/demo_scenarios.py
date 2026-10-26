"""데모 시나리오 3개를 실제 API(api_mock)로 돌려서 결과를 기록한다 (계획서 Phase 8, 8.2).

- 가상 피보호자 DEMO-WARD. 기록은 임시 폴더에 쓰고, 끝나면 버린다 (data/api_mock 은 건드리지 않음).
- 텍스트 감정은 실제 모델(GGARA02)을 쓴다. 얼굴 확인은 AI 서버에 연결하지 않는다:
  1) 정상 하루: 5개 시각 모두 정상 응답 → 하루 결과
  2) 부정·외로움 흐름: 말투에서 부정·외로움이 보이는 대답 → 확인 필요, 애매한 대답 → 판단 보류
  3) 실패 흐름: 12시 분석 오류 → '분석 실패'로 기록, 15시 정상 / 얼굴 서버 연결 불가 → 대답은 저장, 얼굴은 '확인 불가'
결과: logs/demo_scenarios.json (사람 정보·대답 원문은 임시 폴더에만 남고 결과 파일에는 라벨만 기록)
실행 (저장소 루트에서, 키는 .env 의 COMPANION_API_KEY 를 쓴다):
    venv\\Scripts\\python.exe -u eval\\demo_scenarios.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
sys.path.insert(0, str(ROOT / "app" / "companion"))

from dotenv import dotenv_values  # noqa: E402

env = dotenv_values(ROOT / ".env")
os.environ["COMPANION_API_KEY"] = env.get("COMPANION_API_KEY") or "demo-key"
os.environ["COMPANION_STREAM_BASE_URL"] = "http://127.0.0.1:9"  # 닫힌 포트: 얼굴 서버 연결 불가 상황을 흉내
os.environ["COMPANION_STREAM_API_KEY"] = "demo-stream-key"

import api_mock as api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

WARD = "DEMO-WARD"
SLOTS = ["09:00", "12:00", "15:00", "18:00", "21:00"]
QUESTIONS = {
    "09:00": "오늘 아침에 편안하게 주무셨나요?",
    "12:00": "점심은 드셨어요? 무엇을 드셨는지 궁금해요.",
    "15:00": "오후에는 무엇을 하며 지내셨나요?",
    "18:00": "오늘 하루는 어떠셨어요?",
    "21:00": "오늘 하루 편안하게 보내셨나요?",
}

tmp = Path(tempfile.mkdtemp(prefix="companion_demo_"))
api.ANSWERS_FILE = tmp / "answers.json"
api.QUESTIONS_FILE = tmp / "guardian_questions.json"
api.QUESTIONS_FILE.write_text(json.dumps({f"{WARD}|{s}": q for s, q in QUESTIONS.items()}, ensure_ascii=False), encoding="utf-8")

H = {"X-API-Key": os.environ["COMPANION_API_KEY"]}
client = TestClient(api.app)
client.__enter__()  # 서버 시작과 같이 lifespan 실행 (텍스트 모델 미리 읽기)
today = datetime.now().date().isoformat()


def answer(slot: str, text: str, face_sec: int = 0) -> dict:
    body = {"wardId": WARD, "sessionId": f"demo-{slot}", "slot": slot, "text": text}
    if face_sec:
        body["faceCheckSec"] = face_sec
    r = client.post("/companion/answers", json=body, headers=H)
    data = r.json()["data"]
    return {"slot": slot, "http": r.status_code, "label": data["label"], "reason": data["reason"],
            "face": data.get("face"), "followUp": bool(data.get("followUp"))}


def day() -> str:
    return client.get(f"/companion/emotions?wardId={WARD}&date={today}", headers=H).json()["data"]["dayResult"]


results = {}

# 1) 정상 하루
steps = [answer("09:00", "잘 잤어요, 기분이 좋아요"), answer("12:00", "손주가 와서 같이 점심 먹었어요"),
         answer("15:00", "정원에서 꽃을 봤어요, 날씨가 좋았어요"), answer("18:00", "저녁 잘 먹었어요"),
         answer("21:00", "오늘 하루 편안하게 보냈어요")]
results["1_normal_day"] = {"steps": steps, "dayResult": day()}

# 2) 부정·외로움 흐름 (다른 피보호자로 분리)
WARD = "DEMO-WARD-2"
steps = [answer("09:00", "영감이 떠난 뒤로 집이 너무 조용해요"), answer("12:00", "글쎄요 잘 모르겠어요"),
         answer("15:00", "기분이 좋지 않아요, 답답해요")]
results["2_negative_and_loneliness"] = {"steps": steps, "dayResult": day()}

# 3) 실패 흐름
WARD = "DEMO-WARD-3"
original = api.text_label


def broken(_text):
    raise RuntimeError("demo failure")


api.text_label = broken
failed = answer("12:00", "점심 잘 먹었어요")
api.text_label = original
steps = [answer("09:00", "잘 잤어요"), failed, answer("15:00", "오후에 산책했어요", face_sec=1),
         answer("18:00", "저녁 잘 먹었어요")]
results["3_failure_and_face_unreachable"] = {"steps": steps, "dayResult": day()}

client.__exit__(None, None, None)

out_dir = ROOT / "logs"
out_dir.mkdir(exist_ok=True)
(out_dir / "demo_scenarios.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

for name, res in results.items():
    print(name, "→ 하루 결과:", res["dayResult"])
    for s in res["steps"]:
        print("   ", s["slot"], s["http"], s["label"], "| 얼굴:", (s["face"] or {}).get("faceStatus", "-"))
print("결과 저장: logs/demo_scenarios.json")
