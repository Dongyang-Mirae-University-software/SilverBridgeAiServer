"""확인 필요 턴의 사진이 응답에 실려 오는지 확인한다 (DL.8).

- 임시 보호자 ID(CHECK-PHOTO)로 질문과 대화를 만들고, 부정적인 대답 + 화난 얼굴 값을 보낸다 → 확인 필요가 나와야 함
- 동의 켜짐(consentPhoto=true)으로 보내면 응답의 photoForBackend 에 사진이 있어야 하고,
  끄면(false) 없어야 한다.
- 받은 사진은 logs\\photo_check_on.jpg 로 저장해서 직접 열어 볼 수 있다 (logs 폴더는 git에 올라가지 않음).
- 사진은 테스트용으로 만든 빈 이미지다. 실제 사람 얼굴을 쓰지 않는다.

실행 (서버가 켜진 상태에서, 저장소 루트에서):
    venv\\Scripts\\python.exe -u eval\\check_photo_response.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import urllib.request
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
BASE = "http://127.0.0.1:8765"
WARD = "CHECK-PHOTO"
KEY = dotenv_values(ROOT / ".env")["COMPANION_API_KEY"]
HDR = {"X-API-Key": KEY}


def blank_jpeg_b64() -> str:
    img = np.full((120, 160, 3), 120, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, method=method, data=data,
                                 headers={**HDR, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def reply(did: str, consent: bool, photo: str) -> dict:
    body = {"text": "영감이 떠난 뒤로 집이 너무 조용해요", "face": {"label": "Anger", "score": 0.9},
            "consentPhoto": consent, "photo": photo}
    return call("POST", f"/companion/dialogues/{did}/reply", body)["data"]


def main() -> None:
    call("PUT", f"/companion/questions/09:00?wardId={WARD}", {"question": "오늘 아침 편안하게 주무셨나요?"})
    photo = blank_jpeg_b64()

    did = call("POST", "/companion/dialogues", {"wardId": WARD, "sessionId": "check", "slot": "09:00"})["data"]["dialogueId"]
    on = reply(did, True, photo)
    print("동의 켜짐: 결과 =", on["label"], "| 사진 있음 =", bool(on.get("photoForBackend")))
    if on.get("photoForBackend"):
        out = ROOT / "logs" / "photo_check_on.jpg"
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(base64.b64decode(on["photoForBackend"]))
        print("저장 위치:", out)

    did2 = call("POST", "/companion/dialogues", {"wardId": WARD, "sessionId": "check2", "slot": "09:00"})["data"]["dialogueId"]
    off = reply(did2, False, photo)
    print("동의 꺼짐: 결과 =", off["label"], "| 사진 있음 =", bool(off.get("photoForBackend")))

    # 정리: 이 확인으로 만든 기록만 지운다
    data = ROOT / "data" / "api_mock"
    rows_path = data / "answers.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        rows_path.write_text(json.dumps([r for r in rows if r.get("wardId") != WARD], ensure_ascii=False, indent=2), encoding="utf-8")
    q_path = data / "guardian_questions.json"
    if q_path.exists():
        qs = json.loads(q_path.read_text(encoding="utf-8"))
        q_path.write_text(json.dumps({k: v for k, v in qs.items() if not k.startswith(WARD + "|")}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
