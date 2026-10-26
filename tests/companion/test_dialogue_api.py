"""대화 API(api_mock.py)의 시작·텍스트 대답·음성 대답·종료 테스트. 모델과 AI 서버 없이 돌아간다.

실행 (companion-emotion 폴더에서):
    venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

COMP = Path(__file__).resolve().parents[2] / "app" / "companion"
sys.path.insert(0, str(COMP))
import api_mock  # noqa: E402
import dialogue as dlg  # noqa: E402

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
WARD = "W-DLG"


def wav_bytes(seconds: float = 1.0) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype="float32"), 16000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPANION_API_KEY", KEY)
    monkeypatch.setattr(api_mock, "ANSWERS_FILE", tmp_path / "answers.json")
    monkeypatch.setattr(api_mock, "QUESTIONS_FILE", tmp_path / "questions.json")
    monkeypatch.setattr(api_mock, "DIALOGUES", {})
    monkeypatch.setattr(api_mock, "REVIEW_FILE", tmp_path / "review_replies.json")
    (tmp_path / "questions.json").write_text(json.dumps({f"{WARD}|09:00": "오늘 아침 편안하셨나요?"}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_mock, "text_label", lambda text, face=None: {"textEmotion": "기쁨", "textScore": 0.9, "label": "기쁨", "reason": "테스트"})
    monkeypatch.setattr(dlg, "generate_reply", lambda q, a, i: {"reply": "그러셨군요. 무엇을 하셨나요?", "source": "model"})
    return TestClient(api_mock.app), tmp_path


def start(client):
    r = client.post("/companion/dialogues", json={"wardId": WARD, "sessionId": "s1", "slot": "09:00"}, headers=HEADERS)
    assert r.status_code == 200
    return r.json()["data"]["dialogueId"]


def test_text_turn_returns_next_reply_and_stores_turn(client):
    c, tmp = client
    did = start(c)
    r = c.post(f"/companion/dialogues/{did}/reply", json={"text": "잘 잤어요"}, headers=HEADERS)
    data = r.json()["data"]
    assert data["label"] == "기쁨"
    assert data["reply"] == "그러셨군요. 무엇을 하셨나요?"
    assert data["done"] is False
    rows = json.loads((tmp / "answers.json").read_text(encoding="utf-8"))
    assert rows[-1]["dialogueId"] == did and rows[-1]["turn"] == 1


def test_audio_turn_uses_transcript_and_same_flow(client, monkeypatch):
    c, tmp = client
    monkeypatch.setattr(api_mock, "_transcribe", lambda raw: ("손주가 와서 좋았어요", 1.0))
    did = start(c)
    r = c.post(f"/companion/dialogues/{did}/reply/audio", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["transcript"] == "손주가 와서 좋았어요"
    assert data["audioSeconds"] == 1.0
    rows = json.loads((tmp / "answers.json").read_text(encoding="utf-8"))
    assert rows[-1]["text"] == "손주가 와서 좋았어요"


def test_audio_rejects_non_wav_and_empty(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(api_mock, "_transcribe", lambda raw: ("x", 1.0))
    did = start(c)
    bad = c.post(f"/companion/dialogues/{did}/reply/audio", files={"audio": ("a.txt", b"abc", "text/plain")}, headers=HEADERS)
    assert bad.status_code == 422
    empty = c.post(f"/companion/dialogues/{did}/reply/audio", files={"audio": ("a.wav", b"", "audio/wav")}, headers=HEADERS)
    assert empty.status_code == 422


def test_audio_when_nothing_recognized(client, monkeypatch):
    c, _ = client
    from fastapi import HTTPException

    def silent(raw):
        raise HTTPException(status_code=422, detail="SPEECH_NOT_RECOGNIZED")

    monkeypatch.setattr(api_mock, "_transcribe", silent)
    did = start(c)
    r = c.post(f"/companion/dialogues/{did}/reply/audio", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["message"] == "SPEECH_NOT_RECOGNIZED"


def test_finish_then_reply_is_rejected(client):
    c, _ = client
    did = start(c)
    c.post(f"/companion/dialogues/{did}/reply", json={"text": "잘 잤어요"}, headers=HEADERS)
    fin = c.post(f"/companion/dialogues/{did}/finish", headers=HEADERS).json()["data"]
    assert fin["sessionLabel"] == "기쁨"
    assert c.post(f"/companion/dialogues/{did}/reply", json={"text": "더"}, headers=HEADERS).status_code == 409
    assert c.post(f"/companion/dialogues/{did}/reply/audio", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, headers=HEADERS).status_code == 409


def test_unknown_dialogue_and_missing_question(client):
    c, _ = client
    assert c.post("/companion/dialogues/nope/reply", json={"text": "x"}, headers=HEADERS).status_code == 404
    r = c.post("/companion/dialogues", json={"wardId": "NOBODY", "sessionId": "s", "slot": "09:00"}, headers=HEADERS)
    assert r.status_code == 404


def test_requires_key(client):
    c, _ = client
    assert c.post("/companion/dialogues", json={"wardId": WARD, "sessionId": "s", "slot": "09:00"}).status_code == 401


def test_start_returns_limits(client):
    c, _ = client
    data = c.post("/companion/dialogues", json={"wardId": WARD, "sessionId": "s", "slot": "09:00"}, headers=HEADERS).json()["data"]
    assert data["timeLimitSec"] == dlg.TIME_LIMIT_SEC == 120
    assert data["maxTurns"] == dlg.MAX_TURNS


def test_single_answer_audio_endpoint_still_stores(client, monkeypatch):
    c, tmp = client
    monkeypatch.setattr(api_mock, "_transcribe", lambda raw: ("잘 잤어요", 2.0))
    r = c.post("/companion/answers/audio", data={"wardId": WARD, "sessionId": "s1", "slot": "09:00"},
               files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["transcript"] == "잘 잤어요"
    assert data["audioSeconds"] == 2.0
    assert data["label"] == "기쁨"


def test_generated_reply_goes_to_review_list_without_patient_text(client):
    c, tmp = client
    did = start(c)
    c.post(f"/companion/dialogues/{did}/reply", json={"text": "비밀스러운 개인 대답"}, headers=HEADERS)
    items = c.get("/companion/review/replies?status=pending", headers=HEADERS).json()["data"]
    assert len(items) == 1
    assert items[0]["reply"] == "그러셨군요. 무엇을 하셨나요?"
    assert items[0]["aiCheck"] == "통과"
    assert "비밀스러운" not in json.dumps(items, ensure_ascii=False)


def test_review_decision_is_recorded(client):
    c, tmp = client
    did = start(c)
    c.post(f"/companion/dialogues/{did}/reply", json={"text": "잘 잤어요"}, headers=HEADERS)
    item_id = c.get("/companion/review/replies", headers=HEADERS).json()["data"][0]["id"]
    r = c.post(f"/companion/review/replies/{item_id}", json={"decision": "rejected"}, headers=HEADERS)
    assert r.status_code == 200 and r.json()["data"]["status"] == "rejected"
    assert c.get("/companion/review/replies?status=pending", headers=HEADERS).json()["data"] == []
    assert len(c.get("/companion/review/replies?status=rejected", headers=HEADERS).json()["data"]) == 1


def test_review_rejects_bad_input(client):
    c, _ = client
    assert c.post("/companion/review/replies/none", json={"decision": "approved"}, headers=HEADERS).status_code == 404
    did = start(c)
    c.post(f"/companion/dialogues/{did}/reply", json={"text": "잘 잤어요"}, headers=HEADERS)
    item_id = c.get("/companion/review/replies", headers=HEADERS).json()["data"][0]["id"]
    assert c.post(f"/companion/review/replies/{item_id}", json={"decision": "maybe"}, headers=HEADERS).status_code == 422
    assert c.get("/companion/review/replies?status=unknown", headers=HEADERS).status_code == 422


def test_unreadable_audio_is_rejected_by_real_decoder(client):
    c, _ = client
    did = start(c)
    r = c.post(f"/companion/dialogues/{did}/reply/audio", files={"audio": ("a.webm", b"not really audio", "audio/webm")}, headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["message"] == "AUDIO_UNREADABLE"


def test_webm_audio_is_decoded_to_same_text_as_wav(monkeypatch):
    import av
    import numpy as np
    import soundfile as sf

    arr = np.sin(np.linspace(0, 200, 16000 * 2)).astype("float32") * 0.2
    buf = io.BytesIO()
    sf.write(buf, arr, 16000, format="WAV")
    wav = buf.getvalue()
    out = io.BytesIO()
    container = av.open(out, mode="w", format="webm")
    stream = container.add_stream("libopus", rate=48000, layout="mono")
    res = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=48000)
    frame = av.AudioFrame.from_ndarray((arr * 32767).astype("int16").reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = 16000
    for f in res.resample(frame):
        for pkt in stream.encode(f):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)
    container.close()
    webm = out.getvalue()
    a = api_mock._decode_audio(webm)
    b = api_mock._decode_audio(wav)
    assert abs(len(a) - len(b)) < 1600  # 인코딩 여유 (약 0.1초 이내)


def jpeg_b64() -> str:
    import base64

    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def set_label(monkeypatch, label):
    monkeypatch.setattr(api_mock, "text_label", lambda text, face=None: {
        "textEmotion": None, "label": label, "reason": "테스트"})


def test_photo_is_not_returned_without_consent(client, monkeypatch):
    c, _ = client
    set_label(monkeypatch, "확인 필요")
    did = start(c)
    r = c.post(f"/companion/dialogues/{did}/reply", json={"text": "힘들어요", "photo": jpeg_b64()}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["data"]["photoForBackend"] is None


def test_photo_is_returned_only_with_consent_on_review_turn(client, monkeypatch):
    c, _ = client
    set_label(monkeypatch, "확인 필요")
    did = start(c)
    photo = jpeg_b64()
    r = c.post(f"/companion/dialogues/{did}/reply", json={"text": "힘들어요", "consentPhoto": True, "photo": photo}, headers=HEADERS)
    assert r.json()["data"]["photoForBackend"] == photo


def test_photo_not_returned_for_normal_turn_even_with_consent(client, monkeypatch):
    c, _ = client
    set_label(monkeypatch, "기쁨")
    did = start(c)
    r = c.post(f"/companion/dialogues/{did}/reply", json={"text": "좋아요", "consentPhoto": True, "photo": jpeg_b64()}, headers=HEADERS)
    assert r.json()["data"]["photoForBackend"] is None


def test_photo_must_be_jpeg_and_valid_base64(client):
    import base64

    c, _ = client
    did = start(c)
    not_jpeg = base64.b64encode(b"PNG-ish bytes").decode()
    assert c.post(f"/companion/dialogues/{did}/reply", json={"text": "x", "photo": not_jpeg}, headers=HEADERS).status_code == 422
    assert c.post(f"/companion/dialogues/{did}/reply", json={"text": "x", "photo": "!!!not base64"}, headers=HEADERS).status_code == 422
