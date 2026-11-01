"""안부 말벗(checkin) - 모델 없이 계약·상태·오류 코드·개인정보 규칙을 고정한다(가짜 모델·가짜 음성 해독)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.response import error_response
from app.services.checkin import audio as audio_mod
from app.services.checkin import config as cfg_mod
from app.services.checkin import face as face_mod
from app.services.checkin import models as models_mod
from app.services.checkin import service

KEY = "test-key"
SECRET_SENTENCE = "어젯밤에 세 번이나 깼고 요즘 혼자라 적적해"


class FakeModels:
    def __init__(self):
        self.replies = 0
        self.transcripts = []
        self.busy = False

    def transcribe(self, arr):
        return self.transcripts.pop(0) if self.transcripts else ""

    def reply(self, question, answer, last_turn):
        if self.busy:
            raise models_mod.CheckinBusy()
        self.replies += 1
        return "그러셨군요(웃음). 오늘은 무엇을 하셨어요? 그리고 또 하나 질문이에요."

    def summarize(self, utts):
        return {"sleep": None, "meal": "먹음", "pain": None, "mood": "보통", "lonely": None}

    def emotion(self, utts):
        return "외로움"


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("CHECKIN_ENABLED", "true")
    cfg_mod.get_checkin_settings.cache_clear()
    f = FakeModels()
    monkeypatch.setattr(service, "get_models", lambda: f)
    monkeypatch.setattr(audio_mod, "decode", lambda raw: np.zeros(audio_mod.SR * 3, dtype="float32"))
    yield f
    cfg_mod.get_checkin_settings.cache_clear()


@pytest.fixture
def client(fake):
    from app.core.security import require_api_key
    from app.routers.checkin_router import router

    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def handler(_: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR"))

    from app.core.config import Settings, get_settings

    # Settings 는 임포트 때 환경 변수를 굳히므로 키는 의존성으로 바꿔 끼운다
    app.dependency_overrides[get_settings] = lambda: Settings(api_key=KEY)
    app.include_router(router, dependencies=[Depends(require_api_key)])
    return TestClient(app, headers={"X-API-Key": KEY})


def _sid():
    return "ses_" + uuid.uuid4().hex[:16]


def _start(client, sid, user="u1"):
    return client.post("/api/v1/checkin/sessions", json={"sessionId": sid, "userId": user, "slot": "09:00",
                                                          "question": "오늘 아침 편히 주무셨어요?"})


def _turn(client, sid, no, user="u1", frames=None):
    files = [("audio", ("a.webm", b"\x1aE\xdf\xa3fake", "audio/webm"))]
    for i, data in enumerate(frames or []):
        files.append(("frames", (f"f{i}.jpg", data, "image/jpeg")))
    return client.post(f"/api/v1/checkin/sessions/{sid}/turns", data={"userId": user, "turnNo": str(no)}, files=files)


def test_disabled_returns_503_but_delete_still_works(monkeypatch, client):
    monkeypatch.setenv("CHECKIN_ENABLED", "false")
    cfg_mod.get_checkin_settings.cache_clear()
    r = _start(client, _sid())
    assert r.status_code == 503 and r.json()["errorCode"] == "CHECKIN_DISABLED"
    assert client.delete("/api/v1/checkin/logs", params={"userId": "u1"}).status_code == 200


def test_requires_api_key(client):
    r = client.post("/api/v1/checkin/sessions", json={}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_start_is_idempotent_and_conflict_is_409(client):
    sid = _sid()
    assert _start(client, sid).status_code == 200
    assert _start(client, sid).status_code == 200
    r = _start(client, sid, user="other")
    assert r.status_code == 409 and r.json()["errorCode"] == "SESSION_CONFLICT"


def test_turn_flow_reply_is_cleaned_and_resend_returns_stored(client, fake):
    sid = _sid(); _start(client, sid)
    fake.transcripts = ["잘 잤어요"]
    r = _turn(client, sid, 1)
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["transcript"] == "잘 잤어요" and d["done"] is False
    assert "(" not in d["reply"] and d["reply"].count("?") + d["reply"].count(".") <= 2  # 두 문장까지, 괄호 제거
    assert set(d["voice"]) >= {"answerSec", "speechSec", "pauseRatio", "charsPerSec", "loudnessDb"}
    assert d["face"] is None
    again = _turn(client, sid, 1)  # 같은 턴 재전송 -> 다시 만들지 않는다
    assert again.json()["data"] == d and fake.replies == 1


def test_turn_order_and_owner(client, fake):
    sid = _sid(); _start(client, sid)
    assert _turn(client, sid, 2).json()["errorCode"] == "TURN_OUT_OF_ORDER"
    fake.transcripts = ["네"]
    r = _turn(client, sid, 1, user="someone-else")  # 남의 세션은 없는 것과 같게
    assert r.status_code == 404 and r.json()["errorCode"] == "SESSION_NOT_FOUND"


def test_empty_speech_is_422(client, fake):
    sid = _sid(); _start(client, sid)
    fake.transcripts = [""]
    assert _turn(client, sid, 1).json()["errorCode"] == "SPEECH_NOT_RECOGNIZED"


def test_self_harm_gets_fixed_reply_and_flag(client, fake):
    sid = _sid(); _start(client, sid)
    fake.transcripts = ["다 끝내고 죽고 싶어"]
    d = _turn(client, sid, 1).json()["data"]
    assert d["reply"] == service.SELF_HARM_REPLY and d["risk"]["selfHarm"] is True and fake.replies == 0


def test_end_word_finishes_then_more_turns_409(client, fake):
    sid = _sid(); _start(client, sid)
    fake.transcripts = ["이제 그만할게"]
    d = _turn(client, sid, 1).json()["data"]
    assert d["done"] is True and d["reply"] == service.FAREWELL
    fake.transcripts = ["또"]
    assert _turn(client, sid, 2).json()["errorCode"] == "SESSION_FINISHED"


def test_busy_is_429(client, fake):
    sid = _sid(); _start(client, sid)
    fake.transcripts = ["좋아요"]; fake.busy = True
    r = _turn(client, sid, 1)
    assert r.status_code == 429 and r.json()["errorCode"] == "CHECKIN_BUSY"


def test_upload_limits(client, fake, monkeypatch):
    sid = _sid(); _start(client, sid)
    fake.transcripts = ["네"]
    assert _turn(client, sid, 1, frames=[b"PNGnotjpeg"]).json()["errorCode"] == "FRAME_NOT_JPEG"
    monkeypatch.setenv("CHECKIN_MAX_AUDIO_BYTES", str(64 * 1024))
    cfg_mod.get_checkin_settings.cache_clear()
    files = [("audio", ("a.webm", b"x" * (64 * 1024 + 10), "audio/webm"))]
    r = client.post(f"/api/v1/checkin/sessions/{sid}/turns", data={"userId": "u1", "turnNo": "1"}, files=files)
    assert r.status_code == 413 and r.json()["errorCode"] == "AUDIO_TOO_LARGE"


def test_finish_applies_rules_summary_has_no_quote_and_is_idempotent(client, fake):
    sid = _sid(); _start(client, sid)
    fake.transcripts = [SECRET_SENTENCE]
    _turn(client, sid, 1)
    r = client.post(f"/api/v1/checkin/sessions/{sid}/finish", json={"userId": "u1", "selfReport": "외로움"})
    d = r.json()["data"]
    assert d["signals"]["sleep"] == "나쁨"  # R1 "깼"
    assert d["signals"]["lonely"] == "있음" and d["signals"]["mood"] == "나쁨"  # R2·R3
    assert d["emotion"] == "외로움" and d["selfReport"] == "외로움" and d["face"] is None
    assert SECRET_SENTENCE not in d["summary"] and "적적" not in d["summary"]  # 원문을 옮기지 않는다
    assert {"signal": "sleep", "turnNo": 1} in d["evidence"]
    again = client.post(f"/api/v1/checkin/sessions/{sid}/finish", json={"userId": "u1"})
    assert again.json()["data"] == d


def test_finish_without_turns_is_pending(client):
    sid = _sid(); _start(client, sid)
    d = client.post(f"/api/v1/checkin/sessions/{sid}/finish", json={"userId": "u1"}).json()["data"]
    assert d["emotion"] == "판단 보류" and d["turns"] == 0


def test_delete_logs_requires_user_and_removes_only_that_user(client, fake):
    a, b = _sid(), _sid()
    _start(client, a, user="del-me"); _start(client, b, user="keep-me")
    assert client.delete("/api/v1/checkin/logs").json()["errorCode"] == "CHECKIN_USER_ID_REQUIRED"
    n = client.delete("/api/v1/checkin/logs", params={"userId": "del-me"}).json()["data"]
    assert n["sessions"] == 1
    assert _start(client, b, user="keep-me").status_code == 200


def test_retention_purge(fake):
    from app.database.session import SessionLocal
    from app.models.checkin import CheckinSession

    with SessionLocal() as db:
        sid = _sid()
        db.add(CheckinSession(session_id=sid, user_id="old", slot="09:00", question="q",
                              started_at=datetime.utcnow() - timedelta(days=31)))
        db.commit()
        assert service.purge_expired(db)["sessions"] >= 1
        assert db.get(CheckinSession, sid) is None


def test_logs_never_contain_conversation(client, fake, caplog):
    caplog.set_level(logging.DEBUG)
    sid = _sid(); _start(client, sid)
    fake.transcripts = [SECRET_SENTENCE]
    _turn(client, sid, 1)
    client.post(f"/api/v1/checkin/sessions/{sid}/finish", json={"userId": "u1"})
    assert SECRET_SENTENCE not in caplog.text and "적적" not in caplog.text and "오늘 아침" not in caplog.text


def test_parsers():
    assert models_mod.parse_summary('{sleep:"나쁨","meal":"이상","pain":"있음"}') == {
        "sleep": "나쁨", "meal": None, "pain": "있음", "mood": None, "lonely": None}
    assert models_mod.parse_emotion('{"emotion": "걱정"}') == "걱정"
    assert models_mod.parse_emotion('{"emotion": "모름"}') is None


def test_voice_metrics_on_tone_with_silence():
    sr = audio_mod.SR
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    arr = np.concatenate([np.zeros(sr // 2, "float32"), tone, np.zeros(sr // 2, "float32"), tone])
    m = audio_mod.voice_metrics(arr + 1e-4 * np.random.default_rng(0).standard_normal(len(arr)).astype("float32"), "안녕하세요")
    assert m["answerSec"] == 3.0
    assert 1.8 <= m["speechSec"] <= 2.2
    assert 0.15 <= m["pauseRatio"] <= 0.4
    assert m["charsPerSec"] and m["loudnessDb"] < 0


JPEG = b"\xff\xd8\xff\xe0fakejpeg"


class FakeFace:
    def __init__(self, fail=False):
        self.fail = fail

    def analyze(self, frames):
        if self.fail:
            raise RuntimeError("gpu")
        ps = [[0.9, 0.05, 0.05]] * (len(frames) - 1) + [[0.05, 0.05, 0.9]]  # 마지막 장만 부정
        return face_mod.summarize_frames(len(frames), ps, lambda p: "negative" if p[2] >= 0.8 else
                                         "smile" if p[0] >= p[1] else "calm")


def test_face_is_recorded_per_turn_and_merged_at_finish(client, fake, monkeypatch):
    monkeypatch.setattr(face_mod, "get_face", lambda: FakeFace())
    sid = _sid(); _start(client, sid)
    fake.transcripts = ["잘 잤어요", "밥 먹었어요"]
    t1 = _turn(client, sid, 1, frames=[JPEG] * 4).json()["data"]
    assert t1["face"] == {"frames": 4, "faceFrames": 4, "smile": 0.688, "calm": 0.05, "negative": 0.263,
                          "smileRatio": 0.75, "negativeRatio": 0.25}
    t2 = _turn(client, sid, 2).json()["data"]  # 사진 없는 턴
    assert t2["face"] is None
    d = client.post(f"/api/v1/checkin/sessions/{sid}/finish", json={"userId": "u1"}).json()["data"]
    assert d["face"]["frames"] == 4 and d["face"]["smileRatio"] == 0.75


def test_face_failure_keeps_conversation(client, fake, monkeypatch):
    monkeypatch.setattr(face_mod, "get_face", lambda: FakeFace(fail=True))
    sid = _sid(); _start(client, sid)
    fake.transcripts = ["잘 잤어요"]
    r = _turn(client, sid, 1, frames=[JPEG])
    assert r.status_code == 200 and r.json()["data"]["face"] is None


def test_face_without_model_paths_is_off(fake):
    assert face_mod.get_face() is None  # CHECKIN_FACE_MODEL·HEAD 미설정


def test_face_merge_weights_by_face_frames():
    a = {"frames": 2, "faceFrames": 2, "smile": 1.0, "calm": 0.0, "negative": 0.0, "smileRatio": 1.0, "negativeRatio": 0.0}
    b = {"frames": 3, "faceFrames": 0, "smile": None, "calm": None, "negative": None, "smileRatio": None,
         "negativeRatio": None}
    c = {"frames": 2, "faceFrames": 2, "smile": 0.0, "calm": 0.0, "negative": 1.0, "smileRatio": 0.0, "negativeRatio": 1.0}
    m = face_mod.merge([a, None, b, c])
    assert m["frames"] == 7 and m["faceFrames"] == 4 and m["smile"] == 0.5 and m["negativeRatio"] == 0.5
    assert face_mod.merge([None, b])["smile"] is None and face_mod.merge([None]) is None
