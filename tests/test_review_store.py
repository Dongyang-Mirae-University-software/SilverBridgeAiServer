from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers.companion_router import get_question_fetcher, get_text_classifier, get_transcriber
from app.services.companion.review_store import ReviewItem, review_store
from app.services.companion.stt_service import SttResult
from app.services.companion.text_emotion_service import TextEmotionResult
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}


@pytest.fixture
def client(tmp_path):
    review_store.reset_engine(f"sqlite:///{tmp_path / 'review.db'}")
    settings = get_settings().model_copy(update={
        "companion_api_key": KEY,
        "companion_backend_base_url": "http://backend.test",
    })
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_question_fetcher] = lambda: (lambda s, w, sl: ("잘 주무셨어요?", "q-test"))
    app.dependency_overrides[get_transcriber] = lambda: (lambda a: SttResult("테스트 대답입니다", 0.9, 1))
    app.dependency_overrides[get_text_classifier] = lambda: (lambda t: TextEmotionResult("기쁨", 0.9, True, {}))
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _turn(c, text="오늘은 손주가 와서 정말 기분이 좋아요") -> str:
    did = c.post("/companion/dialogues", json={"wardId": "u1", "sessionId": "s1", "slot": "09:00"}, headers=HDR).json()["data"]["dialogueId"]
    c.post(f"/companion/dialogues/{did}/reply", json={"text": text}, headers=HDR)
    return did


def test_ai_reply_is_queued_without_the_answer_text(client):
    _turn(client, text="비밀스러운 대답 원문입니다")
    items = client.get("/companion/review/replies", headers=HDR).json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["status"] == "pending"
    assert items[0]["promptVersion"] == "fallback-v1"  # 모델이 없어 미리 정한 문장이 나갔다
    assert "비밀스러운" not in str(items[0])  # 대답 원문은 검수 목록에 없다


def test_approve_and_reject_change_status(client):
    _turn(client)
    _turn(client)
    items = client.get("/companion/review/replies", headers=HDR).json()["data"]["items"]
    a, b = items[0]["itemId"], items[1]["itemId"]
    r = client.post(f"/companion/review/replies/{a}", json={"decision": "approved"}, headers=HDR)
    assert r.json()["data"]["status"] == "approved"
    assert r.json()["data"]["decidedAt"] is not None
    client.post(f"/companion/review/replies/{b}", json={"decision": "rejected"}, headers=HDR)
    pending = client.get("/companion/review/replies?status=pending", headers=HDR).json()["data"]["items"]
    assert pending == []
    rejected = client.get("/companion/review/replies?status=rejected", headers=HDR).json()["data"]["items"]
    assert [i["itemId"] for i in rejected] == [b]


def test_unknown_item_is_404(client):
    r = client.post("/companion/review/replies/nope", json={"decision": "approved"}, headers=HDR)
    assert r.status_code == 404
    assert r.json()["code"] == "REVIEW_ITEM_NOT_FOUND"


def test_bad_status_filter_is_422(client):
    r = client.get("/companion/review/replies?status=maybe", headers=HDR)
    assert r.status_code == 422
    assert r.json()["code"] == "STATUS_INVALID"


def test_items_past_retention_are_purged(client, monkeypatch):
    did = _turn(client)
    with review_store._sessions()() as s, s.begin():
        for row in s.query(ReviewItem).all():
            row.createdAt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=31)
    removed = review_store.purge_expired()
    assert removed == 1
    assert client.get("/companion/review/replies", headers=HDR).json()["data"]["items"] == []
    assert did  # 대화 자체는 메모리에 그대로 있다
