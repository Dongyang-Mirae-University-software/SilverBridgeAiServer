from __future__ import annotations

import threading
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services.companion import reply_service
from app.services.companion.rewrite_counts import RETENTION_DAYS, rewrite_counts, today_kst
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}


@pytest.fixture
def counts(tmp_path):
    rewrite_counts.reset_engine(f"sqlite:///{tmp_path / 'review.db'}")
    return rewrite_counts


def test_record_and_table_fill_zero(counts):
    d = date(2026, 11, 27)
    counts.record([("followup", "checked"), ("tense", "rewrite")], day=d)
    counts.record([("followup", "checked"), ("vague", "fail"), ("nope", "x"), ("tense", "odd")], day=d)
    row = counts.by_day(d, d)[0]
    assert row == {"day": "2026-11-27", "followup_checked": 2, "tense_rewrite": 1, "tense_keep": 0, "vague_ok": 0, "vague_fail": 1}


def test_concurrent_increments_are_not_lost(counts):
    d = date(2026, 11, 27)

    def worker():
        for _ in range(25):
            counts.record([("followup", "checked")], day=d)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counts.by_day(d, d)[0]["followup_checked"] == 200


def test_rows_older_than_retention_are_purged(counts):
    today = today_kst()
    counts.record([("followup", "checked")], day=today - timedelta(days=RETENTION_DAYS))
    counts.record([("followup", "checked")], day=today - timedelta(days=RETENTION_DAYS - 1))
    assert counts.purge_expired(today) == 1


def test_record_failure_never_raises(counts, monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(counts, "_sessions", boom)
    counts.record([("followup", "checked")])  # 예외 없이 지나간다 - 대답은 그대로 나가야 한다


def _settings(monkeypatch, **kw):
    s = get_settings()
    for k, v in kw.items():
        monkeypatch.setattr(s, k, v)


def test_followup_fix_counts_each_outcome(counts, monkeypatch):
    _settings(monkeypatch, companion_reply_tense_fix="rewrite", companion_reply_vague_rewrite=False,
              companion_reply_strict=False, companion_reply_similar_ratio=0.65)
    out = reply_service._apply_followup_fixes("이불까지 빨아 놓으셨다니 개운하시겠어요. 잘 말랐나요?", "이따 이불 빨 거야", None)
    assert out != "이불까지 빨아 놓으셨다니 개운하시겠어요. 잘 말랐나요?"  # 고쳐 썼다
    reply_service._apply_followup_fixes("오늘 하루 어떠셨어요?", "오늘은 그냥 집에 있었어", None)  # 고칠 것 없음
    row = counts.by_day(today_kst(), today_kst())[0]
    assert row["followup_checked"] == 2 and row["tense_rewrite"] + row["tense_keep"] == 1 and row["vague_ok"] + row["vague_fail"] == 0


@pytest.fixture
def client(counts):
    settings = get_settings().model_copy(update={"companion_api_key": KEY})
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_endpoint_returns_counts_without_text(client, counts):
    counts.record([("followup", "checked"), ("vague", "ok")])
    r = client.get("/companion/ops/rewrite-counts", headers=HDR)
    assert r.status_code == 200
    days = r.json()["data"]["days"]
    assert days and set(days[-1]) == {"day", "followup_checked", "tense_rewrite", "tense_keep", "vague_ok", "vague_fail"}


def test_endpoint_needs_key_and_valid_range(client):
    assert client.get("/companion/ops/rewrite-counts").status_code in (401, 403)
    assert client.get("/companion/ops/rewrite-counts?from=2026-13-01", headers=HDR).status_code == 422
    assert client.get("/companion/ops/rewrite-counts?from=2026-11-27&to=2026-11-01", headers=HDR).status_code == 422
    assert client.get("/companion/ops/rewrite-counts?from=2026-01-01&to=2026-11-27", headers=HDR).status_code == 422
    assert client.get("/companion/ops/rewrite-counts?from=2026-11-01&to=2026-11-27", headers=HDR).status_code == 200
