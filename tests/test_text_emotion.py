from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.services.companion import text_emotion_service as tes
from app.services.companion.text_emotion_service import TEXT_MODEL_DIR, TextEmotionService, TextEmotionUnavailable

LABELS = {"분노", "기쁨", "불안", "당황", "슬픔", "상처"}


def _settings_with_base(monkeypatch, base: Path):
    settings = get_settings().model_copy(update={"model_base_path": str(base)})
    monkeypatch.setattr(tes, "get_settings", lambda: settings)


def test_missing_model_folder_is_unavailable(monkeypatch, tmp_path):
    _settings_with_base(monkeypatch, tmp_path)
    svc = TextEmotionService()
    with pytest.raises(TextEmotionUnavailable):
        svc.classify("안녕하세요")
    assert svc.is_loaded() is False


def test_load_error_is_remembered(monkeypatch, tmp_path):
    _settings_with_base(monkeypatch, tmp_path)
    svc = TextEmotionService()
    with pytest.raises(TextEmotionUnavailable) as first:
        svc._load()
    with pytest.raises(TextEmotionUnavailable) as second:
        svc._load()
    assert str(first.value) == str(second.value)


def test_model_dir_follows_d19_layout(monkeypatch, tmp_path):
    _settings_with_base(monkeypatch, tmp_path)
    svc = TextEmotionService()
    assert svc.model_dir() == tmp_path / "companion" / "text-emotion" / TEXT_MODEL_DIR


_REAL = Path(get_settings().model_base_path) / "companion" / "text-emotion" / TEXT_MODEL_DIR / "model.safetensors"


@pytest.mark.skipif(not _REAL.is_file(), reason="GGARA02 가중치가 models/ 에 없음")
def test_real_model_returns_one_of_six_labels():
    svc = TextEmotionService()
    r = svc.classify("오늘은 손주가 와서 정말 기분이 좋아요")
    assert r.label in LABELS
    assert set(r.scores) == LABELS
    assert sum(r.scores.values()) == pytest.approx(1.0, abs=0.01)
    assert 0.0 <= r.confidence <= 1.0
    assert r.usable == (r.confidence >= get_settings().companion_confidence_min)
