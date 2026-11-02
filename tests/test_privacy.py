from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion import privacy


def _with_salt(monkeypatch, salt: str):
    settings = get_settings().model_copy(update={"companion_pseudonym_salt": salt})
    monkeypatch.setattr(privacy, "get_settings", lambda: settings)


def test_pseudonym_is_stable_and_hides_the_input(monkeypatch):
    _with_salt(monkeypatch, "salt-for-test")
    a = privacy.pseudonym("u123")
    assert a == privacy.pseudonym("u123")
    assert "u123" not in a
    assert len(a) == 24
    assert a != privacy.pseudonym("u124")


def test_different_salt_gives_different_pseudonym(monkeypatch):
    _with_salt(monkeypatch, "one")
    a = privacy.pseudonym("u123")
    _with_salt(monkeypatch, "two")
    assert privacy.pseudonym("u123") != a


def test_no_salt_means_no_pseudonym(monkeypatch):
    _with_salt(monkeypatch, "")
    with pytest.raises(privacy.PseudonymSaltMissing):
        privacy.pseudonym("u123")


@pytest.mark.parametrize("raw,masked", [
    ("제 번호는 010-1234-5678이에요", "제 번호는 [전화번호]이에요"),
    ("02-123-4567로 연락 주세요", "[전화번호]로 연락 주세요"),
    ("주민번호 800101-1234567", "주민번호 [주민번호]"),
    ("계좌 1234567890123 입니다", "계좌 [번호] 입니다"),
    ("오늘은 날씨가 좋아요", "오늘은 날씨가 좋아요"),
])
def test_strip_pii_masks_numbers(raw, masked):
    assert privacy.strip_pii(raw) == masked
