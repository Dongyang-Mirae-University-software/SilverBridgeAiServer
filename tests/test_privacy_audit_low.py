"""#90 점검 낮음 3건(2026-10-29): 국제 형식 전화 가림(L-2) · "다른 지역으로 찾기" 지역 비교(L-3) · 응답 읽기 오류 코드(L-5)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.services.reservation_api_client as rac
from app.services.reservation_api_client import ReservationApiClient
from app.utils.conversation import mask_personal_info, remove_personal_info
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator, make_request


@pytest.mark.parametrize(
    "text",
    ["+82 10-1234-5678", "+82-10-1234-5678", "+821012345678", "+82 (0)10-1234-5678", "+82 2-123-4567", "+82.10.1234.5678"],
)
def test_international_phone_is_masked(text: str) -> None:
    assert mask_personal_info(f"제 번호는 {text} 이에요") == "제 번호는 [전화] 이에요"
    assert "82" not in remove_personal_info(f"배가 아파요 {text}")


@pytest.mark.parametrize("text", ["82-10-1234-5678 번", "2026-10-30 09:00", "1588-1234", "+82 가입"])
def test_other_numbers_are_left(text: str) -> None:
    assert "[전화]" not in mask_personal_info(text)


def test_domestic_rules_unchanged() -> None:
    assert mask_personal_info("010-1234-5678 / 500101-2345678 / a@b.co") == "[전화] / [주민번호] / [이메일]"


ADDR = {"location": None, "address": "서울특별시 강남구 테헤란로 123"}


@pytest.mark.parametrize(
    ("extracted_location", "button"),
    [("", True), ("서울특별시 강남구", True), ("서울특별시 강남구 테헤란로 123", True), ("부산 해운대구", False)],
)
def test_other_region_button_compares_at_city_district_level(extracted_location: str, button: bool) -> None:
    orch = make_orchestrator(FakeReservation(), {"reservation_date": FUTURE, "location": extracted_location})
    out = orch.handle(make_request("근처 내과 예약해줘", context=ADDR), {"rawMessage": "근처 내과 예약해줘"})
    assert out["tool"] == "search_hospital"
    assert bool(out.get("ui")) is button


class _Resp:
    status = 200

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        raise self.error


@pytest.mark.parametrize(("error", "code"), [(TimeoutError(), "UPSTREAM_TIMEOUT"), (ConnectionResetError(), "UPSTREAM_UNREACHABLE")])
def test_read_errors_have_distinct_codes(monkeypatch, error: BaseException, code: str) -> None:
    monkeypatch.setattr(rac.urllib.request, "urlopen", lambda req, timeout: _Resp(error))  # noqa: ARG005
    client = ReservationApiClient(SimpleNamespace(
        reservation_api_base_url="http://x/api/v1", reservation_api_timeout_sec=5, reservation_user_jwt="", reservation_api_key="k",
    ))  # type: ignore[arg-type]
    res = client.list_my_appointments(api_key_override="k")
    assert res["httpStatus"] == 0 and res["data"]["code"] == code
