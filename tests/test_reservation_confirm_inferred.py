"""예약 시작 때 짐작한 진료과·하나뿐인 병원도 먼저 확인받는다(2026-11-17).

11/17 gosky 실측: 다른 상담 뒤 같은 대화에서 "병원 예약하고 싶어요" → 이전 대화의 '내과' + 프로필 지역 + 후보 1개 자동 선택으로
사용자가 고른 적 없는 병원의 날짜 질문이 첫 질문이었다. 이전 대화로만 정한 진료과·병원은 버튼으로 확인받고, 후보가 1개여도 카드로 고르게 한다.
"""
from __future__ import annotations

from typing import Any

from app.services.reservation_orchestrator import (
    OTHER_REGION_OPTION,
    PICK_DEPARTMENT_OPTION,
    SOURCE_BUTTON,
    SOURCE_INFERRED,
    SOURCE_MESSAGE,
    _topic,
    _with_ro,
)
import pytest

from app.services.reservation_orchestrator import _is_plain_yes
from tests.reservation_flow_helpers import FakeReservation, make_orchestrator, make_request

HISTORY_NAEGWA = [
    {"role": "user", "content": "당뇨약 먹는데 내과 다시 가봐야 하나요"},
    {"role": "assistant", "content": "내과 선생님과 상의해 보세요."},
]
ONE_ROW = [{"id": 7, "name": "홍대내과", "hospital": "홍대내과", "department": "내과", "location": "서울 강남구"}]


def _stored(out: dict[str, Any] | None) -> dict[str, Any]:
    meta = (out or {}).get("meta") or {}
    return {k: meta[k] for k in ("pendingAction", "flowState") if isinstance(meta.get(k), dict)}


class _Session:
    def __init__(
        self, fake: FakeReservation | None = None, extracted: dict[str, Any] | None = None, ctx: dict[str, Any] | None = None
    ) -> None:
        self.fake = fake or FakeReservation()
        self.orch = make_orchestrator(self.fake, extracted if extracted is not None else {})
        self.prev: dict[str, Any] | None = None
        self.ctx = ctx

    def turn(self, message: str = "", ui: tuple[str, str] | None = None, history=None) -> dict[str, Any]:
        out = self.orch.handle(
            make_request(message, ui=ui, history=history, context=self.ctx),
            {"storedFlow": _stored(self.prev), "rawMessage": message},
        )
        assert out is not None
        self.prev = out
        return out


def _draft_trace(out: dict[str, Any]) -> dict[str, Any]:
    return [t for t in out["decisionTrace"] if t["event"] == "reservation_draft"][-1]


def _events(out: dict[str, Any]) -> list[str]:
    return [t["event"] for t in out["decisionTrace"]]


def test_department_from_history_is_confirmed_first() -> None:
    s = _Session()
    out = s.turn("병원 예약하고 싶어요", history=HISTORY_NAEGWA)

    assert _draft_trace(out)["department_source"] == SOURCE_INFERRED
    ui = out["ui"]
    assert ui.field == "department" and ui.kind == "select"
    assert [(o.value, o.label) for o in ui.options] == [("내과", "네, 내과"), (PICK_DEPARTMENT_OPTION, "다른 과 고르기")]
    assert out["reply"].startswith("내과로 예약할까요?") and "서울 강남구 근처에서 찾을게요" in out["reply"]
    assert s.fake.searched_regions == []  # 확인 전에는 병원을 찾지 않는다

    # [네, 내과] → 그때 병원을 찾는다(기본 후보 2곳 = 기존 목록).
    out = s.turn("", ui=("department", "내과"))
    assert _draft_trace(out)["department_source"] == SOURCE_BUTTON
    assert out["tool"] == "search_hospital" and len(out["data"]) == 2


def test_pick_other_department_shows_full_list() -> None:
    s = _Session()
    s.turn("병원 예약하고 싶어요", history=HISTORY_NAEGWA)
    out = s.turn("", ui=("department", PICK_DEPARTMENT_OPTION))
    assert out["ui"].field == "department"
    values = [o.value for o in out["ui"].options]
    assert len(values) > 2 and PICK_DEPARTMENT_OPTION not in values
    # 고른 과로 바로 진행(다시 묻지 않음)
    out = s.turn("", ui=("department", "정형외과"))
    assert out["tool"] == "search_hospital"


def test_department_extracted_only_from_history_is_confirmed() -> None:
    # 규칙 파서가 못 잡아도 추출(LLM stub)이 이전 대화로 진료과를 채우면 같은 확인을 거친다.
    s = _Session(extracted={"department": "이비인후과"})
    out = s.turn("병원 예약하고 싶어요")
    assert _draft_trace(out)["department_source"] == SOURCE_INFERRED
    assert out["ui"].field == "department" and out["ui"].options[0].label == "네, 이비인후과"


def test_department_in_this_message_proceeds() -> None:
    s = _Session()
    out = s.turn("내과 예약하고 싶어요", history=HISTORY_NAEGWA)
    assert _draft_trace(out)["department_source"] == SOURCE_MESSAGE
    assert out["tool"] == "search_hospital"


def test_department_from_symptom_in_this_message_proceeds() -> None:
    s = _Session()
    out = s.turn("무릎이 아파서 병원 예약하고 싶어요")
    assert _draft_trace(out)["department_source"] == SOURCE_MESSAGE
    assert out["tool"] == "search_hospital"


def test_nearby_department_button_proceeds_without_confirm() -> None:
    # 진료과 답의 [가까운 ○○ 찾기] = 저장 초안 없이 들어온 department 버튼. 이전 대화에 다른 과가 있어도 묻지 않는다.
    s = _Session()
    out = s.turn("가까운 정형외과 찾기", ui=("department", "정형외과"), history=HISTORY_NAEGWA)
    assert _draft_trace(out)["department_source"] == SOURCE_BUTTON
    assert out["tool"] == "search_hospital"
    assert "hospital_selection_needed" in _events(out)


def test_single_candidate_is_offered_not_picked() -> None:
    fake = FakeReservation()
    fake.search_rows = ONE_ROW
    s = _Session(fake)
    out = s.turn("내과 예약하고 싶어요")

    assert out["tool"] == "search_hospital" and [r["hospital"] for r in out["data"]] == ["홍대내과"]
    assert "서울 강남구 근처 내과는 홍대내과 한 곳이에요" in out["reply"]
    assert out["ui"].field == "location_confirm" and [o.value for o in out["ui"].options] == [OTHER_REGION_OPTION]
    sel = [t for t in out["decisionTrace"] if t["event"] == "hospital_selection_needed"]
    assert sel and sel[-1]["candidate_count"] == 1
    assert "make_appointment" not in fake.names()
    assert not any(t["event"] == "reservation_gap" and t.get("missing") == "date" for t in out["decisionTrace"])

    # [예약하기] → 그 다음에야 날짜를 묻는다.
    out = s.turn("", ui=("hospital_confirm", "홍대내과"))
    assert _draft_trace(out)["hospital_source"] == SOURCE_BUTTON
    assert out["ui"].field == "date"


def test_single_candidate_other_region_asks_region() -> None:
    fake = FakeReservation()
    fake.search_rows = ONE_ROW
    fake.regions = ["서울 강남구", "서울 서초구"]
    s = _Session(fake)
    s.turn("내과 예약하고 싶어요")
    out = s.turn("", ui=("location_confirm", OTHER_REGION_OPTION))
    assert out["ui"].field in {"location", "location_sido"}


def test_multiple_candidates_unchanged() -> None:
    s = _Session()
    out = s.turn("내과 예약하고 싶어요")
    assert out["reply"] == "서울 강남구 기준으로 찾은 내과예요. 예약하려면 병원을 선택해 주세요."
    assert len(out["data"]) == 2 and out["ui"].field == "location_confirm"


def test_hospital_named_only_in_history_is_confirmed_as_card() -> None:
    history = [
        {"role": "user", "content": "강남메디내과 가봤는데 괜찮았어요"},
        {"role": "assistant", "content": "다행이에요."},
    ]
    s = _Session()
    out = s.turn("내과 예약하고 싶어요", history=history)
    # "내과 예약" 은 새 요청이라 이력의 병원은 이미 버려진다(기존 동작) - 병원 검색 목록.
    assert out["tool"] == "search_hospital" and len(out["data"]) == 2

    s = _Session()
    out = s.turn("병원 예약하고 싶어요", history=history)
    # 진료과(이전 대화의 '내과')부터 확인
    assert out["ui"].field == "department"
    out = s.turn("", ui=("department", "내과"))
    assert _draft_trace(out)["hospital_source"] == SOURCE_INFERRED
    assert out["tool"] == "search_hospital" and [r["hospital"] for r in out["data"]] == ["강남메디내과"]
    assert out["reply"].startswith("강남메디내과로 예약할까요?")
    assert out["ui"].field == "location_confirm"
    out = s.turn("", ui=("hospital_confirm", "강남메디내과"))
    assert out["ui"].field == "date"


def test_hospital_named_in_this_message_proceeds() -> None:
    s = _Session()
    out = s.turn("강남메디내과 내과로 예약할래요")
    assert _draft_trace(out)["hospital_source"] == SOURCE_MESSAGE
    assert out["ui"].field == "date"


def test_particles() -> None:
    assert _with_ro("내과") == "내과로" and _with_ro("병원") == "병원으로" and _with_ro("서울") == "서울로"
    assert _topic("내과") == "내과는" and _topic("병원") == "병원은"


def test_typed_yes_to_department_confirm_proceeds() -> None:
    # 보호자가 확인 질문에 버튼 대신 "네"라고 쳐도 같은 질문을 되풀이하지 않는다.
    s = _Session()
    s.turn("병원 예약하고 싶어요", history=HISTORY_NAEGWA)
    out = s.turn("네 예약해 주세요")
    assert _draft_trace(out)["department_source"] == SOURCE_MESSAGE
    assert out["tool"] == "search_hospital"


# ---- 2026-11-17 #312 기능 점검 반영(M-1·L-1·L-2·L-3) ----

WARD_CTX = {
    "wardId": "ward-1", "wardName": "김영희", "wardPhone": "010-9999-0000", "name": "김영희", "phone": "010-9999-0000",
    "location": None, "address": "서울특별시 강남구 테헤란로 1",
}


@pytest.mark.parametrize("typed", ["예약 말고 다른 과로 할래요", "그래도 다른 과로 예약할래요", "네 근데 다른 과로 예약해 주세요", "그래요? 그럼 다른 과로 예약할래요"])
def test_typed_non_yes_is_not_taken_as_yes(typed: str) -> None:
    # M-1: 앞글자만 '네·예·그래·응'인 답("예약…"·"그래도…"·"응급…")이나 다른 과를 말하는 답은 수락이 아니다 - 다시 확인한다.
    s = _Session()
    s.turn("병원 예약하고 싶어요", history=HISTORY_NAEGWA)
    out = s.turn(typed)
    assert _draft_trace(out)["department_source"] == SOURCE_INFERRED
    assert out["ui"].field == "department" and out["ui"].options[0].label == "네, 내과"
    assert s.fake.searched_regions == []


@pytest.mark.parametrize("text,expected", [
    ("네", True), ("네 예약해 주세요", True), ("예, 그걸로 예약", True), ("좋아요 예약해 주세요", True), ("맞아요", True), ("응 예약해줘", True),
    ("예약해 주세요", False), ("그래도 예약", False), ("응급실", False), ("좋아하는 병원으로 예약", False), ("네 근데 다른 과로", False),
    ("아니요", False), ("", False), (None, False),
])
def test_is_plain_yes(text, expected) -> None:
    assert _is_plain_yes(text) is expected


def test_history_hospital_card_without_region_offers_other_hospital() -> None:
    # L-1: 지역을 모르는(프로필 주소 없음) 피보호자에게 이력 병원 카드를 낼 때도 버튼으로 거절할 길이 있다.
    fake = FakeReservation()
    fake.regions = ["서울 강남구", "서울 서초구"]
    s = _Session(fake, ctx={"location": None, "address": None})
    history = [{"role": "user", "content": "강남메디내과 가봤는데 괜찮았어요"}, {"role": "assistant", "content": "다행이에요."}]
    out = s.turn("병원 예약하고 싶어요", history=history)
    assert out["ui"].field == "department" and "근처에서 찾을게요" not in out["reply"]  # 지역을 모르면 지역 문구 없음
    out = s.turn("", ui=("department", "내과"))
    assert [r["hospital"] for r in out["data"]] == ["강남메디내과"]
    assert out["ui"].field == "location_confirm"
    assert [(o.value, o.label) for o in out["ui"].options] == [(OTHER_REGION_OPTION, "다른 병원 찾기")]
    out = s.turn("", ui=("location_confirm", OTHER_REGION_OPTION))
    assert out["ui"].field in {"location", "location_sido"}
    region_field = out["ui"].field
    region_value = "서울 서초구" if region_field == "location" else "서울"
    out = s.turn("", ui=(region_field, region_value))
    if out["ui"] is not None and out["ui"].field == "location":
        out = s.turn("", ui=("location", "서울 서초구"))
    assert out["tool"] == "search_hospital" and len(out["data"]) == 2  # 고른 지역에서 다시 찾은 목록(이력 병원 아님)
    assert _draft_trace(out).get("hospital_source") is None


def test_history_hospital_with_extracted_id_is_resolved_by_name() -> None:
    # L-2: 추출이 hospital_id 를 직접 줘도 이력 출처 병원은 이름으로 다시 찾아 예약 서비스의 이름·id 로 카드를 만든다.
    s = _Session(extracted={"hospital_name": "튼튼내과의원", "hospital_id": 99})
    out = s.turn("병원 예약하고 싶어요")
    assert _draft_trace(out)["hospital_source"] == SOURCE_INFERRED
    assert out["tool"] == "search_hospital"
    assert [(r["id"], r["hospital"]) for r in out["data"]] == [(1, "강남메디내과")]


def test_history_hospital_unknown_to_service_falls_back_to_search() -> None:
    # L-2: 이름으로 다시 찾아도 없는 병원이면 그 병원(추출 id 포함)을 버리고 검색으로 넘긴다.
    fake = FakeReservation()
    fake.resolve_hospital = lambda name, *, api_key_override=None: (None, None)  # type: ignore[method-assign]
    s = _Session(fake, extracted={"hospital_name": "없는의원", "hospital_id": 99, "department": "내과"})
    out = s.turn("내과 예약하고 싶어요")
    assert out["tool"] == "search_hospital" and len(out["data"]) == 2


def test_guardian_proxy_same_rules() -> None:
    # L-3: 보호자 대리 예약(wardId 문맥)도 같은 규칙 - 짐작한 진료과 확인(피보호자 주소 지역), 후보 1개는 카드.
    fake = FakeReservation()
    fake.search_rows = ONE_ROW
    s = _Session(fake, ctx=WARD_CTX)
    out = s.turn("병원 예약하고 싶어요", history=HISTORY_NAEGWA)
    assert out["ui"].field == "department" and out["reply"].startswith("내과로 예약할까요?")
    assert "강남구 근처에서 찾을게요" in out["reply"]
    out = s.turn("", ui=("department", "내과"))
    assert [r["hospital"] for r in out["data"]] == ["홍대내과"] and "한 곳이에요" in out["reply"]
    out = s.turn("", ui=("hospital_confirm", "홍대내과"))
    assert out["ui"].field == "date"


def test_legacy_stored_draft_without_source_is_not_asked() -> None:
    # L-3: 배포 전에 저장된 초안(출처 없음)은 진료과가 그대로면 묻지 않고 이어 간다.
    s = _Session()
    s.turn("내과 예약하고 싶어요")
    flow = s.prev["meta"]["flowState"]
    flow["draft"].pop("department_source", None)
    flow["draft"].pop("hospital_source", None)
    out = s.turn("", ui=("hospital_confirm", "강남메디내과"))
    assert "department_confirm" not in [t.get("missing") for t in out["decisionTrace"]]
    assert out["ui"].field == "date"


def test_pick_other_department_with_hospital_fixed_lists_that_hospital() -> None:
    # L-3: 이번 문장에서 병원을 정했고 진료과만 짐작이면 [다른 과 고르기]는 그 병원의 진료과 목록을 낸다.
    fake = FakeReservation()
    fake.departments_by_hospital = {1: ["내과", "정형외과"]}
    s = _Session(fake)
    out = s.turn("튼튼병원 예약할래요", history=HISTORY_NAEGWA)
    assert _draft_trace(out)["hospital_source"] == SOURCE_MESSAGE
    assert out["ui"].field == "department" and out["reply"] == "내과로 예약할까요?"  # 병원을 정했으면 지역 문구 없음
    out = s.turn("", ui=("department", PICK_DEPARTMENT_OPTION))
    assert out["ui"].field == "department"
    assert [o.value for o in out["ui"].options][:2] == ["내과", "정형외과"]
    assert "강남메디내과" in (out["ui"].label or "")
    out = s.turn("", ui=("department", "정형외과"))
    assert _draft_trace(out)["department_source"] == SOURCE_BUTTON
    assert out["ui"].field == "date"
