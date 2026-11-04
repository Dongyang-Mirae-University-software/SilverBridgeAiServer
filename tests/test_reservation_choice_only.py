"""챗봇 병원 예약을 키보드 없이 고르기만으로 끝낸다(2026-11-04).

피보호자 본인·보호자 대리 각각 첫 문장 한 번 뒤로는 버튼(ui.options)·달력만 눌러 예약 완료까지 간다.
지역은 예약 서비스에 병원이 있는 지역에서 뽑고(같은 시·도 = 가까운 곳), 막히던 곳(검색 0건·빈 시간 조회 실패)에는 다음 길을 버튼으로 준다.
"""
from __future__ import annotations

from typing import Any

from app.services.reservation_intake import CHANGE_DATE_OPTION, RETRY_SLOTS_OPTION
from app.services.reservation_orchestrator import (
    MAX_REGION_OPTIONS,
    MORE_SIDO_OPTION,
    OTHER_DEPARTMENT_OPTION,
    OTHER_REGION_OPTION,
    TYPE_LOCATION_OPTION,
    _same_region,
    _sido_of,
)
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator, make_request

REGIONS = [
    "서울 강남구", "서울 서초구", "경기 성남시", "서울 송파구", "서울 강동구", "서울 마포구",
    "서울 구로구", "서울 영등포구", "부산 해운대구",
]
SELF_NO_ADDRESS = {"name": "홍길동", "phone": "010-1234-5678", "location": None}
WARD_WITH_ADDRESS = {
    "wardId": "ward-1", "wardName": "김영희", "wardPhone": "010-9999-0000", "name": "김영희", "phone": "010-9999-0000",
    "location": None, "address": "서울특별시 강남구 테헤란로 1",
}


def _stored(out: dict[str, Any] | None) -> dict[str, Any]:
    meta = (out or {}).get("meta") or {}
    return {k: meta[k] for k in ("pendingAction", "flowState") if isinstance(meta.get(k), dict)}


class _Session:
    """chat_service 처럼 직전 턴의 저장 상태를 다음 턴에 넘기고, 오간 응답을 모아 둔다."""

    def __init__(self, fake: FakeReservation, ctx: dict[str, Any]) -> None:
        self.fake = fake
        self.orch = make_orchestrator(fake, {})
        self.ctx = ctx
        self.prev: dict[str, Any] | None = None
        self.outs: list[dict[str, Any]] = []

    def say(self, message: str) -> dict[str, Any]:
        return self._turn(message, None)

    def pick(self, field: str, value: str) -> dict[str, Any]:
        ui = self.prev and self.prev.get("ui")
        if field == "hospital_confirm":
            # 병원은 검색 결과 카드(toolData)의 버튼으로 고른다(FE ChatToolCard).
            assert self.prev and self.prev.get("tool") == "search_hospital"
            assert value in [row.get("hospital") for row in self.prev["data"]]
        elif (
            field == "date"
            and ui is not None
            and ui.kind == "select"
            and value not in [o.value for o in ui.options]
        ):
            # 7일 버튼에 없는 날은 사용자가 [달력에서 다른 날 고르기]를 먼저 누른다 - 그러면 달력(kind=date)이 온다.
            assert "__calendar__" in [o.value for o in ui.options], ui
            calendar = self._turn("", ("date", "__calendar__"))
            assert calendar["ui"].kind == "date" and calendar["ui"].field == "date", calendar["ui"]
        elif ui is not None and ui.kind == "select":
            # 화면에 떠 있는 선택지 중에서만 누른다(병원 카드·달력은 화면 컨트롤이라 예외).
            assert ui.field == field and value in [o.value for o in ui.options], (field, value, ui)
        return self._turn("", (field, value))

    def _turn(self, message: str, ui: tuple[str, str] | None) -> dict[str, Any]:
        out = self.orch.handle(make_request(message, ui=ui, context=self.ctx), {"storedFlow": _stored(self.prev), "rawMessage": message})
        assert out is not None
        self.prev = out
        self.outs.append(out)
        return out

    def typed_steps(self) -> list[str]:
        return [o["ui"].field for o in self.outs if o.get("ui") is not None and o["ui"].kind == "text"]


def _values(out: dict[str, Any]) -> list[str]:
    return [o.value for o in out["ui"].options]


def _fake() -> FakeReservation:
    fake = FakeReservation()
    fake.regions = list(REGIONS)
    return fake


# ---- 선택만으로 예약 완료 -----------------------------------------------------------------


def test_ward_self_without_address_books_by_choices_only() -> None:
    """피보호자 본인, 프로필 주소 없음: 시·도 → 지역 → 병원 → 날짜 → 시간 → 확인. 이름·전화는 context 라 묻지 않는다."""
    fake = _fake()
    s = _Session(fake, SELF_NO_ADDRESS)

    first = s.say("신경과 예약은 어떻게 하면 되나요?")
    assert first["ui"].field == "location_sido"
    assert _values(first) == ["서울", "경기", "부산", TYPE_LOCATION_OPTION]  # 병원 많은 시·도 순 + 직접 입력은 맨 끝
    assert fake.region_departments == ["신경과"]  # 그 과가 있는 병원의 지역만

    regions = s.pick("location_sido", "서울")
    assert regions["ui"].field == "location" and regions["ui"].kind == "select"
    assert _values(regions)[:MAX_REGION_OPTIONS] == REGIONS[:2] + REGIONS[3:7]  # 서울만, 최대 6개
    assert _values(regions)[-2:] == [MORE_SIDO_OPTION, TYPE_LOCATION_OPTION]

    found = s.pick("location", "서울 서초구")
    assert found["tool"] == "search_hospital" and fake.searched_regions[0] == "서울 서초구"
    assert _values(found) == [OTHER_REGION_OPTION]  # 고른 지역의 결과에도 다른 지역 버튼

    date_ask = s.pick("hospital_confirm", "서초내과")
    assert date_ask["ui"].kind == "select" and date_ask["ui"].field == "date"
    time_ask = s.pick("date", FUTURE)
    assert time_ask["ui"].field == "time"
    card = s.pick("time", _values(time_ask)[0])
    assert card["ui"].field == "reservation_confirm"
    done = s.pick("reservation_confirm", next(v for v in _values(card) if v.startswith("confirm:")))

    assert fake.names() == ["create"] and "홍길동" in str(fake.calls[0][1])
    assert done["ui"].field == "reservation_followup"
    assert s.typed_steps() == []  # 글 입력 칸이 한 번도 나오지 않았다


def test_guardian_proxy_with_address_books_by_choices_only() -> None:
    """보호자 대리, 피보호자 주소 있음: 프로필 지역으로 바로 찾고, [다른 지역]은 같은 시·도의 다른 구를 고르게 한다."""
    fake = _fake()
    s = _Session(fake, WARD_WITH_ADDRESS)

    found = s.say("내과 예약해줘")
    assert found["tool"] == "search_hospital" and _values(found) == [OTHER_REGION_OPTION]

    regions = s.pick("location_confirm", OTHER_REGION_OPTION)
    assert regions["ui"].field == "location"
    assert "서울 강남구" not in _values(regions)  # 방금 찾은 지역은 뺀다
    assert all(v.startswith("서울 ") for v in _values(regions)[:-2])

    s.pick("location", "서울 송파구")
    assert fake.searched_regions[-1] == "서울 송파구"
    s.pick("hospital_confirm", "강남메디내과")
    time_ask = s.pick("date", FUTURE)
    card = s.pick("time", _values(time_ask)[0])
    assert "김영희" in card["reply"]  # 환자는 BE가 넘긴 피보호자
    s.pick("reservation_confirm", next(v for v in _values(card) if v.startswith("confirm:")))

    assert fake.names() == ["create"] and "김영희" in str(fake.calls[0][1])
    assert s.typed_steps() == []


# ---- 지역 선택지 세부 --------------------------------------------------------------------


def test_profile_region_comes_first_and_other_sido_lists_sidos() -> None:
    fake = _fake()
    s = _Session(fake, {**SELF_NO_ADDRESS, "address": "서울특별시 송파구 올림픽로 1"})
    s.say("내과 예약해줘")
    regions = s.pick("location_confirm", OTHER_REGION_OPTION)
    assert "서울 송파구" not in _values(regions)

    sidos = s.pick("location", MORE_SIDO_OPTION)
    assert sidos["ui"].field == "location_sido" and _values(sidos)[-1] == TYPE_LOCATION_OPTION
    gyeonggi = s.pick("location_sido", "경기")
    assert _values(gyeonggi) == ["경기 성남시", MORE_SIDO_OPTION, TYPE_LOCATION_OPTION]


def test_province_long_name_and_city_district_match() -> None:
    """"경기도 성남시 분당구 …" 주소는 병원 지역 "경기 성남시"와 같은 곳이다."""
    assert _sido_of("경기도 성남시 분당구 판교로 1") == "경기"
    assert _sido_of("서울특별시 강남구") == "서울" and _sido_of("강남구") is None
    assert _same_region("경기 성남시", "경기도 성남시 분당구 판교로 1")
    assert not _same_region("서울 강남구", "서울특별시 강동구 천호대로 1")


def test_type_option_falls_back_to_text_input() -> None:
    """[직접 입력]을 누르면 그때만 글 입력 칸을 준다(시·도·지역 선택지 어느 쪽에서든)."""
    for field in ("location_sido", "location"):
        fake = _fake()
        s = _Session(fake, SELF_NO_ADDRESS)
        s.say("신경과 예약해줘")
        if field == "location":
            s.pick("location_sido", "서울")
        typed = s.pick(field, TYPE_LOCATION_OPTION)
        assert typed["ui"].kind == "text" and typed["ui"].field == "location"


def test_region_lookup_failure_keeps_text_input() -> None:
    """병원 목록 조회가 실패하면(regions=None) 예전처럼 글로 묻는다 - 예약이 막히지 않게."""
    s = _Session(FakeReservation(), SELF_NO_ADDRESS)
    out = s.say("신경과 예약해줘")
    assert out["ui"].kind == "text" and out["ui"].field == "location"


# ---- 막히던 곳 ------------------------------------------------------------------------------


def test_no_hospital_found_offers_other_region_or_department() -> None:
    fake = _fake()
    fake.search_rows = []
    s = _Session(fake, {**SELF_NO_ADDRESS, "location": "서울 강남구"})
    out = s.say("신경과 예약해줘")
    assert out["ui"].field == "location_confirm" and _values(out) == [OTHER_REGION_OPTION, OTHER_DEPARTMENT_OPTION]

    dept = s.pick("location_confirm", OTHER_DEPARTMENT_OPTION)
    assert dept["ui"].field == "department" and "내과" in _values(dept)


def test_slot_lookup_failure_offers_retry_and_other_date() -> None:
    fake = _fake()
    fake.slots_fail = True
    s = _Session(fake, {**SELF_NO_ADDRESS, "location": "서울 강남구"})
    s.say("강남메디내과 내과 예약해줘")
    failed = s.pick("date", FUTURE)
    assert failed["ui"].field == "time" and _values(failed) == [RETRY_SLOTS_OPTION, CHANGE_DATE_OPTION]

    fake.slots_fail = False
    retried = s.pick("time", RETRY_SLOTS_OPTION)
    assert retried["ui"].field == "time" and "09:00" in [o.label for o in retried["ui"].options]


# ---- 진료과 답 → 버튼으로 예약 시작 --------------------------------------------------------


def test_department_answer_button_starts_reservation() -> None:
    """"무슨 과로 가야 해?" 답에 [가까운 ○○ 찾기] 버튼 - 누르면 예약 흐름(주소 없으면 지역 선택지)으로 들어간다."""
    from app.services.chat_service import ChatService

    hist = [{"role": "assistant", "content": "정형외과 진료를 권해드려요."}]
    answer = ChatService._department_question_reply("무슨 과로 가야 해?", hist)
    assert answer is not None and "말씀해 주세요" not in answer["reply"]
    assert answer["ui"].field == "department" and _values(answer) == ["정형외과"]
    assert answer["ui"].options[0].label == "가까운 정형외과 찾기"

    fake = _fake()
    out = make_orchestrator(fake, {}).handle(
        make_request(ui=("department", "정형외과"), context=SELF_NO_ADDRESS), {"storedFlow": {}}
    )
    assert out is not None and out["ui"].field == "location_sido"
    assert fake.region_departments == ["정형외과"]


# ---- 예약 서비스 클라이언트: 지역 목록 --------------------------------------------------------


def test_client_list_regions_counts_department_hospitals(monkeypatch) -> None:
    from app.services.reservation_api_client import ReservationApiClient

    client = ReservationApiClient.__new__(ReservationApiClient)
    monkeypatch.setattr(ReservationApiClient, "_resolve_api_key", staticmethod(lambda _k: None))
    hospitals = [
        {"name": "A내과", "region": "서울 서초구", "departments": ["내과"]},
        {"name": "B내과", "region": "서울  서초구", "departments": ["내과", "신경과"]},
        {"name": "C신경과", "region": "경기 성남시", "departments": ["신경과"]},
        {"name": "D안과", "region": "부산 남구", "departments": ["안과"]},
        {"name": "지역없음", "region": "", "departments": ["신경과"]},
        "깨진 행",
    ]
    monkeypatch.setattr(client, "_request", lambda *a, **k: (200, hospitals), raising=False)
    assert client.list_regions("") == ["서울 서초구", "경기 성남시", "부산 남구"]  # 병원 수 많은 순
    assert client.list_regions("신경과") == ["경기 성남시", "서울 서초구"]  # 같은 수면 이름순
    monkeypatch.setattr(client, "_request", lambda *a, **k: (500, None), raising=False)
    assert client.list_regions("내과") is None


# ---- 2026-11-04 기능 점검 반영 (M-1 · L-1 · L-2 · L-3) ------------------------------------------


def test_department_button_without_stored_draft_drops_old_hospital_and_time() -> None:
    """M-1: 저장 초안 없이 누른 진료과 버튼은 새 요청이다 - 같은 세션 이력의 지난 예약 병원·시간을 되살리지 않고 병원부터 찾는다."""
    hist = [
        {"role": "user", "content": "강남메디내과 내과 예약해줘"},
        {"role": "assistant", "content": "예약 시간을 선택해 주세요."},
        {"role": "user", "content": "14:00"},
        {"role": "assistant", "content": "알겠습니다. 예약은 그대로 둘게요."},
        {"role": "user", "content": "무슨 과로 가야 해?"},
        {"role": "assistant", "content": "정형외과로 가시면 돼요. 가까운 정형외과를 찾아드릴까요?"},
    ]
    fake = _fake()
    out = make_orchestrator(fake, {}).handle(
        make_request("병원 예약: / 정형외과", ui=("department", "정형외과"), history=hist, context={**SELF_NO_ADDRESS, "location": "서울 강남구"}),
        {"storedFlow": {}, "rawMessage": ""},
    )
    assert out is not None and out["tool"] == "search_hospital"  # 옛 병원으로 바로 날짜를 묻지 않는다
    draft = out["meta"]["flowState"]["draft"]
    assert draft["department"] == "정형외과" and draft.get("reservation_time") is None and draft.get("hospital_name") is None


def test_navigation_button_values_stay_out_of_server_history() -> None:
    """L-1: 화면 이동 버튼 값은 다음 턴 라우터·예약 추출(Claude) 이력에 사용자 말로 들어가지 않는다. 고른 지역은 들어간다."""
    from app.services.chat_service import ChatService

    for field, value in (
        ("time", RETRY_SLOTS_OPTION), ("location", TYPE_LOCATION_OPTION), ("location", MORE_SIDO_OPTION),
        ("location_confirm", OTHER_DEPARTMENT_OPTION), ("location_confirm", OTHER_REGION_OPTION), ("time", CHANGE_DATE_OPTION),
    ):
        assert ChatService._history_user_text(f"[선택] {field}: {value}") is None, (field, value)
    assert ChatService._history_user_text("[선택] location: 서울 서초구") == "서울 서초구"


def test_follow_up_region_questions_do_not_repeat_first_question() -> None:
    """L-2: 시·도를 고른 뒤·다른 시·도·다른 지역 단계는 첫 질문("○○과로 가시면 돼요…")을 되풀이하지 않는다."""
    fake = _fake()
    s = _Session(fake, SELF_NO_ADDRESS)
    first = s.say("신경과 예약해줘")
    assert first["reply"].startswith("신경과로 가시면 돼요.")
    assert s.pick("location_sido", "서울")["reply"] == "서울에서 어느 지역으로 찾을까요?"
    assert s.pick("location", MORE_SIDO_OPTION)["reply"] == "어느 시·도에서 찾을까요?"
    s.pick("location_sido", "서울")
    s.pick("location", "서울 서초구")
    assert s.pick("location_confirm", OTHER_REGION_OPTION)["reply"] == "다른 지역 어디에서 찾을까요?"


def test_region_lookup_lets_chat_deadline_through() -> None:
    """L-3: 지역 조회 중 챗 마감은 삼키지 않는다 - 다른 예약 호출처럼 chat_service 의 마감 폴백으로 간다."""
    import pytest

    from app.services.chat_deadline import ChatDeadlineExceeded

    fake = _fake()

    def deadline(*_a, **_k):
        raise ChatDeadlineExceeded("reservation_api")

    fake.list_regions = deadline  # type: ignore[method-assign]
    with pytest.raises(ChatDeadlineExceeded):
        make_orchestrator(fake, {}).handle(make_request("신경과 예약해줘", context=SELF_NO_ADDRESS), {"rawMessage": "신경과 예약해줘"})
