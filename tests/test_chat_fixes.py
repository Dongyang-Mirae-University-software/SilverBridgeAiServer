from __future__ import annotations

from datetime import date, timedelta

from app.core.reservation_prompts import build_user_prompt
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService
from app.services.reservation_intake import ReservationDraft
from app.services.reservation_orchestrator import ReservationOrchestrator
from app.utils.json_extract import extract_json_fields_lenient, extract_json_object


TRUNCATED = '''{
  "summary": "아침 기립성 어지럼일 수 있습니다.",
  "possibleCauses": ["기립성 저혈압", "탈수"],
  "homeCare": ["천천히 일어나기", "수분 섭취"],
  "visitHospitalIf": ["반복되거나 심해질 때'''


def test_lenient_extractor_salvages_truncated_json() -> None:
    assert extract_json_object(TRUNCATED) is None
    parsed = extract_json_fields_lenient(TRUNCATED, ["summary", "finalMessage"], ["possibleCauses", "homeCare", "visitHospitalIf"])
    assert parsed["summary"] == "아침 기립성 어지럼일 수 있습니다."
    assert parsed["possibleCauses"] == ["기립성 저혈압", "탈수"]
    assert parsed["homeCare"] == ["천천히 일어나기", "수분 섭취"]
    assert "visitHospitalIf" not in parsed  # 잘린 항목은 버린다
    assert "finalMessage" not in parsed


def test_intent_keywords_cover_common_phrasings() -> None:
    assert ChatService.classify_intent("병원 좀 잡아줘 이비인후과로") == "hospital_reservation"
    assert ChatService.classify_intent("아침에 일어날 때 자꾸 어지러워요") == "medical_advice"
    assert ChatService.classify_intent("당뇨가 있는데 과일은 얼마나 먹어도 되나요?") == "medical_advice"
    assert ChatService.classify_intent("오늘 날씨가 좋네요") == "general_chat"
    assert MedicalLlmService.classify_intent("계단 오를 때 무릎이 아프고 시려요") == "medical_advice"


def test_extraction_prompt_contains_today() -> None:
    prompt = build_user_prompt(message="내일 오후 2시", history_text="", state_json="{}", location="")
    assert date.today().isoformat() in prompt or "[오늘 날짜" in prompt


def test_past_extracted_date_is_ignored() -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    draft = ReservationDraft(message="내일 오후 2시", reservation_date=tomorrow)
    ReservationOrchestrator._apply_extracted(draft, {"reservation_date": "2023-10-04", "reservation_time": "14:00"})
    assert draft.reservation_date == tomorrow
    assert draft.reservation_time == "14:00"


def test_placeholder_values_are_dropped() -> None:
    assert MedicalLlmService._normalize_text("...") == ""
    assert MedicalLlmService._normalize_list(["...", "충분한 휴식"]) == ["충분한 휴식"]


def test_parsed_json_is_not_backfilled_from_raw_text() -> None:
    raw = '{"summary": "산책은 괜찮습니다.", "possibleCauses": [], "homeCare": [], "visitHospitalIf": [], "emergencyWarning": [], "finalMessage": "물을 챙기세요.", "needsReservation": false}'
    parsed = extract_json_object(raw)
    out = MedicalLlmService._normalize_payload(parsed, raw, "산책 나가도 괜찮을까요?", "general_chat", False)
    assert out["possibleCauses"] == [] and out["homeCare"] == []
    assert out["summary"] == "산책은 괜찮습니다." and out["finalMessage"] == "물을 챙기세요."


def test_risk_is_high_only_with_emergency_signal() -> None:
    generic = ["통증이 심해지면 병원 방문"]
    assert MedicalLlmService.infer_risk("계단 오를 때 무릎이 아파요", "medical_advice", generic)[0] == "medium"
    assert MedicalLlmService.infer_risk("갑자기 가슴이 조이고 식은땀이 나요", "medical_advice", generic)[0] == "high"
    assert MedicalLlmService.infer_risk("숨이 안 쉬어져요", "emergency_guidance", [])[0] == "high"
    assert MedicalLlmService.infer_risk("오늘 날씨가 좋네요", "general_chat", [])[0] == "low"
