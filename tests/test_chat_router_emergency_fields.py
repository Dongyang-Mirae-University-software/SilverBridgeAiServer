"""라우터 심정지·경계 질환 칸(2026-11-04).

- cardiac_arrest: 키워드 밖 심정지 표현에도 가슴압박·AED 줄. 의식 있다는 말(추측·회복)이 있으면 따르지 않는다. true면 응급.
- urgent_condition: 허용 목록(chat_keywords.URGENT_CONDITIONS) 중 하나 → 일반 상담 답에 그 병의 고정 문장(병 이름·오늘 진료·악화 신호).
  허용 목록 밖·응급 턴·라우터 실패면 아무것도 붙이지 않는다.
"""
from __future__ import annotations

import json
import re
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatRequest
from app.services import chat_keywords as kw
from app.services import chat_router_service as crs
from app.services import chat_service as cs
from app.services import medical_llm_service as mls
from app.services.medical_llm_service import FIXED_SENTENCES, MedicalLlmService

CPR = re.compile(r"가슴압박\(가슴 한가운데를 세게, 빠르게")


# ---- 라우터 출력 해석 ------------------------------------------------------------------------------------------

class _Client:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.state = SimpleNamespace(model_name="fake")

    def generate_structured_text(self, system, user, **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.payload, ensure_ascii=False)


def _route(**fields: Any) -> dict[str, Any]:
    base = {"route": "symptom", "normalized_message": "", "symptom_summary": "s", "emergency": False, "self_harm": False, "reason": "r"}
    return crs.ChatRouterService(_Client({**base, **fields}), 4).route("문장", [], None)


def test_cardiac_arrest_true_makes_emergency() -> None:
    out = _route(cardiac_arrest=True)
    assert out["cardiac_arrest"] is True and out["emergency"] is True


def test_missing_new_fields_keep_old_behaviour() -> None:
    out = _route()  # 옛 출력 모양 - 칸이 없으면 false / None
    assert out["cardiac_arrest"] is False and out["urgent_condition"] is None and out["emergency"] is False


@pytest.mark.parametrize("value", ["", "stroke", "DVT", None, 3])
def test_urgent_condition_outside_allowlist_is_dropped(value: Any) -> None:
    assert _route(urgent_condition=value)["urgent_condition"] is None


@pytest.mark.parametrize("flag", ["emergency", "self_harm", "cardiac_arrest"])
def test_urgent_condition_is_dropped_on_emergency_turns(flag: str) -> None:
    assert _route(urgent_condition="dvt", **{flag: True})["urgent_condition"] is None


def test_urgent_condition_allowlisted_value_is_kept() -> None:
    assert _route(urgent_condition="dvt")["urgent_condition"] == "dvt"


def test_schema_and_prompt_list_every_allowed_condition() -> None:
    enum = crs.ROUTE_SCHEMA["properties"]["urgent_condition"]["enum"]
    assert enum == ["", *kw.URGENT_CONDITIONS]
    for key in kw.URGENT_CONDITIONS:
        assert f"- {key}:" in crs.SYSTEM_PROMPT
    assert "cardiac_arrest" in crs.ROUTE_SCHEMA["required"] and "urgent_condition" in crs.ROUTE_SCHEMA["required"]


# ---- 고정 문장 표 ---------------------------------------------------------------------------------------------

def test_every_allowed_condition_has_exactly_one_fixed_sentence() -> None:
    names = [s.name for s in FIXED_SENTENCES if s.name.startswith("_URGENT_CONDITIONS.")]
    assert sorted(n.split(".", 1)[1] for n in names) == sorted(kw.URGENT_CONDITIONS)
    for key in kw.URGENT_CONDITIONS:
        sentence = MedicalLlmService.urgent_condition_sentence(key)
        assert sentence and "오늘" in sentence or "바로" in sentence
        assert re.search(r"119|응급실", sentence)


# ---- 응급 행동 줄 + 라우터 심정지 힌트 ---------------------------------------------------------------------------

def test_router_hint_adds_cpr_for_expression_outside_keywords() -> None:
    message = "아버지가 갑자기 몸이 축 처지고 꼼짝을 안 하세요"
    assert "cardiac_arrest" not in [n for n, _ in MedicalLlmService.emergency_action_sentences(message, True)]
    names = [n for n, _ in MedicalLlmService.emergency_action_sentences(message, True, arrest_hint=True)]
    assert names[0] == "cardiac_arrest"


@pytest.mark.parametrize("message", ["정신을 잃을 것 같아요", "엄마가 쓰러지셨다가 금방 깨어나셨어요"])
def test_router_hint_is_ignored_when_person_is_conscious(message: str) -> None:
    names = [n for n, _ in MedicalLlmService.emergency_action_sentences(message, False, arrest_hint=True)]
    assert "cardiac_arrest" not in names


def test_router_hint_does_not_duplicate_keyword_cpr_and_drops_conflicting_lines() -> None:
    names = [n for n, _ in MedicalLlmService.emergency_action_sentences("남편이 쓰러져서 숨을 안 쉬어요", True, arrest_hint=True)]
    assert names.count("cardiac_arrest") == 1
    assert "collapse_fall" not in names and "breathing" not in names


# ---- 생성(MedGemma 대체) ---------------------------------------------------------------------------------------

class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "다리 부기", "features": "", "advice": "내과 진료를 권해드려요.", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt, user_prompt, prefill="", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **kwargs: Any) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader()).generate(message, {}, [], intent=intent, **kwargs)


def test_urgent_condition_sentence_is_added_once_to_general_answer() -> None:
    result = _gen("비행기 타고 온 뒤로 오른쪽 종아리만 부었어요", urgent_condition="dvt")
    assert result["reply"].count("심부정맥혈전증") == 1
    assert "119" in result["reply"] and result["riskLevel"] != "high"


def test_urgent_condition_is_not_added_to_emergency_answer() -> None:
    result = _gen("가슴이 아파요", "emergency_guidance", urgent_condition="dvt")
    assert "심부정맥혈전증" not in result["reply"]


def test_unknown_urgent_condition_adds_nothing() -> None:
    assert "오늘 안에 진료" not in _gen("다리가 부었어요", urgent_condition="nope")["reply"]


def test_router_arrest_hint_reaches_emergency_answer() -> None:
    result = _gen("할머니가 꼼짝을 안 하세요 숨소리도 안 들려요", "emergency_guidance", arrest_hint=True)
    assert CPR.search(result["reply"]) and "AED" in result["reply"]


# ---- 서비스 연결(라우터 칸 → 생성 인자) ----------------------------------------------------------------------------

class _Router:
    def __init__(self, **extra: Any) -> None:
        self.extra = extra

    def route(self, message, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": "symptom", "normalized_message": message, "symptom_summary": message, "emergency": False,
                "self_harm": False, "reason": "t", "engine": "fake", **self.extra}


class _Llm:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate(self, message, ctx, history, **kwargs) -> dict[str, Any]:  # noqa: ARG002
        self.kwargs = kwargs
        return {"reply": "내과 진료를 권해드려요.", "riskLevel": "medium", "recommendedAction": "guardian_contact",
                "reservationRequired": False, "intent": kwargs.get("intent"), "engine": "medgemma", "modelName": "m",
                "summary": "s", "possibleCauses": [], "homeCare": [], "visitHospitalIf": [], "emergencyWarning": []}


def _send(monkeypatch, router: _Router, message: str) -> tuple[_Llm, dict[str, Any]]:
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_router_fields_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    llm = _Llm()
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: router)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: object())
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: llm)
    svc = cs.ChatService()
    svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake")
    try:
        svc.process_message(db, ChatRequest(userId="rf1", message=message, sessionId="s1"))
        row = db.query(ChatLog).order_by(ChatLog.id.desc()).first()
        meta = json.loads(row.context_json or "{}").get("upstreamMeta") or {}
    finally:
        db.close()
    return llm, meta


def test_service_passes_urgent_condition_to_generation(monkeypatch) -> None:
    llm, out = _send(monkeypatch, _Router(urgent_condition="dvt"), "오른쪽 종아리만 붓고 욱신거려요 비행기 오래 탔어요")
    assert llm.kwargs.get("urgent_condition") == "dvt"
    assert out["router"]["urgent_condition"] == "dvt"


def test_service_passes_arrest_hint_and_forces_emergency(monkeypatch) -> None:
    llm, out = _send(monkeypatch, _Router(cardiac_arrest=True, emergency=True), "아버지가 꼼짝을 안 하세요")
    assert llm.kwargs.get("arrest_hint") is True and llm.kwargs.get("intent") == "emergency_guidance"
    assert out["router"]["cardiac_arrest"] is True


def test_service_without_new_fields_calls_generation_as_before(monkeypatch) -> None:
    llm, _ = _send(monkeypatch, _Router(), "무릎이 몇 달째 아파요")
    assert "arrest_hint" not in llm.kwargs and "urgent_condition" not in llm.kwargs


def test_unlisted_urgent_condition_from_router_is_not_passed(monkeypatch) -> None:
    llm, _ = _send(monkeypatch, _Router(urgent_condition="made_up"), "다리가 부었어요 며칠 됐어요")
    assert "urgent_condition" not in llm.kwargs


def test_urgent_module_lists_match() -> None:
    assert [u.name for u in mls._URGENT_CONDITIONS] == list(kw.URGENT_CONDITIONS)


# ---- 피보호자 본인 짧은 답(#173)에서도 잘리지 않는다 --------------------------------------------------------------

def test_ward_concise_answer_keeps_urgent_condition_sentence() -> None:
    from app.services import chat_layout

    result = _gen("비행기 타고 온 뒤로 오른쪽 종아리만 부었어요", urgent_condition="dvt")
    out = chat_layout.apply(dict(result), mls.DRUG_ALCOHOL_SENTENCES, concise=True)
    assert "심부정맥혈전증" in out["reply"] and "119" in out["reply"]


def test_ward_concise_emergency_answer_keeps_cpr_line() -> None:
    from app.services import chat_layout

    result = _gen("할머니가 꼼짝을 안 하세요 숨소리도 안 들려요", "emergency_guidance", arrest_hint=True)
    out = chat_layout.apply(dict(result), mls.DRUG_ALCOHOL_SENTENCES, concise=True)
    assert CPR.search(out["reply"]) and "AED" in out["reply"]
