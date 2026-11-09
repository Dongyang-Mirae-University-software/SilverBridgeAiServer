"""라우터 실패(키워드 분류) 때 응급 판정 범위(2026-11-06 점검 H-1·M-1).

라우터가 실패하면 응급 판단 근거는 chat_keywords.emergency_signals 하나다. 붙여 쓴 꼴만 보던 목록이 시험지 응급 80개 중
21개를 놓쳤다(심정지·흉통·뇌졸중·출혈·경련·질식·아나필락시스·저혈당·의식 변화). 이 판정은 라우터가 정상일 때도
_force_emergency 로 응급을 강제하므로 과잉도 함께 고정한다.
"""
from __future__ import annotations

import glob
import json
import pathlib

import pytest

from app.services import chat_keywords as kw
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService

_EVAL = pathlib.Path(__file__).parent / "eval"
# 라우터만 잡는 응급 문항 - 지금은 없다. 애매한 자해 표현(sh.09)은 라우터 실패 전용 패턴(2026-11-09)이, 다중 턴 위기 이어짐(sh.11)은
# _crisis_continues_without_router 가 맡는다(2026-11-09 재측정 - 전에는 history 를 넘기지 않아 누락으로 셌다).
_ROUTER_ONLY: set[str] = set()


def _fallback_emergency(q: str) -> bool:
    return bool(ChatService._force_emergency(ChatService._keyword_route(q), q).get("emergency"))


def _fallback_turn_emergency(q: str, history: list[dict] | None = None) -> bool:
    """라우터 실패 턴의 응급·위기 판정 - process_message 와 같은 순서(자해 키워드 → 키워드 분류 → 위기 이어짐 → 자해 패턴)."""
    history = history or []
    if MedicalLlmService.self_harm_signals(q):
        return True
    route = ChatService._force_emergency(ChatService._keyword_route(q), q)
    if ChatService._self_harm_trigger(False, route, history) is not None or route.get("emergency"):
        return True
    return ChatService._crisis_continues_without_router(q, history) or bool(kw.fallback_self_harm_signals(q))


def _unique_items():
    seen: set[str] = set()
    for f in sorted(glob.glob(str(_EVAL / "*.json"))):
        for it in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("items", []):
            q = it.get("q") or ""
            key = q + "\x00" + json.dumps(it.get("history") or [], ensure_ascii=False)
            if q and key not in seen:
                seen.add(key)
                yield it


def test_eval_sets_emergency_caught_without_router() -> None:
    missed = [it["id"] for it in _unique_items()
              if (it.get("expect") or {}).get("emergency") is True and not _fallback_turn_emergency(it["q"], it.get("history"))]
    assert set(missed) <= _ROUTER_ONLY, missed


def test_eval_sets_general_not_overtriaged() -> None:
    over = [it["id"] for it in _unique_items()
            if (it.get("expect") or {}).get("emergency") is False and _fallback_turn_emergency(it["q"], it.get("history"))
            and not MedicalLlmService.self_harm_signals(it["q"])]
    assert over == [], over


# 유형별 합성 문장(시험지 문장과 다르게) - 라우터 없이 응급이어야 한다.
@pytest.mark.parametrize("q", [
    "아버지가 가슴을 꽉 쥐어짜는 것 같다고 하세요",
    "할머니를 아무리 깨워도 안 일어나세요",
    "남편이 갑자기 축 늘어져서 대답이 없어요",
    "엄마 숨소리가 전혀 안 들려요",
    "할아버지 입술이 퍼렇게 변했어요",
    "약 먹고 혀가 부어서 숨이 답답해요",
    "넘어져서 머리에서 피가 계속 안 멈춰요",
    "아버지가 피를 많이 토하셨어요",
    "엄마 입이 갑자기 한쪽으로 돌아갔어요",
    "남편 말이 갑자기 어눌해졌어요",
    "갑자기 왼쪽 팔에 힘이 빠져요",
    "할머니가 거품 물고 떠세요",
    "고기 먹다 목에 걸림 말을 못 해요",
    "혈당이 45 나오고 식은땀이 나요",
    "할아버지가 갑자기 저를 못 알아보세요",
])
def test_emergency_variants_caught(q: str) -> None:
    assert _fallback_emergency(q), q


# 응급이 아닌 말 - 기존 예외(알약·비유·말 꼬임 정정)와 이번 패턴의 경계.
@pytest.mark.parametrize("q", [
    "알약이 커서 목에 걸려요",
    "숨이 멎을 만큼 놀랐어요",
    "말이 꼬여서 다시 말할게요",
    "혈당이 120이에요",
    "혈당 300 넘었어요",
    "아침에 깨워도 안 일어나는 손주 때문에 힘들어요",
    "피가 조금 났는데 금방 멈췄어요",
    "갑자기 비가 와서 식은땀 날 뻔했어요",
    "오른쪽 다리에 힘이 좀 없어요 무릎 수술 후라",
    "갑자기 손주가 엉뚱한 소리를 해서 웃었어요",
    "치매 엄마가 저를 못 알아봐요",
    "넘어지면 못 일어나실까봐 걱정이에요",
    "쓰러지지 않게 뭘 조심해야 해요",
    "혈당이 50 정도 나와요",
])
def test_non_emergency_not_caught(q: str) -> None:
    assert not _fallback_emergency(q), q


def test_router_ok_path_keeps_old_keyword_scope() -> None:
    # 거리 패턴은 라우터 실패 때만 - 라우터가 정상이면 응급 판단·위기 해제·되묻기 경계는 라우터 칸과 기존 키워드 그대로다.
    assert not kw.emergency_signals("갑자기 식은땀이 나고 눈앞이 캄캄해요")
    assert kw.fallback_emergency_signals("갑자기 식은땀이 나고 눈앞이 캄캄해요")


def test_worry_exclusion_only_when_all_mentions_are_worry() -> None:
    # 걱정 꼴과 실제 상황이 함께면 응급이다.
    assert kw.emergency_signals("넘어지면 못 일어나실까봐 걱정했는데 지금 쓰러지셔서 못 일어나세요")
