from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService

# 2026-10-24 응급 키워드 보강: 라우터(외부 모델)가 놓쳐도 키워드로 응급 안내가 나가게 한다.
ADDED = {
    "일어나지못하겠": "넘어졌는데 일어나지 못하겠어요",
    "일어나지를못": "넘어져서 일어나지를 못해요",
    "일어날수가없": "넘어져서 일어날 수가 없어요",
    "일어설수가없": "다리에 힘이 빠져 일어설 수가 없어요",
    "발음이꼬": "아버지 발음이 꼬여요",
    "혀가꼬": "갑자기 혀가 꼬여서 말이 안 돼요",
    "말이자꾸꼬": "엄마 말이 자꾸 꼬여요",
    "의식이흐려": "의식이 흐려져요",
    "의식이흐릿": "할머니 의식이 흐릿해요",
    "발음이자꾸꼬": "엄마가 말할 때 발음이 자꾸 꼬여요",
    "가슴이쪼이": "가슴이 쪼이고 아파요",
    "가슴이쪼여": "가슴이 쪼여요",
    "가심이쪼이": "가심이 쪼이고 식은땀이 나",
    "가심이쪼여": "가심이 쪼여",
    "가심이조이": "가심이 조이고 식은땀이 나",
    "가심이조여": "가심이 조여",
    "가슴이꽉조이": "가슴이 꽉 조이는 느낌이에요",
    "가슴이꽉조여": "가슴이 꽉 조여요",
    "가슴이너무조이": "가슴이 너무 조이는데 병원 가야 하나요",
    "가슴이너무조여": "가슴이 너무 조여요",
    "토한것이커피색": "토한 것이 커피색이에요",
    "토한게커피색": "토한 게 커피색이야",
    "커피색으로토": "커피색으로 토했어요",
}

# 응급이 아닌데 비슷한 표현 - 걸리면 키워드를 좁힌다.
NOT_EMERGENCY = [
    "요즘 아침에 일어나지 못한 날이 많아요",
    "말이 꼬여서 다시 말할게요",
    "커피색 옷이 좋아요",
    "일을 다 끝내고 싶어",
    "피곤해서 눈이 풀렸어",
]

# 시험지 v3 에서 라우터만 믿던 문항(2026-10-24 O-2에서 가벼운 모델이 놓친 qa.C4·var.06 포함).
EVAL_IDS_NOW_KEYWORD = ("qa.B3", "qa.C4", "qa.C5", "var.01", "var.06", "rs.01")


def _eval_items() -> dict[str, dict]:
    path = Path(__file__).resolve().parent / "eval" / "eval_set_v3.json"
    return {item["id"]: item for item in json.loads(path.read_text(encoding="utf-8"))["items"]}


@pytest.mark.parametrize("keyword,message", list(ADDED.items()))
def test_added_keyword_is_emergency(keyword: str, message: str) -> None:
    assert keyword in MedicalLlmService.EMERGENCY_KEYWORDS
    assert keyword in MedicalLlmService.emergency_signals(message), message
    assert ChatService.classify_intent(message) == "emergency_guidance", message


@pytest.mark.parametrize("message", NOT_EMERGENCY)
def test_similar_everyday_phrasing_is_not_emergency(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message) == [], message
    assert MedicalLlmService.self_harm_signals(message) == [], message


def test_broad_keywords_stay_narrowed() -> None:
    # 넓은 형태로 되돌리면 위 일상 표현이 응급으로 잡힌다.
    for broad in ("일어나지못", "말이꼬", "커피색", "눈이풀렸", "다끝내고싶"):
        assert broad not in MedicalLlmService.EMERGENCY_KEYWORDS, broad


@pytest.mark.parametrize("item_id", EVAL_IDS_NOW_KEYWORD)
def test_router_only_eval_items_are_now_caught_by_keyword(item_id: str) -> None:
    question = _eval_items()[item_id]["q"]
    assert MedicalLlmService.emergency_signals(question), item_id


def test_force_emergency_routes_qa_c4_without_router_emergency() -> None:
    # 라우터가 emergency=False·general 로 줘도 키워드가 응급 경로로 보낸다.
    question = _eval_items()["qa.C4"]["q"]
    routed = ChatService._force_emergency({"route": "general", "emergency": False, "symptom_summary": ""}, question)
    assert routed["route"] == "symptom" and routed["emergency"] is True


def test_no_non_emergency_eval_item_is_newly_caught() -> None:
    added = set(ADDED)
    for item in _eval_items().values():
        if item["expect"].get("emergency") is True:
            continue
        hits = set(MedicalLlmService.emergency_signals(item["q"])) & added
        assert not hits, (item["id"], hits)
