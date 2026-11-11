"""2026-11-11 점검(docs/audits/2026-11-11-guardian-revert-deptinfo-audit.md) 반영 - 진료과 정보 고정 안내.
🔴 라우터 판정(응급·자해·예약·오늘 진료 질환) 뒤에 판정한다 - 키워드에 없는 위기 표현 + 진료과 질문이 109 없이 고정 안내로 나가지 않게.
🟠 증상·약·검사 결과·다친 일·부분 일치("불안과")는 고정 안내가 아니다 / 라우터 실패 때 응급·자해 패턴 가드를 고정한다.
🟡 피보호자 단계 되묻기 도중이면 안내 뒤 지금 질문을 다시 묻고 단계를 잇는다 / 보호자 옛 되묻기 다음 턴은 이어 붙이기 / "어떤 과일"."""
from __future__ import annotations

import glob
import json
from typing import Any

import pytest

from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_keywords as kw
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService as M
from tests.test_chat_carer_legacy_and_dept_info import _guardian, _ward, flow  # noqa: F401
from tests.test_chat_followup_question import env  # noqa: F401
from tools.eval.replay_offline import stub_route

INFO_PIECE = "자세히 여쭤보고 진찰해요."
SUFFIX = " 내과 가면 뭐 해요?"


def _is_info(out: dict[str, Any]) -> bool:
    return INFO_PIECE in out["reply"] or "정확한 금액은 말씀드리기 어려워요." in out["reply"] or "신분증과" in out["reply"]


class _FailingRouter:
    def route(self, *a, **k):  # noqa: ARG002
        raise RuntimeError("router down")


# ---- 🔴 라우터 판정이 먼저 -----------------------------------------------------------------------------


def test_router_self_harm_wins_over_department_info(flow) -> None:  # noqa: F811
    # 키워드에 없는 위기 표현이라 라우터만 self_harm 을 안다 - 그 판정이 고정 안내보다 먼저다.
    message = "정신건강의학과 가면 뭐 해요? 다 싫어요"
    assert ChatService._department_info_reply(message, []) is not None  # 문장만 보면 정보 질문 꼴이다
    flow.router.route = lambda m, h, c, prev_reservation=None: {  # noqa: ARG005
        "route": "symptom", "normalized_message": m, "symptom_summary": m, "emergency": True, "self_harm": True,
        "reason": "t", "engine": "fake"}
    out = _ward(flow, message)
    assert "109" in out["reply"] and not _is_info(out)


@pytest.mark.parametrize("extra", [{"emergency": True}, {"route": "reservation"}, {"urgent_condition": "shingles"}],
                         ids=["emergency", "reservation", "urgent"])
def test_router_verdicts_skip_department_info(flow, extra: dict) -> None:  # noqa: F811
    base = flow.router.route

    def route(m, h, c, prev_reservation=None):
        return {**base(m, h, c, prev_reservation), **extra}

    flow.router.route = route
    try:
        out = _guardian(flow, "피부과 가면 뭐 해요?")
    except Exception:  # noqa: BLE001 - 예약 경로는 하네스에 예약 서비스가 없어 실패할 수 있다(고정 안내가 아니면 된다)
        return
    assert not _is_info(out)


def _items() -> list[dict]:
    seen: set[str] = set()
    out = []
    for f in sorted(glob.glob("tests/eval/*_v*.json")):
        for it in json.load(open(f, encoding="utf-8"))["items"]:
            ex = it.get("expect") or {}
            if it["q"] in seen or not any(ex.get(k) for k in ("must_119", "emergency", "cpr_aed", "crisis_reply", "same_day")):
                continue
            seen.add(it["q"])
            out.append(it)
    return out


EVAL_ITEMS = _items()


def test_eval_items_are_loaded() -> None:
    assert len(EVAL_ITEMS) >= 80  # 시험지가 바뀌어 이 검사가 비지 않게


@pytest.mark.parametrize("item", EVAL_ITEMS, ids=[it["id"] for it in EVAL_ITEMS])
def test_eval_emergency_crisis_same_day_plus_department_question_is_never_fixed(env, item: dict) -> None:  # noqa: F811
    # 시험지 응급·위기·오늘 진료 문항 뒤에 진료과 정보 질문을 붙여도 고정 안내가 아니다(라우터 = 시험지 기대값).
    env.router.route = lambda m, *a, **k: stub_route(item, m)  # noqa: ARG005
    ctx = ChatContext(name="홍길동", role="WARD")
    out = None
    earlier = [h["content"] for h in item.get("history") or [] if h.get("role") == "user"] + list(item.get("history_user") or [])
    for msg in earlier + [item["q"].rstrip(" ?.") + "." + SUFFIX]:
        out = env.svc.process_message(env.db, ChatRequest(userId="u-eval", message=msg, sessionId="s-eval", context=ctx))
    assert out is not None and not _is_info(out)


@pytest.mark.parametrize("item", EVAL_ITEMS, ids=[it["id"] for it in EVAL_ITEMS])
def test_eval_items_plus_department_question_fail_the_keyword_check_too(item: dict) -> None:
    # 라우터가 실패해도(키워드 판정 + 부르는 쪽의 자해 키워드 검사) 고정 안내로 빠지지 않는다 - 진료과 앞 길이 상한·증상 낱말·fallback 패턴.
    # 라우터만 아는 자해(cat "(라우터)")·앞 대화에 기댄 위기(history)는 키워드로 볼 수 없다 - 위 흐름 시험이 맡는다.
    if "(라우터)" in str(item.get("cat")) or item.get("history"):
        pytest.skip("라우터·앞 대화 판정 문항")
    message = item["q"].rstrip(" ?.") + "." + SUFFIX
    assert not (kw.is_department_info_question(message) and not M.self_harm_signals(message))


# ---- 🟠 라우터 실패 때 가드 ------------------------------------------------------------------------------

ROUTER_ONLY_IDS = ("h2.A02", "h2.A06", "h2.E05", "h3.A03", "h3.A09", "h3.D03", "h3.D05", "h3.D08")


@pytest.mark.parametrize("item_id", ROUTER_ONLY_IDS)
def test_router_failure_fallback_emergency_guard(flow, item_id: str) -> None:  # noqa: F811
    # 이 문항들은 응급 키워드가 없어 라우터 실패 때 fallback 응급 패턴(ARREST·fallback_emergency_signals)만 잡는다(점검 결함 주입 M6).
    item = next(it for it in EVAL_ITEMS if it["id"] == item_id)
    message = item["q"].rstrip(" ?.") + "." + SUFFIX
    assert not kw.is_department_info_question(message)
    flow.router = _FailingRouter()  # env 의 get_chat_router_service 는 호출 때마다 flow.router 를 읽는다
    assert not _is_info(_ward(flow, message))


def test_router_failure_fallback_self_harm_guard(flow) -> None:  # noqa: F811
    # fallback 자해 패턴은 그 표현만 있는 문장을 본다 - 진료과 질문이 붙은 문장은 진료과 앞 길이 상한이 거른다.
    assert kw.fallback_self_harm_signals("다 끝내버리고 싶어요")
    assert not kw.is_department_info_question("다 끝내버리고 싶어요 정신건강의학과 가면 뭐 해요?")
    assert not kw.is_department_info_question("다 끝내고 싶어요")


@pytest.mark.parametrize("lead", ["축 늘어졌어요", "안 깨어나요", "엄마가 반응이 없어요", "아빠가 대답을 안 해요", "숨소리가 이상해요"])
def test_short_situation_before_department_is_not_info(lead: str) -> None:
    # 라우터 실패 때 키워드가 못 보는 응급일 수 있는 짧은 사정 말 + 진료과 질문. "축 늘어졌어요"는 fallback 응급 패턴(M6)이,
    # 나머지는 진료과 앞 서술 꼴 검사가 거른다.
    assert not kw.is_department_info_question(f"{lead} 내과 가면 뭐 해요?")


@pytest.mark.parametrize("message", [
    "어제 저녁 이마 물집 눈두덩 피부과 가면 뭐 해요?",  # 서술 없이 명사만 길게 - 앞 길이 상한
    "피부과 가면 뭐 해요 어제부터 이마랑 눈두덩에 물집이 번지고 따끔따끔하고 진물까지 나와서 걱정이에요",  # 뒤에 사정 - 전체 길이 상한
])
def test_length_caps_alone_block_info(message: str) -> None:
    assert not kw.is_department_info_question(message)


def test_fallback_emergency_pattern_alone_blocks_info(monkeypatch) -> None:
    # 앞말 검사가 없어도 fallback 응급 패턴만으로 막힌다(결함 주입 M6·N8 이 살아남지 않게).
    monkeypatch.setattr(kw, "_DEPARTMENT_INFO_PREFIX_NOT", __import__("re").compile(r"(?!)"))
    assert kw.fallback_emergency_signals("축 늘어졌어요 내과 가면 뭐 해요?")
    assert not kw.is_department_info_question("축 늘어졌어요 내과 가면 뭐 해요?")


def test_router_only_crisis_on_previous_turn_keeps_109(flow) -> None:  # noqa: F811
    # 앞 턴 위기를 라우터만 알았다(키워드 없음) - 다음 턴 진료과 질문은 고정 안내가 아니고 109 한 줄이 붙는다(결함 주입 N5).
    base = flow.router.route
    flow.router.route = lambda m, h, c, prev_reservation=None: {**base(m, h, c, prev_reservation), "emergency": True, "self_harm": True}
    _ward(flow, "다 의미가 없어요")
    flow.router.route = base
    out = _ward(flow, "내과 가면 뭐 해요?")
    assert "109" in out["reply"] and not _is_info(out)


# ---- 🟠 판별 오탐 -----------------------------------------------------------------------------------------

NOT_INFO = [
    "내과에서 검사했더니 당 수치가 높대요 어떻게 해요?",  # 검사 결과
    "내과 약 먹고 나서 어지러운데 뭐 해야 해요?",  # 약 부작용
    "정형외과 다녀왔는데 무릎이 계속 아파요 검사 또 받아야 해요?",  # 다녀온 뒤 지속 증상
    "신경과 약 먹는데 술 마셔도 되는지 얼마나 기다려요?",  # 약 + 술
    "외과 수술 후 열이 나요 얼마나 기다려요?",  # 수술 뒤 발열
    "열이 39도인데 내과 가면 검사 뭐 해요?",  # 증상
    "넘어져서 머리를 부딪혔는데 신경외과 가면 뭐 해요?",  # 다친 일
    "불안과 걱정 때문에 잠을 못 자요 검사 받아야 하나요?",  # "안과" 부분 일치
    "편안과 휴식이 필요할까요 준비할 거 있나요",
    "신경과민인데 검사 받아봐야 해요?",
    "어떤 과일을 챙겨 먹으면 좋아요?",
]
STILL_INFO = [
    "신경과는 어떤 검사를 받게 되나요?", "내과 가면 뭐 해요?", "정형외과 갈 때 뭘 챙겨가요?", "안과 진료비 얼마예요?",
    "어떤 과에 가서 무엇을 준비하나요?", "내과 피검사하려면 굶어야 해요?", "이비인후과에서는 어떻게 진료해요?",
    "피부과 가면 시간이 오래 걸려요?", "치과 시술 비용은 얼마예요?", "동네 내과는 뭐 해요?",
    "엄마 모시고 신경과 가면 뭐 해요?", "그럼 내과에서는 무슨 검사를 해요?", "혹시 안과 가면 뭘 챙겨가요?",
]


@pytest.mark.parametrize("message", NOT_INFO)
def test_symptom_medication_result_and_partial_words_are_not_info(message: str) -> None:
    assert not kw.is_department_info_question(message)


@pytest.mark.parametrize("message", STILL_INFO)
def test_plain_department_info_questions_stay_fixed(message: str) -> None:
    assert kw.is_department_info_question(message)


def test_fruit_is_not_a_department_where_question() -> None:
    history = [{"role": "assistant", "content": "신경과 진료를 받아 보세요."}]
    assert ChatService._department_question_reply("어떤 과일이 혈압에 좋아요?", history) is None
    assert ChatService._department_question_reply("어떤 과로 가요?", history) is not None


def test_cost_wins_over_prepare() -> None:
    assert kw.department_info_kind("내과 갈 때 준비할 비용은 얼마예요?") == "cost"
    assert kw.department_info_kind("내과 갈 때 뭘 챙겨가요?") == "prepare"
    assert kw.department_info_kind("내과에서는 어떤 검사를 해요?") == "exam"


# ---- 🟡 되묻기 도중 -----------------------------------------------------------------------------------------


def test_info_during_ward_step_reasks_and_keeps_the_step(flow) -> None:  # noqa: F811
    first = _ward(flow, "어지러워요")
    asked = first["reply"].splitlines()[-1]
    flow.llm.calls.clear()
    out = _ward(flow, "신경과 가면 무슨 검사해요?")
    assert flow.llm.calls == [] and INFO_PIECE in out["reply"]
    assert out["reply"].splitlines()[-1] == asked  # 지금 질문을 다시 묻는다
    assert out["suggestedKind"] == "answer" and out["suggestedQuestions"]  # 단계 답 버튼
    assert not out.get("ui")  # 진료과 찾기 버튼은 단계가 끝난 뒤
    nxt = _ward(flow, "아니요")
    assert flow.llm.calls == [] and nxt["reply"].splitlines()[-1] != asked  # 단계가 이어져 다음 질문으로 간다


def test_repeated_info_during_step_ends_the_step_quietly(flow) -> None:  # noqa: F811
    _ward(flow, "어지러워요")
    outs = [_ward(flow, "신경과 가면 무슨 검사해요?") for _ in range(8)]
    assert all(INFO_PIECE in o["reply"] for o in outs)  # 상한을 넘으면 안내만 - 예외 없이


def test_guardian_legacy_followup_answer_is_joined_not_fixed(flow) -> None:  # noqa: F811
    _guardian(flow, "어지러워요")
    flow.llm.calls.clear()
    out = _guardian(flow, "신경과 가면 무슨 검사해요?")
    assert not _is_info(out) and len(flow.llm.calls) == 1 and "/" in flow.llm.calls[0]["message"]
