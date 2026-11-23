"""#336 기능 점검(2026-11-22) 조치 - 벼락두통 패턴 회귀(A-1·A-2·A-3·N-3) · "묻지 말라고" 회피의 남 이야기 예외(A-4·A-5·N-1·N-2).

점검 문장표(BE docs/(2026-11-22) audit-ai-final-fixes-335-336.md)의 문장과 같은 유형의 다른 문장을 함께 고정한다.
두통은 라우터가 실패한 턴의 판정(fallback_emergency_signals)과 그 경로의 결과(_keyword_route)를 함께 본다.
"""
from __future__ import annotations

import pytest

from app.services import chat_keywords as ck
from app.services import chat_service as cs
from app.services.companion.avoidance import is_avoidance

HEADACHE = "뇌출혈:두통"


def _headache(text: str) -> bool:
    return HEADACHE in ck.fallback_emergency_signals(text)


# ---- 두통: 걸려야 하는 말 ------------------------------------------------------------------------------------------

CAUGHT_BEFORE = [  # 고치기 전에도 걸리던 말 - 그대로
    "살면서 이런 두통은 처음이에요",
    "태어나서 최악의 두통이에요",
    "평생 이렇게 머리가 아픈 건 처음이야",
    "이런 두통은 처음이에요",
    "태어나서 처음 두통 때문에 병원에 갔어요",
    "살면서 처음 느껴보는 엄청난 두통이에요",
]
A1_NEVER_AND_WORST = [  # A-1: "처음·최악" 없이 "겪어본 적이 없다·없었다·제일 심한·가장 아픈"
    "살면서 이런 두통은 겪어본 적이 없어요",
    "평생 이런 두통은 없었어요",
    "태어나서 이렇게 머리 아픈 적이 없어요",
    "살면서 제일 심한 두통이에요",
    "평생 가장 아픈 두통이에요",
    "이런 두통은 살면서 겪어본 적이 없어요",
    "태어나서 이만큼 머리가 아팠던 적은 없어요",
    "평생 이 정도 두통은 없었는데요",
    "살면서 두통이 제일 심해요",
    "생전 가장 지독한 두통이에요",
]
A2_GAP_AND_LIFE_WORD = [  # A-2: 머리와 아파 사이 낱말, "생전"
    "난생 처음 머리가 이렇게 아파요",
    "이런 두통은 생전 처음이에요",
    "생전 처음 겪는 두통이에요",
    "난생 처음 겪어서 겁나는데 머리가 너무 아파요",
    "살면서 머리가 이렇게 아플 줄은 처음 알았어요",
]
N3_WITHOUT_LIFE_WORD = [  # N-3: 평생 낱말 없이 "이런·이렇게"만
    "이렇게 머리 아픈 적이 없었어요",
    "머리가 평생 이렇게 아픈 적이 없었어요",
    "머리가 이렇게 아픈 건 처음이에요",
    "이런 두통은 겪어본 적이 없어요",
    "이토록 머리가 아팠던 때가 없었어요",
]
A3_NOT_A_DENIAL = [  # A-3: 부정처럼 보여도 벼락두통을 말하는 꼴은 그대로 119
    "살면서 두통이 처음은 아니지만 이번엔 최악이에요",
    "평생 이런 두통은 처음이 아닌가 싶어요",
]


@pytest.mark.parametrize(
    "text", CAUGHT_BEFORE + A1_NEVER_AND_WORST + A2_GAP_AND_LIFE_WORD + N3_WITHOUT_LIFE_WORD + A3_NOT_A_DENIAL)
def test_thunderclap_headache_is_caught_when_router_failed(text) -> None:
    assert _headache(text), text
    assert cs.ChatService._keyword_route(text)["emergency"] is True, text


# ---- 두통: 걸리면 안 되는 말 -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # 만성·약 이야기 (M-3 그대로)
    "머리가 늘 아파",
    "두통약 먹었어",
    "평생 두통을 달고 살았어요",
    "평생 두통약을 먹고 살아요",
    "난생 처음 두통약을 먹어 봤어요",
    "평생 두통 없이 살았어요",
    "생전 두통약 모르고 살았어요",
    # A-1: 지시어 없이 "적이 없다"는 아픈 적이 없다는 말이다
    "살면서 머리가 아파본 적이 없어요",
    "살면서 이런 두통약은 먹어본 적이 없어요",
    "이런 두통은 없었으면 좋겠어요",
    # A-1: 심함 낱말 없는 "제일·가장"
    "살면서 제일 좋았던 건 두통이 없던 때예요",
    "평생 가장 힘든 건 남편 두통 수발이었어요",
    # A-2: "안"이 끼면 아픈 것이 아니다
    "평생 머리는 안 아팠는데 처음으로 허리가 아파요",
    "살면서 이런 일은 처음이라 머리가 복잡해요",
    # A-3: 부정, 진료 이야기
    "살면서 두통이 처음은 아니에요",
    "살면서 두통이 처음이 아니에요",
    "이런 두통은 처음이 아니에요",
    "태어나서 처음으로 두통 클리닉에 가봤어요",
    "살면서 두통 외래는 처음 가봐요",
    "살면서 두통 센터는 처음이에요",
    # N-3: 지시어가 두통에 붙지 않거나, 전에도 있었다는 말이거나, 다른 곳이 아픈 말
    "이렇게 더우면 머리가 아파요 에어컨이 없었거든요",
    "이런 두통은 예전에도 있었고 없었던 적이 없어요",
    "이렇게 머리 아플 때 먹는 약이 없었어요",
    "이런 날은 머리가 아파서 입맛이 없어요",
    "머리가 이렇게 아플 줄 몰랐어요",
    "이런 두통에는 뭐가 좋아요",
    "머리는 이제 안 아파요 이런 건 처음이네요",
    "머리가 아니라 허리가 이렇게 아픈 건 처음이에요",
    "머리보다 허리가 이렇게 아픈 적이 없었어요",
    "머리가 이렇게 안 아픈 건 처음이에요",
])
def test_ordinary_headache_talk_is_not_caught(text) -> None:
    assert not _headache(text), text
    assert cs.ChatService._keyword_route(text)["emergency"] is False, text


@pytest.mark.parametrize("text", [
    # 알고 남긴 과잉 안내(안전한 방향) - 좁히려면 진짜 벼락두통 꼴을 놓친다(#336 점검 A-3 결정: 부정·진료 낱말만 뺀다)
    "어제 난생 처음 술을 마셔서 두통이 있어요",        # 무관한 "처음"
    "난생 처음 술 마시고 두통이 있어요",
    "평생 이런 두통이 처음은 아닌데요",                # "아닌데" 뒤에 심하다는 말이 올 수 있어 빼지 않는다
    "평생 머리가 아팠어요 처음엔 약도 먹었고요",        # 문장부호 없는 두 절 (전부터 그랬다)
])
def test_known_over_trigger_is_kept(text) -> None:
    assert _headache(text), text


def test_headache_with_self_harm_word_still_has_both_signals() -> None:
    text = "평생 처음 겪는 두통인데 죽고 싶어요"  # 자해 키워드는 라우터 전에 고정 응답(chat_service) - 여기서는 신호만
    assert _headache(text)
    assert any(keyword in ck.compact(text) for keyword in ck.SELF_HARM_KEYWORDS)


# ---- 회피: "묻지 말라고" ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # M-5 그대로
    "자꾸 묻지 말라고 했잖아",
    "물어보지 말라고 했지",
    "캐묻지 말라고 하잖아요",
    # A-5: 준말
    "묻지 말라구 했잖아",
    "묻지 말랬잖아",
    "자꾸 물어보지 말랬지",
    # 본인을 가리키는 말은 남 이야기가 아니다
    "내가 묻지 말라고 했잖아",
    "제가 묻지 말라고 했잖아요",
    "나한테 자꾸 묻지 말라고 했잖아",
    "저한테 묻지 말라고 했잖아요",
    "나더러 묻지 말라고 했잖아",
    "이 늙은이한테 자꾸 묻지 말라고 했잖아",
    # "이·가"로 끝나지만 사람 주어가 아닌 낱말
    "나이 묻지 말라고 했잖아",
    "일일이 캐묻지 말라고 했잖아",
    "쓸데없이 묻지 말라고 했지",
    "굳이 묻지 말라고 했잖아",
    "밥 먹다가 묻지 말라고 했잖아",
    "뭐가 그리 궁금해 묻지 말라고 했잖아",
    "어디가 불편한지 자꾸 묻지 말라고 했잖아",
    # 부르는 말·이야기 주제로 나온 사람 낱말
    "선생님, 자꾸 묻지 말라고 했잖아요",
    "자식 얘기 묻지 말라고 했잖아",
    "아들 얘기는 묻지 말라고 했지",
    "그거 묻지 말라고 했잖아",
    # 전해 들은 말투가 아니다 (N-1 경계)
    "묻지 말라고 했는데도 자꾸 묻네",
    "묻지 말라고 했더니 또 묻네",
    "묻지 말라고 하는데도 자꾸 그러네",
    # N-2: 말끝에서 끊는 꼴
    "묻지 말라니까",
    "묻지 말라니까요",
    "캐묻지 말라니까!",
    "자꾸 묻지 말래도",
    "그런 건 묻지 말라고",
    "묻지 말라고요",
    "물어보지 말라구…",
])
def test_telling_companion_to_stop_asking_is_avoidance(text) -> None:
    assert is_avoidance(text), text


@pytest.mark.parametrize("text", [
    # 전부터 남 이야기
    "아들이 묻지 말라고 했어",
    "친구가 자꾸 캐묻지 말라고 하더라",
    # A-4: 목록에 없던 사람·낮춤말·"한테"
    "엄마가 묻지 말라고 했어",
    "의사가 묻지 말라고 했어",
    "집사람이 묻지 말라고 했어",
    "아들놈이 묻지 말라고 했어",
    "아들한테 묻지 말라고 했어",
    "선생님이 묻지 말라고 했어",
    "어머니께서 묻지 말라고 하셨어",
    "간호사가 캐묻지 말라고 하더라",
    "딸내미가 물어보지 말라고 했어",
    "할멈이 묻지 말라고 했어",
    "손주들한테 묻지 말라고 했어",
    "우리 영감한테 묻지 말라고 했어",
    "복지사 선생님이 캐묻지 말라고 하셨어",
    "영희한테 묻지 말라고 했어",
    "누가 묻지 말라고 했어",
    # A-5 준말 + 남 이야기
    "엄마가 묻지 말랬어",
    "딸이 물어보지 말랬어",
    "며느리가 묻지 말라구 했어",
    "그 양반이 묻지 말라구 했어",
    # N-1: 주어 없는 전해 들은 말투
    "묻지 말라고 하더라",
    "묻지 말라고 했대",
    "묻지 말라고 하셨어",
    "묻지 말라고 했다더라",
    "엄마가 그러는데 묻지 말라고 했대",
    # N-2: 말끝 꼴이라도 남 이야기·되묻는 말·문장 가운데는 아니다
    "아들이 묻지 말라니까",
    "엄마가 묻지 말래도",
    "의사 선생님이 묻지 말라고요",
    "의사가 묻지 말라고요?",
    "묻지 말라고요?",
    "엄마가 묻지 말라니까 안 물었어",
    # 몸이 불편한 말·"묻지"가 아닌 꼴은 전과 같다
    "숨이 차서 묻지 말라니까",
    "숨이 차서 묻지 말라고 했잖아",
    "그 얘기 하지 말라고 했지",
    "그런 얘기 그만 좀 하라고 했지",
    "누구한테 묻지 말고 병원 가래",
])
def test_other_peoples_words_are_not_avoidance(text) -> None:
    assert not is_avoidance(text), text


def test_wider_person_list_is_used_only_for_stop_asking() -> None:
    # 넓힌 사람 목록은 "묻지 말라고" 꼴에서만 쓴다 - 다른 회피 문장은 전과 같은 판정(_OTHER_SUBJECT 그대로)
    assert is_avoidance("엄마가 보고 싶어서 말하기 싫어")
    assert not is_avoidance("아들이 보고 싶어서 말하기 싫어")


@pytest.mark.parametrize("text,expected", [
    # 알고 남긴 한계 - 지금 동작을 고정한다
    ("옆집 김씨가 묻지 말라고 했어", True),     # 목록에 없는 사람 + "가": 남 이야기인데 회피로 본다("이·가" 일반화는 진짜 회피를 놓친다)
    ("네가 묻지 말라고 했잖아", True),
    ("묻지 말라니까 자꾸 묻네", False),         # 말끝이 아닌 "말라니까"는 보지 않는다(남 이야기와 구별할 수 없다)
    ("우리 할머니한테 묻지 말라고 했어", True),  # 본인을 "할머니"라 부르는 말과 구별할 수 없어 회피로 본다(2026-11-23 P-2)
    ("엄마한테 자꾸 묻지 말라고 했잖아", False),  # "묻지" 바로 앞의 "한테"는 남 이야기로 둔다(엄마에게 묻지 말라는 말일 수 있다)
])
def test_known_limits_of_stop_asking(text, expected) -> None:
    assert is_avoidance(text) is expected, text


# ---- 2026-11-23 #340 기능 점검(BE docs/(2026-11-22) audit-ai-headache-avoidance-340.md) 조치 ----------------------------

@pytest.mark.parametrize("text", [
    # P-1 ① 두통 낱말이 평생·처음보다 앞
    "두통이 평생 처음입니다",
    "머리 아픈 건 난생 처음이에요",
    "머리가아파 죽겠어 살면서처음",
    "두통이 생전 처음이라 무서워요",
    # P-1 ② 두통 주어 + 지시어
    "두통이 이렇게 심한 건 처음이에요",
    "두통이 이렇게 심한 적이 없었어요",
    # P-1 ③ "이런 두통이·도"
    "이런 두통이 처음이에요",
    "이런 두통도 처음이에요",
    # P-1 ④ 강조어 4자까지, 머리 비유
    "머리가 이렇게 심하게 아픈 건 처음이에요",
    "난생 처음 머리가 터질 것 같이 아파요",
    "평생 처음 머리가 깨질 것 같아요",
    # P-1 ⑤ 과거형 최상급, "최고로"
    "살면서 가장 심했던 두통이에요",
    "평생 최고로 아픈 두통이에요",
    # P-1·P-8 "첨"
    "이런 두통은 첨이야",
    "머리가 이렇게 아픈 건 첨이에요",
])
def test_thunderclap_forms_from_340_audit_are_caught(text) -> None:
    assert _headache(text), text
    assert cs.ChatService._keyword_route(text)["emergency"] is True, text


@pytest.mark.parametrize("text", [
    # P-3 격식체·떨어진 부정
    "살면서 두통이 처음은 아닙니다",
    "살면서 두통이 처음은 절대 아니에요",
    "이런 두통은 처음이 결코 아닙니다",
    # P-4 "머리가 아니라·아주·아래", 아파트
    "살면서 처음으로 머리가 아니라 허리가 아파요",
    "난생 처음 머리를 아주 짧게 잘랐어요",
    "평생 처음 머리 아래쪽에 점이 생겼어요",
    "난생 처음 머리 자르고 아파트 앞에 갔어요",
    # 새 어순 대안의 경계: 이유 절이 끼거나 4자를 넘으면 끊는다
    "머리가 아파서 병원 갔는데 살면서 처음 MRI를 찍었어요",
    "두통이 있었는데 살면서 처음 침을 맞아 봤어요",
    # 새 지시어 대안의 경계
    "두통이 이렇게 안 심한 건 처음이에요",
    "두통약이 이렇게 잘 듣는 건 처음이에요",
    # "첨"이 아닌 낱말
    "평생 두통 이야기를 첨부해서 보냈어요",
])
def test_ordinary_talk_near_new_headache_forms_is_not_caught(text) -> None:
    assert not _headache(text), text


def test_known_over_trigger_dementia_hospital_visit_is_kept() -> None:
    # 알고 남긴 과잉 안내(안전한 방향): "이런 두통으로 고생한 적이 없어요"와 구별할 수 없다
    assert _headache("이런 두통으로 병원 간 적은 없어요")


@pytest.mark.parametrize("text", [
    # P-2 주제 조사 "은·는·도" = 묻지 말라는 이야깃거리
    "아버지는 묻지 말라고 했잖아",
    "선생님은 왜 자꾸 그래요 묻지 말라고 했잖아요",
    "아들 일도 묻지 말라고 했지",
    # P-2 앞 절·다른 문장의 사람 낱말
    "의사가 뭐라 했는지 묻지 말라고 했잖아",
    "엄마가 그러더라고. 근데 나도 묻지 말라고 했잖아",
    "딸이 왔는데 자꾸 묻지 말라고 했잖아",
    # P-2 "한테"가 "묻지"에서 멀거나 본인을 가리킴 (P-7 한계 문장 포함)
    "남편한테 맞고 산 얘기는 묻지 말라고 했잖아",
    "엄마한테 전화했는데 자꾸 묻지 말라고 했잖아",
    "내한테 자꾸 묻지 말라고 했잖아",
    "할머니한테 자꾸 묻지 말라고 했잖아",
    "노인한테 그런 거 묻지 말라고 했잖아",
    # 전언 예외 밖의 꼴은 그대로 회피
    "묻지 말라고 했다고 몇 번을 말해",
])
def test_stop_asking_after_340_audit_is_avoidance(text) -> None:
    assert is_avoidance(text), text


@pytest.mark.parametrize("text", [
    # 주어·바로 앞 "한테"는 그대로 남 이야기 (#340이 고친 문장의 다른 꼴)
    "아버지가 묻지 말라고 했어",
    "선생님께서 묻지 말라고 하셨어",
    "아들한테 그런 거 묻지 말라고 했어",
    "선생님께 묻지 말라고 했어",
    "묻지 말라고 아들이 그랬어",
    # P-6 전해 들은 말투
    "묻지 말라고 했다고 하더라",
    "묻지 말라고 했다고 그러던데",
    "묻지 말라고 하길래 안 물어봤어",
    "묻지 말라고 했나 봐",
    "병원에서 그런 건 묻지 말라고 했어",
    "복지관에서 묻지 말라고 했어",
])
def test_other_peoples_words_after_340_audit_are_not_avoidance(text) -> None:
    assert not is_avoidance(text), text


@pytest.mark.parametrize("text", [
    "할 말 없어…", "할 말 없어 ~ !? ", "묻지 말라고요?", "  \t\n", "", "가 ", "말하기 싫어.　", "a.b.",
])
def test_strip_trailing_matches_old_regex(text) -> None:
    import re
    from app.services.companion.avoidance import _strip_trailing
    assert _strip_trailing(text) == re.sub(r"[\s.…~!?]+$", "", text)


def test_long_input_is_linear() -> None:
    # Q-1: 공백 10만 자 뒤에 글자가 오는 입력이 전에는 36.8초 걸렸다
    import time
    # 둘째 입력은 옛 정규식으로도 선형이다(#341 점검 T-7) - "묻지"가 많은 입력의 기준선으로 둔다.
    # 넷째·다섯째는 #341 점검 F-3의 경계 건너기(쉼표 앞 주어·"그러는데")가 많이 이어지는 입력이다.
    for text in (" " * 100_000 + "가", "묻지 " * 20_000 + "가", ". " * 50_000 + "가",
                 "아들이, " * 20_000 + "묻지 말라고 했어", "그러는데 " * 20_000 + "묻지 말라고 했어"):
        started = time.perf_counter()
        is_avoidance(text)
        assert time.perf_counter() - started < 1.0
    started = time.perf_counter()
    ck.fallback_emergency_signals("머리가 이렇게 " * 700)  # 약 5,000자
    assert time.perf_counter() - started < 0.2


# ---- 2026-11-23 #341 기능 점검(BE docs/(2026-11-23) audit-ai-headache-avoidance-341.md) 조치 ----------------------------

@pytest.mark.parametrize("text", [
    # F-1 본인 + 한테 + 조사, "저희"
    "나한테도 묻지 말라고 했잖아",
    "저한테는 묻지 말라고 했잖아요",
    "나한테만 묻지 말라고 했잖아",
    "나보고는 묻지 말라고 했잖아",
    "저희한테 묻지 말라고 했잖아",
    # T-3 "우리"
    "우리한테 자꾸 묻지 말라고 했잖아",
    # F-2 "께·보고"가 붙은 보통 낱말
    "어저께도 묻지 말라고 했잖아",
    "그저께 묻지 말라고 했잖아",
    "우리 함께 있을 때는 묻지 말라고 했잖아",
    "뉴스보고 그런 거 묻지 말라고 했잖아",
    # F-4 기관이 이야깃거리
    "병원에서 있었던 일은 묻지 말라고 했잖아",
    "요양원에서 지낸 얘기는 묻지 말라고 했잖아",
    # F-6 전언이 아닌 "기에도·그래도"
    "묻지 말라고 하기에도 지쳤어",
    "묻지 말라고 했다고 그래도 또 묻네",
    # T-1 이음말만·문장부호만으로 끊기는 앞 절
    "엄마가 그랬어 근데 나도 묻지 말라고 했잖아",
    "엄마가 그랬어, 나도 묻지 말라고 했잖아",
    "아들이 뭐라든, 나는 묻지 말라고 했잖아",
    "아들이 어디 사는지 묻지 말라고 했잖아",
    # T-2 사람 낱말 뒤가 "이·가"로 끝나지 않음
    "자식이라고 묻지 말라고 했잖아",
    "딸이랑 묻지 말라고 했잖아",
])
def test_stop_asking_after_341_audit_is_avoidance(text) -> None:
    assert is_avoidance(text), text


@pytest.mark.parametrize("text", [
    # F-3 경계가 말한 사람을 잘라내지 않는다
    "아들이, 묻지 말라고 했어",
    "우리 딸이요, 그런 건 묻지 말라고 했어요",
    "엄마가 그러는데 묻지 말라고 했어",
    "아들이 그러더라 묻지 말라고",
    "딸이 그랬는데 묻지 말라고 했어",
    # F-5 한테 + 조사
    "아들한테는 그런 거 묻지 말라고 했어",
    "딸한테도 묻지 말라고 했어",
    # F-2 사람 목록 뒤 "께·보고", 목록 밖 "께서"
    "딸보고 묻지 말라고 했어",
    "교수님께서 묻지 말라고 했어",
    # T-5 전언 "기에"
    "묻지 말라고 하기에 안 물었어",
])
def test_other_peoples_words_after_341_audit_are_not_avoidance(text) -> None:
    assert not is_avoidance(text), text


@pytest.mark.parametrize("place", [
    "병원", "보건소", "복지관", "센터", "경로당", "주민센터", "동사무소", "구청", "경찰서", "요양원",
])
def test_every_institution_right_before_stop_asking_is_other_peoples_words(place) -> None:
    # T-4: 기관 목록 전부. F-4: "묻지" 바로 앞일 때만
    assert not is_avoidance(f"{place}에서 묻지 말라고 했어")
    assert not is_avoidance(f"{place}에서 그런 거 묻지 말라고 했어")
    assert is_avoidance(f"{place}에서 있었던 일은 묻지 말라고 했잖아")


@pytest.mark.parametrize("text", [
    "살면서 두통이 처음은 아닙니다만 이번엔 최악이에요",   # F-7
    "살면서 처음 이렇게 머리가 아파 트림까지 나와요",     # F-8
    "두통이 너무 심해요 평생 처음이에요",               # T-6 사이 6자 (사용자 결정)
    "두통이 심해요 평생 처음이에요",                    # 사이 4자
])
def test_thunderclap_forms_from_341_audit_are_caught(text) -> None:
    assert _headache(text), text
    assert cs.ChatService._keyword_route(text)["emergency"] is True, text


@pytest.mark.parametrize("text", [
    # F-9 좋은 소식, "첨에는·첨벙"
    "두통이 이렇게 빨리 낫는 건 처음이에요",
    "두통이 이렇게 가벼운 건 처음이에요",
    "두통이 이렇게 금방 나은 건 처음이에요",
    "두통이 이렇게 덜한 건 처음이에요",
    "평생 첨에는 머리가 아파도 참았어요",
    "살면서 첨벙 물에 빠져 머리가 아팠어요",
    # T-6 사이 7자는 끊는다
    "두통이 너무너무 심해 평생 처음이에요",
    # 아파트는 그대로 뺀다
    "난생 처음 머리 자르고 아파트 앞에 갔어요",
])
def test_ordinary_talk_after_341_audit_is_not_caught(text) -> None:
    assert not _headache(text), text

# ---- 2026-11-23 #343 기능 점검(BE docs/(2026-11-23) audit-ai-avoidance-clause-343.md) 조치 ----------------------------

@pytest.mark.parametrize("text", [
    # G-1 속도·나쁜 소식은 벼락두통 양상 - 좋은 소식 제외에 걸리지 않는다
    "두통이 이렇게 빨리 심해진 건 처음이에요",
    "두통이 이렇게 금방 심해진 적이 없어요",
    "두통이 이렇게 낫질 않는 건 처음이에요",
    "두통이 이렇게 덜컥 오는 건 처음이에요",
    "두통이 이렇게 빨리 오는 건 처음이에요",
])
def test_bad_news_headache_after_343_audit_is_caught(text) -> None:
    assert _headache(text), text
    assert cs.ChatService._keyword_route(text)["emergency"] is True, text


@pytest.mark.parametrize("text", [
    # G-1 나아졌다는 꼴만 좋은 소식 (낱말마다 하나씩 - T-1)
    "두통이 이렇게 빨리 낫는 건 처음이에요",
    "두통이 이렇게 금방 나은 건 처음이에요",
    "두통이 이렇게 나았던 건 처음이에요",
    "두통이 이렇게 나아진 건 처음이에요",
    "두통이 이렇게 금방 나아졌던 적이 없어요",
    "두통이 이렇게 가벼운 건 처음이에요",
    "두통이 이렇게 덜한 건 처음이에요",
    # G-6 두통 → 평생 낱말 사이에 "없"
    "두통이 없어졌어요 평생 처음으로 운동했어요",
    "두통이 없어요 평생 처음으로 운동했어요",
])
def test_good_news_after_343_audit_is_not_caught(text) -> None:
    assert not _headache(text), text


@pytest.mark.parametrize("text", [
    # G-2 경계 뒤 절이 본인 주어로 시작하면 건너지 않는다
    "아들이 그러는데 나도 묻지 말라고 했잖아",
    "엄마가 그러더라 나도 묻지 말라고 했잖아",
    "딸이 그랬는데 나는 묻지 말라고 했잖아",
    "엄마가 그러는데 내가 묻지 말라고 했잖아",
    "딸이 그러더라 저는 묻지 말라고 했잖아요",
    "아들이, 나는 묻지 말라고 했잖아",
    "아들이 그러는데, 나도 묻지 말라고 했잖아",
    # G-8 "길래"는 건너지 않는다(사용자 결정)
    "아들이 그러길래 묻지 말라고 했잖아",
    # G-3 "님·사람"이 없는 "께"
    "엊그저께 묻지 말라고 했잖아",
    "아침께 묻지 말라고 했잖아",
    "함께 묻지 말라고 했잖아",
    # G-4 기관이 이야깃거리(부사를 더한 뒤에도)
    "병원에서 받은 약은 묻지 말라고 했잖아",
    "복지관에서 만난 사람은 묻지 말라고 했잖아",
])
def test_stop_asking_after_343_audit_is_avoidance(text) -> None:
    assert is_avoidance(text), text


@pytest.mark.parametrize("text", [
    # G-5 쉼표 앞 "그러는데·그러더라고", "사람 + 말이야"(사용자 결정)
    "아들이 그러는데, 묻지 말라고 했어",
    "엄마가 그러더라고, 묻지 말라고",
    "아들이 말이야, 묻지 말라고 했어",
    "우리 딸이 말이에요, 그런 건 묻지 말라고 했어요",
    "할아버지께서요, 묻지 말라고 하셨어",
    # G-3 목록 밖 사람 + 님·사람 + 께·보고
    "교수님께 묻지 말라고 했어",
    "목사님께 그런 거 묻지 말라고 했어",
    "과장님보고 묻지 말라고 했어",
    "그 사람보고 묻지 말라고 했어",
    # G-4 기관 + 부사·저한테
    "병원에서 다시는 묻지 말라고 했어",
    "병원에서 절대 묻지 말라고 했어",
    "병원에서 앞으로 묻지 말라고 했어",
    "병원에서 또 묻지 말라고 했어",
    "병원에서 저한테 묻지 말라고 했어",
    # T-1 기관 사이 말
    "병원에서 자꾸 묻지 말라고 했어",
    "병원에서 그런 얘기 묻지 말라고 했어",
    "병원에서 함부로 묻지 말라고 했어",
])
def test_other_peoples_words_after_343_audit_are_not_avoidance(text) -> None:
    assert not is_avoidance(text), text


@pytest.mark.parametrize("text,expected", [
    # 알고 남긴 한계(#343 점검 G-9·G-11) - 지금 동작을 고정한다
    ("나한테까지도 묻지 말라고 했잖아", False),     # 조사 두 개("까지도")는 본인 예외 밖
    ("아들한텐 그런 거 묻지 말라고 했어", True),     # 준말 "한텐"은 여격으로 보지 않는다
    ("엄마가그러는데 묻지 말라고 했어", True),       # 붙여 쓴 주어
])
def test_known_limits_after_343_audit(text, expected) -> None:
    assert is_avoidance(text) is expected, text


def test_clause_scan_is_linear_with_many_breaks() -> None:
    # G-10: 경계 10만 개 뒤 "묻지"도 경계를 모두 쥐지 않고 한 번에 훑는다
    import time
    for unit in (",", "…", "아들이, ", "그러는데 "):
        text = unit * (100_000 // len(unit)) + " 묻지 말라고 했잖아"
        started = time.perf_counter()
        is_avoidance(text)
        assert time.perf_counter() - started < 1.0