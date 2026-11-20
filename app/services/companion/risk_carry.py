"""위험 이어짐 (2026-11-15, 고도화 6-2b 검수 결정 · 품질 기준 v1.2 '위험 (이어짐)').

위험 판정(v9 + 위험 LoRA r2, 낱말 규칙 kw4)은 그 턴의 말 하나만 본다. 그래서 위험한 말 바로 다음 턴에
"조금 쉬니까 괜찮아진 것 같기도 하고"(가슴 조임 뒤), "그냥 해 본 소리야"(자살 암시 뒤)처럼 넘기거나 얼버무리면
위험이 아닌 턴이 되어 평소 꼬리질문으로 돌아갔다. 직전 턴이 위험이었고 이번 말이 넘기는 말이면 위험을 이어 간다.

도움을 받겠다는 말("119 부를게", "딸한테 전화해 볼게", "아들한테 연락했어")은 넘기는 말이 아니다 - 이어 가지 않는다.
보호자 위험 알림은 백엔드에서 대화당 1번이라(policy-companion.md '위험 알림은 대화당 1번') 이어 간 턴이 알림을 더 만들지 않는다.
"""
from __future__ import annotations

import re

# 넘기거나 얼버무리는 말. '귀찮'은 '괜찮'과 다르다 (다 귀찮아 = 이어 가지 않음, 평가 세트 K1)
DOWNPLAY_PATTERN = re.compile(
    r"괜찮|해\s?본\s?(소리|말)|그냥\s?한\s?(소리|말)|농담|별(거|것)?\s?아니|신경\s?(쓰지|끄)|걱정\s?(마|하지\s?마)"
    r"|누워\s?있|좀\s?쉬|쉬면|쉬니까|됐어|나아지|지나가|참을\s?만"
)
# 도움을 받겠다·받았다는 말 - 넘기는 말이 섞여 있어도 이어 가지 않는다
ACCEPT_PATTERN = re.compile(
    r"119|구급차|전화\s?(할|해|했|하|드릴)|연락\s?(할|해|했|하|드릴)|병원\s?(에)?\s?(갈|가|갔|다녀)|부를|불렀|와\s?달라|오라고"
)

# 도움을 거부하는 말 (11/20 점검 H-1). "병원 가기 싫어"·"119 부르지 마"는 ACCEPT 낱말("병원 가"·"119"·"전화하")을 품고 있어
#   도움을 받겠다는 말로 읽혀 이어짐이 풀렸다. 거부가 보이면 ACCEPT 보다 먼저 넘기는 말로 친다 (위험 직후 도움 거부는 더 위험한 신호)
REFUSE_PATTERN = re.compile(
    r"(가기|부르기|하기|연락하기|전화하기|받기|알리기)\s?싫"
    r"|(부르|불러|전화하|전화|연락하|연락|가|알리|오)지\s?(마|말|않)"
    r"|안\s?(갈|가|부를|불러|할|해|알릴)|소용\s?없|필요\s?없|하지\s?마"
)
# '괜찮'을 부정한 말 (11/20 점검 M-1). 이어 가되 "괜찮다고 하셔도"로 받지 않는다
NEGATED_OK_PATTERN = re.compile(r"안\s?괜찮|괜찮지\s?(않|안)")

# 이어 간 턴의 안내. 같은 안내를 되풀이하지 않고, 넘긴 말을 받아 준 뒤 다시 한 번 권한다.
CARRY_REPLY = (
    "괜찮다고 하셔도 아까 하신 말씀이 마음에 걸려요. 혼자 견디지 마시고 가까운 가족이나 주변 분께 꼭 이야기해 주세요. "
    "급하시면 119에 전화하실 수 있어요."
)
CARRY_BODY_REPLY = (
    # '나아지셨다니'로 받지 않는다 - "괜찮겠지 뭐, 좀 누워 있을게"는 나아졌다는 말이 아니다 (품질 기준 R5 넘겨짚기)
    "괜찮다고 하셔도 아까 같은 증상은 다시 올 수 있어서 걱정돼요. 지금이라도 119나 가까운 가족에게 꼭 알려 주세요."
)
# 도움을 거부했거나 괜찮지 않다고 한 턴의 안내 (11/20 점검 H-1·M-1) - "괜찮다고 하셔도"로 받지 않는다
CARRY_PLAIN_REPLY = (
    "아까 하신 말씀이 마음에 걸려요. 혼자 견디지 마시고 가까운 가족이나 주변 분께 꼭 이야기해 주세요. "
    "급하시면 119에 전화하실 수 있어요."
)
CARRY_BODY_PLAIN_REPLY = "아까 같은 증상은 다시 올 수 있어서 걱정돼요. 지금이라도 119나 가까운 가족에게 꼭 알려 주세요."


# 이어 간 턴의 안내 문장을 고를 때만 쓰는 몸 증상 표시 (판정에는 쓰지 않는다).
#   risk.body_emergency_only 는 kw4 몸 응급 낱말만 봐서 "가슴이 꽉 조이고 숨이 차"·"오른팔에 힘이 없어"를 몸 응급으로 보지 않는다 (I-21)
BODY_HINT_PATTERN = re.compile(r"가슴|숨|팔에?\s?힘|다리에?\s?힘|말이\s?(잘\s?)?안\s?나|어지럽|쓰러|마비|저려|저리|토하|식은땀")
# "숨이 막혀 죽겠어"의 '죽겠'은 감정 관용구라 마음 쪽 표시로 보지 않는다 (11/20 점검 L-2, kw4 관용구 예외와 같은 판단)
MIND_HINT_PATTERN = re.compile(r"죽(?!겠)|없어지|없어져|안\s?깼|안\s?깨|살기\s?싫|자살|자해|끝내")


def is_body_risk(transcript: str, body_keyword: bool = False) -> bool:
    """직전 위험 턴이 몸 응급 쪽인가 - 안내 문장 고르기용. kw4 몸 응급 낱말이거나, 몸 증상 표시가 있고 마음 쪽 표시가 없으면 몸."""
    if body_keyword:
        return True
    return bool(BODY_HINT_PATTERN.search(transcript)) and not MIND_HINT_PATTERN.search(transcript)


def is_refusal(transcript: str) -> bool:
    """도움을 거부하는 말인가 ("병원 가기 싫어", "119 부르지 마"). 위험 직후면 이어 간다 (11/20 점검 H-1)."""
    return bool(REFUSE_PATTERN.search(transcript))


def is_downplay(transcript: str) -> bool:
    """위험 다음 턴의 말이 넘기거나 얼버무리는 말인가 (도움을 받겠다는 말은 아니다).

    도움 거부는 ACCEPT 낱말을 품어도 넘기는 말이다 (11/20 점검 H-1) - 거부를 먼저 본다.
    """
    text = transcript.strip()
    if not text:
        return False
    if is_refusal(text):
        return True
    if ACCEPT_PATTERN.search(text):
        return False
    return bool(DOWNPLAY_PATTERN.search(text))


def carry_reply(prev_body: bool, transcript: str = "") -> str:
    """직전 위험 턴이 몸 응급이었으면 몸 응급용, 아니면 마음 쪽 안내.

    도움을 거부했거나 '괜찮지 않다'고 한 말이면 "괜찮다고 하셔도"로 받지 않는 문장을 쓴다 (11/20 점검 H-1·M-1).
    """
    plain = bool(transcript) and (is_refusal(transcript) or bool(NEGATED_OK_PATTERN.search(transcript)))
    if prev_body:
        return CARRY_BODY_PLAIN_REPLY if plain else CARRY_BODY_REPLY
    return CARRY_PLAIN_REPLY if plain else CARRY_REPLY
