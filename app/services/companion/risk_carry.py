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

# 이어 간 턴의 안내. 같은 안내를 되풀이하지 않고, 넘긴 말을 받아 준 뒤 다시 한 번 권한다.
CARRY_REPLY = (
    "괜찮다고 하셔도 아까 하신 말씀이 마음에 걸려요. 혼자 견디지 마시고 가까운 가족이나 주변 분께 꼭 이야기해 주세요. "
    "급하시면 119에 전화하실 수 있어요."
)
CARRY_BODY_REPLY = (
    # '나아지셨다니'로 받지 않는다 - "괜찮겠지 뭐, 좀 누워 있을게"는 나아졌다는 말이 아니다 (품질 기준 R5 넘겨짚기)
    "괜찮다고 하셔도 아까 같은 증상은 다시 올 수 있어서 걱정돼요. 지금이라도 119나 가까운 가족에게 꼭 알려 주세요."
)


# 이어 간 턴의 안내 문장을 고를 때만 쓰는 몸 증상 표시 (판정에는 쓰지 않는다).
#   risk.body_emergency_only 는 kw4 몸 응급 낱말만 봐서 "가슴이 꽉 조이고 숨이 차"·"오른팔에 힘이 없어"를 몸 응급으로 보지 않는다 (I-21)
BODY_HINT_PATTERN = re.compile(r"가슴|숨|팔에?\s?힘|다리에?\s?힘|말이\s?(잘\s?)?안\s?나|어지럽|쓰러|마비|저려|저리|토하|식은땀")
MIND_HINT_PATTERN = re.compile(r"죽|없어지|없어져|안\s?깼|안\s?깨|살기\s?싫|자살|자해|끝내")


def is_body_risk(transcript: str, body_keyword: bool = False) -> bool:
    """직전 위험 턴이 몸 응급 쪽인가 - 안내 문장 고르기용. kw4 몸 응급 낱말이거나, 몸 증상 표시가 있고 마음 쪽 표시가 없으면 몸."""
    if body_keyword:
        return True
    return bool(BODY_HINT_PATTERN.search(transcript)) and not MIND_HINT_PATTERN.search(transcript)


def is_downplay(transcript: str) -> bool:
    """위험 다음 턴의 말이 넘기거나 얼버무리는 말인가 (도움을 받겠다는 말은 아니다)."""
    text = transcript.strip()
    if not text or ACCEPT_PATTERN.search(text):
        return False
    return bool(DOWNPLAY_PATTERN.search(text))


def carry_reply(prev_body: bool) -> str:
    """직전 위험 턴이 몸 응급이었으면 몸 응급용, 아니면 마음 쪽 안내."""
    return CARRY_BODY_REPLY if prev_body else CARRY_REPLY
