"""받아쓰기 결과 다듬기: 자막 환각 문구를 지우고, 같은 말이 계속 되풀이된 부분을 한 번으로 줄인다.

Whisper 계열은 소리가 끊기거나 잡음이 섞이면 같은 낱말을 몇십 번 되풀이하거나 방송 자막 문구를 지어낸다.
근거: DGX cmp_rep.py (11/05) - turbo 잡음 5dB 조건 글자 오류 37.6% → 26.1%, 다른 조건은 변화 없음.
"""

from __future__ import annotations

import re

REPEAT_MIN = 3  # 이 횟수 이상 연달아 나와야 줄인다 (두 번은 "네 네" 같은 실제 말일 수 있다)
MAX_GROUP_WORDS = 4  # 1~4어절 묶음까지 본다

# 무음·잡음에서 Whisper 가 지어내는 방송 자막 문구 (DGX cmp_stt.py HALLU 와 같다)
HALLUCINATION = re.compile(
    r"(자막\s*제공|배달의\s*민족|시청해\s*주셔서\s*감사합니다|구독과\s*좋아요|MBC\s*뉴스|한국어\s*자막|자막\s*by)[^.?!]*[.?!]?"
)
# 띄어쓰기 없이 같은 2~10글자 묶음이 4번 이상 이어진 반복 ("고마워요고마워요고마워요고마워요")
_GLUED_REPEAT = re.compile(r"(.{2,10}?)\1{3,}")


def strip_hallucination(text: str) -> str:
    return " ".join(HALLUCINATION.sub(" ", text).split())


def collapse_repeats(text: str, rep: int = REPEAT_MIN) -> str:
    """같은 1~4어절 묶음이 rep 번 이상 연달아 나오면 한 번만 남긴다.

    DGX 근거 코드는 반복이 없을 때 묶음 길이만큼 건너뛰어 홀수 위치에서 시작하는 반복을 놓쳤다. 여기서는 한 어절씩 나아간다.
    """
    words = text.split()
    for n in range(1, MAX_GROUP_WORDS + 1):
        out: list[str] = []
        i = 0
        while i < len(words):
            group = words[i:i + n]
            k = 1
            while len(group) == n and words[i + k * n:i + (k + 1) * n] == group:
                k += 1
            if k >= rep:
                out += group
                i += k * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return _GLUED_REPEAT.sub(r"\1", " ".join(words))


def clean_transcript(text: str, collapse: bool) -> str:
    text = strip_hallucination(text)
    return collapse_repeats(text) if collapse else text
