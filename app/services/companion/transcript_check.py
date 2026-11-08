"""음성 인식 결과 검증: 실제 단어가 아닌 말(예: 삼기탕)이 섞였는지 본다.

방법 두 가지를 같이 쓴다.
1. 형태소 분석: 한 글자짜리 명사가 두 개 이상 붙어 나오면 단어가 쪼개진 것으로 의심한다 (삼·기·탕).
2. 어휘 목록: 어르신 생활 어휘에 있는 말은 실제 단어로 인정하고, 목록에서 한 글자만 다른 말이 있으면 후보로 든다 (삼기탕 → 삼계탕).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from kiwipiepy import Kiwi

_LEXICON = Path(__file__).parent / "elder_lexicon.txt"
_NOUN_TAGS = {"NNG", "NNP", "NR"}


@dataclass
class TranscriptCheck:
    suspicious: list[str] = field(default_factory=list)  # 실제 단어가 아닐 수 있는 말
    candidates: dict[str, str] = field(default_factory=dict)  # 의심 단어 → 목록에서 찾은 비슷한 말 (없으면 키가 없다)


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


@lru_cache(maxsize=1)
def _lexicon() -> tuple[str, ...]:
    words = []
    for line in _LEXICON.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return tuple(words)


def _one_letter_off(a: str, b: str) -> bool:
    """같은 길이이고 글자 한 개만 다르면 True (삼기탕 ↔ 삼계탕)."""
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def check_transcript(text: str) -> TranscriptCheck:
    result = TranscriptCheck()
    if not text.strip():
        return result
    lexicon = _lexicon()
    runs: list[list[str]] = []
    current: list[str] = []
    for token in _kiwi().tokenize(text):
        if len(token.form) == 1 and token.tag in _NOUN_TAGS and re.fullmatch(r"[가-힣]", token.form):
            current.append(token.form)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)
    for run in runs:
        word = "".join(run)
        if word in lexicon:
            continue  # 목록에 있는 말이면 실제 단어로 본다
        result.suspicious.append(word)
        match = next((w for w in lexicon if _one_letter_off(word, w)), None)
        if match:
            result.candidates[word] = match
    return result


def confirm_reply(check: TranscriptCheck) -> str:
    """의심 단어가 있을 때의 응답. 뜻을 추측해서 공감하지 않고 다시 확인한다."""
    if check.suspicious:
        word = check.suspicious[0]
        if word in check.candidates:
            return f"혹시 {check.candidates[word]} 이야기이신가요? 맞으면 그렇다고 말씀해 주세요."
    return "잘 알아듣지 못했어요. 한 번 더 말씀해 주시겠어요?"
