"""글 기분 판단 e900 (11/10): EXAONE 3.5 2.4B + 기분 LoRA(어르신 대답으로 학습), 4코드 JOY·CALM·WORRY·DEPRESSED.

재희 버전(GGARA02 KcELECTRA 6감정)을 바꾼다. 근거: CareCall 새 대답 200개 맞힌 정도(F1) 0.32 → 0.76,
10자 미만 짧은 대답도 판단한다 (노션 "AI 안부말벗·정서분석 모델 학습 기록").

- prob_add 합치기(기본): 답을 만들지 않고 '{"mood": "' 다음 첫 토큰이 4코드 중 무엇일지의 확률을 한 번에 읽는다.
  가장 큰 코드가 글 기분이고, 확률은 표정과 합칠 때 쓴다 (DGX cmp/fuse_cmp.py 의 R3 규칙).
- plan41 합치기: 답을 만들어 코드를 읽는다. 따옴표가 빠진 {mood: X} 도 받는다. 읽지 못하면 UNKNOWN (판단 불가).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.llm_client import MOOD_ADAPTER, LlmUnavailable
from app.services.companion.text_emotion_service import TextEmotionResult, TextEmotionUnavailable

MOOD_PROMPT_VERSION = "mood-e900"
MOOD_CODES = ("JOY", "CALM", "WORRY", "DEPRESSED")
UNKNOWN = "UNKNOWN"
PREFIX = '{"mood": "'
_PROMPT = Path(__file__).parent / "prompts" / f"{MOOD_PROMPT_VERSION}.txt"
_CODE = re.compile(r'"?mood"?\s*:\s*"?([A-Za-z]+)')


def parse_mood(raw: str | None) -> str:
    """모델 답에서 기분 코드를 읽는다. 따옴표가 없어도 받는다. 4코드가 아니면 UNKNOWN."""
    m = _CODE.search(raw or "")
    code = m.group(1).upper() if m else None
    return code if code in MOOD_CODES else UNKNOWN


def _user(text: str) -> str:
    return "대답:\n" + text  # 학습 때와 같은 형식 (cmp/ft_mood4.py)


class MoodService:
    """글 기분 분류. 모델은 말벗 LLM 한 개를 같이 쓴다 (어댑터만 바꾼다)."""

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def _client(self):
        if self._llm is not None:
            return self._llm
        from app.services.companion.llm_client import llm_client

        return llm_client

    def classify(self, text: str) -> TextEmotionResult:
        """글을 기분 코드 하나로 나눈다. 모델·어댑터를 쓸 수 없으면 TextEmotionUnavailable (표정만으로 판단)."""
        s = get_settings()
        llm = self._client()
        system = _PROMPT.read_text(encoding="utf-8")
        try:
            if s.companion_fusion_rule == "prob_add":
                probs = llm.message_token_probs(system, _user(text), PREFIX, MOOD_CODES, adapter=MOOD_ADAPTER,
                                                timeout=s.companion_text_timeout_sec)
                scores = {c: float(probs[c]) for c in MOOD_CODES}  # 반올림하지 않는다 (같은 값이 생기면 판정이 시험과 달라진다)
                label = max(MOOD_CODES, key=lambda c: scores[c])
                return TextEmotionResult(label=label, confidence=scores[label], usable=True, scores=scores)
            raw = llm.generate_adapter(_user(text), adapter=MOOD_ADAPTER, system=system, temperature=0.0,
                                       num_predict=16, timeout=s.companion_text_timeout_sec)
        except (LlmUnavailable, AttributeError) as exc:  # AttributeError: 확률·어댑터 기능이 없는 LLM
            raise TextEmotionUnavailable("글 기분 모델을 쓸 수 없습니다") from exc
        code = parse_mood(raw)
        usable = code != UNKNOWN
        return TextEmotionResult(label=code, confidence=1.0 if usable else 0.0, usable=usable, scores={})


mood_service = MoodService()
