"""위험 신호 판정 (노인 자살·자해 신호). 답을 만들지 않고 '있' 글자가 나올 확률을 읽어 판단한다.

확률이 기준 이상이면 위험(riskFlag = true). 말 한마디로 판단하므로 아주 돌려 말한 신호는 놓칠 수 있다.
11/10: 기본은 v9 프롬프트 + 기준 0.99 (COMPANION_RISK_PROMPT=v4 면 재희 v4 + 0.5).
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from app.core.config import get_settings

RISK_PROMPT_VERSION = "risk-v4"  # 프롬프트 파일을 고치면 올린다
RISK_THRESHOLD = 0.5  # 2026-11-09 측정. 검증 세트 B(개발에 안 쓴 새 문장 위험 20·평범 20): v4 18/20 포착, 헛경보 0/20
# 11/10 v9: 노인의 돌려 말한 자살 신호를 넣은 프롬프트. DGX 시험(cmp/risk_v9.py)과 같은 형식:
# system = 프롬프트, user = "대답:\n" + 말, 모델 답 앞에 '{"risk": "' 를 이어 붙이고 '있음'·'없음' 첫 토큰 확률을 비교한다
RISK_V9_PROMPT_VERSION = "risk-v9"
RISK_V9_THRESHOLD = 0.99  # 평범한 안부 통화 강제 알림 100통 중 11 → 1 미만 (노션 "재희 버전에서 무엇을 바꿨나" 6번)
_PROMPT_V9 = Path(__file__).parent / "prompts" / f"{RISK_V9_PROMPT_VERSION}.txt"
_PROMPT = Path(__file__).parent / "prompts" / f"{RISK_PROMPT_VERSION}.txt"
PREFIX = '{"risk": "'  # 모델이 이 뒤에 '있' 또는 '없'을 쓰도록 미리 적어 둔다
CANDIDATES = ("있", "없")


def risk_prompt_version() -> str:
    return RISK_V9_PROMPT_VERSION if get_settings().companion_risk_prompt == "v9" else RISK_PROMPT_VERSION


logger = logging.getLogger(__name__)
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def risk_threshold() -> float:
    """설정값이 있으면 그 값, 없으면 프롬프트에 맞춘 기준 (v9 0.99 · v4 0.5).

    11/10 점검 M-3: 0 이하·1 이상·숫자가 아닌 값(nan·inf)은 위험 모델을 사실상 끄거나 모든 턴을 강제 알림으로 만든다.
    그런 값은 경고 후 기본 기준을 쓴다.
    """
    s = get_settings()
    default = RISK_V9_THRESHOLD if s.companion_risk_prompt == "v9" else RISK_THRESHOLD
    t = s.companion_risk_threshold
    if t is None:
        return default
    if not (math.isfinite(t) and 0 < t < 1):
        _warn_once("threshold", "[COMPANION-RISK] COMPANION_RISK_THRESHOLD 가 0과 1 사이가 아니어서 기본 기준 %.2f 을 씁니다", default)
        return default
    return t


def _finite(p) -> float | None:
    """확률이 숫자가 아니면(NaN·무한대) 계산 못 한 것으로 본다 - 그래야 신호 추출의 riskFlag 로 넘어간다 (M-3)."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if math.isfinite(p) else None


def _risk_probability_v9(llm, transcript: str) -> float | None:
    probs_fn = getattr(llm, "message_token_probs", None)
    if probs_fn is None:
        return None
    try:
        # 위험은 어댑터를 끈 기본 모델로 본다 (기분·대답 어댑터는 위험 판정에 학습되지 않았다)
        probs = probs_fn(_PROMPT_V9.read_text(encoding="utf-8"), "대답:\n" + transcript, PREFIX, CANDIDATES,
                         adapter=None, timeout=get_settings().companion_llm_timeout_sec)
    except Exception as exc:  # noqa: BLE001 - 위험 판정이 실패해도 대화를 막지 않는다
        logger.warning("[COMPANION-RISK] model probability unavailable (%s)", type(exc).__name__)  # 원문·메시지 없이
        return None
    return _finite(probs.get("있"))


def risk_probability(llm, transcript: str) -> float | None:
    """'있'일 확률. 계산할 수 없으면 None (대화는 계속된다)."""
    if not transcript.strip():
        return None
    if get_settings().companion_risk_prompt == "v9":
        return _risk_probability_v9(llm, transcript)
    probs_fn = getattr(llm, "next_token_probs", None)
    if probs_fn is None or not transcript.strip():
        return None
    prompt = _PROMPT.read_text(encoding="utf-8").replace("{{TRANSCRIPT}}", transcript)
    try:
        probs = probs_fn(prompt, PREFIX, CANDIDATES, timeout=get_settings().companion_llm_timeout_sec)
    except Exception as exc:  # noqa: BLE001 - 위험 판정이 실패해도 대화를 막지 않는다
        logger.warning("[COMPANION-RISK] model probability unavailable (%s)", type(exc).__name__)
        return None
    return _finite(probs.get("있"))


def is_risk(probability: float | None) -> bool:
    return probability is not None and probability >= risk_threshold()


# 백엔드 결정 E (2026-11-03): 응급·자해 낱말은 모델과 따로, 규칙으로 먼저 잡는다. 모델이 꺼져 있어도 알림이 나간다.
RISK_KEYWORDS = (
    "죽고 싶", "죽고싶", "죽겠", "살기 싫", "살기싫", "살고 싶지", "살고싶지", "사라지고 싶", "사라지고싶",
    "끝내고 싶", "끝내고싶", "목숨", "자살", "자해", "약을 모", "약 모", "따라가고 싶", "따라가고싶",
    "숨이 안 쉬", "숨이 안쉬", "숨을 못", "가슴이 너무 아", "쓰러졌", "119", "꼼짝을 못", "꼼짝 못",
)


# 11/10 kw4: '아파 죽겠어' 같은 관용 표현의 '죽겠'은 위험으로 보지 않는다 (DGX cmp/jh_kw2.py 목록 그대로)
IDIOM = re.compile(r"(아파|아퍼|바빠|답답해|피곤해|미안해|힘들어|더워|추워|배고파|졸려|심심해|궁금해|귀찮아|가려워|짜증나|배불러|"
                   r"미치겠|좋아|웃겨|속상해|걱정돼|무서워|외로워|지겨워|서러워|저려|쑤셔|결려)(서)?\s*죽겠")
# '자살'·'자해' 앞에 붙은 한글이 다른 낱말의 끝일 때만 뺀다 (11/10 점검 H-4).
# 예전 kw4 는 앞이 한글이면 모두 뺐는데, 띄어쓰기를 빠뜨린 "그냥자살하고싶어"·"난자살할거야"까지 놓쳤다.
# 이제는 ① 이미 아는 '○자' 낱말(혼자 살다·여자살인·손자 살) 이거나 ② 띄어 쓴 "○자 살/해" 꼴일 때만 뺀다.
BOUND_KEYWORDS = frozenset({"자살", "자해"})
JA_WORDS = frozenset({"혼자", "손자", "여자", "남자", "부자", "왕자", "의자", "사자", "이자", "감자", "과자", "모자", "피자",
                      "액자", "상자", "탁자", "제자", "효자", "박자", "숫자", "글자", "문자", "한자", "투자", "환자", "기자",
                      "원자", "전자", "양자", "유자", "가자", "하자", "주자", "타자", "아자"})
# kw4 에만 더하는 낱말 (재희 목록 RISK_KEYWORDS 는 그대로) - 두 규칙 모두 놓치던 말 (M-4)
KW4_EXTRA = ("죽어버리", "죽고 시퍼", "죽고시퍼", "죽어 버리")
# 낱말 사이에 끼어 비교를 막는 문장부호 ("자.살", "자-해") - 지우고 비교한다 (M-4)
_PUNCT = re.compile(r"[.\-,·~!?'\"…_/]")


def _keyword_risk_jaehee(transcript: str) -> bool:
    text = transcript.replace(" ", "")
    return any(k.replace(" ", "") in text for k in RISK_KEYWORDS)


def _bound_hit(text: str, word: str) -> bool:
    """'자살'·'자해'가 위험한 뜻으로 쓰였는지. 다른 낱말의 끝 '자'에 '살/해'가 이어진 경우만 뺀다."""
    for m in re.finditer(word[0] + r"(\s*)" + word[1], text):
        i = m.start()
        prev = text[i - 1] if i > 0 else ""
        if not re.match(r"[가-힣]", prev):
            return True  # 앞이 한글이 아니면(문장 처음·띄어쓰기 뒤) 위험
        if prev + "자" in JA_WORDS:
            continue  # 혼자살다·여자살인·손자 살 쪘어
        if m.group(1):
            continue  # "○자 살" 처럼 띄어 쓴 꼴은 다른 낱말로 본다
        return True  # 그냥자살·난자살·엄마자해
    return False


def _keyword_risk_kw4(transcript: str) -> bool:
    text = IDIOM.sub("", _PUNCT.sub("", transcript))
    flat = text.replace(" ", "")
    for k in dict.fromkeys(RISK_KEYWORDS + KW4_EXTRA):
        kk = k.replace(" ", "")
        if kk in BOUND_KEYWORDS:
            if _bound_hit(text, kk):
                return True
        elif kk in flat:
            return True
    return False


def keyword_risk(transcript: str) -> bool:
    """응급·자해 낱말이 하나라도 있으면 True. 띄어쓰기 차이는 무시한다 (kw4 는 위 두 가지 오탐만 뺀다)."""
    if get_settings().companion_risk_keyword_rule == "jaehee":
        return _keyword_risk_jaehee(transcript)
    return _keyword_risk_kw4(transcript)
