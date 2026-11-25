from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.llm_client import REPLY_ADAPTER, REPLY_LAST_ADAPTER, LlmClient, LlmUnavailable
from app.services.companion import risk_carry
from app.services.companion.risk import body_emergency_only, keyword_risk

logger = logging.getLogger(__name__)

REPLY_PROMPT_VERSION = "reply-v2"  # 응답 프롬프트 파일을 고치면 올린다 (v2: 대답 속 낱말을 이어 가는 꼬리질문)
# 11/15: 대화 흐름(말벗 문장 + 어르신 말 한 줄 요약)을 넣은 프롬프트 (COMPANION_REPLY_HISTORY=summary).
#   꼬리질문 reply-v4-followup · 마지막 인사 reply-v2-last(직전 질문도 넣는다). off 면 위 reply-v2 그대로
#   11/21 v3 → v4: 흐름 블록을 맨 앞 참고로 옮기고 뒤는 대답 LoRA(cc7)가 배운 v2 본문 그대로, 길이는 "정확히 두 문장".
#   v3 는 흐름을 이어 붙여 세 문장(2문장 초과 11 vs off 6)·화제 이탈이 늘었다 (reply-v3-followup.txt 는 기록으로 둔다)
REPLY_HISTORY_PROMPT_VERSION = "reply-v4"
HISTORY_EMPTY = "(아직 없음 - 첫 대답입니다)"
HISTORY_NO_ANSWER = "(대답 없음)"
HISTORY_NO_SUMMARY = "(요약 없음)"
HISTORY_UNCLEAR = "(잘 알아듣지 못한 말)"


def reply_prompt_version(is_last_turn: bool = False) -> str:
    """지금 설정에서 쓰는 대답 프롬프트 버전 (검수 목록·대화 결과의 promptVersion).

    is_last_turn=True 인 마지막 인사가 summary 에서 COMPANION_REPLY_HISTORY_LAST=false 면 실제로는 흐름 없는 v1-last 를 쓴다
    (build_reply_prompt) - off 와 같은 파일이라 reply-v2 로 적는다 (2026-11-21 L-8)."""
    s = get_settings()
    if s.companion_reply_history != "summary":
        return REPLY_PROMPT_VERSION
    if is_last_turn and not s.companion_reply_history_last:
        return REPLY_PROMPT_VERSION
    return REPLY_HISTORY_PROMPT_VERSION
# 대답과 관계없는 문장을 거를 때 기준으로 삼지 않는 흔한 낱말의 앞 두 글자 (조사가 붙은 말도 걸리도록 앞 글자로 본다)
COMMON_STEMS = {"그냥", "오늘", "어제", "요즘", "정말", "너무", "조금", "많이", "그래", "있어", "없어",
                "했어", "같아", "좋아", "괜찮", "아니", "그럼", "이제", "다시"}
_PROMPT_DIR = Path(__file__).parent / "prompts"

# S6-2: 규칙 검사에 쓰는 목록. 한 곳에만 둔다 (C-23 결정 전까지 이 목록을 그대로 쓴다)
DISEASE_TERMS = ("우울증", "치매", "불안증", "공황장애", "조현병", "조울증")
PAST_TERMS = ("지난", "어제", "예전", "옛날", "여행")
# 11/13: '과거 가정' 완화 (DGX 재현, AI main 460200f 사본).
#   ① 어르신이 이번 턴에 지난 때를 먼저 말했으면(예전에는·옛날에·전에는·젊었을 때) 대답의 예전·옛날은 넘겨짚기가 아니다.
#      같은 낱말(어제·지난·여행)을 어르신이 먼저 말한 경우도 마찬가지다.
#   ② '예전 같지 않다·예전만 못하다'는 지난 일을 넘겨짚는 말이 아니라 지금 상태를 말하는 관용 표현이다
#      ("몸이 예전같지 않으셔서 피곤하시겠네요" - 앞 턴에서 어르신이 한 말을 받았는데 탈락했다)
#      11/14 점검 M-1: '예전처럼·예전만 하'는 뺐다 - "요즘도 예전처럼 자주 산책하세요?"는 지난 일을 넘겨짚는다
ELDER_PAST_PATTERN = re.compile(r"예전|옛날|전에|전엔|젊었을|젊을\s?때")
ELDER_PAST_ALLOWS = ("예전", "옛날")
PAST_IDIOM_PATTERN = re.compile(r"예전\s?같지\s?않|예전만\s?못")
# 11/13: '대답과 무관' 보정 - 어르신 말의 뜻을 다른 낱말로 받은 대답을 통과시킨다 (같은 뜻 묶음).
#   앞 두 글자 비교만으로는 못잤어↔못 주무셨다니, 못했어↔못하셨군요, 몸이↔몸 상태, 아파서↔불편하시겠어요가 모두 걸렸다.
#   어르신 말과 대답이 같은 묶음에 함께 걸리면 이어 받은 대답으로 본다. "공감 한 마디 + 질문" 꼴을 통째로 허용하지 않는 것은
#   걸러야 할 대답("잠이 많이 중요하시겠네요. …편안했던 순간은 언제였나요?")도 같은 꼴이기 때문이다.
#   한계: '못 ~'만 받은 대답은 화제가 달라도 통과한다 (대답은 질문 문맥을 보고 만들어져 드물다고 보고 수용).
#   11/14 점검 L-1: 어르신이 되물은 말("어디 아픈 곳이라도 있나요?")은 묶음으로 보지 않는다 - 물음을 대답으로 받은
#   "아프신 곳은 없으시다니 다행이네요"가 통과했다. 이때는 앞 두 글자 비교만 한다
RELATED_GROUPS = (
    re.compile(r"잠(?!깐|시)|잤|(?<![남여])자고|(?<![남여])자요|주무|깼|깨서|깨요|수면"),  # 잠 ('잠깐·잠시·남자요'는 뺀다, 11/14 L-4)
    re.compile(r"(?<![잘연])못\s?[가-힣]"),  # 못 함 ('잘못·연못'은 뺀다)
    re.compile(r"몸|아프|아파|아픈|아팠|쑤시|쑤셔|결리|저리|저려|불편|편찮|몸살|피곤|건강|무릎|허리|어깨"),  # 몸 불편
)

# S6-3: 규칙을 통과하지 못하면 이 목록에서 고른다. 모든 문장이 규칙을 통과해야 한다 (시험으로 확인)
# 11/11: 대체 문장은 어르신 말을 받아 주기만 하고 화제를 바꾸지 않는다 (식사·하루 일과 질문 금지).
#   힘든 이야기에 "요즘 식사는 잘 챙겨 드시고 계세요?"가 나간 일이 있었다 - '대답과 무관'에 걸린 모델 대답이 이 문장으로 바뀌었다
FALLBACK_FOLLOWUP = (
    "그러셨군요. 조금 더 들려주실 수 있을까요?",
    "말씀해 주셔서 고맙습니다. 그 이야기 조금 더 해 주실 수 있으세요?",
    "네, 잘 들었어요. 더 하고 싶으신 말씀 있으세요?",  # 11/12 M-1: 짧은 대답에도 맞게 시점(그때)을 뺐다
)
# 그 턴의 글 기분이 부정(슬픔·화남·우울)이면 이 목록에서 고른다 (11/11)
# 11/25: 꼬리질문 대체 문장은 두 문장까지다 (품질 세트 R8 '2문장 초과' - 세 문장이던 첫 문장 "그러셨군요."를 뺐다)
FALLBACK_EMPATHY = (
    "마음이 많이 쓰이셨겠어요. 조금 더 이야기해 주실 수 있을까요?",
    "많이 속상하셨겠어요. 어떤 점이 가장 마음에 걸리세요?",
    "힘든 이야기 꺼내 주셔서 고맙습니다. 제가 더 들어 드려도 될까요?",
)
NEGATIVE_TURN_LABELS = ("슬픔", "화남", "우울")
# 이 글 기분의 턴은 '대답과 무관' 검사를 하지 않는다 (11/12, COMPANION_REPLY_RELAX_CALM).
#   DGX 측정(CareCall 200): '대답과 무관'만으로 대체된 62개 중 약 50개가 멀쩡한 대답이었다.
#   힘든 이야기(93)는 풀면 살아난 대답의 절반이 질문 주제에 끌려간 오해라 부정 기분 턴은 그대로 둔다
RELAX_TURN_LABELS = ("평온", "기쁨")
# '권유' 규칙: 꼬리질문 대신 무엇을 하라고 권하는 문장을 거른다 (11/12).
#   11/12 점검 M-1: 처음 정규식("산책·운동·음악 … 어떠세요")은 주제를 이어 가는 물음("요즘 운동은 어떠세요?",
#   "산책 다녀오시니 기분은 어떠세요?", "말씀해 보세요?")까지 걸러서, 권하는 꼴만 보도록 좁혔다:
#   ① 해·드셔·들어·가 … + 보시는/보시면/보세요/보는 건 (단 '말씀해 보세요'는 이야기를 청하는 말이라 뺀다)
#   ② 권유 낱말 바로 뒤 어떠세요 ("가벼운 산책 어떠세요?" - 조사 '은·는'이 붙은 물음은 통과)
#      스트레칭·명상, '가벼운·간단한 …'은 어르신이 먼저 꺼내는 주제가 드물어 예전처럼 넓게 본다 (DGX 대조 11/12)
#   ③ 하시는·드시는 건 어떠세요  ④ 하시면 좋아요·좋겠·좋을 것 ("하시면 좋을까요?"는 물음이라 통과)
ADVICE_PATTERN = re.compile(
    r"(?:(?<!말씀)해|드셔|들어|맞아|걸어|나가|가|연락해|시작해)\s?보(?:시는|시면|시길|세요|시죠|실래요|는\s?건|는\s?거)"
    r"|(?:스트레칭|산책|걷기|운동|명상|음악)\s?(?:한\s?번\s?)?(?:어떠신가요|어떨까요|어떠세요)"
    r"|(?:스트레칭|명상|가벼운|간단한)[^.?!]*(?:어떠신가요|어떨까요|어떠세요)"
    r"|(?:하|드)시는\s?건\s?(?:어떠신가요|어떨까요|어떠세요)"
    r"|하시면\s?좋(?:아요|습니다|겠|을\s?것|을\s?거)"
    # ⑤ 11/14 점검 H-1: "어떻게 하면 … 생각해 보셨나요?"(방법을 궁리해 보라는 권유). '과거 가정'이 우연히 걸러 주던 대답이다
    r"|어떻게\s?하면[^.?!]*(?:생각해\s?보|해\s?보|찾아\s?보)(?:셨|시|세|실)"
)
# 어르신이 아무 말도 하지 않은 턴(무응답)의 문장 (11/12 점검 H-1). 말을 들은 것처럼 받지 않고, 편하게 다시 말을 건넨다.
# 무응답 턴에는 대화를 다시 여는 질문이 화제 전환이 아니다
FALLBACK_NO_RESPONSE = (
    "천천히 생각하셔도 괜찮아요. 편하실 때 말씀해 주시겠어요?",
    "제 말이 잘 들리셨을까요? 편하실 때 한 번 더 말씀해 주시겠어요?",
    "괜찮으시면 지금 기분이 어떠신지 들려주시겠어요?",
)
# 어르신이 말을 피하는 턴의 문장 (11/12, avoidance.is_avoidance). 캐묻지 않고 물러나되, 대화는 이어 가도록 물음으로 끝낸다.
#   "조금 더 들려주실 수 있을까요?"(받아 주는 대체 문장)·"특별히 걱정되는 일이 있으신가요?"(cc7)는 더 말하라고 캐묻게 됐다
FALLBACK_AVOID = (
    "괜찮아요, 말씀하고 싶지 않으시면 안 하셔도 돼요. 지금 마음은 좀 편안하세요?",
    "네, 편하실 때 말씀해 주세요. 오늘 몸은 괜찮으세요?",  # 11/25 세 문장 → 두 문장
)
# 검수 목록에 남는 대체 문장 버전. 대체 문장 목록을 고치면 올린다 (v2: 11/11~12 화제 전환 제거·공감·무응답 목록)
FALLBACK_VERSION = "fallback-v5"  # v3: 말을 피하는 턴 문장 (11/12) · v4: 짧은 대답이 이어질 때 화제 전환 질문 (11/13) · v5: 두 문장까지 (11/25)
FALLBACK_LAST = (
    "오늘도 이야기 나눠 주셔서 고맙습니다. 편안한 하루 보내세요.",
)
# 위험 신호가 켜진 턴의 응답: 평소 규칙(꼬리질문 등)을 따르지 않고 안내 문장으로 답한다 (내용 확인 필요)
SAFETY_REPLY = (
    "많이 힘드셨겠어요. 지금 마음이 많이 무거우신 것 같아요. 혼자 견디지 마시고 가까운 가족이나 "
    "주변 분께 바로 연락해 주세요. 급하시면 119에 전화하실 수 있어요."
)

# 몸 응급 낱말(숨·가슴·쓰러짐·뇌졸중 징후)만 걸린 위험 턴의 안내 (11/12 영향 범위 점검 Q-1, COMPANION_BODY_SAFETY_REPLY).
#   SAFETY_REPLY("마음이 많이 무거우신 것 같아요")는 "혀가 굳어서 말을 못 하겠어" 같은 몸 응급에 맞지 않았다
BODY_SAFETY_REPLY = (
    "많이 놀라셨겠어요. 몸이 많이 불편하시거나 위급한 상황이면 바로 119에 전화하시고, 가까운 가족에게도 알려 주세요."
)


def safety_reply(transcript: str) -> str:
    """위험 턴의 안내 문장. 몸 응급 낱말만 걸렸으면 몸 응급용, 아니면 기존 안내 (스위치를 끄면 늘 기존 안내)."""
    if not get_settings().companion_body_safety_reply:
        return SAFETY_REPLY
    if body_emergency_only(transcript):
        return BODY_SAFETY_REPLY
    # 11/15 I-21: 위험 낱말이 하나도 없이 모델로만 위험인데 몸 증상이 보이면("가슴이 꽉 조이고 숨이 차", "오른팔에 힘이 없어")
    #   몸 응급 안내. 다른 위험 낱말(자해·죽음)이 있거나 마음 쪽 표시가 있으면 지금처럼 마음 쪽 안내. 판정은 바꾸지 않는다
    if not keyword_risk(transcript) and risk_carry.is_body_risk(transcript):
        return BODY_SAFETY_REPLY
    return SAFETY_REPLY


@dataclass
class ReplyResult:
    text: str
    isFollowUp: bool
    source: str  # LLM | FALLBACK
    failures: list[str]  # 마지막으로 걸린 규칙 이름 (없으면 빈 목록)


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[가-힣]+", text) if len(w) >= 2 and w[:2] not in COMMON_STEMS]


def _shares_meaning(transcript: str, body: str) -> bool:
    """어르신 말과 대답이 같은 뜻 묶음(RELATED_GROUPS)에 함께 걸리는가 (11/13). 어르신 말이 물음이면 보지 않는다 (11/14 L-1)."""
    if transcript.rstrip().endswith("?"):
        return False
    return any(g.search(transcript) and g.search(body) for g in RELATED_GROUPS)


def _presumes_past(body: str, transcript: str) -> bool:
    """대답이 어르신이 말하지 않은 지난 일을 넘겨짚는가 ('과거 가정', 11/13 완화)."""
    rest = PAST_IDIOM_PATTERN.sub("", body)  # '예전 같지 않다' 같은 지금 상태 표현은 지운 뒤 본다
    elder_past = bool(ELDER_PAST_PATTERN.search(transcript))
    for term in PAST_TERMS:
        if term not in rest:
            continue
        if term in transcript or (elder_past and term in ELDER_PAST_ALLOWS):
            continue
        return True
    return False


def is_relaxed_mood(text_mood: dict | None) -> bool:
    """글 기분이 쓸 수 있는 결과이고 평온·기쁨인가 ('대답과 무관' 검사를 빼는 턴)."""
    return bool(text_mood and text_mood.get("usable") and text_mood.get("turnLabel") in RELAX_TURN_LABELS)


# 11/21 대답 검사 보강 (COMPANION_REPLY_STRICT, 기본 켬) - gosky 실측·품질 세트에서 남은 문제
# ① 마지막 인사가 어르신이 앞으로 하겠다는 일을 이미 한 일처럼 받음 ("이따가 김치찌개 데워 먹으려고" → "마무리하시는 모습이 보기 좋네요")
STRICT_FUTURE_MARK = re.compile(r"려고|[을를할갈볼올먹]\s?거(야|예요|에요|라)|기로\s?했|이따|내일|다음\s?주|다음\s?달|주말에|나중에")
STRICT_DONE_FORM = re.compile(r"(드|하|다녀오|가|만드|끓이|보내|사|데우|데워|먹으)셨(네요|군요|어요|다니|구나|다는|겠어요)|잘\s?드셨"
                              r"|모습이\s?보기\s?좋|마무리하시는|마무리하셨")
# ② 막연한 질문 (어르신이 먼저 꺼낸 말이면 보지 않는다)
STRICT_VAGUE_Q = re.compile(r"가장\s?(좋아하|즐거|기억에\s?남|행복하|좋았)[^.?!]{0,8}(시간|순간)|하고\s?싶은\s?활동|어떤\s?활동|다른\s?활동|취미")
# ③ 계획 캐묻기 - 사용자 결정(11/21): 계획·예정·언제쯤·하실 생각만 본다 ('왜·무슨 이유'는 보지 않는다)
STRICT_PLAN_Q = re.compile(r"계획|예정이|언제쯤|하실\s?생각")
# 11/25 막연한 질문 다시 만들기(COMPANION_REPLY_VAGUE_REDO): ②보다 넓다 - 판정 세트(followup_v1)·cc8d 출력에서 본 형태를 일반화했다.
#   "특별히 ~ 있으신가요"는 어르신 대답에 내용이 있을 때만 막연한 질문으로 본다(내용 없는 대답에는 달리 물을 거리가 없다).
VAGUE_REDO_Q = re.compile(STRICT_VAGUE_Q.pattern + r"|활동(은|이|을)?\s?(있|하시|어떠|어떻)|(어떻게|뭐\s?하며)\s?(지내|보내)실\s?(계획|예정)"
                          r"|그때까지|생각하시는\s?(것|게|일)|(어떤|무슨)\s?계획이\s?있으")
VAGUE_SPECIAL_Q = re.compile(r"특별히[^.?!]{0,14}(있으|있나|있었|없으)")
VAGUE_ECHO_WORDS = ("활동", "계획", "특별히", "취미", "그때")  # 어르신이 먼저 쓴 낱말을 받아 묻는 것은 막연한 질문으로 보지 않는다
VAGUE_REDO_HINT = ("\n(다시 쓰기) 질문에는 어르신 대답에 나온 낱말을 넣어 구체적으로 한 가지만 묻습니다. "
                   "'특별히'·'활동'·'계획'이라는 말은 쓰지 않습니다.")
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


def _question_part(body: str) -> str:
    """마지막 문장(꼬리질문). 물음이 없으면 빈 문자열."""
    sents = [x for x in _SENT_SPLIT.split(body.strip()) if x]
    return sents[-1] if sents and sents[-1].endswith("?") else ""


# 11/23 대답 첫머리 낱말 깨짐 고치기: 문장 맨 앞의 "경로당"은 토큰 3개(경·로·당)로 쪼개져, 대답 LoRA 가 '경' 다음에 흔한 '량'을
#   고른다("경량에서…"·"경량당에서…"·"경 경로당에서…"). 문장 중간(" 경로당" 1토큰)에서는 깨지지 않는다.
#   어르신 말에 그 낱말이 있을 때만, 깨진 표기를 바로잡는다 (DGX 재현: 기본 모델 0/30 · cc7 3/30 · cc8b 7/30).
#   다른 낱말도 같은 현상이 확인되면 이 표에 더한다.
SPLIT_WORD_FIXES = {
    # 순서가 중요하다: 조사가 바로 붙은 '경량'(경량에서)을 먼저, 한 글자가 더 붙은 꼴(경량리에)은 그다음
    "경로당": re.compile(r"경\s?량\s?당|경\s+경로당|경로딩|경노당|경량(?=[에은는이가을를과와도의서로]|\s|$)"
                      r"|경량[가-힣](?=[에은는이가을를과와도의서로]|\s|$)"),
}


def fix_split_words(text: str, transcript: str) -> str:
    """어르신 말에 있는 낱말이 대답에서 깨진 표기로 나오면 바로잡는다. 원문은 로그에 남기지 않는다."""
    for word, broken in SPLIT_WORD_FIXES.items():
        if word in (transcript or "") and broken.search(text):
            text = broken.sub(word, text)
            logger.info("[COMPANION-REPLY-SPLITFIX] word=%s", word)
    return text


def trim_reply(text: str, is_last_turn: bool) -> str:
    """④ 꼬리질문이 세 문장 이상이면 첫 문장(공감)과 마지막 질문만 남긴다 (11/21, 대체하지 않고 모델 문장을 살린다).

    마지막 질문이 물음으로 끝나지 않으면 그대로 둔다 (규칙 검사가 따로 본다). 마지막 인사는 줄이지 않는다.
    """
    s = get_settings()
    if is_last_turn or not (s.companion_reply_trim or s.companion_reply_strict):
        return text
    sents = [x for x in _SENT_SPLIT.split(text.strip()) if x]
    if len(sents) <= 2 or not sents[-1].endswith("?"):
        return text
    logger.info("[COMPANION-REPLY-TRIM] sentences=%d", len(sents))  # 원문은 남기지 않는다
    return f"{sents[0]} {sents[-1]}"


def is_vague_question(body: str, transcript: str) -> bool:
    """꼬리질문(마지막 문장)이 막연한가 (11/25). 어르신이 먼저 꺼낸 말(같은 표현)이면 보지 않는다."""
    q = _question_part(body)
    if not q or any(w in q and w in transcript for w in VAGUE_ECHO_WORDS):
        return False
    if VAGUE_REDO_Q.search(q) and not VAGUE_REDO_Q.search(transcript):
        return True
    return bool(VAGUE_SPECIAL_Q.search(q)) and not is_flat_answer(transcript)


# 11/25 꼬리질문 시제 고쳐 쓰기 (COMPANION_REPLY_TENSE_FIX). 어르신이 앞으로 하실 일("이따 ~하려고")을 대답이 이미 한 일처럼 받는다
#   ("국수 삶는 건 어려우셨나요?"). 마지막 인사 ① 검사(STRICT_FUTURE_MARK)는 시간 낱말만으로 앞일로 보고 과거형 질문을 못 잡아 따로 둔다.
#   앞일은 동사 끝으로만 본다(시간 낱말만 보면 "주말에 왔어"도 걸린다). 짐작("괜찮아질 거야", "비가 올 거 같아")은 앞일이 아니다.
#   DGX 대조(CareCall 상담 대화, 사람 대답): 앞일 절이 있는 587쌍 중 5쌍이 걸렸고 그중 2쌍은 사람 대답이 실제로 틀렸다.
TENSE_FUT_END = re.compile(r"려고|려구|려던|기로\s?했|(?<![가-힣])[온간]대|할\s?까\s?해|해야\s?(?:해|돼|지|겠)|[갈볼]\s?까")
TENSE_GUESS = re.compile(r"거\s?같|질\s?거|될\s?거|있을\s?거|없을\s?거|수\s?있을")
TENSE_CLAUSE = re.compile(r"[,.]|(?<=[었았했왔갔켰줬잤])고\s|(?<=[었았했왔])는데\s")
TENSE_PAST = re.compile(r"(?:셨|었|았|했|왔|갔)(?:나요|어요|군요|네요|다니|구나|다는|겠어요|을\s?것|던|습니까|지요|죠|는지|을까요|니)"
                        r"|어떠셨|어땠")
# 미리 정한 일·하려던 마음을 묻는 과거형은 맞다 ("가시기로 하셨군요", "메뉴는 정하셨어요?", "준비는 해 두셨어요?")
TENSE_OK_PAST = re.compile(r"(?:려고|기로|실\s?거라고|실\s?생각|겠다고)\s?(?:하셨|하신|하시)|정하셨|두셨|준비하셨|계획하셨|예약하셨|생각하셨|마음먹으셨")
TENSE_TIME_STEMS = ("이따", "내일", "모레", "다음", "주말", "나중", "저녁", "오후", "오늘", "그래", "그리", "있다")
# 앞일 절의 동사가 오다·가다·먹다면 대답의 그 동사 과거형도 본다 ("손주가 온대" → "손주가 오셨다니")
TENSE_VERBS = (
    (re.compile(r"(?<![가-힣])온대|올\s?거|오려|오기로|온다"), re.compile(r"오셨|왔다|왔군|다녀가셨|와\s?주셨")),
    (re.compile(r"(?<![가-힣])간대|갈\s?거|가려|가\s?보려|가기로|간다"), re.compile(r"(?<![하드])가셨|다녀오셨|갔다|갔군|가\s?보셨")),
    (re.compile(r"먹으려|먹을\s?거|먹기로"), re.compile(r"드셨|잡수셨|먹었")),
)
TENSE_REDO_HINT = "\n(다시 쓰기) 어르신이 앞으로 하실 일은 앞으로의 일로 받고 묻습니다. 이미 하신 일처럼 묻지 않습니다."
_PART_SPLIT = re.compile(r"(?<=[.?!,])\s+")
# 질문 어미를 앞일로: 상태(어렵다·좋다…)는 지금으로, 동작은 앞으로
_Q_HOW = re.compile(r"어떠셨(?:나요|어요|습니까)\?$")
_Q_STATE = re.compile(r"(어려우|좋으|힘드|즐거우|괜찮으|재미있으|맛있으|편하|바쁘|기쁘|설레|피곤하|있으)셨(?:나요|어요|습니까)\?$")
_Q_ACTION = re.compile(r"([가-힣])셨(?:나요|어요|습니까)\?$")


def _rieul_geo(clause: str) -> bool:
    """받침 ㄹ 글자 + '거'로 끝나는 말("만날 거야", "갈 거")인가. '즐거웠어'처럼 '거' 뒤에 글자가 이어지면 아니다."""
    for m in re.finditer(r"([가-힣])\s?거(?:야|예요|에요|라|여|다|든|고)?(?![가-힣])", clause):
        if (ord(m.group(1)) - 0xAC00) % 28 == 8:
            return True
    return False


def _is_future_clause(clause: str) -> bool:
    return bool(TENSE_FUT_END.search(clause) or (_rieul_geo(clause) and not TENSE_GUESS.search(clause)))


def _clauses(transcript: str) -> list[str]:
    return [c for c in TENSE_CLAUSE.split(transcript or "") if c and c.strip()]


def _tense_bad_sentences(body: str, transcript: str) -> set[int]:
    """앞일을 한 일로 받은 문장 번호 (_SENT_SPLIT 기준). 어르신 말에 앞일 절이 없으면 빈 집합."""
    cls = _clauses(transcript)
    fut = [c for c in cls if _is_future_clause(c)]
    if not fut:
        return set()
    done = [c for c in cls if not _is_future_clause(c)]
    fut_words = [w for c in fut for w in re.findall(r"[가-힣]+", c) if len(w) >= 2 and w[:2] not in TENSE_TIME_STEMS]
    elder_words = [w for w in re.findall(r"[가-힣]+", transcript) if len(w) >= 2]
    verbs = [vp for vf, vp in TENSE_VERBS
             if any(vf.search(c) for c in fut) and not any(vf.search(c) or vp.search(c) for c in done)]
    sents = [x for x in _SENT_SPLIT.split(body.strip()) if x]
    bad = set()
    for i, sent in enumerate(sents):
        plain = TENSE_OK_PAST.sub("", sent)
        for part in _PART_SPLIT.split(plain):
            if TENSE_PAST.search(part) and any(w[:2] in part for w in fut_words):
                bad.add(i)
            elif any(vp.search(part) for vp in verbs):
                bad.add(i)
        # 어르신 말이 전부 앞일이면, 어르신 말을 받은 질문의 과거형도 본다 ("주말에 온대" → "주말 동안 하신 일이 있으셨나요?").
        #   나가기 전 준비를 묻는 질문("우산은 챙기셨나요?")은 어르신 낱말이 없어 걸리지 않는다
        if (not done and i == len(sents) - 1 and plain.endswith("?") and TENSE_PAST.search(plain)
                and any(w[:2] in plain for w in elder_words)):
            bad.add(i)
    return bad


def is_tense_error(body: str, transcript: str) -> bool:
    """꼬리질문이 어르신이 앞으로 하실 일을 이미 한 일처럼 받는가 (11/25)."""
    return bool(_tense_bad_sentences(body, transcript))


def _future_question(q: str) -> str | None:
    """과거 어미 질문을 앞일 질문으로 바꾼다. 바꿀 수 없으면 None.

    '이셨'(이다: "어떤 맛이셨나요")·'않으셨'(않다)은 앞일로 바꾸면 말이 안 돼("맛이실 건가요") 포기한다.
    """
    if re.search(r"(?:이|않으|없으)셨(?:나요|어요|습니까)\?$", q):
        return None
    for pat, rep in ((_Q_HOW, "어떠실 것 같으세요?"), (_Q_STATE, r"\1신가요?"), (_Q_ACTION, r"\1실 건가요?")):
        if pat.search(q):
            return pat.sub(rep, q)
    return None


def rewrite_tense(body: str, transcript: str) -> str | None:
    """앞일을 한 일로 받은 대답을 고쳐 쓴다: 질문은 어미만 앞일로, 틀린 공감 문장은 "그러시군요."로. 못 고치면 None."""
    sents = [x for x in _SENT_SPLIT.split(body.strip()) if x]
    bad = _tense_bad_sentences(body, transcript)
    if not sents or not bad:
        return None
    last = len(sents) - 1
    question = sents[last]
    if last in bad:
        question = _future_question(question)
        if question is None:
            return None
    head = [s for i, s in enumerate(sents[:last]) if i not in bad]
    if len(head) < last:  # 틀린 공감 문장은 받아 주는 말 하나로
        head = ["그러시군요."] + head
    text = " ".join(head + [question])
    return None if is_tense_error(text, transcript) else text


# 11/25 막연한 질문 고쳐 쓰기 (COMPANION_REPLY_VAGUE_REWRITE): 다시 만들기(VAGUE_REDO)는 다시 만들어도 막연했고, 대체 문장은 대화를
#   밋밋하게 만들었다. 모델 공감 문장은 두고 질문만 어르신 말의 낱말로 만든 틀 질문으로 바꾼다.
VAGUE_PLAN_Q = re.compile(r"하실\s?계획이\s?있으|어떤\s?계획을")  # is_vague_question 에 더해 계획을 캐묻는 막연한 질문도 본다
_PARTICLE_TAIL = re.compile(r"(?:에서|에게|한테|이랑|하고|까지|부터|으로|에만|이나|처럼|보다|은|는|을|를|에|도|랑|로|와|과|의|만|이|가)$")
# 이름말이 아닌 말(부사·꾸밈말·홀로 못 쓰는 말). 틀 질문에 넣으면 "혼자 이야기", "나쁜 이야기", "다음은 어떻게"가 된다
NOT_TOPIC_WORDS = frozenset({"혼자", "다음", "하루", "요즘", "아까", "방금", "그냥", "정말", "너무", "같이", "함께", "조금", "많이", "다시",
                             "혹시", "그래도", "나쁜", "좋은", "이따", "나중", "자꾸", "벌써", "아직", "계속", "항상", "가끔", "매일"})
_VERBISH_TAIL = tuple("어아야요고대해했지네다서면게던는니려까")
# 앞일 틀 질문의 조사: 시간 낱말은 "~에는", 장소는 "~에서는", 사람은 "~과는", 그 밖은 "~은" (11/25 "주말은 어떻게 하실 거예요?"가 어색했다)
TIME_NOUNS = ("주말", "내일", "모레", "저녁", "오후", "오늘", "아침", "점심")  # '~요일'도 시간 낱말로 본다
PLACE_NOUNS = ("경로당", "회관", "복지관", "시장", "병원", "교회", "성당", "목욕탕", "미용실", "센터", "공원", "마트", "텃밭", "약국",
               "은행", "아들네", "딸네", "동네", "학교")
PERSON_STEMS = ("딸", "아들", "손주", "사촌", "조카", "손녀", "손자", "따님", "아드님", "친구", "며느리", "사위", "남편", "아내", "영감", "언니",
                "동생", "오빠", "누나", "형님", "이웃", "자식", "애들", "아이들")


def is_vague_for_rewrite(body: str, transcript: str) -> bool:
    """고쳐 쓸 막연한 질문인가: is_vague_question + 계획을 캐묻는 질문(어르신이 먼저 꺼낸 말이면 보지 않는다)."""
    if is_vague_question(body, transcript):
        return True
    q = _question_part(body)
    return bool(q and VAGUE_PLAN_Q.search(q) and not VAGUE_PLAN_Q.search(transcript)
                and not any(w in q and w in transcript for w in VAGUE_ECHO_WORDS))


def _josa(word: str, with_final: str, without_final: str) -> str:
    return word + (with_final if (ord(word[-1]) - 0xAC00) % 28 else without_final)


def _topic_word(transcript: str, body: str) -> tuple[str, bool] | None:
    """틀 질문에 넣을 낱말과 그 낱말이 앞일 절에 있는지. 어르신 말에 있고 모델 대답에도 나온 이름말(2자 이상)만 쓴다.

    앞일 절의 낱말을 먼저, 시간 낱말(주말·내일…)은 맨 나중에 고른다. 못 찾으면 None.
    '이'로 끝나는 낱말은 대답에서 그 뒤에 글자가 바로 붙어 쓰였을 때만 그대로 쓴다(분갈이라니·분갈이는 → 분갈이).
    "마음이 쓰여"처럼 띄어 쓰였으면 '이'는 조사라 뗀다(마음이 → 마음).
    """
    picks: list[tuple[int, int, str, bool]] = []
    for ci, c in enumerate(_clauses(transcript)):
        fut = _is_future_clause(c)
        for raw in re.findall(r"[가-힣]+", c):
            stripped = _PARTICLE_TAIL.sub("", raw)
            # 조사를 떼면 한 글자가 되는 낱말(집에 → 집)은 쓰지 않는다 - 조사째 쓰면 "집에 이야기"가 된다
            #   '이'로 끝나는 이름말(분갈이)은 그대로를 먼저 본다. '가'는 거의 늘 조사라 뗀다 ("손주가와는" 방지)
            #   '과·와'로 끝나는 낱말은 대답에 그 꼴 그대로 나오면 이름말이다 (정형외과 → "정형외" 방지)
            if raw[-1] == "이":
                forms = [raw, stripped]
            elif raw[-1] in "과와" and raw in body:
                forms = [raw]
            else:
                forms = [stripped]
            for f in forms:
                if (len(f) < 2 or f.endswith(_VERBISH_TAIL) or f in FLAT_ANSWER_WORDS or f in NOT_TOPIC_WORDS
                        or f not in body):
                    continue
                # 대답에서 '~하·해·히'로 이어지면 이름말이 아니다 ("답답하고" → "답답"을 떼어 "답답 이야기"가 되지 않게)
                if re.search(re.escape(f) + r"(?:하|해|히|스러)", body):
                    continue
                if f == raw and raw[-1] == "이" and not re.search(re.escape(f) + r"[가-힣]", body):
                    continue
                rank = 2 if (f[:2] in TENSE_TIME_STEMS or f.endswith("요일")) else (0 if fut else 1)
                picks.append((rank, ci, f, fut))
                break
    if not picks:
        return None
    _, _, word, fut = min(picks, key=lambda x: (x[0], x[1]))
    return word, fut


def rewrite_vague(body: str, transcript: str) -> str | None:
    """막연한 질문을 '공감 문장 + 낱말 틀 질문'으로 바꾼다. 낱말을 못 찾으면 None."""
    sents = [x for x in _SENT_SPLIT.split(body.strip()) if x]
    picked = _topic_word(transcript, body)
    if not sents or picked is None:
        return None
    word, fut = picked
    head = [s for s in sents[:-1] if not is_tense_error(s, transcript)] or ["그러시군요."]
    if fut and (word in TIME_NOUNS or word.endswith("요일")):
        q = word + "에는 뭐 하실 거예요?"
    elif fut and word.endswith(PLACE_NOUNS):
        q = word + "에서는 뭐 하실 거예요?"
    elif fut and word.startswith(PERSON_STEMS):
        q = _josa(word, "과는", "와는") + " 뭐 하실 거예요?"
    elif fut:
        q = _josa(word, "은", "는") + " 어떻게 하실 거예요?"
    else:
        q = word + " 이야기 조금 더 들려주시겠어요?"
    return " ".join(head[:1] + [q])


def _apply_followup_fixes(llm: LlmClient, prompt: str, text: str, transcript: str, text_mood: dict | None,
                          redone: bool) -> tuple[str, bool]:
    """규칙을 통과한 꼬리질문의 시제·막연한 질문을 고쳐 쓴다 (11/25, 스위치마다 기본 끔). 고친 문장도 규칙에 걸리면 원문을 둔다.

    반환: (문장, 다시 만들기를 썼는가). 다시 만들기는 한 대답에 한 번만이라 '반복' 다시 만들기와 같은 표시를 쓴다 (11/14 L-4).
    로그에는 동작 이름만 남긴다 (대답·어르신 말 원문 없음).
    """
    s = get_settings()
    if s.companion_reply_tense_fix != "off" and is_tense_error(text, transcript):
        fixed, action = None, "keep"
        if s.companion_reply_tense_fix == "redo" and not redone:
            redone = True
            try:
                raw = _generate(llm, prompt + TENSE_REDO_HINT, False, temperature=s.companion_reply_redo_temperature)
                t2 = trim_reply(fix_split_words(re.sub(r"\s+", " ", raw).strip(), transcript), False)
                if not check_reply(t2, False, transcript, text_mood) and not is_tense_error(t2, transcript):
                    fixed, action = t2, "redo"
            except LlmUnavailable:
                pass
        if fixed is None:
            t3 = rewrite_tense(text, transcript)
            if t3 is not None and not check_reply(t3, False, transcript, text_mood):
                fixed, action = t3, "rewrite"
        logger.info("[COMPANION-REPLY-TENSE] action=%s", action)
        text = fixed or text
    if s.companion_reply_vague_rewrite and is_vague_for_rewrite(text, transcript):
        v = rewrite_vague(text, transcript)
        ok = v is not None and not check_reply(v, False, transcript, text_mood) and not is_tense_error(v, transcript)
        logger.info("[COMPANION-REPLY-VAGUE-REWRITE] ok=%s", ok)
        if ok:
            text = v
    return text, redone


def strict_failures(body: str, is_last_turn: bool, transcript: str) -> list[str]:
    """11/21 검사 보강 ①~③. 걸린 항목 이름 목록. ③ 계획 질문은 COMPANION_REPLY_STRICT_PLAN(기본 끔)일 때만 (11/22)."""
    out = []
    if is_last_turn:
        if STRICT_FUTURE_MARK.search(transcript) and STRICT_DONE_FORM.search(body):
            out.append("앞일을 한 일로")
        return out
    q = _question_part(body)
    if q and STRICT_VAGUE_Q.search(q) and not STRICT_VAGUE_Q.search(transcript):
        out.append("막연한 질문")
    if get_settings().companion_reply_strict_plan and q and STRICT_PLAN_Q.search(q) and not STRICT_PLAN_Q.search(transcript):
        out.append("계획 질문")
    return out


def check_reply(text: str, is_last_turn: bool, transcript: str = "", text_mood: dict | None = None) -> list[str]:
    """규칙을 하나씩 본다. 걸린 항목 이름 목록을 돌려준다. 빈 목록이면 통과.

    text_mood: 그 턴의 글 기분 (pipeline 의 textMood). COMPANION_REPLY_RELAX_CALM 이 켜져 있으면
    평온·기쁨 턴은 '대답과 무관'을 보지 않고, 꼬리질문 턴 전체에 '권유'를 본다 (11/12).
    없거나 부정·판단 불가면 '대답과 무관'은 전과 같다.
    """
    relax = get_settings().companion_reply_relax_calm
    failures: list[str] = []
    body = text.strip()
    if not body:
        return ["비어 있음"]
    if len(body) > get_settings().companion_reply_max_chars:
        failures.append("길이 초과")
    if any(term in body for term in DISEASE_TERMS):
        failures.append("질병명")
    if _presumes_past(body, transcript):
        failures.append("과거 가정")
    if not is_last_turn and not body.endswith("?"):
        failures.append("물음으로 끝나지 않음")
    # 꼬리질문은 어르신 대답에 나온 낱말 가운데 하나를 이어 받아야 한다 (2글자 앞부분으로 비교: 토스트 ↔ 토스트를)
    # 11/13: 같은 뜻 묶음(잠·못 함·몸 불편)에 어르신 말과 대답이 함께 걸려도 이어 받은 것으로 본다
    if not is_last_turn and transcript and not (relax and is_relaxed_mood(text_mood)):
        words = _content_words(transcript)
        if words and not any(w[:2] in body for w in words) and not _shares_meaning(transcript, body):
            failures.append("대답과 무관")
    if relax and not is_last_turn and ADVICE_PATTERN.search(body):
        failures.append("권유")
    s = get_settings()
    if s.companion_reply_strict:
        failures += strict_failures(body, is_last_turn, transcript)
    elif s.companion_reply_last_check and is_last_turn:  # 11/23 ① 만 따로
        failures += strict_failures(body, True, transcript)
    return failures


def _signal_line(prev_signals: dict | None) -> str:
    if not prev_signals:
        return "(직전 신호 없음: 첫 턴입니다)"
    return "직전 턴 신호: " + ", ".join(f"{k}={v}" for k, v in prev_signals.items())


def history_lines(history, max_pairs: int) -> str:
    """대화 흐름을 프롬프트 줄로. history = ((말벗 문장, 어르신 말 요약 또는 None), ...) 오래된 것부터.

    최근 max_pairs 쌍만 넣는다 (0 이하면 전부). 어르신 말은 한 줄 요약이다 - 대답 원문은 AI가 보관하지 않는다.
    """
    pairs = list(history or ())
    if max_pairs > 0:
        pairs = pairs[-max_pairs:]
    if not pairs:
        return HISTORY_EMPTY
    out = []
    for bot, elder in pairs:
        out.append(f"- 말벗: {' '.join((bot or '').split())}")
        out.append(f"- 어르신: {' '.join((elder or HISTORY_NO_SUMMARY).split())}")
    return "\n".join(out)


def _plain_last_prompt(s) -> str:
    """흐름 없는 마지막 인사 프롬프트 파일. v1b(11/23)는 앞으로 하실 일을 앞일로 받으라는 규칙·예시를 더했다."""
    return "reply-v1b-last.txt" if s.companion_reply_last_prompt == "v1b" else "reply-v1-last.txt"


def build_reply_prompt(question: str, transcript: str, prev_signals: dict | None, is_last_turn: bool,
                       history=()) -> str:
    """대답 프롬프트. COMPANION_REPLY_HISTORY=summary 면 대화 흐름을 넣은 v4(꼬리질문)·v2-last(마지막 인사)를 쓴다.

    COMPANION_REPLY_HISTORY_LAST=false 면 summary 에서도 마지막 인사는 흐름 없는 v1-last 를 쓴다 (11/21 - v2-last 가 앞 대화의
    음식을 끌어와 하지 않은 일을 한 것처럼 말하는 치명이 반복돼, 두 가지를 나란히 시험한다).
    """
    s = get_settings()
    if s.companion_reply_history == "summary":
        if is_last_turn:
            name = "reply-v2-last.txt" if s.companion_reply_history_last else _plain_last_prompt(s)
        else:
            name = "reply-v4-followup.txt"
    else:
        name = _plain_last_prompt(s) if is_last_turn else "reply-v2-followup.txt"
    template = (_PROMPT_DIR / name).read_text(encoding="utf-8")
    return (template
            .replace("{{PREV_SIGNALS}}", _signal_line(prev_signals))
            .replace("{{HISTORY}}", history_lines(history, s.companion_reply_history_turns))
            .replace("{{QUESTION}}", question)
            .replace("{{TRANSCRIPT}}", transcript))


# 11/13: 별 내용 없는 짧은 대답("그냥 그래"·"몰라"·"계획 없어")이 이어지면 꼬리질문 모델이 "좋아하시는 활동"을 되풀이했다
#   (실기기 sessionId=212, DGX 재생 cc7 30턴 중 15턴 · 재학습 cc8 도 기준 미달). 이런 턴이 이어지면 아직 묻지 않은 일상 화제로 넘어간다.
# pipeline.SHORT_ANSWER_WORDS(받아쓰기 확신도가 낮아도 되묻지 않는 짧은 부정 대답)와 목적이 달라 따로 둔다 - 한쪽에 낱말을 더하면 다른 쪽도 볼 것
FLAT_ANSWER_WORDS = frozenset({
    "그냥", "그래", "그래요", "그렇지", "그렇죠", "그저", "몰라", "몰라요", "모르겠어", "모르겠어요", "없어", "없어요", "없다",
    "없다니까", "없다고", "없는데", "없지", "없잖아", "별로", "별일", "별거", "글쎄", "글쎄요", "뭐", "됐어", "됐다", "괜찮아",
    "괜찮아요", "아니", "아니요", "아뇨", "할", "거", "알", "계획", "있지", "있어", "지내", "쉬어", "그런", "말", "딱히",
    "그랬어", "그랬어요", "그랬지",  # 11/25 "그냥 그랬어"도 별 내용 없는 대답 (판정 세트 S01)
})
FLAT_ANSWER_MAX_CHARS = 14
# (이미 물었는지 볼 정규식, 화제 전환 질문). 앞에서부터 아직 묻지 않은 것을 쓴다. 모든 문장이 규칙(물음·질병명·과거 가정·권유)을 통과해야 한다.
# 11/14 점검 M-2: 말벗의 물음 문장만 본다(안전 안내 "가까운 가족…"은 물음이 아니다). 오탐을 막는다 - "힘드셨"(식사 아님)·"잠깐/잠시"(잠 아님)·
#   "저녁노을"(식사 아님)·"마음이 불편"(몸 아님)
TOPIC_SWITCH = (
    (r"식사|밥|끼니|반찬|점심|아침 식사|저녁(?!노을)|(?<!힘)드셨|잡수", "그러셨군요. 오늘 식사는 잘 챙겨 드셨어요?"),
    (r"잠(?!깐|시)|주무", "네, 알겠어요. 요즘 밤에 잠은 잘 주무세요?"),
    (r"몸|아프|컨디션|건강|불편하신 (곳|데)", "그러시군요. 요즘 몸은 좀 어떠세요?"),
    (r"가족|자녀|아드님|따님|손주|자제분", "그러셨군요. 요즘 가족분들과는 자주 연락하세요?"),
    (r"이웃|친구|동네", "네, 그러시군요. 요즘 이웃분들과는 이야기 나누고 지내세요?"),
    (r"바깥|산책|밖에|나가셨|나가시", "그러셨군요. 오늘 바깥에 잠깐이라도 나가셨어요?"),
)


def _asked_sentences(said) -> str:
    """말벗 문장들에서 물음 문장만 이어 붙인다 (화제 판정용)."""
    out = []
    for t in said or ():
        out += [x for x in re.split(r"(?<=[.?!])\s+", (t or "").strip()) if x.strip().endswith("?")]
    return " ".join(out)


def is_flat_answer(transcript: str) -> bool:
    """별 내용 없는 짧은 대답인가: 문장부호를 뺀 낱말이 모두 FLAT_ANSWER_WORDS 이고 공백 뺀 길이가 14자 이하."""
    words = "".join(ch if ch.isalnum() else " " for ch in transcript or "").split()
    return (bool(words) and len("".join(words)) <= FLAT_ANSWER_MAX_CHARS
            and all(w in FLAT_ANSWER_WORDS for w in words))


def topic_switch_reply(said) -> str | None:
    """이 대화에서 말벗이 한 말(said)에 아직 나오지 않은 화제의 전환 질문. 다 썼으면 None."""
    text = _asked_sentences(said)
    for pattern, sentence in TOPIC_SWITCH:
        if not re.search(pattern, text):
            return sentence
    return None


def is_negative_mood(text_mood: dict | None) -> bool:
    """글 기분(턴 응답의 textMood)이 쓸 수 있는 결과이고 부정(슬픔·화남·우울)인가."""
    return bool(text_mood and text_mood.get("usable") and text_mood.get("turnLabel") in NEGATIVE_TURN_LABELS)


def fallback_options(is_last_turn: bool, negative: bool = False, no_response: bool = False,
                     avoid: bool = False) -> tuple[str, ...]:
    """대체 문장 목록. 마지막 턴은 마무리 문장, 무응답 턴은 다시 말을 건네는 문장,
    말을 피하는 턴은 물러나는 문장, 부정 기분 턴은 공감 문장, 그 밖은 받아 주는 문장."""
    if is_last_turn:
        return FALLBACK_LAST
    if no_response:
        return FALLBACK_NO_RESPONSE
    if avoid:
        return FALLBACK_AVOID
    return FALLBACK_EMPATHY if negative else FALLBACK_FOLLOWUP


def fallback_text(turn_no: int, is_last_turn: bool, negative: bool = False, no_response: bool = False,
                  avoid: bool = False) -> str:
    options = fallback_options(is_last_turn, negative, no_response, avoid)
    return options[turn_no % len(options)]  # 턴마다 다른 문장이 나오도록 번갈아 쓴다


def _fallback(turn_no: int, is_last_turn: bool) -> str:
    return fallback_text(turn_no, is_last_turn)


def _normalized(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def reply_similarity(a: str, b: str) -> float:
    """두 문장이 얼마나 비슷한가 (0~1, 공백·문장부호를 뺀 글자 비교). 11/13 세션 203: 2번↔5번 대답 0.76, 다른 대답끼리 0.47 이하."""
    na, nb = _normalized(a), _normalized(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def is_similar_to_past(text: str, past_replies) -> bool:
    """이 대화의 지난 말벗 문장(여는 질문 포함) 가운데 COMPANION_REPLY_SIMILAR_RATIO 이상 비슷한 것이 있는가. 0이면 보지 않는다."""
    ratio = get_settings().companion_reply_similar_ratio
    if ratio <= 0:
        return False
    return any(reply_similarity(text, p) >= ratio for p in past_replies or ())


def _generate(llm: LlmClient, prompt: str, is_last_turn: bool, temperature: float | None = None) -> str:
    """꼬리질문 턴은 대답 어댑터(11/13부터 cc7, 이전 cc5)로 만든다 (11/10, COMPANION_REPLY_LORA). 프롬프트는 reply-v2 그대로다.

    어댑터는 꼬리질문 프롬프트로만 학습돼서 마지막 인사 턴은 기본 모델로 만든다.
    11/23: cc8b 는 마지막 인사도 배웠다 - COMPANION_REPLY_LAST_LORA 를 켜면 마지막 인사도 어댑터로 만든다 (기본 끔).
    어댑터가 없으면(파일 없음, 어댑터 기능이 없는 LLM) 기본 모델로 만든다 (재희 동작).
    """
    s = get_settings()
    kwargs = {"temperature": s.companion_llm_temperature if temperature is None else temperature, "num_predict": s.companion_llm_num_predict,
              "timeout": s.companion_llm_timeout_sec}
    if s.companion_reply_lora and (not is_last_turn or s.companion_reply_last_lora):
        gen = getattr(llm, "generate_adapter", None)
        has = getattr(llm, "has_adapter", None)
        # 11/24 마지막 인사는 전용 어댑터(reply_last, cc8c)가 올라 있으면 그것을 쓴다
        if is_last_turn and gen is not None and has is not None and has(REPLY_LAST_ADAPTER):
            return gen(prompt, adapter=REPLY_LAST_ADAPTER, **kwargs)
        if gen is not None and (has is None or has(REPLY_ADAPTER)):
            return gen(prompt, adapter=REPLY_ADAPTER, **kwargs)
    return llm.generate(prompt, **kwargs)


def generate_reply(
    llm: LlmClient,
    question: str,
    transcript: str,
    prev_signals: dict | None,
    is_last_turn: bool,
    turn_no: int,
    text_mood: dict | None = None,
    past_replies=(),
    history=(),
) -> ReplyResult:
    """응답 문장을 만든다. (S6-3, D20)

    LLM 실패(연결 불가·시간 초과)와 규칙 위반을 COMPANION_LLM_RETRY 번 더 시도한다 (기본 0회 - 11/09 결정, 바로 대체 문장).
    그래도 안 되면 미리 정한 문장을 쓴다. 대화는 끊기지 않는다.
    text_mood 는 규칙 검사에 넘긴다 (평온·기쁨 턴 완화, 11/12). 없으면 전 규칙.
    past_replies: 이 대화에서 말벗이 이미 한 문장 (11/13). 규칙을 통과한 대답이 그 가운데 하나와 매우 비슷하면
    COMPANION_REPLY_REDO_TEMPERATURE 로 한 번 다시 만들고, 그래도 비슷하거나 규칙에 걸리면 대체 문장을 쓴다 (실패 이름 "반복").
    history: 이번 턴 앞까지의 대화 흐름 (11/15). COMPANION_REPLY_HISTORY=summary 일 때만 프롬프트에 들어간다.
    """
    s = get_settings()
    prompt = build_reply_prompt(question, transcript, prev_signals, is_last_turn, history)
    failures: list[str] = []
    redone = False
    llm_code = "-"  # LLM 실패 원인 코드 (TIMEOUT·ERROR·LOAD·ADAPTER, 11/14 점검 L-3). 메시지는 남기지 않는다
    for _ in range(1 + s.companion_llm_retry):
        try:
            raw = _generate(llm, prompt, is_last_turn)
        except LlmUnavailable as exc:
            failures = ["LLM 실패"]
            llm_code = getattr(exc, "code", "UNAVAILABLE")
            continue
        text = trim_reply(fix_split_words(re.sub(r"\s+", " ", raw).strip(), transcript), is_last_turn)
        failures = check_reply(text, is_last_turn, transcript, text_mood)
        if not failures and not is_last_turn:  # 11/25 시제·막연한 질문 고쳐 쓰기 (스위치마다 기본 끔)
            text, redone = _apply_followup_fixes(llm, prompt, text, transcript, text_mood, redone)
        vague = not failures and _vague_redo_on(is_last_turn) and is_vague_question(text, transcript)
        if not failures and (vague or is_similar_to_past(text, past_replies)):
            if redone:  # 11/14 점검 L-4: 다시 생성은 한 대답에 한 번만 (재시도 설정과 곱해지지 않게)
                failures = ["막연한 질문" if vague else "반복"]
                continue
            redone = True
            text, failures = _redo_once(llm, prompt + (VAGUE_REDO_HINT if vague else ""), is_last_turn, transcript,
                                        text_mood, past_replies)
        if not failures:
            return ReplyResult(text=text, isFollowUp=not is_last_turn, source="LLM", failures=[])
    # 11/12 점검 L-2: 대체된 이유(규칙 이름)만 남긴다 - 대답·어르신 말 본문은 남기지 않는다
    logger.info("[COMPANION-REPLY-FALLBACK] failures=%s relaxed=%s last=%s llm=%s",
                ",".join(failures), is_relaxed_mood(text_mood), is_last_turn, llm_code)
    return ReplyResult(text=_fallback(turn_no, is_last_turn), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=failures)


def _vague_redo_on(is_last_turn: bool) -> bool:
    return not is_last_turn and get_settings().companion_reply_vague_redo


def _redo_once(llm: LlmClient, prompt: str, is_last_turn: bool, transcript: str, text_mood: dict | None,
               past_replies) -> tuple[str, list[str]]:
    """지난 문장과 비슷하거나 막연한 질문인 대답을 한 번 다시 만든다 (온도만 올림, 막연하면 호출하는 쪽이 힌트 한 줄을 붙인다). 반환: (문장, 실패 이름 목록)."""
    try:
        raw = _generate(llm, prompt, is_last_turn, temperature=get_settings().companion_reply_redo_temperature)
    except LlmUnavailable:
        return "", ["반복"]
    text = trim_reply(fix_split_words(re.sub(r"\s+", " ", raw).strip(), transcript), is_last_turn)
    failures = check_reply(text, is_last_turn, transcript, text_mood)
    if not failures and _vague_redo_on(is_last_turn) and is_vague_question(text, transcript):
        failures = ["막연한 질문"]
    if not failures and is_similar_to_past(text, past_replies):
        failures = ["반복"]
    logger.info("[COMPANION-REPLY-REDO] ok=%s", not failures)  # 대답 본문은 남기지 않는다
    return text, failures


def no_response_reply(turn_no: int, is_last_turn: bool) -> ReplyResult:
    """어르신이 답하지 않은 턴의 응답. LLM을 부르지 않고 무응답 문장을 쓴다 (S7-8, 11/12 H-1)."""
    return ReplyResult(text=fallback_text(turn_no, is_last_turn, no_response=True), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=["무응답"])


def timeout_reply(turn_no: int, is_last_turn: bool) -> ReplyResult:
    """대답 생성이 턴 상한을 넘긴 턴의 응답. 어르신 말은 있으므로 받아 주는 문장을 쓴다 (11/12 H-1).

    부정 기분이면 pipeline 이 공감 문장으로 바꾼다. 실패 이름은 전과 같은 "무응답"이다(검수·로그 호환).
    """
    return ReplyResult(text=fallback_text(turn_no, is_last_turn), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=["무응답"])


def avoid_reply(turn_no: int, is_last_turn: bool) -> ReplyResult:
    """어르신이 말을 피한 턴의 응답. LLM을 부르지 않고 물러나는 문장을 쓴다 (마지막 턴은 마무리 문장, 11/12).

    위험 턴이면 pipeline 이 SAFETY_REPLY 로 덮어쓴다 (위험 판정이 우선)."""
    return ReplyResult(text=fallback_text(turn_no, is_last_turn, avoid=True), isFollowUp=not is_last_turn,
                       source="AVOID", failures=["말 피함"])
