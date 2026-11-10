"""되묻기 단계 문답(2026-11-09) - 한 턴에 짧은 질문 하나 + 누르기 쉬운 답 버튼.

짧은 증상 한마디("머리가 아파요")에 질문 여러 개를 한꺼번에 보이면 어르신이 다 읽거나 한 번에 답하기 어렵다.
증상 묶음별 고정 문답표로 최대 3질문(위험 신호 → 언제부터 → 묶음별 동반 증상)을 하나씩 묻고, 끝나면 MedGemma 를 한 번 부른다.

- 모델이 만들지 않는다 - 질문·버튼·끝맺음 문장은 이 표의 문장만 나간다(질문 턴은 MedGemma 0회, 라우터는 매 턴 그대로).
- 판정 순서는 그대로다: 응급 키워드·라우터 응급·자해·위기·예약은 chat_service 가 이 단계보다 먼저 본다.
- 위험 신호 질문(Q1)에 "네" → 그 자리에서 끝낸다. 119 묶음은 규칙 응급 답(119), 관절·기본 묶음은 오늘 진료 권유(2026-11-09 사용자 결정).
- 진행 상태는 서버가 쓴 직전 기록(upstreamMeta.followupStep)에서만 복원한다. 클라이언트 값은 보지 않는다.
- 문장 조건(tests/test_followup_steps.py 가 고정): 질문 22자 이하, 버튼 12자 이하·응급/자해/위험 신호 낱말 없음·인사·종료 문장 아님,
  MedGemma 에 넘기는 답 문장(phrase)에 응급·자해·위험 신호 낱말 없음(이어 붙인 문장이 판정을 바꾸지 않게).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services import chat_keywords

MAX_QUESTIONS = 3
INTRO = "하나씩 여쭤볼게요."
# 버튼이 아닌 짧은 말을 못 알아들었을 때 한 번 더 묻는 앞말(같은 질문을 더 쉽게).
_YESNO_RETRY = "'네'나 '아니요'로 알려 주세요."
_ONSET_RETRY = "'오늘부터', '며칠 됐어요'처럼 알려 주세요."
# 위험 신호에 "네"인데 119 묶음이 아닐 때(관절·기본) - 오늘 진료 권유로 끝낸다.
_VISIT_TODAY = "오늘 바로 {department} 진료를 받아 보세요."
_VISIT_TODAY_PROXY = "오늘 바로 {department} 진료를 받으시게 해 주세요."
_VISIT_WORSE = "갑자기 더 심해지면 바로 119에 전화하세요."
# 이 길이(공백 뺀 글자) 이상인 자유 입력은 증상 설명으로 보고 남은 질문을 건너뛴다(시험지 fq.11~20 "어제부터요, 열도 조금 있어요").
_DESCRIPTION_MIN = 6
# 이보다 긴 자유 입력은 알아들은 답이어도 설명으로 본다("일주일째인데 점점 심해져요" - 기간 + 정도).
_SHORT_ANSWER_MAX = 8

# "예전에도…"·"응급실 가야 하나요?"·"그래도 좀…"은 "네"가 아니다(2026-11-10 전체 점검 H-5 - 위험 신호 질문에서 119 로 갔다).
_YES = re.compile(r"^(네|넹|넵|예(?!전|를|컨|약|방|상|민)|응(?!급|답|원)|웅|맞아|맞어|맞습|맞대|그래(?!도|서|야)|그려|그랴|그렇|있어|있습|있대|있으|나요|나와|나네|심해|심하|그런것같|그런거같|ㅇㅇ|yes)", re.I)
# 홀로 쓴 짧은 긍정(앞 글자 규칙에 넣으면 "어제부터요"가 "네"가 된다 - 2026-11-09 점검 H-1).
_YES_ALONE = re.compile(r"(어|엉|웅|응|ㅇ|y|ok|오케이)(요|유)?", re.I)
# 위험 신호 질문(Q1)에서 "네"로 읽는 강조 말(2026-11-09 점검 H-1 - "너무 아파요"·"엄청 심해요"가 못 알아들음이었다).
# 강조 없는 "아파요" 한마디는 "아주 심하게"인지 알 수 없어 한 번 더 묻는다.
_INTENSE = re.compile(r"너무|많이|엄청|되게|디게|무척|굉장|진짜|정말|겁나|심해|심하|심했|죽겠|못참|참기힘|견디기힘")
# 정도 말 - 질문마다 정한 쪽(Question.mild)으로 읽는다(2026-11-09 점검 L-1).
_MILD = re.compile(r"^(조금|쬐금|약간|살짝|좀|그냥|그럭저럭|그저그래|그저그렇|보통|가끔|어쩌다)")
# 되묻는 말("네?"·"예?") - "네"가 아니다(2026-11-09 점검 M-1, 위험 신호 질문에서 119 로 가던 과잉).
_PARDON = re.compile(r"^\s*(네|예|응|뭐|뭐라|뭐라고|무슨|머|엥|에|진짜|정말|그래)\s*(요|유)?\s*[?？]+\s*$")
# 예/아니요 질문의 "괜찮아요"는 종료 인사가 아니라 "아니요"다(2026-11-09 점검 M-3).
_OKAY = re.compile(r"(네|예|응|아니요|아뇨|지금은|이제는|이제|저는|전|그건|그런건|그거는)?괜찮(아요|아|습니다|어요|아유|은데요|네요|다|더라고요|구요|고요)?")
# 부정이 섞이면 "네" 꼴로 시작해도 "아니요"다("심하지는 않아요", "그렇지 않아요").
# "안 답답해요"·"숨이 안 가빠요"·"안 힘들어요"(2026-11-10 전체 점검 H-5 - 질문의 증상 낱말 때문에 "네"로 읽혔다).
_NEG = re.compile(
    r"않|아니|아뇨|없|안아|안나|안해|안심|안그|안저|안메|안빠|안차|안조|안답답|안가빠|안가쁘|안숨차|안힘들|안어지|안부었|안붓|안쑤|안결|안막|안뻐근|안떨|"
    r"괜찮|ㄴㄴ|^no$", re.I)
_NO = re.compile(r"^(아니|아뇨|아녀|아닙|안그|없어|없습|없대|없으|없네|안나|안해|안심|별로|전혀|그렇지않|그런건아니|ㄴㄴ|ㄴ$|no$|노$)", re.I)
_UNKNOWN = re.compile(r"^(모르|몰라|몰러|글쎄|잘모르|기억이안|기억안|기억이잘|기억못|기억이없)")
_ONSET = re.compile(r"오늘|아까|방금|조금전|아침|점심|저녁|밤|새벽|어제|그제|그저께|며칠|사흘|이틀|나흘|닷새|일주|주일|주째|달|개월|년|오래|요즘|최근|한참|\d+(일|주|시간)|부터|째|됐")
_NOT_LETTER = re.compile(r"[^가-힣ㄱ-ㅎA-Za-z0-9]")


@dataclass(frozen=True)
class Question:
    kind: str  # "yesno" | "onset"
    text: str
    proxy: str
    chips: tuple[str, ...]
    proxy_chips: tuple[str, ...]
    yes_phrase: str = ""  # MedGemma 에 넘길 답 문장(네)
    no_phrase: str = ""  # (아니요)
    # 정도 말("조금요")을 어느 쪽으로 읽나 - 심한 정도를 묻는 질문은 "no", 증상이 있는지 묻는 질문은 "yes"(2026-11-09 점검 L-1).
    mild: str = "yes"
    # 이 질문에서 "네"로 읽는 증상 말(그 질문이 묻는 증상 그대로 - "말이 어눌해요", "38도요").
    yes_words: re.Pattern[str] | None = None

    def pick(self, proxy: bool) -> str:
        return self.proxy if proxy else self.text

    def pick_chips(self, proxy: bool) -> list[str]:
        return list(self.proxy_chips if proxy else self.chips)


@dataclass(frozen=True)
class Group:
    name: str
    pattern: re.Pattern[str]
    red_flag: Question
    red_flag_action: str  # "119" | "visit"
    extra: Question
    department: str  # 오늘 진료 권유(visit)에 쓰는 진료과
    skipped_red_flag: str  # Q1 을 건너뛰고 결과로 간 답에 넣는 위험 신호 기준 한 줄(본인용)
    skipped_red_flag_proxy: str


_YN = ("네", "아니요")


def _yn(text: str, proxy: str, yes: str, no: str, mild: str = "yes", yes_words: str | None = None) -> Question:
    return Question("yesno", text, proxy, _YN, _YN, yes, no, mild, re.compile(yes_words) if yes_words else None)


ONSET = Question(
    "onset", "언제부터 그러셨어요?", "언제부터 그러셨대요?",
    ("오늘부터요", "며칠 됐어요", "오래됐어요"), ("오늘부터래요", "며칠 되셨대요", "오래되셨대요"),
)

# 위에서부터 처음 맞는 묶음(followup_answer_chips 옛 표와 같은 순서 - 저림·어지럼을 머리보다 먼저, 관절을 기침·열보다 먼저).
# 위험 신호 낱말("숨이 좀..")이 있는 말은 맨 앞 chest 묶음 - 묶음 정규식이 아니라 chat_keywords.has_red_flag 로 고른다.
_GROUPS: tuple[Group, ...] = (
    Group(
        "chest", re.compile(r"$^"),
        _yn("숨이 차거나 가슴이 조이세요?", "숨차하시거나 가슴이 조인대요?", "", "그런 증상은 없어요",
            yes_words=r"숨|가슴|조여|조이|답답|헐떡|헉헉"), "119",
        _yn("기침이나 열도 있으세요?", "기침이나 열도 있으시대요?", "기침이나 열도 있어요", "기침이나 열은 없어요"),
        "내과", "", "",
    ),
    Group(
        "numb", re.compile(r"저리|저려|저린|저림"),
        _yn("한쪽 팔다리 힘이 갑자기 빠지세요?", "한쪽 팔다리 힘이 갑자기 빠지셨대요?", "", "힘이 빠지지는 않아요",
            yes_words=r"힘이빠|힘빠|빠져|빠지|안움직|못움직|마비|어눌"), "119",
        _yn("밤에 더 저리세요?", "밤에 더 저리시대요?", "밤에 더 저려요", "밤에 더 저리지는 않아요"),
        "신경과", "한쪽 팔다리 힘이 갑자기 빠지거나 말이 어눌해지면 바로 119에 전화하세요.",
        "한쪽 팔다리 힘이 갑자기 빠지시거나 말이 어눌해지시면 바로 119에 전화하세요.",
    ),
    Group(
        "dizzy", re.compile(r"어지"),
        _yn("말이 어눌하거나 팔다리 힘이 빠지세요?", "말씀이 어눌하거나 팔다리 힘이 빠지세요?", "", "말이나 팔다리는 괜찮아요",
            yes_words=r"어눌|말이이상|발음|힘이빠|힘빠|빠져|빠지|마비"), "119",
        _yn("속도 메스꺼우세요?", "속도 메스꺼우시대요?", "속도 메스꺼워요", "속은 괜찮아요"),
        "내과", "말이 어눌해지거나 팔다리 힘이 빠지면 바로 119에 전화하세요.",
        "말씀이 어눌해지시거나 팔다리 힘이 빠지시면 바로 119에 전화하세요.",
    ),
    Group(
        "head", re.compile(r"머리|두통"),
        _yn("갑자기 아주 심하게 아프세요?", "갑자기 아주 심하게 아파하세요?", "", "갑자기 아주 심한 건 아니에요", mild="no",
            yes_words=r"깨질|터질|벼락|망치"), "119",
        _yn("열도 나세요?", "열도 나신대요?", "열도 나요", "열은 없어요", yes_words=r"\d+(\.\d+)?도|미열|고열|열나|열이나|열있|열이있|뜨거|오한"),
        "신경과", "갑자기 아주 심하게 아프거나 말이 어눌해지면 바로 119에 전화하세요.",
        "갑자기 아주 심하게 아파하시거나 말씀이 어눌해지시면 바로 119에 전화하세요.",
    ),
    Group(
        "belly", re.compile(r"배|속|구토|토할|설사|변비|소화"),
        _yn("참기 힘들 만큼 아프세요?", "참기 힘들 만큼 아파하세요?", "", "참을 만해요", mild="no", yes_words=r"뒹굴|데굴|못참|참기힘"), "119",
        _yn("설사도 하세요?", "설사도 하신대요?", "설사도 해요", "설사는 안 해요"),
        "내과", "참기 힘들 만큼 아프거나 피를 토하면 바로 119에 전화하세요.",
        "참기 힘들 만큼 아파하시거나 피를 토하시면 바로 119에 전화하세요.",
    ),
    Group(
        "joint",
        # 손·발은 통증 말만, 목은 뻐근·결림만(옛 답 칩 표와 같은 기준, 2026-11-03 점검 L-1).
        re.compile(r"무릎|허리|어깨|관절|뼈|손목|발목|다리|팔|손가락|발가락|손이아|발이아|손이쑤|발이쑤|목이뻐근|목뻐근|목이결|목결"),
        _yn("다친 뒤로 많이 부었어요?", "다치신 뒤로 많이 부었대요?", "", "다쳐서 부은 건 아니에요", yes_words=r"넘어|다쳤|다친|다쳐|삐었|삐끗|접질|부딪"), "visit",
        _yn("걸을 때 더 아프세요?", "걸으실 때 더 아프시대요?", "걸을 때 더 아파요", "걸을 때 더 아프지는 않아요"),
        "정형외과", "", "",
    ),
    Group(
        "cough", re.compile(r"기침|가래|감기|열|목|코|콧물"),
        _yn("숨쉬기 힘드세요?", "숨쉬기 힘들어하세요?", "", "숨쉬기는 괜찮아요", yes_words=r"숨|헐떡|헉헉|답답"), "119",
        _yn("열도 나세요?", "열도 나신대요?", "열도 나요", "열은 없어요", yes_words=r"\d+(\.\d+)?도|미열|고열|열나|열이나|열있|열이있|뜨거|오한"),
        "내과", "숨쉬기 힘들어지면 바로 119에 전화하세요.", "숨쉬기 힘들어하시면 바로 119에 전화하세요.",
    ),
    Group(
        "default", re.compile(r""),
        _yn("갑자기 아주 심해졌어요?", "갑자기 아주 심해지셨대요?", "", "갑자기 심해진 건 아니에요", mild="no"), "visit",
        _yn("점점 심해지고 있어요?", "점점 심해지신대요?", "점점 심해져요", "점점 심해지지는 않아요"),
        "가까운 병원", "", "",
    ),
)
_BY_NAME = {g.name: g for g in _GROUPS}


def group_for(message: str) -> Group:
    if chat_keywords.has_red_flag(message):
        return _BY_NAME["chest"]
    compact = re.sub(r"\s+", "", message or "")
    return next(g for g in _GROUPS[1:] if g.pattern.search(compact))


def questions(group: Group) -> tuple[Question, ...]:
    return (group.red_flag, ONSET, group.extra)[:MAX_QUESTIONS]


def all_questions() -> list[Question]:
    """표의 모든 질문(테스트용)."""
    out: list[Question] = []
    for g in _GROUPS:
        for q in questions(g):
            if q not in out:
                out.append(q)
    return out


def all_groups() -> tuple[Group, ...]:
    return _GROUPS


def new_state(message: str, proxy: bool) -> dict[str, Any]:
    """첫 질문 턴의 상태(기록 upstreamMeta.followupStep). 답 본문은 넣지 않는다 - 증상 문장은 이력의 사용자 발화로 되찾는다."""
    # turns: 이 되묻기를 시작한 뒤의 사용자 턴 수(다시 묻기 포함) - 증상 문장을 이력에서 그만큼 앞에서 찾는다.
    return {"group": group_for(message).name, "index": 0, "answers": [], "retried": False, "proxy": bool(proxy), "turns": 1}


def valid_state(state: Any) -> dict[str, Any] | None:
    """기록에서 읽은 상태의 모양 확인 - 서버가 쓴 값이지만 옛 기록·손상 값이면 단계 흐름을 쓰지 않는다(None)."""
    if not isinstance(state, dict) or state.get("group") not in _BY_NAME:
        return None
    index, answers = state.get("index"), state.get("answers")
    if not isinstance(index, int) or not 0 <= index < MAX_QUESTIONS or not isinstance(answers, list) or len(answers) != index:
        return None
    if not all(isinstance(a, str) and len(a) <= 40 for a in answers):
        return None
    turns = state.get("turns")
    # 질문 3개 + 다시 묻기는 질문마다 한 번 - 6턴을 넘는 값은 손상으로 본다.
    if not isinstance(turns, int) or not index + 1 <= turns <= MAX_QUESTIONS * 2:
        return None
    return {"group": state["group"], "index": index, "answers": list(answers),
            "retried": bool(state.get("retried")), "proxy": bool(state.get("proxy")), "turns": turns}


def current_question(state: dict[str, Any]) -> Question:
    return questions(_BY_NAME[state["group"]])[state["index"]]


def question_lines(state: dict[str, Any], *, first: bool, red_flag_line: str | None = None) -> list[str]:
    """질문 턴 reply 줄. 질문은 "· " 줄(chat_layout 이 questions 칸으로 옮긴다)."""
    q = current_question(state)
    lines: list[str] = []
    if first:
        lines.append(INTRO)
        if red_flag_line:
            lines.append(red_flag_line)  # 위험 신호 낱말이 있는 말 - 119 기준은 지금처럼 첫 턴에 바로(2026-11-02 그대로)
    if state.get("retried"):
        lines.append(_YESNO_RETRY if q.kind == "yesno" else _ONSET_RETRY)
    lines.append(f"· {q.pick(state['proxy'])}")
    return lines


def chips(state: dict[str, Any]) -> list[str]:
    return current_question(state).pick_chips(state["proxy"])


def _compact(text: str) -> str:
    return _NOT_LETTER.sub("", text or "")


def interpret(state: dict[str, Any], message: str) -> tuple[str, str | None]:
    """이번 답을 지금 질문의 답으로 해석한다. (판정, 결과 문장 조각 - 지금 질문에 답했으면 그 답 문장)
    판정: yes / no / answer(기간 답) / unknown(모름 - 넘어간다) / describe(설명 - 남은 질문 건너뜀) / unclear(다시 묻기).
    순서(2026-11-09 점검 반영): 되묻는 말 → "괜찮아요" → 부정 → 긍정·강조·그 질문의 증상 말 → 정도 말 → 위험 신호 낱말 →
    다른 질문의 답·새 증상 → 모름 → 길이."""
    q = current_question(state)
    c = _compact(message)
    if not c:
        return "unclear", None
    red_flag_question = state["index"] == 0
    if q.kind == "yesno":
        if _PARDON.match(message or ""):
            return "unclear", None
        if _OKAY.fullmatch(c):
            return "no", q.no_phrase
        negated = bool(_NO.match(c) or _NEG.search(c))
        about_this = bool(q.yes_words and q.yes_words.search(c))  # 이 질문이 묻는 증상을 부정("가슴은 괜찮아요")
        if negated and not about_this and not _NO.match(c) and (_ONSET.search(c) or chat_keywords.FOLLOWUP_SYMPTOM_NOUN.search(c)):
            # 다른 질문에 대한 부정("열은 없고 괜찮아요")은 이 질문의 답이 아니다 - 설명으로 본다.
            return "describe", None
        if negated:
            return ("no" if len(c) <= _SHORT_ANSWER_MAX else "describe"), q.no_phrase
        yes = bool(
            _YES.match(c) or _YES_ALONE.fullmatch(c) or (q.yes_words and q.yes_words.search(c))
            # 강조 말은 기간·"점점"이 없을 때만 - "일주일째인데 점점 심해져요"는 갑자기 심한 게 아니라 설명이다(시험지 fq.13)
            # 심한 정도를 묻는 위험 신호 질문(mild="no" - 머리·배·기본 묶음)에만 쓴다. 관절("다친 뒤로 부었어요?")처럼 있는지를 묻는 질문에는 아니다(fq.19).
            or (red_flag_question and q.mild == "no" and _INTENSE.search(c) and not _ONSET.search(c) and "점점" not in c)
            or (red_flag_question and red_flag_action(state) == "119" and chat_keywords.has_red_flag(c))
        )
        if yes:
            # 위험 신호(Q1) "네"는 길이와 상관없이 "네"다("네 갑자기 너무 아파요") - 119·오늘 진료 쪽으로 기운다.
            if red_flag_question or len(c) <= _SHORT_ANSWER_MAX:
                return "yes", q.yes_phrase or None
            return "describe", q.yes_phrase or None
        if _MILD.match(c) and len(c) <= _SHORT_ANSWER_MAX:
            return ("yes", q.yes_phrase or None) if q.mild == "yes" else ("no", q.no_phrase)
        # 위험 신호 낱말이 든 답은 설명으로 - 결과 턴 생성 후처리가 위험 신호 119 줄을 붙인다.
        if chat_keywords.has_red_flag(c):
            return "describe", None
        # 다른 질문의 답("어제부터요", "열은 없고 괜찮아요")·새 증상 말("배가 아파요")은 설명으로 본다 - 같은 질문을 되풀이하지 않는다.
        if _ONSET.search(c) or chat_keywords.FOLLOWUP_SYMPTOM_NOUN.search(c):
            return "describe", None
    elif _ONSET.search(c):
        return ("answer" if len(c) <= _SHORT_ANSWER_MAX else "describe"), message.strip()
    if _UNKNOWN.match(c) and len(c) <= _SHORT_ANSWER_MAX:
        return "unknown", None
    if len(c) >= _DESCRIPTION_MIN:
        return "describe", None
    return "unclear", None


def is_okay_answer(state: dict[str, Any], message: str) -> bool:
    """예/아니요 질문에 "괜찮아요"(앞에 "네"·"지금은" 정도만 붙은 말)로 답했는가 - 종료 인사로 보지 않고 "아니요"로 읽는다."""
    return current_question(state).kind == "yesno" and bool(_OKAY.fullmatch(_compact(message)))


def red_flag_action(state: dict[str, Any]) -> str:
    return _BY_NAME[state["group"]].red_flag_action


def visit_today_lines(state: dict[str, Any], symptom: str = "") -> list[str]:
    """위험 신호 "네"(관절·기본 묶음) 끝맺음. 진료과는 증상 문장의 진료과(눈 → 안과 등), 없으면 묶음 기본값(2026-11-09 점검 L-2)."""
    g = _BY_NAME[state["group"]]
    department = chat_keywords.symptom_department(symptom) if symptom else None
    first = (_VISIT_TODAY_PROXY if state["proxy"] else _VISIT_TODAY).format(department=department or g.department)
    return [first, _VISIT_WORSE]


def skipped_red_flag_line(state: dict[str, Any]) -> str | None:
    """Q1(위험 신호)에 답하지 않고 결과로 간 답에 넣는 그 묶음의 119 기준 한 줄. chest 묶음은 생성 후처리가 이미 붙인다."""
    g = _BY_NAME[state["group"]]
    line = g.skipped_red_flag_proxy if state["proxy"] else g.skipped_red_flag
    return line or None


def generation_text(symptom: str, answers: list[str], message: str | None = None) -> str:
    """MedGemma 에 넘길 "증상 / 답 / 답" 문장(옛 되묻기 다음 턴 "앞 질문 / 이번 답"과 같은 꼴)."""
    parts = [symptom.strip(), *[a for a in answers if a]]
    if message and message.strip():
        parts.append(message.strip())
    return " / ".join(p for p in parts if p)
