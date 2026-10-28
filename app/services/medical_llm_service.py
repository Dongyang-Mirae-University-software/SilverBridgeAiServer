from __future__ import annotations

from functools import lru_cache
import logging
import re
from typing import Any, NamedTuple

from app.core.config import Settings, get_settings
from app.models.medgemma_loader import MedGemmaLoader
from app.prompts.medical_chat_prompts import SYSTEM_PROMPT, build_user_prompt
from app.services.reservation_orchestrator import _symptom_department
from app.utils.conversation import format_recent_history, normalize_history
from app.utils.json_extract import extract_json_fields_lenient, extract_json_object


_LOG = logging.getLogger(__name__)

_STRING_KEYS = ["empathy", "symptoms", "features", "advice", "reply"]
_LIST_KEYS: list[str] = []
_EMERGENCY_REPLY = "지금은 응급 상황일 수 있어요. 지금 바로 119에 전화하세요. 혼자 계시면 문을 열어 두고, 가족이나 이웃에게도 알려 주세요."
# 자해·자살 신호에는 증상 정리 틀 대신 이 고정 응답을 낸다(근거: 보건복지부 자살예방 상담전화 109, 24시간).
_SELF_HARM_REPLY = (
    "많이 힘드시군요. 지금 위험하다고 느끼시면 바로 119에 전화해 주세요. "
    "혼자 견디기 어려우면 자살예방 상담전화 109(24시간)로 이야기해 보세요. "
    "곁에 있는 가족이나 가까운 분에게도 꼭 알려 주세요."
)
# 가족(제3자)의 위기를 전하는 보호자에게는 곁에 있어 달라는 안내로 바꾼다. 보호자 본인의 위기일 수도 있어 역할값이 아니라 문장의 주어로 가른다.
_SELF_HARM_REPLY_FAMILY = (
    "많이 놀라셨겠어요. 지금 위험해 보이시면 바로 119에 전화해 주세요. "
    "자살예방 상담전화 109는 가족도 24시간 상담할 수 있어요. "
    "혼자 계시지 않게 곁에 있어 주세요."
)
_SELF_HARM_WARNING = "지금 위험하다고 느끼시면 바로 119에 전화하세요. 자살예방 상담전화 109는 24시간 받습니다."
_THIRD_PERSON_SUBJECT = re.compile(r"(엄마|어머니|어머님|아빠|아버지|아버님|할머니|할아버지|부모님|남편|아내|와이프)\s*(가|이|께서|는|은)")
# 약 복용 문의 + 증상인데 답에 병원 권유가 없을 때 덧붙인다. 판정 기준은 평가 실행기 must_hospital 과 같은 단어다
# ("의사·약사에게 물어보세요"만으로는 진료 권유가 아니다).
_CLINIC_SENTENCE = "증상이 계속되거나 걱정되시면 병원 진료를 받아 보세요."
# 당뇨약 + 술 문의에 붙이는 저혈당 문장(2026-10-27 점검 M-2 - qa.D5 에서 저혈당 미언급 4/4·방향 반대 1/4). 약 종류별 규칙은 _DRUG_ALCOHOL_RULES.
_ALCOHOL_HYPOGLYCEMIA_SENTENCE = (
    "당뇨약을 드시는 중에 술을 마시면 저혈당(식은땀·떨림·어지럼)이 늦게 올 수 있어요. 드시기 전에 의사·약사와 꼭 상의하세요."
)
_DIABETES_MED_WORDS = re.compile(r"당뇨|혈당약|혈당 ?약|인슐린")
# "수술"·"기술"·"예술"·"마술"·"미술"의 "술"은 술이 아니다.
_ALCOHOL_WORDS = re.compile(r"(?<![수기예마미])술|음주|소주|맥주|막걸리|와인|양주|알코올|알콜")
# 술이 혈당을 올린다고 쓴 문장(방향이 반대, 2026-10-27 재측정 4/5). "저혈당"이 든 문장은 건드리지 않는다.
_GLUCOSE_UP = re.compile(r"혈당[^.!?。]{0,12}(높|오르|올라|올리|올려|상승)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+")


class _DrugAlcoholRule(NamedTuple):
    """약 + 술 문의 안전 규칙(2026-10-27, 약+술 탐색 측정에서 사실 반대·허용 표현·핵심 위험 누락).
    drug = 질문의 약 이름 / sentence = 모델 출력과 상관없이 붙일 고정 문장 / covered = 답에 이미 있으면 문장을 붙이지 않는 핵심어 /
    wrong = 이 약에서 방향이 반대인 문장(covered 가 든 문장은 남긴다)."""

    name: str
    drug: re.Pattern[str]
    sentence: str
    covered: re.Pattern[str]
    wrong: re.Pattern[str] | None


_DRUG_ALCOHOL_RULES = (
    _DrugAlcoholRule("diabetes", _DIABETES_MED_WORDS, _ALCOHOL_HYPOGLYCEMIA_SENTENCE, re.compile("저혈당"), _GLUCOSE_UP),
    _DrugAlcoholRule(
        "sedative",
        re.compile(r"수면제|수면 ?유도제|수면 ?보조제|신경 ?안정제|안정제|진정제|항불안제|졸피뎀|스틸녹스|자낙스|알프라졸람|로라제팜"),
        "수면제나 안정제를 술과 함께 드시면 숨이 약해지거나 깨어나기 어려울 만큼 위험할 수 있어요. 술과 함께 드시지 마세요.",
        re.compile(r"(함께|같이) (드시지|복용하지|먹지) 마"),
        None,
    ),
    _DrugAlcoholRule(
        "acetaminophen",
        re.compile(r"타이레놀|아세트아미노펜|펜잘|게보린|써스펜"),
        "타이레놀(아세트아미노펜)을 드시는 중에 술을 마시면 간이 상할 수 있어요. 함께 드시지 마시고, 술을 자주 드시면 의사·약사와 상의하세요.",
        re.compile(r"간(이|을|에| 손상| 기능| 독성)"),
        None,
    ),
    _DrugAlcoholRule(
        "anticoagulant",
        re.compile(r"와파린|쿠마딘|항응고제|피 ?묽게|피를 ?묽게|혈액 ?희석"),
        "와파린 같은 피를 묽게 하는 약을 드시는 중에 술을 마시면 출혈 위험이 커지고 약 효과가 달라질 수 있어요. 드시기 전에 의사·약사와 꼭 상의하세요.",
        re.compile(r"출혈|피가 (잘 )?안 멈|피가 멈추지"),
        # 술이 응고(혈전) 위험을 높인다고 쓴 문장 - 실제로는 출혈 위험(2026-10-27 측정 2/3).
        re.compile(r"(응고|혈전)[^.!?。]{0,12}(높|커|증가|올라|올리|상승)"),
    ),
    _DrugAlcoholRule(
        "blood_pressure",
        re.compile(r"고혈압약|혈압약|혈압 ?약"),
        # 술은 바로는 혈압약 효과를 키워 혈압을 떨어뜨리고, 자주 마시면 혈압을 올린다 - 모델이 한쪽만 써도 어긋나지 않게 두 방향을 함께 쓴다
        # (2026-10-27 측정: 모델 "혈압이 더 높아질" + 옛 고정 문장 "혈압이 너무 떨어져"가 반대로 읽힘 1/3).
        "혈압약을 드시는 중에 술을 마시면 혈압이 갑자기 떨어져 어지럽거나 넘어질 수 있고, 자주 마시면 오히려 혈압이 올라 약 효과가 떨어질 수 있어요. 드시기 전에 의사·약사와 상의하세요.",
        re.compile(r"어지|낙상|넘어|쓰러"),
        None,
    ),
    _DrugAlcoholRule(
        "antibiotic",
        re.compile(r"항생제|아목시실린|오구멘틴|메트로니다졸|플라질|클래리스로마이신|레보플록사신|시프로플록사신|독시사이클린"),
        "항생제를 드시는 동안에는 술을 피하세요. 일부 항생제는 술과 함께 드시면 심한 두근거림·구토가 생길 수 있어요. 드시는 약이 괜찮은지 약사에게 확인해 주세요.",
        re.compile(r"(술|음주)[^.!?。]{0,25}(피하|삼가|드시지 마|마시지 마|좋지 않|금하)|금주"),
        None,
    ),
)
# 약 + 술 문의에서 술을 조금은 괜찮다고 허용하는 문장(2026-10-27 측정 "한 잔 정도는 괜찮을 수"). 부정형("괜찮지 않아요")은 남긴다.
_ALCOHOL_ALLOWANCE = re.compile(r"(한두 ?잔|한 ?잔|두 ?잔|조금|적당히|소량|가볍게)[^.!?。]{0,15}(괜찮|마셔도 (돼|됩|괜)|드셔도 (돼|됩|괜)|문제(가)? ?없|무방)")
# 약을 스스로 줄이거나 끊으라고 읽히는 문장(2026-10-27 측정: "필요하면 당뇨약을 조절" / 금식 때 "당뇨약을 복용하지 않거나").
# 같은 문장에 의사·약사 상의가 없으면 고정 문장을 덧붙인다 - 모델 문장은 지우지 않는다(맞는 내용일 수 있어 덧붙이기만).
_SELF_ADJUST = re.compile(r"(약|복용량|용량)[^.!?。]{0,20}(조절|줄이|줄여|늘리|늘려|끊|중단|건너뛰|복용하지 않|드시지 않|먹지 않)")
_SELF_ADJUST_CONSULTED = re.compile(r"의사|약사|병원|진료|처방|상의|상담")
_SELF_ADJUST_SENTENCE = "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요."
_ALLOWANCE_NEGATED = re.compile(r"괜찮지 않|괜찮은 (것|게) 아니|안 괜찮|되지 않|안 ?돼")
_HOSPITAL_WORDS = re.compile(r"병원|진료|응급실|119")
_MEDICATION_WORD = re.compile(r"약|항생제|진통제|수면제|소화제|해열제|아스피린")
_MEDICATION_ASK = re.compile(r"먹어도|드셔도|복용해도|끊어도|중단해도|먹으면|드시면|먹을까")
_SYMPTOM_WORDS = (
    "열이", "열나", "발열", "아파", "아프", "통증", "기침", "어지", "두통", "구토", "설사", "저리", "저려", "가려",
    "붓", "부었", "숨이", "가슴", "메스꺼", "콧물",
)


class MedicalLlmService:
    # 응급 신호 목록의 단일 출처(chat_service도 이 목록을 쓴다). 공백을 뺀 문장에 부분 일치시킨다("숨을 못 쉬겠어요"→"숨을못쉬").
    # 경계 케이스("숨이 차요", "가슴이 답답해요", "코피가 안 멈춰요")는 일부러 뺐다 - docs/chat-eval-set.md 참고.
    # "일어나지못"·"말이꼬"는 "아침에 일어나지 못한 날"·"말이 꼬여서 다시 말할게요"에 걸려 "일어나지못하겠"·"말이자꾸꼬"로 좁혔다.
    EMERGENCY_KEYWORDS = (
        # 호흡
        "숨이안쉬", "숨을못쉬", "숨쉬기힘", "숨쉬기가힘", "숨이막", "호흡곤란", "호흡이곤란", "질식", "목에걸려", "입술이파래", "입술이파랗",
        # 심장·가슴
        "가슴통증", "가슴이아", "가슴이조이", "가슴이조여", "가슴을쥐어", "가슴이찢어", "심장이아",
        "가슴이쪼이", "가슴이쪼여", "가심이쪼이", "가심이쪼여", "가심이조이", "가심이조여", "가슴이꽉조이", "가슴이꽉조여", "가슴이너무조이", "가슴이너무조여",
        # 의식·실신·낙상
        "의식없", "의식이없", "의식을잃", "정신을잃", "기절", "실신", "쓰러짐", "쓰러졌", "쓰러져", "쓰러지", "깨어나지", "불러도반응", "못일어나",
        "의식이흐려", "의식이흐릿", "일어나지못하겠", "일어나지를못", "일어날수가없", "일어설수가없",
        # 출혈
        "피가멈추지", "피가안멈", "피를토", "토혈", "피가많이", "출혈이심", "피가계속", "토한것이커피색", "토한게커피색", "커피색으로토",
        # 경련
        "경련", "발작", "거품을물",
        # 뇌졸중 의심
        "말이어눌", "말이안나", "발음이이상", "발음이꼬", "발음이자꾸꼬", "혀가꼬", "말이자꾸꼬", "한쪽팔", "한쪽다리", "한쪽이마비", "한쪽얼굴", "얼굴이한쪽", "입이돌아", "팔다리에힘이", "갑자기앞이안보", "벼락", "갑자기심한두통", "망치로맞은",
        # 중독·자해
        "약을많이먹", "약을한꺼번에", "과다복용", "농약", "죽고싶", "자살", "극단적인생각", "극단적인선택", "극단적선택",
        # 기타
        "응급", "심한화상", "화상이심", "온몸에두드러기",
    )
    # 자해·자살 신호(공백을 뺀 문장에 부분 일치). 일상 표현에 걸리지 않게 좁게 둔다 -
    # "그만 살 빼야지"·"밥 다 먹어버릴까"·"무릎 아파 죽겠어"·"극단적인 다이어트"·"약을 다 먹었어요"(약이 떨어짐)는 걸리면 안 된다.
    # "죽어버리"·"사는게싫"은 과장 표현에 걸릴 수 있다(수용 - 위기 응답 쪽 오탐).
    SELF_HARM_KEYWORDS = (
        "죽고싶", "자살", "자해", "극단적인생각", "극단적인선택", "극단적선택", "살기싫", "살기가싫", "사는게싫",
        "살고싶지않", "살고싶지가않", "그만살고", "그만살래", "사라지고싶", "죽어버리", "목숨을끊", "목숨끊", "뛰어내리고싶",
        "약을다먹어버릴", "약다먹어버릴", "약을다먹을까", "수면제를다먹어버릴", "수면제를다먹을까", "수면제를한꺼번에", "농약마시",
    )
    EMERGENCY_WARNING = _EMERGENCY_REPLY
    RESERVATION_KEYWORDS = ("예약", "병원 찾", "근처 병원", "진료 가능", "응급실", "잡아", "접수", "병원 좀", "병원 추천", "진료 받")
    MEDICAL_KEYWORDS = (
        "아파", "아프", "통증", "열", "기침", "두통", "어지럽", "어지러", "어지럼", "구토", "충혈", "메스꺼", "설사", "변비",
        "더부룩", "속이", "소화", "체했", "무릎", "허리", "관절", "시려", "저리", "붓", "부었", "가려", "피부", "발진",
        "당뇨", "혈압", "혈당", "콜레스테롤", "불면", "잠이", "수면", "약을", "약 먹", "복용", "숨이", "가슴", "심장",
        "기운", "피곤", "식욕", "체중", "눈이", "귀가", "목이", "코가",
    )

    def __init__(self, settings: Settings, loader: MedGemmaLoader) -> None:
        self._settings = settings
        self._loader = loader

    @classmethod
    def emergency_signals(cls, message: str) -> list[str]:
        """문장에 든 응급 신호 키워드. 비어 있지 않으면 응급이다. 판단 근거로 meta·로그에 남긴다."""
        compact = re.sub(r"\s+", "", (message or "").lower())
        return [keyword for keyword in cls.EMERGENCY_KEYWORDS if keyword in compact]

    @classmethod
    def self_harm_signals(cls, message: str) -> list[str]:
        """자해·자살 신호 키워드. 비어 있지 않으면 증상 상담 대신 고정 응답을 낸다."""
        compact = re.sub(r"\s+", "", (message or "").lower())
        return [keyword for keyword in cls.SELF_HARM_KEYWORDS if keyword in compact]

    @staticmethod
    def about_family(message: str) -> bool:
        """문장의 주어가 가족(제3자)인가 - "엄마가 죽고 싶다고 하세요"."""
        return bool(_THIRD_PERSON_SUBJECT.search(message or ""))

    @classmethod
    def self_harm_in_history(cls, history: list[dict[str, str]]) -> bool:
        """직전 사용자 턴 2개 안에 자해 신호가 있었는가(키워드). 이번 턴이 응급일 때만 쓴다."""
        users = [item.get("content", "") for item in history or [] if item.get("role") == "user"][-2:]
        return any(cls.self_harm_signals(text) for text in users)

    @staticmethod
    def self_harm_response(family: bool = False, trigger: str = "keyword") -> dict[str, Any]:
        """자해·자살 신호 고정 응답. 증상 칸 없이 공감·119·109·주변 알림만. 응답 키는 다른 경로와 같다.
        예약 안내가 붙을 상황이 아니므로 reservationRequired 는 false 다(일반 응급 경로 값은 그대로)."""
        return {
            "reply": _SELF_HARM_REPLY_FAMILY if family else _SELF_HARM_REPLY,
            "riskLevel": "high",
            "recommendedAction": "hospital_visit",
            "reservationRequired": False,
            "intent": "emergency_guidance",
            "engine": "rule",
            "modelName": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [_SELF_HARM_WARNING],
            "meta": {"selfHarm": True, "selfHarmTrigger": trigger, "selfHarmFamily": family},
        }

    @staticmethod
    def drug_alcohol_rules(message: str) -> list[_DrugAlcoholRule]:
        """약 + 술을 함께 묻는 문장이면 해당 약 규칙들(여러 개일 수 있다). 술이 없으면 빈 목록."""
        text = message or ""
        if not _ALCOHOL_WORDS.search(text):
            return []
        return [rule for rule in _DRUG_ALCOHOL_RULES if rule.drug.search(text)]

    @staticmethod
    def apply_drug_alcohol_rules(message: str, reply: str) -> tuple[str, dict[str, int], list[str]]:
        """약 + 술 문의 답 손보기: ① 술을 조금은 괜찮다고 허용한 문장 ② 그 약에서 방향이 반대인 문장을 빼고
        ③ 답에 핵심 위험이 없으면 고정 문장을 붙인다. (고친 답, 뺀 문장 수 {이유: 수}, 붙인 규칙 이름). 다른 상담은 그대로."""
        rules = MedicalLlmService.drug_alcohol_rules(message)
        if not rules:
            return reply, {}, []

        def drop_reason(sentence: str) -> str | None:
            if (
                (_ALCOHOL_WORDS.search(sentence) or "잔" in sentence)
                and _ALCOHOL_ALLOWANCE.search(sentence)
                and not _ALLOWANCE_NEGATED.search(sentence)
            ):
                return "allowance"
            for rule in rules:
                if rule.wrong and _ALCOHOL_WORDS.search(sentence) and rule.wrong.search(sentence) and not rule.covered.search(sentence):
                    return rule.name
            return None

        dropped: dict[str, int] = {}
        lines: list[str] = []
        for line in (reply or "").split("\n"):
            sentences = _SENTENCE_SPLIT.split(line)
            kept = []
            for sentence in sentences:
                reason = drop_reason(sentence)
                if reason is None:
                    kept.append(sentence)
                else:
                    dropped[reason] = dropped.get(reason, 0) + 1
            if len(kept) == len(sentences):
                lines.append(line)  # 뺄 문장이 없는 줄은 그대로
                continue
            new_line = " ".join(kept).strip()
            if new_line:
                lines.append(new_line)
        out = "\n".join(lines)
        # 핵심 위험이 이미 있는지는 모델 답만 보고 정한다 - 앞서 붙인 고정 문장의 단어("어지럼")로 다른 약 문장이 빠지지 않게.
        model_text = out
        added = []
        for rule in rules:
            if not rule.covered.search(model_text):
                out = f"{out}\n{rule.sentence}"
                added.append(rule.name)
        return out, dropped, added

    @staticmethod
    def needs_self_adjust_sentence(reply: str) -> bool:
        """약을 스스로 조절·중단하라고 읽히는 문장이 있는데 그 문장에 의사·약사 상의가 없고, 고정 문장도 아직 없으면 True."""
        text = reply or ""
        if _SELF_ADJUST_SENTENCE in text:
            return False
        for line in text.split("\n"):
            for sentence in _SENTENCE_SPLIT.split(line):
                if _SELF_ADJUST.search(sentence) and not _SELF_ADJUST_CONSULTED.search(sentence):
                    return True
        return False

    @staticmethod
    def needs_clinic_sentence(message: str, reply: str) -> bool:
        """약 복용 문의에 증상이 함께 있는데 답에 병원·진료 권유가 없으면 True."""
        text = message or ""
        compact = re.sub(r"\s+", "", text)
        if not (_MEDICATION_WORD.search(compact) and _MEDICATION_ASK.search(compact)):
            return False
        if not any(word in compact for word in _SYMPTOM_WORDS):
            return False
        return not _HOSPITAL_WORDS.search(reply or "")

    @classmethod
    def classify_intent(cls, message: str) -> str:
        text = (message or "").strip().lower()
        if cls.emergency_signals(text):
            return "emergency_guidance"
        if any(keyword in text for keyword in cls.RESERVATION_KEYWORDS):
            return "hospital_reservation"
        if any(keyword in text for keyword in cls.MEDICAL_KEYWORDS):
            return "medical_advice"
        return "general_chat"

    HIGH_RISK_KEYWORDS = ("가슴", "숨이", "호흡", "의식", "실신", "쓰러", "출혈", "피가", "마비", "경련", "파래", "식은땀")

    @staticmethod
    def infer_risk(message: str, intent: str, emergency_warning: list[str]) -> tuple[str, str, bool]:
        text = (message or "").strip().lower()
        # 모델은 거의 항상 일반적인 응급 주의 문구를 붙이므로, 사용자 메시지에 응급 신호가 있을 때만 high로 본다.
        has_high_signal = any(keyword in text for keyword in MedicalLlmService.HIGH_RISK_KEYWORDS)
        if intent == "emergency_guidance" or (emergency_warning and has_high_signal):
            return "high", "hospital_visit", True
        if intent in {"hospital_reservation", "medical_advice"}:
            return "medium", "guardian_contact", intent == "hospital_reservation"
        if any(keyword in text for keyword in MedicalLlmService.MEDICAL_KEYWORDS):
            return "medium", "guardian_contact", False
        return "low", "home_monitoring", False

    @staticmethod
    def format_answer(data: dict[str, Any], emergency: bool = False) -> str:
        """말풍선 문장. 모델이 쓴 공감·증상·특징·권고를 줄바꿈으로 잇는다. 비어 있는 값만 기본 문장으로 메운다."""
        symptoms = str(data.get("symptoms") or "").strip()
        if not symptoms:
            return str(data.get("reply") or "").strip() or "증상을 조금 더 구체적으로 말씀해 주시면 도와드릴게요."

        lines = [str(data.get("empathy") or "").strip() or "말씀해 주신 내용을 정리했어요.", f"· 증상: {symptoms}"]
        features = str(data.get("features") or "").strip()
        if features and not re.search(r"없음|없습니다|불명확|모름|알 수 없|언급되지|말씀하지|제공되지", features):  # "유발 요인 불명확" 같은 빈말은 줄을 빼는 게 낫다
            lines.append(f"· 특징: {features}")
        advice = str(data.get("advice") or "").strip()
        if emergency:
            # 응급이면 모델이 진료과를 권해도 쓰지 않는다. 경고 문구는 말풍선 위 빨간 박스(emergencyWarning)가 맡는다.
            advice = "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요."
        lines.append(advice or "증상이 계속되면 가까운 병원 진료를 권해드려요. 갑자기 심해지면 바로 119에 연락하세요.")
        return "\n".join(lines)

    PLACEHOLDERS = {"...", "…", "문자열", "-", "n/a", "없음", "정보 없음", "해당 없음", "알 수 없음", "해당사항 없음"}

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            return "" if text.lower() in MedicalLlmService.PLACEHOLDERS else text
        if isinstance(value, (int, float)):
            return str(value).strip()
        if isinstance(value, list):
            for item in value:
                text = MedicalLlmService._normalize_text(item)
                if text:
                    return text
        return ""

    @staticmethod
    def _normalize_list(value: Any) -> list[str]:
        if isinstance(value, str):
            raw_items = re.split(r"[\n•·,;/]+", value)
        elif isinstance(value, list):
            raw_items = value
        else:
            return []

        items: list[str] = []
        for item in raw_items:
            text = MedicalLlmService._normalize_text(item)
            if text and text.lower() not in {"none", "null"}:
                items.append(text)
        return items[:3]

    @staticmethod
    def _normalize_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on", "예", "네"}
        return False

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        chunks = re.split(r"(?<=[.!?。])\s+|\n+", text or "")
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    @classmethod
    def _sanitize_unstructured_text(cls, raw_text: str) -> str:
        lines: list[str] = []
        for line in (raw_text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("<unused") or s.startswith("```"):
                break
            if s.startswith("Constraint Checklist") or s.startswith("Mental Sandbox"):
                break
            if s.startswith("MedGemma") or s.startswith("위험도"):
                break
            if s.startswith("The user") or s.startswith("I need to"):
                break
            lines.append(s)
        return " ".join(lines).strip()

    @classmethod
    def _plain_sentence(cls, raw_text: str) -> str:
        """모델 원문의 첫 문장. 잘린 JSON 조각이면 빈 값(사용자에게 JSON을 보여주지 않는다)."""
        text = cls._sanitize_unstructured_text(raw_text)
        if not text or text.lstrip().startswith("{") or '"' in text[:20]:
            return ""
        sentences = cls._split_sentences(text)
        return sentences[0] if sentences else ""

    @classmethod
    def _fallback_payload(cls, message: str, intent: str, emergency_hint: bool, raw_text: str) -> dict[str, Any]:
        """JSON을 못 받았을 때. 증상 상담이면 사용자 문장을 증상으로 두고, 일반 대화면 모델 문장을 그대로 쓴다."""
        empty = {key: "" for key in _STRING_KEYS}
        if emergency_hint or intent == "emergency_guidance":
            return {**empty, "symptoms": (message or "").strip()[:60], "advice": _EMERGENCY_REPLY}
        if intent == "medical_advice":
            return {**empty, "symptoms": (message or "").strip()[:60]}
        return {**empty, "reply": cls._plain_sentence(raw_text)}

    @classmethod
    def _normalize_payload(
        cls,
        parsed: dict[str, Any] | None,
        raw_text: str,
        message: str,
        intent: str,
        emergency_hint: bool,
    ) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return cls._fallback_payload(message, intent, emergency_hint, raw_text)
        out = {key: cls._normalize_text(parsed.get(key)) for key in _STRING_KEYS}
        if intent in {"medical_advice", "emergency_guidance"}:
            if not out["symptoms"]:
                # 모델이 증상을 비우면 사용자 문장으로 메운다 - 상담인데 일반 답변 틀로 빠지지 않게.
                out["symptoms"] = (message or "").strip()[:60]
            department = _symptom_department(message or "")
            if not out["advice"] and department:
                out["advice"] = f"{department} 진료를 권해드려요. 갑자기 심해지면 바로 119에 연락하세요."
        else:
            # 인사·일반 질문인데 모델이 증상 칸을 채우면("고마워요") 상담 틀로 빠지므로 비운다.
            out["symptoms"] = ""
            out["reply"] = out["reply"] or cls._plain_sentence(raw_text) or "네, 말씀해 주세요. 궁금한 점이 있으면 언제든 물어보세요."
        return out

    def generate(
        self,
        message: str,
        user_context: dict[str, Any] | None,
        history: list[dict[str, str]],
        *,
        intent: str | None = None,
        processed_symptom: str | None = None,
    ) -> dict[str, Any]:
        intent = intent or self.classify_intent(message)
        signals = self.emergency_signals(message)
        emergency_hint = intent == "emergency_guidance" or bool(signals)

        recent, summary = normalize_history(
            history,
            keep_turns=self._settings.chat_history_keep_turns,
        )
        history_lines = format_recent_history(recent)
        if summary:
            history_lines = f"(이전 요약) {summary}\n{history_lines}".strip()

        user_prompt = build_user_prompt(
            message=message,
            user_context=user_context,
            history_text=history_lines or "(없음)",
            intent=intent,
            emergency_hint=emergency_hint,
            processed_symptom=processed_symptom,
            # 예약 흐름과 같은 증상→진료과 규칙을 힌트로 준다. 상담에서 권한 과와 예약에서 찾는 과가 어긋나지 않게.
            department_hint=_symptom_department(message or ""),
        )

        if not self._settings.chat_enable_llm:
            raise RuntimeError("CHAT_ENABLE_LLM=false")

        self._loader.ensure_loaded()
        # 응급 턴은 생성 슬롯 줄을 오래 서지 않는다 - 넘기면 MedGemmaBusyError -> chat_service가 119 안내 고정 문구로 답한다.
        wait_timeout_sec = (
            self._settings.medgemma_gen_wait_timeout_emergency_sec
            if emergency_hint
            else self._settings.medgemma_gen_wait_timeout_sec
        )
        raw = self._loader.generate_structured_text(
            SYSTEM_PROMPT, user_prompt, prefill="{", wait_timeout_sec=wait_timeout_sec
        )
        parsed = extract_json_object(raw)
        if parsed is None:
            # 출력이 max_new_tokens에 걸려 잘린 경우: 완성된 필드만이라도 살린다.
            parsed = extract_json_fields_lenient(raw, _STRING_KEYS, _LIST_KEYS)
            _LOG.warning(
                "medgemma json parse fallback (lenient=%s, chars=%d)",  # 모델 출력(상담 내용)은 남기지 않는다
                bool(parsed), len(raw),
            )
        normalized = self._normalize_payload(parsed, raw, message, intent, emergency_hint)

        # 응급이면 상단에 경고 박스 한 줄만. 그 외 목록(원인·자가관리·방문기준)은 더 이상 내지 않는다 - 말풍선 한 개로 끝낸다.
        emergency_warning = [_EMERGENCY_REPLY] if emergency_hint or intent == "emergency_guidance" else []
        risk_level, recommended_action, reservation_required = self.infer_risk(message, intent, emergency_warning)
        reply = self.format_answer(normalized, emergency=bool(emergency_warning))
        if not emergency_warning:
            reply, dropped, added = self.apply_drug_alcohol_rules(message, reply)
            if dropped or added:
                # 상담 내용은 남기지 않는다 - 규칙 이름·뺀 문장 수만.
                _LOG.info("[CHAT-SAFETY] drug_alcohol dropped=%s added=%s", dropped or {}, ",".join(added) or "-")
            if self.needs_self_adjust_sentence(reply):
                reply = f"{reply}\n{_SELF_ADJUST_SENTENCE}"
                _LOG.info("[CHAT-SAFETY] self_adjust_sentence=added")
        if not emergency_warning and self.needs_clinic_sentence(message, reply):
            # 모델이 "의사·약사에게 물어보세요"로만 끝내도 진료 권유가 빠지지 않게 한 줄 덧붙인다(모델 출력에 기대지 않는다).
            reply = f"{reply}\n{_CLINIC_SENTENCE}"
            _LOG.info("[CHAT-SAFETY] clinic_sentence=added")

        return {
            "reply": reply,
            "riskLevel": risk_level,
            "recommendedAction": recommended_action,
            "reservationRequired": reservation_required,
            "intent": intent,
            "engine": "medgemma",
            "modelName": self._loader.state.model_name,
            "summary": normalized.get("symptoms") or None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": emergency_warning,
            # 근거: 어떤 키워드로 응급을 봤는지(없으면 라우터 판단). chat_service가 router 정보를 덧붙여 로그에 남긴다.
            "meta": {"emergencySignals": signals},
        }


@lru_cache
def get_medgemma_loader() -> MedGemmaLoader:
    return MedGemmaLoader(get_settings())


@lru_cache
def get_medical_llm_service() -> MedicalLlmService:
    settings = get_settings()
    return MedicalLlmService(settings, get_medgemma_loader())
