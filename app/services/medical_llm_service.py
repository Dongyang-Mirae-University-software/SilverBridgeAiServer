from __future__ import annotations

from functools import lru_cache
import logging
import re
from typing import Any, NamedTuple

from app.core.config import Settings, get_settings
from app.models.medgemma_loader import MedGemmaLoader
from app.prompts.medical_chat_prompts import SYSTEM_PROMPT, build_user_prompt
from app.services.chat_deadline import call_timeout, gen_min_sec
from app.services import chat_keywords
from app.services.chat_keywords import symptom_department as _symptom_department
from app.utils.conversation import format_recent_history, normalize_history
from app.utils.json_extract import extract_json_fields_lenient, extract_json_object


class FixedSentence(str):
    """답에 붙이는 고정 문장(2026-10-31 구조 정리 B). 문자열 그대로 쓰이고(본인 상담 답은 바이트 그대로),
    이름·주어·대리 상담 문장을 함께 들고 있다. 주어 - self: 본인에게 하는 지시 / carer: 돌보는 사람에게 하는 지시 /
    neutral: 누구에게나 맞음 / own_variant: 자체 변형이 따로 있음(자해 응답 가족용·되묻기 질문)."""

    name: str
    subject: str
    proxy: str | None

    def __new__(cls, text: str, name: str, subject: str, proxy: str | None = None) -> "FixedSentence":
        obj = super().__new__(cls, text)
        obj.name, obj.subject, obj.proxy = name, subject, proxy
        return obj

    def pick(self, proxy: bool) -> str:
        """대리 상담(보호자가 가족 대신 묻는 상담)이면 대리 문장, 아니면 본래 문장."""
        return self.proxy if proxy and self.proxy else str(self)


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
# 술을 끊거나 줄이라는 권고(2026-10-31 L-1) - 이 구절을 떼고 약 임의 조절을 본다("약 드시는 동안은 술을 끊으세요"가 빠졌다).
_ALCOHOL_REDUCE = re.compile(r"(술|음주)[을를은는도]?\s*(끊|줄이|줄여|중단|삼가)")


def _self_adjust_hit(sentence: str) -> bool:
    return bool(_SELF_ADJUST.search(_ALCOHOL_REDUCE.sub(" ", sentence)))


_SELF_ADJUST_CONSULTED = re.compile(r"의사|약사|병원|진료|처방|상의|상담")
_SELF_ADJUST_SENTENCE = "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요."
# 질문 자체가 약을 끊거나 줄이거나 거르려는 것("약 끊어도 되죠?", 공백 뺀 문장) - 답 본문과 상관없이 상의 문장을 붙인다
# (2026-10-29 E-1: 답 본문 단어만 보면 공감 문장이 질문을 되풀이할 때만 붙어 qa.F1에서 실행마다 붙었다 빠졌다 했다).
# "약속을 끊어도"·"혈압약 먹는데 술 끊어도"(술을 끊는 것)는 걸리지 않게 "약속"과 사이의 "술"을 뺀다.
_SELF_ADJUST_ASK = re.compile(r"약(?!속)[^.!?。술]{0,12}(끊어도|끊으면|끊을까|중단해도|중단하면|안먹어도|안드셔도|안먹으면|줄여도|줄이면|건너뛰어도)")
# 프롬프트 예시 2의 advice 를 모델이 그대로 베낀 문장(2026-10-28 측정: 술 질문 9/27에 끼어 어색 / "술을 더 드시거나"로 꼬임).
# 약 + 술 문의에서만 뺀다 - 상의 안내는 고정 문장이 맡는다. 약을 더 먹거나 바꾸려는 질문(예시 2 상황)에서는 맞는 말이라 건드리지 않는다.
_EXAMPLE_COPY = re.compile(
    r"(약|술)을 더 드시거나 바꾸기 전에|다니시는 병원 진료 때 함께 여쭤보셔도"
    # 2026-10-30 J-5: "술을 마시거나 바꾸기 전에"(qa.D5, 예시 문장이 꼬인 변형)
    r"|(약|술)을 (더 )?(드시|마시|드셔|마셔)[^.!?。]{0,6}바꾸기 전에"
)
# 모델이 지어낸 진료과("당뇨 관리의학과", 2026-10-28 측정 1회). 실제 있는 "…의학과"만 남긴다.
_REAL_MEDICINE_DEPTS = frozenset({
    "가정의학과", "재활의학과", "응급의학과", "핵의학과", "직업환경의학과", "예방의학과",
    "마취통증의학과", "진단검사의학과", "영상의학과", "정신건강의학과", "병리과", "방사선종양학과",
    "비뇨의학과",  # 2026-10-29: 빠져 있어 빈뇨 답의 맞는 진료과 문장이 지워졌다
})
_MEDICINE_DEPT = re.compile(r"[가-힣]+의학과")


def _is_unknown_department(name: str) -> bool:
    """"…의학과" 덩어리가 실제 진료과 이름으로 끝나지 않으면 지어낸 이름이다(앞 글자가 붙어 와도 판정되게 끝맺음으로 본다)."""
    return not any(name.endswith(real) for real in _REAL_MEDICINE_DEPTS)


# ---- 답변 품질 후처리(2026-10-29 D-1·D-2·D-4) - 응급·약 + 술 답에는 쓰지 않는다 ----
# 답에 나온 진료과 이름(긴 이름부터 - "정형외과"를 "외과"로 자르지 않게). 세부 내과는 내과 계열로 본다.
_ANSWER_DEPARTMENTS = (
    "정신건강의학과", "마취통증의학과", "가정의학과", "재활의학과", "응급의학과", "소아청소년과", "순환기내과", "소화기내과",
    "내분비내과", "호흡기내과", "류마티스내과", "신장내과", "이비인후과", "비뇨의학과", "비뇨기과", "산부인과", "정형외과",
    "신경외과", "흉부외과", "성형외과", "신경과", "피부과", "안과", "내과", "외과",
)
_ANSWER_DEPARTMENT = re.compile("|".join(_ANSWER_DEPARTMENTS))
# 일반 증상 답에 병원·진료 권유가 없을 때 붙이는 문장(D-4). 진료과 힌트가 있으면 그 과를 넣는다.
_CLINIC_SENTENCE_WITH_DEPT = "증상이 계속되거나 걱정되시면 {department} 진료를 받아 보세요."
# 증상 없이 생활관리·복약 상황을 말한 상담("혈압이 높은 날 운동해도 되나요", "혈압약 먹었어요")에 붙이는 문장.
_CONSULT_SENTENCE = "궁금하신 점은 병원 진료 때 의사와 상의해 보세요."
_CLINIC_WORDS = re.compile(r"병원|진료|응급실")
# 응급은 아니지만 질문에 위험 신호 단어(가슴·숨이 등)가 있는데 답에 119가 없을 때 붙인다(D-3 - 규칙 8이 119를 위험 신호 때만 쓰게 하면서
# 모델이 그 문장을 빠뜨린 경우의 바닥, 2026-10-29 DGX "계단 오르면 숨이 차요").
_RED_FLAG_119_SENTENCE = "숨쉬기 힘들거나 가슴이 조이고 식은땀이 나는 등 갑자기 심해지면 바로 119에 연락하세요."
_COMPACT_DROP = re.compile(r"[\s.,!?~…]+")
# 어르신 돌봄 서비스라 답에 소아청소년과가 나오면 힌트 과(없으면 내과)로 바꾼다(2026-10-29 E: qa.A7 빈뇨에 소아청소년과).
# 사용자가 직접 소아청소년과·소아과를 말했으면 그대로 둔다.
_PEDIATRICS = "소아청소년과"
_PEDIATRICS_ASKED = re.compile(r"소아청소년과|소아과")
# 이미 약을 더·여러 번 먹었다는 말(과다 복용) - 이때는 119 문장을 남긴다(프롬프트 규칙 4 "이미 더 드셨다고 하면 119").
_OVERDOSE_TOLD = re.compile(
    r"(더|또|두 ?번|두 ?알|여러 ?번|여러 ?알|많이|한꺼번에|두 ?배)[^.!?。]{0,10}(먹었|드셨|복용했|먹어 ?버렸)|과다 ?복용|잘못 (먹|드)"
)



class _DrugDirectRule(NamedTuple):
    """약 질문 유형별 직접 안내(2026-10-29 F-1: 판정자 12회 모두 "의사·약사에게 물어보세요"만 있고 핵심 거절이 없다고 지적).
    ask = 질문(공백 뺀 문장)이 이 유형인가 / sentence = 답에 덧붙일 고정 문장 / covered = 답에 이미 같은 뜻이 있으면 생략."""

    name: str
    ask: re.Pattern[str]
    sentence: str
    covered: re.Pattern[str]


_DOUBLE_DOSE_SENTENCE = "놓친 약을 다음에 두 번 몫으로 한꺼번에 드시지는 마세요."
# 수면제를 약국에서 살 수 있느냐는 질문(G-1, 공백 뺀 문장).
_SLEEPING_PILL_PURCHASE = re.compile(r"(수면제|수면유도제|수면보조제|잠오는약)[^.!?。]{0,15}(약국|살수|사도|사야|사먹|구입|구매)")
_MISSED_DOSE_ASK = re.compile(r"약(?!속)[^.!?。]{0,12}(깜빡|잊어|잊고|못먹|안먹었|빠뜨|걸렀|놓쳤)")
_DRUG_DIRECT_RULES = (
    _DrugDirectRule(
        # 2026-10-30 I-4 qa.D1: 기준 R1(기억나면 바로)·R2(다음 시간이 가까우면 건너뛰기)가 답에 없었다(두 번 몫 금지만 있음).
        "missed_dose_timing",
        _MISSED_DOSE_ASK,
        "놓친 약은 생각났을 때 바로 드시되, 다음 드실 시간이 가까우면 놓친 것은 건너뛰고 원래 시간에 한 번만 드세요. 약마다 다를 수 있으니 약사에게 확인해 보세요.",
        re.compile(r"(?s)(?=.*(생각나|생각났|기억나|기억났|떠오르|떠올))(?=.*건너뛰)"),  # "생각났을"은 "생각나"와 다른 글자다
    ),
    _DrugDirectRule(
        "double_dose",
        # 놓친 약(깜빡·잊어·안 먹었) 또는 몰아서·두 번 몫으로 먹으려는 질문. 이미 더 먹었다는 말(과다 복용)은 아래에서 뺀다.
        re.compile(r"약(?!속)[^.!?。]{0,12}(깜빡|잊어|잊고|못먹|안먹었|빠뜨|걸렀|놓쳤)|(몰아서|한꺼번에|한번에|두알|두번|2알|2회분|하루치)[^.!?。]{0,12}(먹어도|드셔도|복용해도|먹으면|먹을까)"),
        _DOUBLE_DOSE_SENTENCE,
        re.compile(r"(한꺼번에|두 ?번 ?몫|두 배|2회분|몰아서)[^.!?。]{0,20}(마세요|마시|않)"),
    ),
    _DrugDirectRule(
        "painkiller_limit",
        re.compile(r"진통제[^.!?。]{0,15}(많이|아무때나|여러|몇알|더)|(많이|아무때나|몇알)[^.!?。]{0,10}진통제"),
        "진통제는 정해진 양과 간격을 넘겨 드시지 마세요.",
        re.compile(r"정해진 (양|용량)|권장 (용량|양)|넘겨 드시지|과다 ?복용"),
    ),
    _DrugDirectRule(
        "leftover_antibiotic",
        re.compile(r"(남은|예전|전에받은|먹다남은|남아있는)[^.!?。]{0,6}항생제"),
        "남은 항생제를 임의로 드시지 마세요. 열이 나면 진료를 받아 보세요.",
        re.compile(r"항생제[^.!?。]{0,20}(드시지 마|먹지 마|임의로)"),
    ),
    _DrugDirectRule(
        # 2026-10-30 I-4 qa.F4: 기준 R1(본인에게 처방된 것만)·R2(바이러스에는 듣지 않음)가 답에 없었다.
        "antibiotic_virus",
        re.compile(r"(남은|예전|전에받은|먹다남은|남아있는)[^.!?。]{0,6}항생제|(감기|독감|열|기침|콧물)[^.!?。]{0,20}항생제|항생제[^.!?。]{0,20}(감기|독감|열|기침)"),
        "감기·독감 같은 바이러스에는 항생제가 듣지 않고, 항생제는 그때 본인에게 처방된 것만 드셔야 해요.",
        re.compile(r"바이러스"),
    ),
    _DrugDirectRule(
        # 2026-10-30 K-4 qa.D2: 기준 R2(어르신은 감기약에도 부작용)·R3(같은 성분이 겹치는지 라벨 확인)가 답에 없었다.
        "cold_medicine_combo",
        re.compile(r"감기약[^.!?。]{0,20}(같이|함께|먹어도|드셔도|복용해도)|(같이|함께)[^.!?。]{0,15}감기약"),
        "어르신은 감기약에도 부작용이 생길 수 있어요. 드시는 약과 같은 성분이 겹치지 않는지 약 상자의 성분표를 약사와 함께 확인하세요.",
        re.compile(r"성분"),
    ),
    _DrugDirectRule(
        "bp_med_stop",
        re.compile(r"혈압[^.!?。]{0,20}약[^.!?。]{0,8}(끊|중단|안먹어도|줄여)|혈압약[^.!?。]{0,8}(끊|중단|안먹어도|줄여)"),
        "혈압약을 임의로 끊으면 혈압이 다시 오를 수 있어요.",
        re.compile(r"혈압[^.!?。]{0,15}(다시 )?(오르|오를|올라|높아|상승)"),
    ),
    _DrugDirectRule(
        # 2026-10-29 G-1 qa.D3: 모델이 "수면제는 약국에서 살 수 없다"고 단정했다(일반의약품 수면유도제가 있어 사실 오류, 판정자 4명).
        "sleeping_pill_purchase",
        _SLEEPING_PILL_PURCHASE,
        "약국에서 살 수 있는 수면유도제(일반의약품)도 있지만, 수면제는 처방이 필요해요. 드시기 전에 약사와 상의하세요.",
        re.compile(r"일반의약품|수면유도제|수면 ?보조제"),
    ),
)
# 수면제 구입 질문에서 "약국에서 살 수 없다"고 단정한 문장(G-1) - 이 질문에서만 뺀다.
_SLEEPING_PILL_WRONG = re.compile(
    r"(약국|처방)[^.!?。]{0,25}(살 수 없|구입할 수 없|구매할 수 없|구입하실 수 있는 약은 아니|살 수 있는 약은 아니|사실 수 없)"
    # 2026-10-30 J-1: "약국에서 직접 구입할 수 있는 약은 아니에요"·"살 수 있는 약이 아니에요" 변형도 같은 단정이다.
    r"|(약국|처방)[^.!?。]{0,25}(살|사실|구입할|구입하실|구매할|구매하실) 수 있는 약(은|이) 아니"
)
# 2026-10-30 K-2: 같은 질문에서 "처방이 필요할 수 있어요"는 고정 문장("수면제는 처방이 필요해요")과 엇갈린다.
_SLEEPING_PILL_HEDGE = re.compile(r"처방[^.!?。]{0,6}필요할 수 있")
# 생활관리 질문 고정 문장(2026-10-29 G-3, 약 질문 직접 안내와 같은 방식 - 판정자가 생성 차이로 내용이 빠진다고 본 E1·E2·E4).
_LIFESTYLE_RULES = (
    _DrugDirectRule(
        "diabetes_fruit",
        re.compile(r"(당뇨|혈당)[^.!?。]{0,20}과일|과일[^.!?。]{0,20}(당뇨|혈당)"),
        "과일은 당도가 낮은 것을 조금씩, 식후에 드시는 것이 좋아요.",
        re.compile(r"(당도|양)[^.!?。]{0,15}(조절|낮은|적게|조금)|조금씩"),
    ),
    _DrugDirectRule(
        "high_bp_exercise",
        re.compile(r"혈압[^.!?。]{0,15}(높|올라|오른)[^.!?。]{0,15}운동|운동[^.!?。]{0,15}혈압[^.!?。]{0,10}(높|올라|오른)"),
        "혈압이 평소보다 높게 나온 날은 운동을 쉬거나 가볍게 하시고, 가슴 통증·어지럼이 있으면 바로 멈추세요.",
        # 2026-10-30 K-1: 모델의 애매한 "운동을 하신 후에도 혈압이 높으면 … 멈추세요"에 걸려 생략되던 것을 없앤다 - 늘 붙인다(같은 문장만 생략).
        re.compile(r"(?!)"),
    ),
    _DrugDirectRule(
        "arthritis_exercise",
        re.compile(r"관절[^.!?。]{0,15}운동|운동[^.!?。]{0,15}관절"),
        "걷기·수영·실내 자전거처럼 관절에 무리가 덜 가는 운동을, 통증이 생기지 않는 범위에서 해 보세요.",
        re.compile(r"(?s)(?=.*수영)(?=.*통증)"),
    ),
    _DrugDirectRule(
        # 2026-10-30 I-4 qa.E3: 기준 R1(포화·트랜스지방 줄이기)·R2(식이섬유 늘리기)가 생성마다 빠졌다.
        "cholesterol_diet",
        re.compile(r"(콜레스테롤|고지혈|이상지질|중성지방)[^.!?。]{0,20}(식단|음식|식사|식이|먹|드)|(식단|음식|식사)[^.!?。]{0,20}(콜레스테롤|고지혈|이상지질|중성지방)"),
        "기름진 고기·튀김·가공식품의 포화지방·트랜스지방은 줄이고, 채소·콩·통곡물·해조류처럼 식이섬유가 많은 음식을 늘려 보세요.",
        re.compile(r"(?s)(?=.*(포화|트랜스))(?=.*식이 ?섬유)"),
    ),
)
# 2026-10-30 I-1: 이미 이어지거나 나빠지고 있다는 말(공백 뺀 문장) - 이때 "계속되거나 심해지면 진료"로만 끝나지 않게 한 줄 덧붙인다.
# "며칠째"는 넣지 않는다(기침 3주처럼 기준 기간이 더 긴 증상이 있다). 열·분비물·고름·피 섞임은 이미 나타난 진료 신호다.
_PERSISTING = re.compile(
    r"(일주일|이주|보름|한달|몇주|몇달|\d+주|\d+개월|[1-9]\d+일|[7-9]일)(째|넘|이상|동안|가까이|이나|됐|되었|지났)"
    r"|(점점|갈수록|더)심해|심해(져|졌|지고)|안낫|낫지않|안나아|나아지지않"
    r"|물이나|고름|진물|피가섞|피섞|열(이|도|은|나|까지)|발열|미열|고열"
)
_URGED_VISIT = re.compile(r"미루지|되도록 ?(빨리|일찍)|빨리 진료|바로 진료|일찍 진료|진료를 받으세요(?!\S)")
_PERSISTING_SENTENCE = "말씀하신 상태라면 미루지 말고 {department} 진료를 받아 보세요."
# 2026-10-30 I-3: 시스템 프롬프트 문구가 답에 그대로 나온 문장(E3 1회 "…119 문장은 붙이지 않습니다"). 사용자에게 쓸 일이 없는 말만 고른다.
_PROMPT_LEAK = re.compile(
    r"규칙 ?\d|119 ?문장은|\[(응급 키워드|권장 진료과|현재 증상|사용자 (정보|질문)|최근 대화|의도)|advice|empathy|symptoms|features|JSON|json|예시 ?\d|형식만 보여|자리표시|키 이름|진료과 힌트"
)


class _EmergencyAction(NamedTuple):
    """응급 유형별 행동 한 줄(2026-10-29 F-2). 119 안내 뒤에 붙인다 - 119 문장은 지우거나 바꾸지 않는다."""

    name: str
    signal: re.Pattern[str]  # 질문(공백 뺀 문장)에 이 유형의 응급 신호가 있는가
    sentence: str


_EMERGENCY_ACTIONS = (
    _EmergencyAction(
        "chest_stroke",
        re.compile(r"가슴이?(아|조|쪼|통증|찢)|가심이|심장이아|말이어눌|발음이(이상|꼬|자꾸)|혀가꼬|한쪽(팔|다리|얼굴|이)|얼굴이한쪽|입이돌아|팔다리에힘|갑자기앞이안보|벼락|갑자기심한두통|망치로맞은"),
        "직접 운전하거나 혼자 움직이지 마세요.",
    ),
    _EmergencyAction("seizure", re.compile(r"경련|발작|거품을물"), "억지로 붙잡거나 입에 물건을 넣지 마세요."),
    # 2026-10-29 G-2. 질식·목에 걸림은 앉은 자세 안내가 맞지 않아 뺀다(신호 정규식이 "질식"·"목에걸려"를 보지 않는다).
    _EmergencyAction(
        "breathing",
        re.compile(r"숨이안쉬|숨을못쉬|숨쉬기힘|숨쉬기가힘|숨이막히|호흡곤란|호흡이곤란|입술이파"),
        "편하게 앉은 자세로 숨을 쉬게 하고, 꽉 끼는 옷은 풀어 주세요.",
    ),
    _EmergencyAction(
        "vomiting_blood",
        re.compile(r"피를토|토혈|커피색|피가섞인구토|피섞인구토"),
        "토한 것이 목에 걸리지 않게 옆으로 눕히고, 아무것도 먹거나 마시지 마세요.",
    ),
    _EmergencyAction(
        "collapse_fall",
        re.compile(r"쓰러|넘어져|넘어졌|못일어나|일어나지못|일어나지를못|일어날수가없|일어설수가없"),
        "무리하게 일으키지 말고 편한 자세로 눕혀 두세요.",
    ),
)
# 2026-10-30 J-3: 응급 답의 설명 한 줄(판정자 6명 공통 지적 - B2 운전 금지·C5 출혈 신호·F3 심근경색 가능성).
_BREATHING_NO_DRIVE = "직접 운전해서 병원에 가지 마세요."
_VOMIT_BLOOD_INFO = "커피색이나 검은색으로 토하는 것은 위나 장의 출혈일 수 있어요."
_CHEST_INDIGESTION = re.compile(r"소화불량|소화가안|체한|체했|체해서|체기|속이(쓰|더부룩|안좋)")
_CHEST_INDIGESTION_INFO = "소화불량처럼 느껴져도 심근경색 증상일 수 있어요."
# 2026-10-30 J-4: 일반 증상 답의 위험 신호·자가관리 고정 문장(A1·A2·A8). 119라는 말은 넣지 않는다(일반 답의 막연한 119를 늘리지 않게).
_SYMPTOM_RULES = (
    _DrugDirectRule(
        "cough_red_flag",
        re.compile(r"기침|목이(아프|아파|따가)|목(통증|감기)"),
        "숨이 차거나 피 섞인 기침이 나오거나 삼키기 어려우면 바로 응급실 진료를 받으세요.",
        re.compile(r"피 ?섞인|피가 섞|객혈"),
    ),
    _DrugDirectRule(
        "orthostatic_dizziness",
        re.compile(r"(일어나|일어설|일어날|일어서)[^.!?。]{0,10}어지|어지[^.!?。]{0,10}(일어나|일어설|일어날|일어서)"),
        "누웠다 일어날 때는 천천히 일어나시고, 물을 충분히 드세요.",
        re.compile(r"(?s)(?=.*천천히)(?=.*(물|수분))"),
    ),
    _DrugDirectRule(
        "sudden_vision_loss",
        re.compile(r"눈이?(침침|흐릿|뿌옇|잘안보)|시력|앞이(흐릿|뿌옇|침침)"),
        "갑자기 앞이 안 보이거나 시야가 가려지면 바로 응급실이나 안과로 가세요.",
        re.compile(r"갑자기[^.!?。]{0,15}(안 ?보|시력|시야)"),
    ),
    _DrugDirectRule(
        # 2026-10-30 K-4 qa.A6: 기준 R3(얼굴·팔·다리 저림이나 힘 빠짐). 119 낱말은 쓰지 않는다(일반 답).
        "numbness_stroke_sign",
        re.compile(r"저려|저리|저린|저림"),
        "한쪽 얼굴·팔·다리가 갑자기 저리거나 힘이 빠지면 바로 응급실로 가세요.",
        re.compile(r"한쪽[^.!?。]{0,15}(저리|저려|힘이 빠|마비)"),
    ),
)
# 2026-10-30 K-3: 약 이야기가 없는 질문의 답에 붙는 모델 상투 문장("약 복용이나 (다른) 대처 방법에 대해 …") - 프롬프트 예시 2 영향.
_MEDICATION_BOILERPLATE = re.compile(r"약 ?복용(이나|과|및)[^.!?。]{0,20}대처")
# 2026-10-30 J-2: 증상 칸에 나온 모델의 분석 문장("복약 문의. … 언급되지 않음") - 특징 칸 거르기(format_answer)와 같은 말들.
_ANALYSIS_NOTE = re.compile(r"언급되지|언급 ?없|언급이 없|말씀하지 않|제공되지|명시되지|정보 ?없")
# 저혈당은 "저혈당"이라는 말 또는 혈당 수치 70 미만일 때만 - "혈당이 300"에 단 음식을 권하면 반대가 된다.
_HYPOGLYCEMIA_WORD = re.compile(r"저혈당")
_GLUCOSE_VALUE = re.compile(r"혈당(?:이|은|가|수치가?)?(\d{2,3})")
_HYPOGLYCEMIA_SENTENCE = "의식이 있으면 사탕이나 단 음료를 드시고, 의식이 흐리면 억지로 먹이지 마세요."
# 2026-10-30 K-4 qa.C6: 기준 R2(다시 재고, 계속 낮으면 의사에게 연락)
_HYPOGLYCEMIA_RECHECK = "15분쯤 뒤 혈당을 다시 재 보고, 계속 낮거나 증상이 이어지면 의사에게 알리세요."

# 증상 상담 되묻기(2026-10-29 C-1, 설계 docs/design/2026-10-29-chat-followup-question.md A안).
# 기간·상황·정도를 말한 문장(공백 뺀 문장)이면 되묻지 않는다 - 이미 답할 재료가 있다.
_FOLLOWUP_CONTEXT = re.compile(
    r"어제|그제|그저께|오늘|아침|점심|저녁|밤|새벽|며칠|일주일|이주|한주|주일|한달|개월|년|째|부터|동안|요즘|최근|매일|자주|가끔|"
    r"계속|자꾸|항상|늘|식후|식사|먹고나|먹으면|계단|걸을|걷|일어날|일어나면|누우|누워|앉|운동|너무|심하|심해|많이|조금|약간|살짝|"
    r"좀|못참|견디|참을|[0-9]|번|때마다|할때|하면|무슨과|어느과|병원|예약|진료"
)
_FOLLOWUP_SYMPTOM_NOUN = chat_keywords.FOLLOWUP_SYMPTOM_NOUN
# 감정·괴로움 표현이 섞이면 되묻지 않는다 - 위기 신호일 수 있는 말("다 그만두고 싶어")에 고정 질문으로 답하지 않게.
_FOLLOWUP_DISTRESS = re.compile(r"그만두|그만하|힘들|힘드|싫어|싫다|우울|죽겠|못살|외로|불안|무서|괴로")
_FOLLOWUP_INTRO = "말씀해 주셔서 고마워요. 더 정확히 안내해 드리려고 몇 가지 여쭤볼게요."
_FOLLOWUP_QUESTIONS = (
    "· 언제부터 그러셨나요?",
    "· 얼마나 심하신가요? (참을 만한지, 점점 심해지는지)",
    "· 같이 나타나는 증상(열·통증 등)이 있나요?",
)
# 보호자가 어르신 대신 상담할 때(대리 상담 wardId 또는 가족이 주어인 문장) 묻는 대상만 바꾼다(Q4).
_FOLLOWUP_QUESTIONS_PROXY = (
    "· 어르신께서 언제부터 그러셨나요?",
    "· 얼마나 심해 보이시나요? (참으실 만한지, 점점 심해지는지)",
    "· 같이 나타나는 증상(열·통증 등)이 있나요?",
)
_FOLLOWUP_CLOSING = "갑자기 심해지면 바로 진료를 받으세요."


def _department_family(name: str) -> str:
    return "내과" if name.endswith("내과") else name


def _compact(text: str) -> str:
    return _COMPACT_DROP.sub("", text or "")


_ALLOWANCE_NEGATED = re.compile(r"괜찮지 않|괜찮은 (것|게) 아니|안 괜찮|되지 않|안 ?돼")
_HOSPITAL_WORDS = re.compile(r"병원|진료|응급실|119")
_MEDICATION_WORD = re.compile(r"약|항생제|진통제|수면제|소화제|해열제|아스피린")
_MEDICATION_ASK = re.compile(r"먹어도|드셔도|복용해도|끊어도|중단해도|먹으면|드시면|먹을까")
_SYMPTOM_WORDS = chat_keywords.MEDICATION_SYMPTOM_WORDS
_GENERAL_SYMPTOM_WORDS = chat_keywords.GENERAL_SYMPTOM_WORDS


# ---- 고정 문장 표(2026-10-31 구조 정리 B, 점검 L-3·L-6) ----------------------------------------------------------
# 답에 붙이는 고정 문장 전부의 주어와 대리 상담 문장. 새 고정 문장은 이 표에 넣어야 붙일 수 있다
# (FIXED_SENTENCES 를 순회하는 테스트가 빼기 규칙 충돌·주어 self 의 대리 문장 누락을 막는다).
# 키: 모듈 상수 이름 또는 "규칙 표.규칙 이름". 값: (주어, 대리 상담 문장 또는 None).
_SENTENCE_TABLE: dict[str, tuple[str, str | None]] = {
    "_EMERGENCY_REPLY": ("self", "지금은 응급 상황일 수 있어요. 지금 바로 119에 전화하세요. 구급대가 올 때까지 곁에 있어 주시고, 다른 가족에게도 알려 주세요."),
    "_SELF_HARM_REPLY": ("own_variant", None),
    "_SELF_HARM_REPLY_FAMILY": ("own_variant", None),
    "_SELF_HARM_WARNING": ("own_variant", None),
    "_CLINIC_SENTENCE": ("neutral", None),
    "_SELF_ADJUST_SENTENCE": ("neutral", None),
    "_CLINIC_SENTENCE_WITH_DEPT": ("neutral", None),
    "_CONSULT_SENTENCE": ("neutral", None),
    "_RED_FLAG_119_SENTENCE": ("neutral", None),
    "_DOUBLE_DOSE_SENTENCE": ("self", "놓친 약을 다음에 두 번 몫으로 한꺼번에 드시게 하지는 마세요."),
    "_PERSISTING_SENTENCE": ("neutral", None),
    "_BREATHING_NO_DRIVE": ("self", "직접 운전해서 병원에 모시고 가지 마세요."),
    "_VOMIT_BLOOD_INFO": ("neutral", None),
    "_CHEST_INDIGESTION_INFO": ("neutral", None),
    "_HYPOGLYCEMIA_SENTENCE": ("self", "의식이 있으시면 사탕이나 단 음료를 드시게 하고, 의식이 흐리시면 억지로 먹이지 마세요."),
    "_HYPOGLYCEMIA_RECHECK": ("self", "15분쯤 뒤 혈당을 다시 재 드리고, 계속 낮거나 증상이 이어지면 의사에게 알려 주세요."),
    "_FOLLOWUP_INTRO": ("own_variant", None),
    "_FOLLOWUP_CLOSING": ("own_variant", None),
    "_DRUG_ALCOHOL_RULES.diabetes": ("self", "어르신이 당뇨약을 드시는 중에 술을 드시면 저혈당(식은땀·떨림·어지럼)이 늦게 올 수 있어요. 드시기 전에 의사·약사와 꼭 상의해 주세요."),
    "_DRUG_ALCOHOL_RULES.sedative": ("self", "어르신이 수면제나 안정제를 술과 함께 드시면 숨이 약해지거나 깨어나기 어려울 만큼 위험할 수 있어요. 술과 함께 드시지 않게 해 주세요."),
    "_DRUG_ALCOHOL_RULES.acetaminophen": ("self", "어르신이 타이레놀(아세트아미노펜)을 드시는 중에 술을 드시면 간이 상할 수 있어요. 함께 드시지 않게 해 주시고, 술을 자주 드시면 의사·약사와 상의해 주세요."),
    "_DRUG_ALCOHOL_RULES.anticoagulant": ("self", "어르신이 와파린 같은 피를 묽게 하는 약을 드시는 중에 술을 드시면 출혈 위험이 커지고 약 효과가 달라질 수 있어요. 드시기 전에 의사·약사와 꼭 상의해 주세요."),
    "_DRUG_ALCOHOL_RULES.blood_pressure": ("self", "어르신이 혈압약을 드시는 중에 술을 드시면 혈압이 갑자기 떨어져 어지럽거나 넘어지실 수 있고, 자주 드시면 오히려 혈압이 올라 약 효과가 떨어질 수 있어요. 드시기 전에 의사·약사와 상의해 주세요."),
    "_DRUG_ALCOHOL_RULES.antibiotic": ("self", "어르신이 항생제를 드시는 동안에는 술을 피하시게 해 주세요. 일부 항생제는 술과 함께 드시면 심한 두근거림·구토가 생길 수 있어요. 드시는 약이 괜찮은지 약사에게 확인해 주세요."),
    "_DRUG_DIRECT_RULES.missed_dose_timing": ("self", "놓친 약은 생각났을 때 바로 드시게 하되, 다음 드실 시간이 가까우면 놓친 것은 건너뛰고 원래 시간에 한 번만 드시게 해 주세요. 약마다 다를 수 있으니 약사에게 확인해 보세요."),
    "_DRUG_DIRECT_RULES.double_dose": ("self", "놓친 약을 다음에 두 번 몫으로 한꺼번에 드시게 하지는 마세요."),
    "_DRUG_DIRECT_RULES.painkiller_limit": ("self", "진통제는 정해진 양과 간격을 넘겨 드시지 않게 해 주세요."),
    "_DRUG_DIRECT_RULES.leftover_antibiotic": ("self", "남은 항생제를 임의로 드시지 않게 해 주세요. 열이 나시면 진료를 받아 보시게 해 주세요."),
    "_DRUG_DIRECT_RULES.antibiotic_virus": ("neutral", None),
    "_DRUG_DIRECT_RULES.cold_medicine_combo": ("neutral", None),
    "_DRUG_DIRECT_RULES.bp_med_stop": ("neutral", None),
    "_DRUG_DIRECT_RULES.sleeping_pill_purchase": ("self", "약국에서 살 수 있는 수면유도제(일반의약품)도 있지만, 수면제는 처방이 필요해요. 드시게 하기 전에 약사와 상의해 주세요."),
    "_LIFESTYLE_RULES.diabetes_fruit": ("self", "과일은 당도가 낮은 것을 조금씩, 식후에 드시게 하는 것이 좋아요."),
    "_LIFESTYLE_RULES.high_bp_exercise": ("self", "혈압이 평소보다 높게 나온 날은 운동을 쉬시거나 가볍게 하시게 하고, 가슴 통증·어지럼이 있으시면 바로 멈추시게 해 주세요."),
    "_LIFESTYLE_RULES.arthritis_exercise": ("self", "걷기·수영·실내 자전거처럼 관절에 무리가 덜 가는 운동을, 통증이 생기지 않는 범위에서 하시게 해 주세요."),
    "_LIFESTYLE_RULES.cholesterol_diet": ("neutral", None),
    "_SYMPTOM_RULES.cough_red_flag": ("neutral", None),
    "_SYMPTOM_RULES.orthostatic_dizziness": ("self", "누웠다 일어나실 때는 천천히 일어나시게 하고, 물을 충분히 드시게 해 주세요."),
    "_SYMPTOM_RULES.sudden_vision_loss": ("neutral", None),
    "_SYMPTOM_RULES.numbness_stroke_sign": ("neutral", None),
    "_EMERGENCY_ACTIONS.chest_stroke": ("self", "직접 운전해서 모시고 가지 마시고, 혼자 움직이시게 두지 마세요."),
    "_EMERGENCY_ACTIONS.seizure": ("carer", None),
    "_EMERGENCY_ACTIONS.breathing": ("carer", None),
    "_EMERGENCY_ACTIONS.vomiting_blood": ("carer", None),
    "_EMERGENCY_ACTIONS.collapse_fall": ("carer", None),
}


def _fixed(key: str, text: str) -> FixedSentence:
    subject, proxy = _SENTENCE_TABLE[key]  # 표에 없는 고정 문장은 여기서 KeyError - 표를 먼저 채운다
    return FixedSentence(text, key, subject, proxy)


for _name in [k for k in _SENTENCE_TABLE if "." not in k]:
    globals()[_name] = _fixed(_name, globals()[_name])
_DRUG_ALCOHOL_RULES = tuple(r._replace(sentence=_fixed(f"_DRUG_ALCOHOL_RULES.{r.name}", r.sentence)) for r in _DRUG_ALCOHOL_RULES)
_DRUG_DIRECT_RULES = tuple(r._replace(sentence=_fixed(f"_DRUG_DIRECT_RULES.{r.name}", r.sentence)) for r in _DRUG_DIRECT_RULES)
_LIFESTYLE_RULES = tuple(r._replace(sentence=_fixed(f"_LIFESTYLE_RULES.{r.name}", r.sentence)) for r in _LIFESTYLE_RULES)
_SYMPTOM_RULES = tuple(r._replace(sentence=_fixed(f"_SYMPTOM_RULES.{r.name}", r.sentence)) for r in _SYMPTOM_RULES)
_EMERGENCY_ACTIONS = tuple(a._replace(sentence=_fixed(f"_EMERGENCY_ACTIONS.{a.name}", a.sentence)) for a in _EMERGENCY_ACTIONS)
del _name
# 고정 문장 전부(표 순서). 테스트·점검은 이 목록만 본다.
FIXED_SENTENCES: tuple[FixedSentence, ...] = tuple(
    [globals()[k] for k in _SENTENCE_TABLE if "." not in k]
    + [r.sentence for r in _DRUG_ALCOHOL_RULES + _DRUG_DIRECT_RULES + _LIFESTYLE_RULES + _SYMPTOM_RULES + _EMERGENCY_ACTIONS]
)
assert len(FIXED_SENTENCES) == len(_SENTENCE_TABLE)


class MedicalLlmService:
    # 응급·자해 목록은 chat_keywords 가 단일 출처다.
    EMERGENCY_KEYWORDS = chat_keywords.EMERGENCY_KEYWORDS
    SELF_HARM_KEYWORDS = chat_keywords.SELF_HARM_KEYWORDS
    EMERGENCY_WARNING = _EMERGENCY_REPLY
    RESERVATION_KEYWORDS = chat_keywords.RESERVATION_KEYWORDS
    MEDICAL_KEYWORDS = chat_keywords.MEDICAL_KEYWORDS

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
    def apply_drug_alcohol_rules(message: str, reply: str, proxy: bool = False) -> tuple[str, dict[str, int], list[str]]:
        """약 + 술 문의 답 손보기: ① 술을 조금은 괜찮다고 허용한 문장 ② 그 약에서 방향이 반대인 문장을 빼고
        ③ 답에 핵심 위험이 없으면 고정 문장을 붙인다. (고친 답, 뺀 문장 수 {이유: 수}, 붙인 규칙 이름). 다른 상담은 그대로."""
        rules = MedicalLlmService.drug_alcohol_rules(message)
        if not rules:
            return reply, {}, []

        def drop_reason(sentence: str) -> str | None:
            if _EXAMPLE_COPY.search(sentence):
                return "example_copy"
            if _self_adjust_hit(sentence) and sentence.strip() != _SELF_ADJUST_SENTENCE:
                # 2026-10-30 J-5(qa.D5 X2): 술 마실 때 약을 거르라는 문장은 "상의"가 같이 있어도 뺀다 - 고정 상의 문장이 대신한다.
                return "self_adjust"
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
        if dropped.get("self_adjust") and _SELF_ADJUST_SENTENCE not in out:
            out = f"{out}\n{_SELF_ADJUST_SENTENCE}"
            added.append("self_adjust")
        for rule in rules:
            if not rule.covered.search(model_text):
                out = f"{out}\n{rule.sentence.pick(proxy)}"
                added.append(rule.name)
        return out, dropped, added

    @staticmethod
    def drop_unknown_department_sentences(reply: str) -> tuple[str, int]:
        """실제 없는 "…의학과" 이름이 든 문장을 뺀다(모델이 진료과를 지어낸 경우). (고친 답, 뺀 문장 수)."""
        removed = 0
        lines: list[str] = []
        for line in (reply or "").split("\n"):
            sentences = _SENTENCE_SPLIT.split(line)
            kept = [x for x in sentences if not any(_is_unknown_department(m) for m in _MEDICINE_DEPT.findall(x))]
            if len(kept) == len(sentences):
                lines.append(line)
                continue
            removed += len(sentences) - len(kept)
            new_line = " ".join(kept).strip()
            if new_line:
                lines.append(new_line)
        return "\n".join(lines), removed

    @staticmethod
    def needs_self_adjust_sentence(reply: str, message: str = "") -> bool:
        """약을 스스로 조절·중단하라고 읽히는 문장이 있는데 그 문장에 의사·약사 상의가 없거나, 질문이 약을 끊거나 줄이려는 것이고,
        고정 문장이 아직 없으면 True."""
        text = reply or ""
        if _SELF_ADJUST_SENTENCE in text:
            return False
        if _SELF_ADJUST_ASK.search(re.sub(r"\s+", "", message or "")):
            return True
        for line in text.split("\n"):
            for sentence in _SENTENCE_SPLIT.split(line):
                if _self_adjust_hit(sentence) and not _SELF_ADJUST_CONSULTED.search(sentence):
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

    @staticmethod
    def is_echo(symptoms: str, message: str) -> bool:
        """증상 칸이 질문을 그대로 옮긴 것인가(D-2). 모델이 증상을 비워 서버가 질문 앞 60자로 채운 경우도 여기에 든다."""
        s, q = _compact(symptoms), _compact(message)
        if not s or not q:
            return False
        return s == q or q in s or (len(s) >= 15 and q.startswith(s))

    @staticmethod
    def align_department(advice: str, hint: str | None) -> tuple[str, bool]:
        """advice 의 진료과가 서버 힌트와 다르면 힌트로 바꾼다(D-1, 예약 안내와 같은 과). (고친 advice, 바꿨는가).
        과가 둘 이상이면 모델이 고른 이유가 있을 수 있어 그대로 둔다. 같은 계열(소화기내과 / 내과)이면 그대로 둔다."""
        if not hint or not advice:
            return advice, False
        found = {_department_family(x) for x in _ANSWER_DEPARTMENT.findall(advice)}
        if len(found) != 1 or _department_family(hint) in found:
            return advice, False
        return _ANSWER_DEPARTMENT.sub(hint, advice), True

    @staticmethod
    def replace_pediatrics(advice: str, message: str, hint: str | None) -> tuple[str, bool]:
        """답의 소아청소년과를 힌트 과(없으면 내과)로 바꾼다. 사용자가 직접 소아청소년과·소아과를 말했으면 그대로. (고친 advice, 바꿨는가)."""
        if _PEDIATRICS not in (advice or "") or _PEDIATRICS_ASKED.search(message or ""):
            return advice, False
        return advice.replace(_PEDIATRICS, hint or "내과"), True

    @staticmethod
    def drop_medication_119(message: str, reply: str) -> tuple[str, int]:
        """약 복용 문의 답에서 119 문장을 뺀다(프롬프트 규칙 4를 코드로 보장 - 2026-10-29 qa.F1 "혈압이 다시 떨어지면 119" 1/3).
        과다 복용을 말한 질문은 그대로 둔다. 응급 답은 호출하는 쪽(품질 후처리 대상 밖)에서 빠진다. (고친 답, 뺀 문장 수)."""
        compact = re.sub(r"\s+", "", message or "")
        if not (_MEDICATION_WORD.search(compact) and _MEDICATION_ASK.search(compact)) or _OVERDOSE_TOLD.search(message or ""):
            return reply, 0
        removed = 0
        lines: list[str] = []
        for line in (reply or "").split("\n"):
            sentences = _SENTENCE_SPLIT.split(line)
            kept = [x for x in sentences if "119" not in x]
            if len(kept) == len(sentences):
                lines.append(line)
                continue
            removed += len(sentences) - len(kept)
            new_line = " ".join(kept).strip()
            if new_line:
                lines.append(new_line)
        return "\n".join(lines), removed

    @staticmethod
    def drug_direct_sentences(message: str, reply: str, proxy: bool = False) -> list[tuple[str, str]]:
        """약 질문 유형별로 덧붙일 (규칙 이름, 문장). 과다 복용을 말한 질문의 "두 번 몫" 안내는 빼고, 답에 같은 뜻이 있으면 생략한다."""
        compact = re.sub(r"\s+", "", message or "")
        overdose = bool(_OVERDOSE_TOLD.search(message or ""))
        out: list[tuple[str, str]] = []
        for rule in _DRUG_DIRECT_RULES + _LIFESTYLE_RULES + _SYMPTOM_RULES:
            if not rule.ask.search(compact) or rule.covered.search(reply or "") or rule.sentence.pick(proxy) in (reply or ""):
                continue
            if rule.name in ("double_dose", "missed_dose_timing") and overdose:
                continue
            if any(sentence == rule.sentence.pick(proxy) for _, sentence in out):
                continue
            out.append((rule.name, rule.sentence.pick(proxy)))
        return out

    @staticmethod
    def persisting_visit_sentence(message: str, reply: str, hint: str | None) -> str | None:
        """이미 이어지거나 나빠진 증상(I-1)인데 답이 조건부 권유로만 끝나면 덧붙일 문장. 약 복용 문의는 대상 밖."""
        compact = re.sub(r"\s+", "", message or "")
        if not _PERSISTING.search(compact) or _URGED_VISIT.search(reply or ""):
            return None
        if _MEDICATION_WORD.search(compact) and _MEDICATION_ASK.search(compact):
            return None
        return _PERSISTING_SENTENCE.format(department=f"{hint}" if hint else "가까운 병원")

    @staticmethod
    def drop_medication_boilerplate(message: str, reply: str) -> tuple[str, int]:
        """질문에 약 이야기가 없으면 "약 복용이나 … 대처" 상투 문장을 뺀다(K-3). 약 + 술 질문은 대상 밖. (고친 답, 뺀 문장 수)."""
        compact = re.sub(r"\s+", "", message or "")
        if _MEDICATION_WORD.search(compact) or MedicalLlmService.drug_alcohol_rules(message):
            return reply, 0
        removed = 0
        lines: list[str] = []
        for line in (reply or "").split("\n"):
            sentences = _SENTENCE_SPLIT.split(line)
            kept = [x for x in sentences if not _MEDICATION_BOILERPLATE.search(x)]
            if len(kept) == len(sentences):
                lines.append(line)
                continue
            removed += len(sentences) - len(kept)
            new_line = " ".join(kept).strip()
            if new_line:
                lines.append(new_line)
        return ("\n".join(lines) if removed else reply), removed

    @staticmethod
    def drop_prompt_leak(reply: str) -> tuple[str, int]:
        """프롬프트 규칙 문구가 든 문장을 뺀다(I-3). 다 빠지면 상의 권유 한 줄. (고친 답, 뺀 문장 수)."""
        removed = 0
        lines: list[str] = []
        for line in (reply or "").split("\n"):
            sentences = _SENTENCE_SPLIT.split(line)
            kept = [x for x in sentences if not _PROMPT_LEAK.search(x)]
            if len(kept) == len(sentences):
                lines.append(line)
                continue
            removed += len(sentences) - len(kept)
            new_line = " ".join(kept).strip()
            if new_line:
                lines.append(new_line)
        out = "\n".join(lines).strip()
        if removed and not out:
            out = _CONSULT_SENTENCE
        return (out if removed else reply), removed

    @classmethod
    def needs_followup_question(cls, message: str) -> bool:
        """정보가 부족한 일반 증상 한마디("어지러워", "기침이 나요")인가 - 되묻기 대상(C-1). 문장만 본다:
        응급·자해 신호, 약·약 + 술·약 직접 안내·생활관리 질문, 기간·상황·정도 표현, 서로 다른 증상 둘 이상이면 False.
        턴 조건(응급 경로·위기 뒤·직전 턴 되묻기·버튼)은 chat_service 가 본다."""
        compact = re.sub(r"\s+", "", message or "")
        if not compact or len(compact) > 25:
            return False
        if cls.emergency_signals(message) or cls.self_harm_signals(message):
            return False
        # 응급 키워드는 아니어도 위험 신호 단어(가슴·숨이 등)가 있으면 되묻지 않는다 - 위험 신호 119 한 줄이 나가는 답으로 둔다
        # (2026-10-29 v7 bd.02 "가슴이 답답해요"가 되묻기로 가 119 안내가 빠졌다).
        # 공백을 뺀 문장으로 본다(2026-10-31 M-1: "숨차요"·"숨 차요"가 되묻기로 가 119 한 줄이 빠졌다).
        if chat_keywords.has_red_flag(message):
            return False
        if _MEDICATION_WORD.search(compact) or cls.drug_alcohol_rules(message):
            return False
        if any(rule.ask.search(compact) for rule in _DRUG_DIRECT_RULES + _LIFESTYLE_RULES):
            return False
        if _FOLLOWUP_CONTEXT.search(compact) or _FOLLOWUP_DISTRESS.search(compact):
            return False
        nouns = set(_FOLLOWUP_SYMPTOM_NOUN.findall(compact))
        if len(nouns) >= 2:
            return False
        return bool(nouns) or bool(_symptom_department(message or ""))

    @staticmethod
    def followup_question_response(message: str, proxy: bool = False) -> dict[str, Any]:
        """되묻기 답(MedGemma 를 부르지 않는다). 진료과 한 줄을 함께 둔다 - 답하지 않고 떠나는 분도 갈 곳은 알도록(Q2)."""
        department = _symptom_department(message or "")
        lines = [_FOLLOWUP_INTRO, *(_FOLLOWUP_QUESTIONS_PROXY if proxy else _FOLLOWUP_QUESTIONS)]
        lines.append(f"{department} 진료를 생각해 보셔도 좋아요." if department else "가까운 병원 진료를 생각해 보셔도 좋아요.")
        lines.append(_FOLLOWUP_CLOSING)
        return {
            "reply": "\n".join(lines),
            "riskLevel": "medium",
            "recommendedAction": "guardian_contact",
            "reservationRequired": False,
            "intent": "medical_advice",
            "engine": "rule",
            "modelName": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [],
            "meta": {"followupAsked": True, "followupProxy": proxy},
        }

    @staticmethod
    def drop_sleeping_pill_misstatement(message: str, reply: str) -> tuple[str, int]:
        """수면제 구입 질문에서만 "약국에서 살 수 없다" 단정 문장을 뺀다(G-1). (고친 답, 뺀 문장 수)."""
        if not _SLEEPING_PILL_PURCHASE.search(re.sub(r"\s+", "", message or "")):
            return reply, 0
        removed = 0
        lines: list[str] = []
        for line in (reply or "").split("\n"):
            sentences = _SENTENCE_SPLIT.split(line)
            kept = [x for x in sentences if not (_SLEEPING_PILL_WRONG.search(x) or _SLEEPING_PILL_HEDGE.search(x))]
            if len(kept) == len(sentences):
                lines.append(line)
                continue
            removed += len(sentences) - len(kept)
            new_line = " ".join(kept).strip()
            if new_line:
                lines.append(new_line)
        return "\n".join(lines), removed

    @staticmethod
    def emergency_action_sentences(message: str, proxy: bool = False) -> list[tuple[str, str]]:
        """응급 답에 덧붙일 유형별 행동 (이름, 문장). 질문에 그 유형의 신호가 없으면 빈 목록(지금 응급 답 그대로)."""
        compact = re.sub(r"\s+", "", message or "")
        out = [(a.name, a.sentence.pick(proxy)) for a in _EMERGENCY_ACTIONS if a.signal.search(compact)]
        names = {name for name, _ in out}
        if "breathing" in names and "chest_stroke" not in names:  # 가슴 통증 줄에 이미 운전 금지가 있다
            out.append(("breathing_no_drive", _BREATHING_NO_DRIVE.pick(proxy)))
        if "vomiting_blood" in names:
            out.append(("vomiting_blood_info", _VOMIT_BLOOD_INFO))
        if "chest_stroke" in names and _CHEST_INDIGESTION.search(compact):
            out.append(("chest_indigestion", _CHEST_INDIGESTION_INFO))
        values = [int(v) for v in _GLUCOSE_VALUE.findall(compact)]
        if _HYPOGLYCEMIA_WORD.search(compact) or any(v < 70 for v in values):
            out.append(("hypoglycemia", _HYPOGLYCEMIA_SENTENCE.pick(proxy)))
            out.append(("hypoglycemia_recheck", _HYPOGLYCEMIA_RECHECK.pick(proxy)))
        return out

    @staticmethod
    def clinic_guarantee_sentence(message: str, reply: str, hint: str | None) -> str | None:
        """증상 상담 답에 병원·진료 권유가 없으면 덧붙일 문장(D-4). 약 복용 문의는 C-2 규칙(needs_clinic_sentence)이 맡는다."""
        if _CLINIC_WORDS.search(reply or ""):
            return None
        compact = re.sub(r"\s+", "", message or "")
        has_symptom = any(word in compact for word in _GENERAL_SYMPTOM_WORDS)
        if _MEDICATION_WORD.search(compact) and (_MEDICATION_ASK.search(compact) or not has_symptom):
            return None
        if hint:
            return _CLINIC_SENTENCE_WITH_DEPT.format(department=hint)
        if has_symptom:
            return _CLINIC_SENTENCE
        return _CONSULT_SENTENCE

    @classmethod
    def needs_red_flag_sentence(cls, message: str, reply: str) -> bool:
        """질문에 위험 신호 단어가 있는데 답에 119 안내가 없으면 True(응급 판정은 바꾸지 않는다 - 안내 한 줄만)."""
        return "119" not in (reply or "") and chat_keywords.has_red_flag(message)

    @classmethod
    def classify_intent(cls, message: str) -> str:
        text = (message or "").strip().lower()
        if cls.emergency_signals(text):
            return "emergency_guidance"
        if chat_keywords.has_reservation_word(text):
            return "hospital_reservation"
        # 위험 신호·증상 단어만 있는 문장("숨차요"·"답답해요")도 상담이다 - 라우터 실패 때 일반 대화로 빠져 119 한 줄을 놓치지 않게.
        if chat_keywords.has_medical_word(text) or chat_keywords.has_symptom_word(text) or chat_keywords.has_red_flag(text):
            return "medical_advice"
        return "general_chat"

    HIGH_RISK_KEYWORDS = chat_keywords.RED_FLAG_KEYWORDS

    @staticmethod
    def infer_risk(message: str, intent: str, emergency_warning: list[str]) -> tuple[str, str, bool]:
        text = (message or "").strip().lower()
        # 모델은 거의 항상 일반적인 응급 주의 문구를 붙이므로, 사용자 메시지에 응급 신호가 있을 때만 high로 본다.
        has_high_signal = chat_keywords.has_red_flag(message)
        if intent == "emergency_guidance" or (emergency_warning and has_high_signal):
            return "high", "hospital_visit", True
        if intent in {"hospital_reservation", "medical_advice"}:
            return "medium", "guardian_contact", intent == "hospital_reservation"
        if chat_keywords.has_medical_word(text):
            return "medium", "guardian_contact", False
        return "low", "home_monitoring", False

    @staticmethod
    def format_answer(data: dict[str, Any], emergency: bool = False, show_symptoms: bool = True) -> str:
        """말풍선 문장. 모델이 쓴 공감·증상·특징·권고를 줄바꿈으로 잇는다. 비어 있는 값만 기본 문장으로 메운다.
        show_symptoms=False 면 증상 줄만 뺀다(질문을 그대로 옮긴 경우, D-2) - 상담 틀은 그대로다."""
        symptoms = str(data.get("symptoms") or "").strip()
        if not symptoms:
            return str(data.get("reply") or "").strip() or "증상을 조금 더 구체적으로 말씀해 주시면 도와드릴게요."

        lines = [str(data.get("empathy") or "").strip() or "말씀해 주신 내용을 정리했어요."]
        if show_symptoms:
            lines.append(f"· 증상: {symptoms}")
        features = str(data.get("features") or "").strip()
        if features and not re.search(r"없음|없습니다|불명확|모름|알 수 없|언급되지|말씀하지|제공되지", features):  # "유발 요인 불명확" 같은 빈말은 줄을 빼는 게 낫다
            lines.append(f"· 특징: {features}")
        advice = str(data.get("advice") or "").strip()
        if emergency:
            # 응급이면 모델이 진료과를 권해도 쓰지 않는다. 경고 문구는 말풍선 위 빨간 박스(emergencyWarning)가 맡는다.
            advice = "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요."
        # 빈 advice 기본 문장에는 119를 넣지 않는다 - 119는 위험 신호 문장 때만(D-3, 2026-10-31 L-2).
        lines.append(advice or "증상이 계속되면 가까운 병원 진료를 권해드려요. 계속되거나 심해지면 꼭 진료를 받아 보세요.")
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
            if _ANALYSIS_NOTE.search(out["symptoms"]):
                out["symptoms"] = " ".join(
                    x for x in _SENTENCE_SPLIT.split(out["symptoms"]) if not _ANALYSIS_NOTE.search(x)
                ).strip()
            if not out["symptoms"]:
                # 모델이 증상을 비우면 사용자 문장으로 메운다 - 상담인데 일반 답변 틀로 빠지지 않게.
                out["symptoms"] = (message or "").strip()[:60]
            department = _symptom_department(message or "")
            if not out["advice"] and department:
                out["advice"] = f"{department} 진료를 권해드려요. 계속되거나 심해지면 꼭 진료를 받아 보세요."
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
        proxy: bool = False,
    ) -> dict[str, Any]:
        # proxy: 보호자가 가족 대신 묻는 대리 상담(chat_service 가 wardId·가족 주어로 정한다) - 고정 문장을 대리 문장으로 고른다(L-3).
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

        # 예약 흐름과 같은 증상→진료과 규칙을 힌트로 준다. 상담에서 권한 과와 예약에서 찾는 과가 어긋나지 않게.
        department_hint = _symptom_department(message or "")
        user_prompt = build_user_prompt(
            message=message,
            user_context=user_context,
            history_text=history_lines or "(없음)",
            intent=intent,
            emergency_hint=emergency_hint,
            processed_symptom=processed_symptom,
            department_hint=department_hint,
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
        # 생성은 중간에 끊을 수 없다 - 챗 전체 마감 안에서는 생성할 시간(CHAT_GEN_MIN_REMAINING_SEC)이 남을 만큼만 줄을 선다(O-3).
        # 모자라면 ChatDeadlineExceeded -> chat_service가 고정 문구(응급이면 119 안내)로 답한다. 응급 3초 상한은 그대로다.
        wait_timeout_sec = call_timeout("medgemma_wait", float(wait_timeout_sec), reserve=gen_min_sec(), minimum=0.5)
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
        emergency_warning = [_EMERGENCY_REPLY.pick(proxy)] if emergency_hint or intent == "emergency_guidance" else []
        risk_level, recommended_action, reservation_required = self.infer_risk(message, intent, emergency_warning)
        # 답변 품질 후처리(D-1·D-2·D-4)는 응급이 아닌 증상 상담에만 쓴다 - 응급·약 + 술 답은 기존 안전 규칙만 손댄다.
        quality = intent == "medical_advice" and not emergency_warning and not self.drug_alcohol_rules(message)
        hide_symptoms = quality and self.is_echo(normalized["symptoms"], message)
        aligned = pediatrics = False
        if quality:
            # 모델 advice 칸에서만 바꾼다 - 뒤에서 붙이는 고정 문장은 건드리지 않는다.
            normalized["advice"], pediatrics = self.replace_pediatrics(normalized["advice"], message, department_hint)
            if not _PEDIATRICS_ASKED.search(message or ""):  # 사용자가 소아과를 직접 물었으면 그 과를 힌트로 덮지 않는다
                normalized["advice"], aligned = self.align_department(normalized["advice"], department_hint)
        reply = self.format_answer(normalized, emergency=bool(emergency_warning), show_symptoms=not hide_symptoms)
        if not emergency_warning:
            # 프롬프트 문구 노출(I-3)은 다른 후처리보다 먼저 뺀다 - 뒤의 병원 권유 보장이 빈자리를 채운다. 응급 답은 건드리지 않는다.
            reply, leaked = self.drop_prompt_leak(reply)
            if leaked:
                _LOG.info("[CHAT-SAFETY] prompt_leak_dropped=%s", leaked)
            reply, boilerplate = self.drop_medication_boilerplate(message, reply)
            if boilerplate:
                _LOG.info("[CHAT-QUALITY] medication_boilerplate_dropped=%s", boilerplate)
            reply, dropped, added = self.apply_drug_alcohol_rules(message, reply, proxy)
            if dropped or added:
                # 상담 내용은 남기지 않는다 - 규칙 이름·뺀 문장 수만.
                _LOG.info("[CHAT-SAFETY] drug_alcohol dropped=%s added=%s", dropped or {}, ",".join(added) or "-")
            reply, unknown_dept = self.drop_unknown_department_sentences(reply)
            if unknown_dept:
                _LOG.info("[CHAT-SAFETY] unknown_department_dropped=%s", unknown_dept)
                if not _HOSPITAL_WORDS.search(reply):
                    # 진료과 문장을 빼서 병원 권유가 사라졌으면 일반 권유 한 줄로 채운다.
                    reply = f"{reply}\n{_CLINIC_SENTENCE}"
            if self.needs_self_adjust_sentence(reply, message):
                reply = f"{reply}\n{_SELF_ADJUST_SENTENCE}"
                _LOG.info("[CHAT-SAFETY] self_adjust_sentence=added")
            if not self.drug_alcohol_rules(message):
                # 약 + 술 질문은 약 + 술 규칙만 손댄다.
                reply, misstated = self.drop_sleeping_pill_misstatement(message, reply)
                if misstated:
                    _LOG.info("[CHAT-SAFETY] sleeping_pill_misstatement_dropped=%s", misstated)
                direct = self.drug_direct_sentences(message, reply, proxy)
                for _, sentence in direct:
                    reply = f"{reply}\n{sentence}"
                if direct:
                    _LOG.info("[CHAT-SAFETY] drug_direct=%s", ",".join(name for name, _ in direct))
        else:
            # 응급 답: 119 안내(advice 줄)는 그대로 두고 그 뒤에 유형별 행동 한 줄. 경고 박스(emergencyWarning)는 바꾸지 않는다.
            actions = self.emergency_action_sentences(message, proxy)
            for _, sentence in actions:
                reply = f"{reply}\n{sentence}"
            if actions:
                _LOG.info("[CHAT-SAFETY] emergency_action=%s", ",".join(name for name, _ in actions))
        if not emergency_warning and self.needs_clinic_sentence(message, reply):
            # 모델이 "의사·약사에게 물어보세요"로만 끝내도 진료 권유가 빠지지 않게 한 줄 덧붙인다(모델 출력에 기대지 않는다).
            reply = f"{reply}\n{_CLINIC_SENTENCE}"
            _LOG.info("[CHAT-SAFETY] clinic_sentence=added")
        clinic_added = red_flag_added = False
        persisting = None
        medication_119 = 0
        if quality:
            reply, medication_119 = self.drop_medication_119(message, reply)
            persisting = self.persisting_visit_sentence(message, reply, department_hint)
            if persisting:
                reply = f"{reply}\n{persisting}"
            extra = self.clinic_guarantee_sentence(message, reply, department_hint)
            if extra:
                reply = f"{reply}\n{extra}"
                clinic_added = True
            if self.needs_red_flag_sentence(message, reply):
                reply = f"{reply}\n{_RED_FLAG_119_SENTENCE}"
                red_flag_added = True
        if hide_symptoms or aligned or clinic_added or red_flag_added or pediatrics or medication_119 or persisting:
            # 상담 내용은 남기지 않는다 - 플래그·문장 수만.
            _LOG.info(
                "[CHAT-QUALITY] echo_hidden=%s department_aligned=%s clinic_added=%s red_flag_119=%s pediatrics=%s medication_119_dropped=%s persisting_visit=%s",
                hide_symptoms, aligned, clinic_added, red_flag_added, pediatrics, medication_119, bool(persisting),
            )

        return {
            "reply": reply,
            "riskLevel": risk_level,
            "recommendedAction": recommended_action,
            "reservationRequired": reservation_required,
            "intent": intent,
            "engine": "medgemma",
            "modelName": self._loader.state.model_name,
            "summary": None if hide_symptoms else (normalized.get("symptoms") or None),
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": emergency_warning,
            # 근거: 어떤 키워드로 응급을 봤는지(없으면 라우터 판단). chat_service가 router 정보를 덧붙여 로그에 남긴다.
            # redFlag119·drugAlcohol: 이어서 물어볼 질문을 붙이지 않는 답(2026-10-31 M-2).
            "meta": {"emergencySignals": signals, "redFlag119": red_flag_added, "drugAlcohol": bool(self.drug_alcohol_rules(message))},
        }


@lru_cache
def get_medgemma_loader() -> MedGemmaLoader:
    return MedGemmaLoader(get_settings())


@lru_cache
def get_medical_llm_service() -> MedicalLlmService:
    settings = get_settings()
    return MedicalLlmService(settings, get_medgemma_loader())
