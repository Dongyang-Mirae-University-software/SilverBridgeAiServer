"""챗봇 판정 낱말의 단일 출처(2026-10-31 점검 H-1·M-1·M-3).

생성 경로·생성 실패 대체 응답·되묻기·진료과 판단·이어서 물어볼 질문이 모두 이 모듈의 객체만 참조한다.
목록을 다른 파일에 다시 적지 말 것 - 목록이 갈라져 대체 응답만 위험 신호를 놓쳤다(H-1).
증상 단어 목록은 목적별로 범위가 달라 합치지 않았다(약 문의 C-2·병원 권유 D-4·예약 요약·되묻기). 합집합이 필요한 곳은
ANY_SYMPTOM_WORDS 를 쓴다.
"""
from __future__ import annotations

import re


def compact(text: str) -> str:
    """공백을 뺀 소문자 문장 - 응급·자해·위험 신호·합집합 증상 단어 판정 기준."""
    return re.sub(r"\s+", "", (text or "").lower())


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
    # 반응·호흡 없음(심정지 의심, 2026-11-02): "쓰러" 없이 "남편이 숨을 안 쉬어요"만 써도 라우터에 기대지 않게.
    # 본인 호흡곤란("숨이 안 쉬어져요")은 위의 "숨이안쉬"가 잡는다. "자다가 숨을 멈췄다가"(수면 무호흡)는 넣지 않는다.
    "숨을안쉬", "숨도안쉬", "숨을쉬지않", "숨을안쉼", "호흡이없", "숨이멎",
    "불러도대답", "불러도대꾸", "흔들어도반응", "흔들어도대답", "흔들어도안깨", "흔들어도눈을", "깨워도안깨", "깨워도반응", "꼬집어도반응", "눈을안떠",
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

# 응급 키워드에 걸리지만 응급이 아닌 말(2026-11-02 홀드아웃 v1 M08: "알약이 커서 목에 걸려요" → 질식 "목에걸려").
# 약 제형 낱말이 있고 숨·말·얼굴색·침 삼킴 이상이 없고 음식 낱말이 없을 때만 "목에걸려" 신호를 뺀다.
# 음식이 걸린 질식("떡이 목에 걸려")·침을 못 삼키거나 흘리는 말은 그대로 응급이다(2026-11-03 점검 H-1).
# 약이 걸려 숨을 못 쉬면 "숨을못쉬"·"숨이막" 같은 다른 응급 키워드도 잡는다.
_PILL_FORM = re.compile(r"알약|약이|약을|약알|캡슐|정제|서방정|장용정|가루약|한알|두알")
_AIRWAY_DISTRESS = re.compile(
    r"숨|호흡|말을못|말이안|소리가안|목소리가안|얼굴이|입술|파래|파랗|컥컥|켁켁|캑캑|꺽꺽|"
    r"침(을|도|이)?(못|안)삼|침(을|도|이)?(못|안)넘|침을?흘|침이?흘|못삼키|못삼켜|삼키지못|삼킬수가없|안넘어가|넘어가지않|넘기지못"
)
# 목에 걸리는 음식만(한 글자 "면"·"밥"은 "먹으면"·"밥 먹고 약"에 걸려 넣지 않는다).
_CHOKING_FOOD = re.compile(r"떡|사탕|고기|생선|가시|과자|견과|땅콩|빵|젤리|뼈|음식이|음식물")
# 비유 표현(2026-11-03 점검 M-1): "숨이 멎을 만큼 놀랐어요"는 응급·심정지 신호가 아니다. 지운 뒤 판정한다.
_FIGURATIVE = re.compile(r"숨이멎(을|는|을것|을거)(만큼|듯|것같|뻔|줄|정도|같)")


def _plain(text: str) -> str:
    """판정용 공백 뺀 문장에서 비유 표현을 지운 것."""
    return _FIGURATIVE.sub(" ", compact(text))


# 라우터 실패 때만(fallback_emergency_signals) 보는 응급 표현(2026-11-06 라우터 실패 점검 H-1 - 시험지 응급 80개 중 21개가 키워드 경로에서
# 응급이 아니었다: 사이에 낱말이 끼거나("가슴을 누가 쥐어짜") 활용형이 다르거나("거품 물고"·"목에 걸림") 수치("혈당 40")).
# 라우터가 실패하면 응급 판단은 이 모듈뿐이다. 유형 단위의 짧은 거리 패턴만 둔다(문장 맞춤 금지). (이름, 패턴) - 공백 뺀 문장.
_GAP = r"[^.!?。]"
EMERGENCY_PATTERNS = (
    ("흉통:쥐어짜", re.compile(rf"가슴{_GAP}{{0,8}}쥐어짜")),
    ("무반응:깨워도", re.compile(rf"(깨워도|깨우는데|흔들어도|불러도|꼬집어도){_GAP}{{0,8}}((안|못)일어(?!나는|나서)|그대로|미동|가만히|꿈쩍)")),
    ("무반응:늘어짐", re.compile(r"축늘어|미동도없|미동도안")),
    ("무호흡:숨소리", re.compile(rf"숨소리가?{_GAP}{{0,4}}안들")),
    ("무호흡:가슴", re.compile(rf"가슴이?{_GAP}{{0,4}}(안움직|오르락내리락을?안)")),
    ("청색증", re.compile(rf"입술이{_GAP}{{0,3}}(파래|파랗|퍼래|퍼렇|새파)")),
    ("호흡곤란", re.compile(r"숨이(잘|좀|제대로)안쉬")),
    ("아나필락시스", re.compile(rf"(혀|입술|목구멍|얼굴){_GAP}{{0,8}}(붓|부어|부었){_GAP}{{0,12}}(숨|호흡)")),
    ("출혈", re.compile(rf"피(가|를)?{_GAP}{{0,6}}(안멈|멈추지않|멈추질않)")),
    ("토혈", re.compile(rf"피를{_GAP}{{0,8}}(토하|토했|토해)")),
    ("뇌졸중:입", re.compile(rf"입이{_GAP}{{0,10}}돌아(갔|가)")),
    ("뇌졸중:말", re.compile(rf"(말|발음)(이|도){_GAP}{{0,5}}어눌|(말|발음)도(좀|자꾸)?꼬(이|여)|말이잘안나")),
    # 한쪽 힘 빠짐은 "갑자기"가 있거나 같은 쪽 팔·다리가 함께일 때만("무릎 수술 후 오른쪽 다리에 힘이 없어요"는 아니다).
    ("뇌졸중:한쪽", re.compile(rf"갑자기{_GAP}{{0,12}}(오른|왼)쪽?(팔|다리){_GAP}{{0,15}}힘이{_GAP}{{0,3}}(빠지|빠져|빠졌|없어|없고)|(오른|왼)쪽?팔{_GAP}{{0,10}}(오른|왼)?쪽?다리{_GAP}{{0,8}}힘이{_GAP}{{0,3}}(빠지|빠져|빠졌|없어|없고)")),
    ("경련", re.compile(r"거품(을)?물")),
    ("의식변화", re.compile(rf"갑자기{_GAP}{{0,20}}(못알아|헛소리|횡설수설|눈이풀)")),
    ("실신전조", re.compile(rf"갑자기{_GAP}{{0,10}}(식은땀|눈앞이캄캄|눈앞이깜깜)")),
)
# 저혈당 수치(혈당 70 미만) - "혈당이 40밖에", "혈당 55". 세 자리(혈당 300)는 아니다.
# 수치만 말한 경우("혈당이 50 정도 나와요")는 단 음식 안내 상담이다 - 떨림·식은땀·멍함 같은 증상이 함께일 때만 응급.
_LOW_GLUCOSE_SYMPTOM = re.compile(r"떨|식은땀|멍|어지|정신|의식|쓰러|이상해|기운이없|힘이없|횡설")
_LOW_GLUCOSE = re.compile(r"혈당(이|가|수치가|수치|이요)?(\d{1,2})(?!\d)")
# 질식 낱말 활용형("목에 걸림"·"목에 걸렸어") - 알약 예외는 "목에걸려"와 같게 본다.
_CHOKE_FORMS = ("목에걸려", "목에걸림", "목에걸렸")
# 미리 걱정하는 말(2026-11-06 점검 M-1 - "넘어지면 못 일어나실까봐 걱정"이 응급이 됐다): 이 키워드 바로 뒤가 가정·걱정 꼴이면 뺀다.
_ALMOST_AFTER = re.compile(r"^[가-힣]{0,3}뻔")
_WORRY_SENSITIVE = ("못일어나", "쓰러지", "쓰러져", "쓰러짐")
_WORRY_AFTER = re.compile(r"^[가-힣]{0,3}(까봐|까싶|까걱정|면어떡|면어쩌|면어떻|지않게|지않도록|ㄹ때|면뭘)")


def emergency_signals(text: str) -> list[str]:
    """문장에 든 응급 키워드(공백 뺀 소문자 문장에 부분 일치). 비어 있지 않으면 응급이다. 라우터 성공·실패 모두 쓴다."""
    c = _plain(text)
    hits = [keyword for keyword in EMERGENCY_KEYWORDS if keyword in c]
    for keyword in _WORRY_SENSITIVE:
        if keyword in hits and all(_WORRY_AFTER.search(c[m.end():]) for m in re.finditer(keyword, c)):
            hits.remove(keyword)
    if "목에걸려" in hits and _PILL_FORM.search(c) and not _AIRWAY_DISTRESS.search(c) and not _CHOKING_FOOD.search(c):
        hits.remove("목에걸려")
    return hits


def fallback_emergency_signals(text: str) -> list[str]:
    """라우터가 실패했을 때만 더 보는 응급 신호(거리 패턴·저혈당 수치 + 증상·질식 활용형). 라우터가 정상이면 쓰지 않는다 -
    라우터의 응급 판단·위기 해제·되묻기 경계는 그대로 두고, 라우터가 없을 때 놓치던 응급만 잡는다(2026-11-06 점검 H-1)."""
    c = _plain(text)
    hits = [name for name, pattern in EMERGENCY_PATTERNS
            if any(not _ALMOST_AFTER.search(c[m.end():]) for m in pattern.finditer(c))]
    if _LOW_GLUCOSE_SYMPTOM.search(c) and any(int(m.group(2)) < 70 for m in _LOW_GLUCOSE.finditer(c)):
        hits.append("저혈당:수치")
    choke = [k for k in _CHOKE_FORMS[1:] if k in c]
    if choke and not (_PILL_FORM.search(c) and not _AIRWAY_DISTRESS.search(c) and not _CHOKING_FOOD.search(c)):
        hits += choke
    return hits


# 라우터가 실패했을 때만 보는 애매한 자해 표현(2026-11-09, 시험지 sh.09). 라우터가 있으면 self_harm 칸이 맡던 몫이라
# SELF_HARM_KEYWORDS(모든 경로에서 라우터를 건너뛰는 목록)에는 넣지 않는다 - 라우터가 "일·숙제를 끝내고 싶다"를 가려 준다.
# 판정 문장은 공백·문장 부호·이모지·자모(ㅠㅜ)를 지운 꼴이다.
# A: 삶·인생·목숨을 끝내고 싶다(문장 어디든). 바로 뒤가 부정("~싶지 않아요"·"~싶다는 생각은 없어요")·걱정 꼴("~싶어하실까봐")이면
#    빼고(#193 점검 M-1), 과거 경험("~싶었던 적이 있어요")은 위기 신호로 잡는다.
# B: 문장 전체가 "(감정 이유)(그냥·이제…) 다·전부·모든 걸 끝내(버리)고 싶어"뿐이거나, 가족이 그렇게 말했다는 전달뿐 - 끝낼 대상
#    낱말이 붙으면("숙제 다 끝내고 싶어"·"빨리 다 끝내고 싶어"·"일이 힘들어서 다 끝내고 싶어"·"다 끝내고 쉬고 싶어"·"다 끝내고 싶었는데")
#    걸리지 않는다. 감정 이유·가족 전달·사투리 어미·오타 "끝네"는 #193 점검 L-1.
_FALLBACK_SELF_HARM_STRIP = re.compile(r"[^가-힣a-z0-9]")
_FALLBACK_SELF_HARM_LIFE = re.compile(r"(삶|인생|목숨)(을|를|도|이|은)?(그만|다|이제|이만)?끝(내|네)(고|버리고|뿌고|삐고)싶")
_FALLBACK_SELF_HARM_NOT = re.compile(r"^(지|진|지는|지도|지만은|진않|진안)(않|안)|^[가-힣]{0,6}(생각|마음)(은|도|이|까지)?(전혀|절대|별로)?(없|안|아니)|^(어하실|어하시|어할|어하|을|으실)?(까봐|까걱정|까싶)")
_FALLBACK_SELF_HARM_CORE = (
    r"(그냥|이제|정말|진짜|차라리|너무|이대로|그만|이만|요즘)*"
    r"(지쳐서|힘들어서|괴로워서|외로워서|지쳐|힘들어|괴로워|외로워)?"
    r"(그냥|이제|정말|진짜|차라리|너무|이대로|그만|이만)*"
    r"(다|전부|모두|모든걸|모든것을|모든것|모든게|모든거)(그냥|이제|그만)?끝(내|네)(고|버리고|뿌고|삐고)"
)
_FALLBACK_SELF_HARM_BARE = re.compile(
    r"(나|난|저|전|내가|제가)?" + _FALLBACK_SELF_HARM_CORE
    # "싶은데요"·"싶다고요"·"싶다는 생각이 들어요"(2026-11-10 전체 점검 H-4 - 말끝만 달라도 놓쳤다). 지난 일("싶었는데")은 넣지 않는다.
    + r"싶(어|다|어요|네|네요|습니다|음|구먼|구만|구나|당께|다께|당|데이|어예|어유|어라|은데|은데요|다고|다고요|"
    r"다는생각(이|만|도)?(들어|들어요|듭니다|드네|드네요|나|나요|납니다))?"
)
_FALLBACK_SELF_HARM_FAMILY = re.compile(
    r"(엄마|아빠|어머니|아버지|어머님|아버님|할머니|할아버지|할매|할배|부모님|남편|아내|와이프|아들|딸|며느리|사위|손자|손녀|"
    r"어르신|형|누나|언니|오빠|동생)(가|이|께서|는|은)?" + _FALLBACK_SELF_HARM_CORE
    + r"(싶다고|싶대|싶다셔|싶으시대|싶으시다고|싶어해|싶어하)(요|해요|하세요|하셔요|하시네요|하네요|하십니다|하셨어요|했어요|그래요|그러세요|"
    r"세요|셔요|시네요|십니다|셨어요|네요)?"
)


# C: 죽음·사라짐의 방법을 묻는 꼴(2026-11-11 기능 점검 F-3) - "어떻게 하면 (안 아프게·조용히) 죽을/사라질/안 깰 수 있나",
#    "어떻게 하면 편하게 갈 수 있을까", "(다) 끝내려면 어떻게". 증상 말("아파 죽겠어요"·"어떻게 하면 통증이 사라질까")은 잡지 않는다 -
#    "어떻게 하면" 바로 뒤(부사 하나까지)에 죽음 낱말이 와야 한다.
_FALLBACK_SELF_HARM_HOW = re.compile(
    r"(어떻게|어찌|어떡)(하면|해야|하나|할까)?(그냥|조용히|빨리|쉽게|편하게|안아프게|고통없이)?"
    r"(죽(을|는|어|으려|으면|게|일)(?!맛|끓|쑤|드|먹|만들|데우|식|담)|사라(질|지|져)|생을마감|목숨을?끊|잠들고안(깨|깰)|안(깨|깰)수)"
    r"|(어떻게|어찌)(하면|해야)편하게(갈|가|떠날|떠나)"
    r"|(안아프게|고통없이|조용히|편하게)죽(을|는|어|으려|게)|죽는방법"
    r"|^(나|난|저|전|그냥|이제)*(다|전부|모두|모든걸)?끝내(려면|려고)(어떻게|어떡)"
)


def fallback_self_harm_signals(text: str) -> list[str]:
    """라우터가 실패했을 때만 더 보는 자해 신호(2026-11-09). 라우터가 정상이면 쓰지 않는다(라우터 self_harm 칸이 맡는다)."""
    c = _FALLBACK_SELF_HARM_STRIP.sub("", compact(text))
    hits = []
    if any(not _FALLBACK_SELF_HARM_NOT.search(c[m.end():]) for m in _FALLBACK_SELF_HARM_LIFE.finditer(c)):
        hits.append("자해:삶을끝내")
    if _FALLBACK_SELF_HARM_BARE.fullmatch(c) or _FALLBACK_SELF_HARM_FAMILY.fullmatch(c):
        hits.append("자해:다끝내고싶")
    if _FALLBACK_SELF_HARM_HOW.search(c):
        hits.append("자해:방법")
    return hits


# 라우터 실패 때 심정지 의심(가슴압박 줄, 2026-11-08): 라우터가 있으면 cardiac_arrest 칸이 맡던 몫. 위 거리 패턴 중 반응·호흡 없음과
# 줄여 쓴 꼴("숨 없음")·"흔들어도 꼼짝 안 해요"만 더 본다. "몸이 축 처져요"는 피로 표현과 겹쳐 넣지 않는다.
# 조사를 뺀 꼴("숨 안 쉬어요"·"숨 안쉼"·"호흡 없어요", 2026-11-09 점검 H-1)도 본다. 본인 호흡곤란("숨 안 쉬어져요")은 뺀다.
# 목격자의 추측("숨 안 쉬는 것 같아요")은 붙인다(2026-11-09 사용자 결정 - 확신이 없어도 가슴압박을 안내하는 기준과 같게).
_FALLBACK_ARREST_PREFIX = ("무반응:", "무호흡:")
_FALLBACK_ARREST_EXTRA = re.compile(
    rf"숨(이|도)?없(음|어|다|고|네)|호흡(이|도)?없|숨안쉬(?!어지|어져)|숨안쉼|(흔들어도|불러도|깨워도){_GAP}{{0,8}}꼼짝"
)
# 지난 일·곧 회복된 일("꼼짝 안 하던 때가"·"꼼짝 안 하다가 깼어요")·본인 느낌("숨이 없어질 것 같이", 2026-11-09 점검 M-1).
_FALLBACK_ARREST_NOT_NOW = re.compile(r"^(도|을|를)?(안|못)?하(던|더니|다가)|^질|^어질")


def fallback_arrest_signal(text: str) -> bool:
    """라우터가 실패한 턴에 가슴압박 줄을 붙일 근거. 의식이 있다는 말(arrest_conscious)이 있으면 False."""
    c = _plain(text)
    hit = any(name.startswith(_FALLBACK_ARREST_PREFIX) for name in fallback_emergency_signals(text)) or any(
        not _ARREST_NOT_NOW.search(c[m.end():m.end() + 8]) and not _FALLBACK_ARREST_NOT_NOW.search(c[m.end():m.end() + 8])
        for m in _FALLBACK_ARREST_EXTRA.finditer(c)
    )
    return hit and not arrest_conscious(compact(text))


# 의료·예약과 무관한 질문(2026-11-08, 고정 안내). 주제 낱말 + 묻거나 부탁하는 꼴이 함께이고 건강 낱말이 하나도 없을 때만.
# 애매하면 False(지금처럼 MedGemma) - 응급을 엉뚱한 질문으로 막는 것이 더 위험하다. 맞장구("날씨가 참 좋네요")는 묻는 꼴이 아니라 제외.
_OFF_TOPIC_SUBJECT = re.compile(
    r"날씨|기온|미세먼지|주식|주가|코인|비트코인|환율|부동산|아파트값|정치|대통령|선거|국회|"
    r"야구|축구|농구|배구|올림픽|경기결과|몇대몇|연예인|아이돌|드라마|영화추천|노래추천|가사|"
    r"로또|복권|레시피|요리법|맛집|여행지|관광지|번역|영어로|일본어로|수학|숙제|코딩|프로그래밍|파이썬|"
    r"게임|퀴즈|끝말잇기|운세|사주|타로|별자리|궁합"
)
_OFF_TOPIC_ASK = re.compile(r"알려|어때|어떄|추천|뭐야|뭐예요|뭐에요|뭔가요|누구|누가|어디|언제|할까|살까|팔까|해줘|해주세요|해줄래|줘요|줘|\?|나요|까요|인가요|궁금|맞아|맞나")
# 건강·몸·생활 관리 낱말 - 하나라도 있으면 상담 쪽이다("날씨 추운데 산책해도 돼?"·"당뇨인데 김치찌개 먹어도 돼?").
_OFF_TOPIC_HEALTH = re.compile(
    r"몸|건강|아프|아파|아픈|통증|쑤|저리|결리|약|병원|의사|간호|진료|예약|혈압|혈당|당뇨|콜레스테롤|운동|산책|걷|"
    r"먹어도|먹으면|드셔도|드시면|마셔도|마시면|괜찮|위험|숨|어지|열이|기침|가래|감기|독감|잠|수면|치매|기억|"
    r"넘어|낙상|다쳐|다친|증상|검사|수술|입원|응급|119|109|체중|살이|살쪄|살빠|소화|배가|배를|머리|가슴|허리|무릎|관절|"
    r"눈이|눈물|귀가|피부|가렵|간지|붓|부어|토하|설사|변비|소변|대변|피가|심장|폐|뼈|우울|불안|외로|죽|자살|힘들|무서|"
    # 의식·반응·위기 의심·다친 곳·피로(2026-11-09 점검 C-1 - "드라마 보다 정신 잃음 어떡해?"가 고정 안내로 막혔다). 오타 "무릅" 포함.
    r"정신|의식|기절|쓰러|꼼짝|반응|안깨|깨워|깨어|졸았|졸다|경련|발작|끝내고|끝내버|사라지고|살기싫|"
    r"발목|손목|팔이|팔을|팔에|다리|삐끗|삐었|무릅|시리|시린|지쳐|지치|피곤"
)
# 가족·옆 사람 호칭 - 그 사람의 일을 묻는 말은 상담일 수 있어 고정 안내로 보내지 않는다(2026-11-09 점검 C-1, 애매하면 MedGemma).
_OFF_TOPIC_PERSON = re.compile(
    r"엄마|아빠|어머니|아버지|어머님|아버님|할머니|할아버지|할매|할배|남편|아내|와이프|손자|손녀|아들|며느리|사위|어르신|회원분"
)


def is_off_topic_question(text: str) -> bool:
    """의료·예약과 무관한 질문인가(고정 안내 판정). 건강 낱말·증상 낱말·응급·예약 낱말이 하나라도 있으면 False."""
    c = compact(text)
    if not c or not _OFF_TOPIC_SUBJECT.search(c) or not _OFF_TOPIC_ASK.search(c):
        return False
    # 라우터가 실패한 턴의 응급 판정(거리 패턴·저혈당 수치)과 반응·호흡 없음도 보고 하나라도 있으면 상담으로(2026-11-09 점검 C-1).
    if _OFF_TOPIC_PERSON.search(c) or fallback_emergency_signals(text) or ARREST_SIGNAL.search(c):
        return False
    if _OFF_TOPIC_HEALTH.search(c) or has_symptom_word(text) or has_red_flag(text) or emergency_signals(text):
        return False
    return not has_reservation_word(text) and not is_health_lifestyle_question(text)


# 심정지 의심 신호(응급 행동 줄 cardiac_arrest, 2026-11-02). 반응이 없거나 숨을 쉬지 않는다는 말 - 본인 호흡곤란
# ("숨이 안 쉬어져요"·"숨을 못 쉬겠어요")은 앉은 자세 줄(breathing)로 두고 여기에 넣지 않는다. 공백 뺀 문장.
_ARREST_WORDS = re.compile(
    r"의식이?없|의식을잃|정신을잃|깨어나지|불러도(반응|대답|대꾸)|흔들어도(반응|대답|안깨|눈을)|깨워도(안깨|반응)|꼬집어도반응|눈을안떠|"
    r"숨을안쉬(?!어지|어져)|숨도안쉬|숨을쉬지않|숨을안쉼|호흡이없|숨이멎"
)
# 의식이 있는 사람의 추측·회복 표현(2026-11-03 점검 M-2): "정신을 잃을 것 같아요"·"잃었다가 금방 깼어요"·
# "의식이 없어질 것처럼"은 가슴압박 줄 대상이 아니다(응급 키워드 판정은 그대로 - 쓰러짐·어지럼 줄로 간다).
_ARREST_NOT_NOW = re.compile(r"^(을|ㄹ)?(것같|것처럼|듯|뻔|까봐|까요)|^을(것|거|까|듯|뻔)|^어질|^어지려|^었다가|^다가|^던")
# "안 깼어요"·"못 깨어났어요"는 회복이 아니다.
# "지금은 괜찮아요"·"지금은 멀쩡해요"도 회복이다(2026-11-09 점검 M-1). "지금은 괜찮을까요"(묻는 꼴)는 넣지 않는다.
_ARREST_RECOVERED = re.compile(
    r"(?<![안못])(깼|깨셨|깨어났|깨어나셨|깨났)|(정신|의식)이?(돌아왔|돌아오셨|들었|드셨|들어요|드세요)|"
    r"지금은(괜찮아요|괜찮아졌|괜찮습니다|멀쩡해|멀쩡하)"
)
# 지금 의식이 있다는 말(2026-11-05 점검 M-1): "말은 해요"·"대답은 하세요"·"숨은 쉬어요". 눈을 뜬 것은 넣지 않는다 -
# 심정지·발작에서도 눈은 뜨고 있을 수 있다("눈은 뜨고 있는데 초점이 없어요"는 심정지 문장이다). 부정("말을 못 해요"·
# "숨을 안 쉬어요"·"움직이지 않아요")은 낱말 사이에 안·못·않이 끼어 걸리지 않게 붙여 쓴 꼴만 본다(공백 뺀 문장).
_ARREST_AWAKE = re.compile(
    r"(말|말씀|대답)(은|을|도|이)?(잘)?(해요|하세요|하셔|하시|해서|하는데|합니다|한다|하고있)|"
    # "숨 잘 쉬세요"(2026-11-09 점검 M-1). "안 움직여요"·"못 움직이세요"는 의식 있음이 아니다(2026-11-09 점검 H-2 - 가슴압박 줄이 빠졌다).
    r"숨(은|을|도)?(잘)?(쉬어요|쉬세요|쉬셔|쉬고있|쉬는데|쉰다)|(?<![안못])움직(여요|이세요|이셔)|"
    r"(정신|의식)(은|이)(있어요|있으세요|있으셔|있는데|또렷)"
)


_ARREST_PAST = re.compile(r"었|았|셨|더니")
# "의식이 없어지셨어요"·"숨이 멎어 버리셨어요"는 지난 일이 아니라 지금 상태다(상태가 바뀐 꼴, 2026-11-10 2차 점검 P-1).
_ARREST_STATE_CHANGE = re.compile(r"^(이|을|도)?(없)?(어|아)(지|져|버리|버렸)")
# 회복 뒤에 이 말이 끼면 그 뒤의 반응·호흡 없음은 시제와 관계없이 다시 나빠진 것이다("깼는데 다시 숨을 안 쉬었어요").
_ARREST_AGAIN = re.compile(r"다시|또|도로|이제는|지금은|방금")


def _arrest_past(c: str, m) -> bool:
    """반응·호흡이 없다는 말(m)이 지난 일로 적혔는가."""
    tail = c[m.end():m.end() + 8]
    if _ARREST_STATE_CHANGE.match(tail):
        return False
    return bool(_ARREST_PAST.search(tail[:6]))


# 두 사람이 섞인 문장(2026-11-10 전체 점검 M-9): "아버지는 의식이 없어요. 어머니는 숨은 쉬세요"의 "숨은 쉬세요"는 다른 사람 이야기다.
# 의식이 있다는 말 바로 앞의 사람과 반응·호흡이 없다는 말 바로 앞의 사람이 둘 다 적혀 있고 서로 다를 때만 그 말을 빼고 본다.
# 한쪽이라도 사람이 적혀 있지 않으면 지금처럼 같은 사람으로 본다("의식이 없어요. 숨은 쉬세요" - 숨을 쉬면 가슴압박 대상이 아니다).
# 신고하는 본인("저는 숨은 쉬어요")과 "환자분"도 사람으로 본다(2차 점검 P-3 - 가장 흔한 "환자 + 신고자" 꼴이 빠져 있었다).
# 공백을 뺀 문장이라 "나는"("열이 나는데")·"제가"("문제가")는 낱말 속에 걸려 넣지 않는다.
_ARREST_PERSON = re.compile(
    r"(엄마|아빠|어머니|아버지|어머님|아버님|할머니|할아버지|할매|할배|부모님|남편|아내|와이프|아들|딸|며느리|사위|손자|손녀|누나|언니|오빠|동생|"
    r"어르신|환자분|환자|저)(께서는|께서|는|은|가|이|도)"
)


def _person_before(c: str, pos: int) -> str | None:
    last = None
    for m in _ARREST_PERSON.finditer(c, 0, pos):
        last = m.group(1)
    return last


def _awake(c: str) -> bool:
    """지금 의식이 있다는 말이 (반응·호흡이 없다는) 그 사람 이야기로 있는가."""
    live = [m for m in _ARREST_WORDS.finditer(c)
            if not _ARREST_NOT_NOW.search(c[m.end():m.end() + 8]) and not _arrest_past(c, m)]
    patients = {p for p in (_person_before(c, m.start()) for m in live) if p}
    for a in _ARREST_AWAKE.finditer(c):
        who = _person_before(c, a.start())
        if who and patients and who not in patients:
            continue  # 다른 사람이 숨을 쉰다·말을 한다는 말
        return True
    return False


def _recovered(c: str) -> bool:
    """회복했다는 말이 있고, 그 뒤에 다시 반응·호흡이 없다는 말이 없는가(2026-11-10 전체 점검 H-3).
    "잃었다가 금방 깼어요"는 회복, "깼다가 다시 의식이 없어요"·"아까 깨셨는데 지금 숨을 안 쉬어요"는 회복이 아니다."""
    for r in _ARREST_RECOVERED.finditer(c):
        # 뒤에 나온 말이 지난 일("지금은 멀쩡해요 숨을 안 쉬었었대요")이면 다시 나빠진 것이 아니다.
        again = any(
            bool(_ARREST_AGAIN.search(c[r.end():m.start()]))
            or (not _ARREST_NOT_NOW.search(c[m.end():m.end() + 8]) and not _arrest_past(c, m))
            for m in _ARREST_WORDS.finditer(c, r.end())
        )
        if not again:
            return True
    return False


class _ArrestSignal:
    """ARREST_SIGNAL.search(공백 뺀 문장) - 지금 반응·호흡이 없다는 말이 하나라도 있으면 그 매치를 준다."""

    def search(self, text: str):
        c = _FIGURATIVE.sub(" ", text or "")
        if _recovered(c) or _awake(c):
            return None
        for m in _ARREST_WORDS.finditer(c):
            if not _ARREST_NOT_NOW.search(c[m.end():m.end() + 8]):
                return m
        return None


ARREST_SIGNAL = _ArrestSignal()


def arrest_conscious(text: str) -> bool:
    """의식이 있다는 말(추측·회복·지금 의식 있음)이 있는가 - 라우터가 심정지로 봐도 가슴압박 줄을 붙이지 않는 근거(2026-11-04).
    공백 뺀 문장을 받는다."""
    c = _FIGURATIVE.sub(" ", text or "")
    if _recovered(c) or _awake(c):
        return True
    return any(_ARREST_NOT_NOW.search(c[m.end():m.end() + 8]) for m in _ARREST_WORDS.finditer(c))


# 오늘 진료가 필요한 경계 질환(라우터 urgent_condition 칸의 허용 값, 2026-11-04 사용자 승인 - 고정 문장 예외).
# 라우터는 이 중 하나 또는 ""만 낼 수 있다(스키마 enum). 문장은 medical_llm_service._URGENT_CONDITIONS 에 있다.
# 값을 더하면 라우터 지시문(chat_router_service)·고정 문장 표·테스트를 함께 바꾼다.
# 갑자기 생긴 혼란(섬망)은 넣지 않는다 - NHS 가 바로 999 로 분류하고, 넣었더니 라우터가 응급 대신 이 칸을 골랐다(2026-11-04 v3 판정).
URGENT_CONDITIONS: dict[str, str] = {
    "dvt": "한쪽 다리(주로 종아리)만 붓고 아프거나 색이 변함 - 다리 혈전",
    "cellulitis": "피부 한 곳이 붉게 붓고 뜨겁고 아프며 퍼짐 - 봉와직염",
    "kidney_infection": "열·오한과 함께 옆구리나 허리 한쪽 통증, 소변 이상 - 신우신염",
    "shingles_eye": "이마·눈 주변 한쪽에 물집·띠 모양 통증 - 눈 주변 대상포진",
    "giant_cell_arteritis": "50세 이상의 새 관자놀이 두통·머리 피부 통증·씹을 때 턱 통증 - 거대세포동맥염",
    "head_injury_anticoagulant": "피를 묽게 하는 약(와파린 등)을 먹는 중 머리를 부딪힘",
    "acute_glaucoma": "갑자기 한쪽 눈이 아프고 충혈·흐려짐, 불빛 주위 무지개 - 급성 녹내장",
    "urinary_retention": "소변이 마려운데 전혀 나오지 않음 - 급성 요폐",
}

# 라우터 실패 때 키워드 분류가 약 질문을 상담으로 보게 하는 약 제형 낱말(2026-11-03 점검 L-1). 판정 낱말 목록(MEDICAL_KEYWORDS)은
# 위기 이어짐·예약 재개 판정이 함께 쓰므로 넓히지 않고, 분류(classify_intent)에서만 더 본다.
MEDICATION_FORM_WORDS = ("알약", "서방정", "장용정", "캡슐", "가루약", "물약", "패치", "좌약")

# 위험 신호 단어 - 응급 키워드는 아니지만 119 안내가 필요한 말. 공백을 뺀 문장에 부분 일치시킨다(M-1: "숨차요"·"숨 차요"를 놓쳤다).
# 생성 경로(되묻기 제외·119 한 줄)와 생성 실패 대체 응답(high)이 같은 목록을 쓴다. 응급 키워드로 올리지 않는다("숨이 차요" 경계 결정 유지).
RED_FLAG_KEYWORDS = ("가슴", "숨이", "호흡", "의식", "실신", "쓰러", "출혈", "피가", "마비", "경련", "파래", "식은땀", "숨차")

# 예약 의도(라우터 실패 때 키워드 분류). 원문에 부분 일치.
RESERVATION_KEYWORDS = ("예약", "병원 찾", "근처 병원", "진료 가능", "응급실", "잡아", "접수", "병원 좀", "병원 추천", "진료 받")
# 예약 흐름 안에서 예약 이야기를 이어 가는 말인지(위의 의도 분류보다 넓다 - "병원"·"진료"만 있어도 된다).
RESERVATION_CONTEXT_WORDS = ("예약", "병원", "진료", "접수", "내 예약", "예약 내역", "예약 조회", "예약 목록")

# 상담(의료) 의도. 원문에 부분 일치 - 위기 이어짐·예약 재개 판정도 이 목록·방식을 그대로 쓴다(바꾸면 위기 경로가 달라진다).
MEDICAL_KEYWORDS = (
    "아파", "아프", "통증", "열", "기침", "두통", "어지럽", "어지러", "어지럼", "구토", "충혈", "메스꺼", "설사", "변비",
    "더부룩", "속이", "소화", "체했", "무릎", "허리", "관절", "시려", "저리", "붓", "부었", "가려", "피부", "발진",
    "당뇨", "혈압", "혈당", "콜레스테롤", "불면", "잠이", "수면", "약을", "약 먹", "복용", "숨이", "가슴", "심장",
    "기운", "피곤", "식욕", "체중", "눈이", "귀가", "목이", "코가",
)

# 증상 이름 - 서로 다른 것이 둘 이상이면 동반 증상을 이미 말한 것으로 본다.
# "가려움"은 "피부"와 같은 증상을 두 번 세게 되어 넣지 않는다(2026-10-29 v7 fq.08 "피부가 가려워요"가 동반 증상으로 잡혔다) - "가려워요"만 말해도 피부과 힌트로 되묻는다.
FOLLOWUP_SYMPTOM_NOUN = re.compile(
    r"기침|열|두통|머리|어지|배|속|구토|토할|설사|허리|무릎|어깨|관절|눈|귀|코|목|피부|저리|저려|소변|변비|가슴|숨|콧물|피곤|잠|이가|이빨|손|발|다리|팔"
)

# 약 문의 C-2 판정 증상 단어(공백 뺀 문장).
MEDICATION_SYMPTOM_WORDS = (
    "열이", "열나", "발열", "아파", "아프", "통증", "기침", "어지", "두통", "구토", "설사", "저리", "저려", "가려",
    "붓", "부었", "숨이", "가슴", "메스꺼", "콧물",
)
# D-4 에서 "증상을 말한 상담"으로 볼 단어. 위 목록(약 문의 C-2 판정)은 넓히지 않는다.
GENERAL_SYMPTOM_WORDS = MEDICATION_SYMPTOM_WORDS + (
    "소변", "쥐가", "피곤", "잠이안", "잠을못", "변비", "변을", "침침", "부어", "쑤시", "결리", "쓰려", "답답", "더부룩", "시려",
)

# MedicalLlmService.MEDICAL_KEYWORDS 기준. 한 글자 "열"은 "열 시"와 겹쳐 "열이"·"열나"로 좁히고, 진료과 추론에 쓰는 부위 표현을 더했다.
SUMMARY_SYMPTOM_WORDS = (
    "아파", "아프", "통증", "열이", "열나", "기침", "두통", "어지럽", "어지러", "어지럼", "구토", "충혈", "메스꺼", "설사", "변비",
    "더부룩", "속이", "소화", "체했", "무릎", "허리", "관절", "시려", "저리", "붓", "부었", "가려", "피부", "발진",
    "당뇨", "혈압", "혈당", "콜레스테롤", "불면", "잠이", "수면", "숨이", "가슴", "심장", "기운", "피곤", "식욕", "체중",
    "눈이", "귀가", "목이", "코가", "콧물", "편도", "두드러기", "시력", "감기", "결려", "쑤셔", "따가", "울렁",
)

# 증상 단어 합집합 - 생성 실패 대체 응답이 "안정 상태" 문구를 쓰지 않을 문장, 라우터 실패 때 상담 의도(H-1). 공백 뺀 문장에 부분 일치.
ANY_SYMPTOM_WORDS = tuple(dict.fromkeys(
    w.replace(" ", "") for w in MEDICAL_KEYWORDS + GENERAL_SYMPTOM_WORDS + SUMMARY_SYMPTOM_WORDS + ("복통",)
))


# 위험 신호 낱말에 걸리지만 위험 신호가 아닌 말(2026-11-01 점검 M-2: "커피 가게" → "피가"). 지운 뒤에 본다 -
# "커피가 쏟아져 피가 나요"의 "피가"는 그대로 잡힌다. "코피가"는 위험 신호로 둔다.
_RED_FLAG_FALSE = re.compile(r"커피가")


# 건강을 지키는 생활 관리·집 안전 질문(2026-11-04 홀드아웃 v1 L05 - 가족이 "집에 안 넘어지게 뭘 손봐 드리면" 하고 물으면
# 라우터가 증상 단어가 없다고 4회 중 2회 general 로 보내 한두 문장만 나갔다). 주제 낱말과 묻는 말이 함께 있을 때만 상담으로 올린다 -
# "산책 다녀왔어요"·"식사하셨어요?" 같은 잡담은 그대로 둔다. MEDICAL_KEYWORDS(위기 이어짐 판정과 공유)는 넓히지 않는다.
# 2026-11-05 기능 점검 F-1: 넓은 낱말(물을·손잡이·보일러·반찬 - 좁히거나 뺌)과 비건강 질문(집수리·보험·물건·방송·날씨)이 라우터의 맞는 general 판단을
# 뒤집었다 - 물은 마시는 말과 함께일 때만, 손잡이는 안전·욕실·화장실 손잡이만, 아래 제외어가 있으면 올리지 않는다. F-3: "잠은 몇 시간".
_LIFESTYLE_TOPIC = re.compile(
    r"낙상|넘어지|넘어질|넘어져|미끄러|욕실|문턱|(안전|욕실|화장실)손잡이|난간|운동|걷기|근력|스트레칭|식단|반찬[^.!?。]{0,8}(먹어도|드셔도)|소금|싱겁|짜게|"
    r"물(을|은|도)?[^.!?。]{0,8}(마시|마셔|드시|드셔)|물몇잔|수분|낮잠|잠을|잠은|몇시간자|몇시간주무|수면|"
    r"더위|폭염|열사병|추위|한파|난방|치매|기억력|깜빡깜빡"
)
_LIFESTYLE_ASK = re.compile(
    r"얼마나|어떻게|어떤|뭘|뭐를|뭐가|무엇을|몇|해야|하면좋|좋을까|좋은지|좋나요|괜찮을까|괜찮나|되나|될까|되는지|"
    r"손봐|조심|예방|챙겨|챙길|방법|대비"
)
_LIFESTYLE_NOT_HEALTH = re.compile(
    r"고장|수리|닦|엎질|보험|게임|바지|잠옷|요금|난방비(가|는|이)?얼마|전기세|가스비|채널|방송|대회|운동회|시합|"
    r"날씨|예보|기온|(내일|모레|주말|이번주)(더위|추위|한파|폭염)|어디서사|어디서팔|가격"
)


def is_health_lifestyle_question(text: str) -> bool:
    """생활 관리·집 안전을 묻는 말인가(라우터 general -> 상담으로 올리는 근거, 라우터 실패 때 키워드 분류, 되묻기 제외)."""
    c = compact(text)
    return bool(_LIFESTYLE_TOPIC.search(c) and _LIFESTYLE_ASK.search(c) and not _LIFESTYLE_NOT_HEALTH.search(c))


# 진료과 정보 질문(2026-11-11): 진료과 이름(또는 "무슨/어느/어떤 과") 뒤에 그 과에서 무엇을 하는지·준비·비용을 묻는 말.
# "신경과는 어떤 검사를 받게 되나요?" - 어디로 가나(진료과 고정 답)도, 증상 상담(증상 요약 틀)도 아니다.
# 묻는 말이 진료과 앞에 있으면("검사 받으려면 어느 과 가요?") 어디로 가나 질문이다.
# 공백을 살린 문장에 맞춘다 - 앞이 한글이면 진료과가 아니다("불안과"·"편안과"의 안과), "신경과민"·"과일"도 아니다(2026-11-11 점검 B-2).
DEPARTMENT_MENTION = re.compile(
    r"(?<![가-힣])(정신건강의학과|소아청소년과|이비인후과|가정의학과|응급의학과|재활의학과|비뇨의학과|비뇨기과|정형외과|신경외과|흉부외과|"
    r"산부인과|소아과|피부과|신경과(?!민)|치과|안과|외과|내과)|(무슨|어느|어떤)\s*과(?!일)"
)
_DEPARTMENT_INFO_ASK = re.compile(
    r"검사|문진|진찰|준비|챙기|챙겨|가져가|가지고가|들고가|비용|진료비|검사비|얼마|절차|순서|오래걸|시간이걸|금식|굶|"
    r"무엇을하|무엇을해|뭘하|뭘해|뭐하|뭐해|무슨일을|어떤걸하|어떤것을하|어떻게진료|어떻게봐|어떻게보"
)


def department_info_ask_after(text: str, start: int = 0) -> bool:
    """공백 뺀 문장 c 의 start 뒤에 진료과 정보를 묻는 말이 있는가(진료과 고정 답 제외 판정)."""
    return bool(_DEPARTMENT_INFO_ASK.search(compact(text)[start:]))


_DEPARTMENT_INFO_COST = re.compile(r"비용|진료비|검사비|얼마|돈이|돈은|돈들")
_DEPARTMENT_INFO_PREPARE = re.compile(r"준비|챙기|챙겨|가져가|가지고가|들고가|금식|굶")


def department_info_kind(text: str) -> str:
    """진료과 정보 질문의 종류 - cost(비용) > prepare(준비물·금식) > exam(검사·문진·무엇을 하나). is_department_info_question 뒤에 쓴다."""
    c = compact(text)
    if _DEPARTMENT_INFO_COST.search(c):
        return "cost"
    if _DEPARTMENT_INFO_PREPARE.search(c):
        return "prepare"
    return "exam"


# 진료과 낱말이 있어도 증상·약·검사 결과·다녀온 뒤·다친 일을 묻는 말은 상담이다 - 고정 안내로 답하면 약 + 술 규칙·증상 상담·
# 오늘 진료 줄이 빠진다(2026-11-11 점검 B-2: "내과에서 검사했더니 당 수치가…", "내과 약 먹고 나서 어지러운데…").
_DEPARTMENT_INFO_NOT = re.compile(
    r"결과|수치|나왔|다녀왔|다녀온|갔다왔|갔다온|받았는데|했더니|하고나서|수술후|수술하고|수술받|수술뒤|"
    r"약먹|약을먹|약드|약을드|먹는데|먹고|드시는데|드시고|복용|넘어|부딪|다쳤|다친|다쳐|삐었|접질"
)


_DEPARTMENT_INFO_MAX_PREFIX = 8  # 진료과 앞 글자 수(공백 뺀) 상한
_DEPARTMENT_INFO_MAX_LEN = 40  # 문장 전체 글자 수(공백 뺀) 상한
# 진료과 앞이 서술(끝맺음·이음 말·부정)이면 사정을 말한 문장이다 - "안 깨어나요 내과 가면 뭐 해요?"(라우터 실패 때 키워드가 못 보는 응급).
# "동네"·"우리"·"엄마 모시고"·"그럼"·"혹시" 같은 앞말은 그대로 둔다.
_DEPARTMENT_INFO_PREFIX_NOT = re.compile(r"요|다|니까|는데|은데|해서|어서|아서|워서|했|없|안|못|않|말고")


def department_mention(text: str) -> re.Match[str] | None:
    """진료과 이름 또는 "무슨/어느/어떤 과"(공백을 하나로 줄인 문장 기준)."""
    return DEPARTMENT_MENTION.search(" ".join((text or "").split()))


def is_department_info_question(text: str) -> bool:
    """진료과에서 무엇을 하는지·준비·비용만 묻는 말인가(고정 안내 대상). 아래가 하나라도 있으면 False:
    위험 신호·응급(라우터 실패 때 판정·반응 없음 포함)·예약 낱말 / 라우터 실패 때 보는 자해 패턴 / 증상 낱말 / 약·검사 결과·다친 일.
    부르는 쪽(chat_service)은 라우터 판정(응급·자해·예약·오늘 진료 질환) 뒤에 부른다 - 키워드에 없는 위기·응급은 라우터가 맡는다."""
    spaced = " ".join((text or "").split())
    m = DEPARTMENT_MENTION.search(spaced)
    if not m or not _DEPARTMENT_INFO_ASK.search(compact(spaced[m.end():])):
        return False
    c = compact(text)
    # 진료과가 문장 앞쪽에 있는 짧은 질문만("동네 내과는…", "엄마 모시고 신경과 가면…"). 앞에 증상·사정을 길게 말한 문장
    # ("이마에 물집이 … 내과 가면 뭐 해요?")은 낱말 목록에 없는 증상·위기 표현일 수 있어 상담으로 보낸다(점검 h3.C04·sh.09).
    prefix = compact(spaced[: m.start()])
    if len(prefix) > _DEPARTMENT_INFO_MAX_PREFIX or len(c) > _DEPARTMENT_INFO_MAX_LEN or _DEPARTMENT_INFO_PREFIX_NOT.search(prefix):
        return False
    if ARREST_SIGNAL.search(c) or fallback_emergency_signals(text) or fallback_self_harm_signals(text):
        return False
    # 증상 낱말은 진료과 이름을 뺀 문장으로 본다("피부과"의 피부는 증상이 아니다).
    if has_symptom_word(DEPARTMENT_MENTION.sub(" ", spaced)) or _DEPARTMENT_INFO_NOT.search(c):
        return False
    return not (has_red_flag(text) or emergency_signals(text) or has_reservation_word(text))


def has_red_flag(text: str) -> bool:
    c = _RED_FLAG_FALSE.sub(" ", compact(text))
    return any(keyword in c for keyword in RED_FLAG_KEYWORDS)


def has_medical_word(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(keyword in t for keyword in MEDICAL_KEYWORDS)


def has_symptom_word(text: str) -> bool:
    c = compact(text)
    return any(word in c for word in ANY_SYMPTOM_WORDS)


def has_reservation_word(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(keyword in t for keyword in RESERVATION_KEYWORDS)


# 증상 단어 → 진료과. 새 요청의 진료과가 직전 요청에서 이어지지 않게 하는 최소 규칙(LLM 추론보다 우선).
# 상담 진료과 힌트(medical_llm_service)도 이 표를 쓴다. 위에서부터 먼저 걸린 과를 쓴다.
# 한 글자 단어("목"·"속"·"열"·"눈"·"배"·"코"·"귀")는 조사·증상 말이 붙을 때만 본다(2026-10-29 D-6) -
# 부분 일치일 때 "목요일"·"약속"·"계속"·"속상"·"열심히"·"배우자"·"눈치"·"코로나"·"귀찮"·"머리카락"·"허리띠"·"가려고"에 걸렸다.
SYMPTOM_DEPARTMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"머리가|머리 ?(아프|아파|통증|띵)|두통|어지러|어지럼|어지럽"), "신경과"),
    (re.compile(r"눈이|눈알|눈꺼풀|눈 ?(아프|아파|따가|가려|침침|충혈)|충혈|시력"), "안과"),
    (re.compile(r"귀가|귀에|귓|코가|코피|콧|코 ?막|목이|목구멍|목 ?(아프|아파|따가|부었|붓)|목감기|목소리|편도"), "이비인후과"),
    (re.compile(r"피부|두드러기|가려[워운우움]|가렵"), "피부과"),
    (re.compile(r"허리가|허리 ?(아프|아파|통증|삐)|무릎|관절|어깨|뼈|삐었|삐끗"), "정형외과"),
    # 저림은 정형외과 다음 - "허리가 아프고 다리가 저려요"는 정형외과로 둔다(2026-10-29 E: 손발 저림이 힌트 없이 정형외과로 답해졌다).
    (re.compile(r"저리|저려|저린|저림"), "신경과"),
    (re.compile(r"소변|빈뇨|배가|배 ?(아프|아파)|배탈|아랫배|윗배|속이|속 ?(쓰|더부룩|메스)|더부룩|소화|설사|구토|기침|열이|열나|발열|미열|고열|감기"), "내과"),
)

# 열이 있다는 말(I-2). "열심히"·"열어"·"열쇠"는 걸리지 않게 뒤 글자를 정한다.
FEVER = re.compile(r"열(이|도|은|나|까지|감)|발열|미열|고열")


def symptom_department(text: str) -> str | None:
    for pattern, department in SYMPTOM_DEPARTMENTS:
        if pattern.search(text or ""):
            # 열이 함께 있는 어지럼·두통·저림은 내과(2026-10-30 I-2: "어지러워요 / 열도 조금 있어요"에 신경과만 권했다).
            # 귀·눈처럼 부위가 분명한 과는 앞 줄에서 이미 정해지므로 신경과일 때만 바꾼다.
            if department == "신경과" and FEVER.search(text or ""):
                return "내과"
            return department
    return None


# ── 답 모양(2026-11-01 가독성): 줄 머리 이모지·카드 자리 낱말 ──────────────────────────────────────────
# 판정 낱말이 아니라 "보여 주는 모양"용이다. 위험도·119·응급·자해·약 + 술·되묻기·진료과 판정은 이 목록을 보지 않는다.
# 이모지는 모든 판정이 끝난 뒤 chat_layout 이 붙이고, 다음 턴 이력은 strip_layout_emoji 로 지운 문장을 쓴다.

# 응급·자해·위기 답 전용(그 답에는 이것 하나만).
LAYOUT_EMERGENCY_EMOJI = "🚨"
LAYOUT_EMPATHY_EMOJI = "💬"
LAYOUT_SUMMARY_EMOJI = "📝"
LAYOUT_QUESTION_EMOJI = "❓"
LAYOUT_DEFAULT_EMOJI = "💡"

# 문장 뜻 → 이모지. 위에서부터 처음 맞는 것 하나(순서가 규칙이다).
LAYOUT_EMOJI_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("⚠️", re.compile(r"119|응급실")),
    ("🚫", re.compile(r"마세요|마시고|않게|피하|금지|안 ?돼")),
    ("🩺", re.compile(r"의사|약사|상의|의료진")),
    ("🏥", re.compile(r"병원|진료")),
    ("💊", re.compile(r"(?<![예])약(?![속해])|복용|항생제|진통제|수면제|타이레놀")),
    ("🩸", re.compile(r"혈당|혈압")),
    ("🌡️", re.compile(r"열(이|도|은|나|감)|발열|미열|고열|체온")),
    ("💧", re.compile(r"물을|물은|물도|수분|음료")),
    ("🍚", re.compile(r"식사|식후|식전|음식|과일|식단|드시는 것이 좋")),
    ("🚶", re.compile(r"운동|걷기|산책|스트레칭")),
    ("🛏️", re.compile(r"잠|수면|주무|쉬시|쉬세요|휴식|누워|눕")),
    ("🧊", re.compile(r"찜질|얼음")),
    ("📞", re.compile(r"가족|이웃|알려|연락|곁에")),
)

# 허용 목록(19개). 답에 이 밖의 이모지는 나가지 않는다.
LAYOUT_EMOJI_ALLOWLIST: frozenset[str] = frozenset(
    [LAYOUT_EMERGENCY_EMOJI, LAYOUT_EMPATHY_EMOJI, LAYOUT_SUMMARY_EMOJI, LAYOUT_QUESTION_EMOJI, LAYOUT_DEFAULT_EMOJI]
    + [emoji for emoji, _ in LAYOUT_EMOJI_RULES]
)

# 카드 자리: 병원·의사·119 이야기나 "~하면 진료" 같은 조건 문장은 "이럴 땐 병원으로", 그 밖의 안내는 "집에서 이렇게".
LAYOUT_HOSPITAL = re.compile(r"119|응급실|병원|진료|의사|약사|상의|의료진")
LAYOUT_CONDITIONAL = re.compile(r"계속되|지속되|심해지|생기면|생기거나|나타나면|이어지면|되면|있으면|나면")

# 이모지 지우기(다음 턴 이력·판정용, 모델이 문장 안에 쓴 이모지도). 그림 문자 범위 + 변형 선택자 + 뒤 공백 하나.
# 뒤 공백은 스페이스·탭만 - 줄바꿈까지 먹으면 줄 끝 이모지 뒤의 다음 줄이 합쳐진다(2026-11-01 점검 L-1).
_EMOJI_CHARS = re.compile(r"[\U0001F000-\U0001FAFF⌀-⏿☀-➿⬀-⯿️‍]+[ \t]?")


def strip_layout_emoji(text: str) -> str:
    """이모지를 지운 문장. 판정·이력은 이 문장으로 본다(이모지 때문에 판정이 바뀌지 않게)."""
    return _EMOJI_CHARS.sub("", text or "")


def layout_emoji(sentence: str) -> str:
    """문장 뜻에 맞는 줄 머리 이모지 하나(허용 목록 안)."""
    plain = strip_layout_emoji(sentence)
    for emoji, pattern in LAYOUT_EMOJI_RULES:
        if pattern.search(plain):
            return emoji
    return LAYOUT_DEFAULT_EMOJI
