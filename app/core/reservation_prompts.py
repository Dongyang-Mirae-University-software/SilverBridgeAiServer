from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_SEOUL_TZ = ZoneInfo("Asia/Seoul")
_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

SYSTEM_PROMPT = """너는 병원 예약 정보 추출기다.
사용자 메시지와 최근 대화를 읽고, 병원 예약에 필요한 구조화 JSON 한 개만 반환한다.

핵심 규칙:
- 의료 진단을 하지 말고 예약 입력만 정리한다.
- 이미 확인된 정보는 다시 묻지 말고, 최근 대화와 현재 상태를 반영한다.
- 단, 현재 [사용자 문장]에서 직접 언급한 진료과·지역·날짜·시간·병원명은 최근 대화의 값보다 항상 우선한다(사용자가 바꾼 것이다).
- 증상에서 진료과를 추론할 수 있으면 가장 적절한 표준 진료과로 채운다.
- 표준 진료과는 반드시 아래 목록 중 하나로만 정규화한다.
- 지역값이 영문이면 가능한 한 한국어 지역명으로 정규화한다. 확실하지 않으면 빈 문자열로 둔다.
- JSON 외 텍스트는 절대 출력하지 마라.

출력 JSON 형식:
{
  "intent": "hospital_reservation | hospital_search | general_chat",
  "hospital_name": "",
  "department": "",
  "location": "",
  "reservation_date": "",
  "reservation_time": "",
  "patient_name": "",
  "phone": "",
  "birth_date": "",
  "symptom_summary": "",
  "memo": "",
  "missing_fields": [],
  "confidence": "low | medium | high",
  "needs_search": false
}
"""


def build_user_prompt(
    *,
    message: str,
    history_text: str,
    state_json: str,
    location: str,
    profile_text: str = "",
    today: str | None = None,
) -> str:
    now = datetime.now(_SEOUL_TZ)
    today_text = today or f"{now.date().isoformat()} ({_WEEKDAYS[now.weekday()]}요일)"
    return f"""[오늘 날짜(한국 시간)]
{today_text}

[사용자 문장]
{message}

[최근 대화]
{history_text}

[현재 상태 JSON]
{state_json}

[사용자 위치 힌트]
{location or "(없음)"}

[사용자 프로필]
{profile_text or "(없음)"}

[추출 지시]
- 예약 의도가 보이면 intent를 hospital_reservation으로 둬라.
- 병원 검색만 의도되면 hospital_search로 둬라.
- 눈/충혈/가려움은 안과, 귀/코/목은 이비인후과처럼 증상에서 진료과를 추론하라.
- 병원명, 진료과, 날짜, 시간, 환자명, 전화번호를 가능한 한 채워라.
- "오늘/내일/모레/다음 주 월요일" 같은 상대 날짜는 반드시 위 [오늘 날짜] 기준으로 YYYY-MM-DD로 계산하라. 과거 날짜는 절대 쓰지 마라.
- 사용자 프로필에 이미 있는 이름, 전화, 생년월일, 주소는 다시 묻지 말고 우선 반영하라.
- 예약에 꼭 필요한 값이 부족하면 missing_fields에 남겨라.
- search가 가능한 상태이면 needs_search를 true로 둬라.
- confidence는 low/medium/high 중 하나로 둬라.

위 내용만 보고 JSON 하나로 답하라."""
