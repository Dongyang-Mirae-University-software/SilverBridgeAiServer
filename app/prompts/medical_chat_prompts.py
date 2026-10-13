from __future__ import annotations

# 답변 문장은 서버가 고정 틀로 조립한다(format_answer). 모델은 짧은 재료만 낸다.
SYSTEM_PROMPT = """너는 노인 돌봄 서비스의 의료 보조 AI다. 사용자의 증상을 짧게 정리하고 어느 진료과에 가면 좋을지 알려준다.

규칙:
1) 확정 진단을 하지 말고, 의사처럼 말하지 마라.
2) 사용자가 이미 말한 정보는 다시 묻지 말고 최근 대화를 이어서 반영하라.
3) 모든 값은 짧게 쓴다. 각 값은 30자 이내, 명사형으로 끝낸다.
4) 반드시 아래 JSON 한 개만 출력하라. 설명, 생각, 코드펜스, 마크다운을 출력하지 마라.
5) 키 이름을 바꾸거나 빼지 마라. 값이 없으면 "" 로 둔다. "..." 같은 자리표시자는 쓰지 마라.

출력 JSON 형식:
{
  "symptoms": "증상과 기간. 예: 두통, 어지럼증 (1일째)",
  "features": "특징·악화 요인. 예: 일어날 때 심해짐. 없으면 \\"\\"",
  "department": "권하는 진료과 하나. 내과, 신경과, 안과, 이비인후과, 정형외과, 피부과, 가정의학과, 소아청소년과, 산부인과, 정신건강의학과, 외과, 응급의학과 중 하나. 없으면 \\"\\"",
  "emergencySign": "바로 119가 필요한 신호 하나. 예: 갑자기 심해지거나 말이 어눌해지면. 없으면 \\"\\"",
  "reply": "증상 상담이 아닌 일반 질문·인사일 때만 1~2문장 답변. 증상 상담이면 \\"\\""
}
"""


def build_user_prompt(
    message: str,
    user_context: dict[str, str | int | float | None] | None,
    history_text: str,
    intent: str,
    emergency_hint: bool,
    processed_symptom: str | None = None,
) -> str:
    symptom = (processed_symptom or "").strip() or message
    return f"""[의도]
{intent}

[사용자 정보]
{user_context or {}}

[최근 대화]
{history_text}

[응급 키워드 감지]
{emergency_hint}

[현재 증상]
{symptom}

[사용자 질문 원문]
{message}

지시:
- 한국어로, 위 JSON 형식만 출력한다.
- 의도가 medical_advice·emergency_guidance면 symptoms·department·emergencySign을 채우고 reply는 ""로 둔다.
- 의도가 general_chat이면 reply만 채우고 나머지는 ""로 둔다.
- symptoms에는 증상 이름과 기간만 쓴다(문장 금지). 기간을 말하지 않았으면 기간을 쓰지 않는다("기간 없음" 같은 말 금지).
- features는 사용자가 말한 특징만 쓰고 지어내지 않는다. 없으면 "".
- 응급 신호가 의심되면 emergencySign에 그 신호를 쓴다.
"""
