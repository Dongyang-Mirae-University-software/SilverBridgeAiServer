# 챗 응답 "이어서 물어볼 질문" (`suggestedQuestions`, 2026-10-30)

FE 빠른 질문 칸이 고정 문장 4개라 대화가 이어지지 않았다(사용자 요청: "기록을 보고 계속 질문으로 진행"). 답마다 사용자가 다음에 누를 만한 질문을 최대 3개 붙인다.

## 응답 계약

- `POST /api/v1/chat` 응답 `data.suggestedQuestions: string[]` - 0~3개, 각 2~18자, 제어문자·숫자 3자리 이상 덩어리 없음, 사용자 문장과 같은 것 제외
- 빈 목록인 경우: `FOLLOW_UP_ENABLED=false`, 응급(`riskLevel=high`·`emergency_guidance`·라우터 emergency·응급 키워드), 자해·위기 후속(`selfHarm`·`crisisFollowup`), 응급 후속 안내(`emergencyFollowup`), 되묻기(`followupAsked`), 생성 실패(`generationFailed`), 예약 흐름(`route=reservation`·`hospital_reservation`·`ui` 선택지·`tool_result`)
- 챗 기록(`chat_logs`)에는 저장하지 않는다 - 기록 목록·상세 응답은 그대로

## 만드는 방법

1. 짧은 Claude 호출(`FOLLOW_UP_MODEL`, 기본 `claude-haiku-5-5`, 출력 상한 1024) - 입력은 이번 사용자 문장(500자)과 답 본문(1500자)뿐(프로필·이름·전화 없음). 대리 상담(`wardId` 또는 가족 주어 문장)이면 가족에 대해 묻는 말투
2. 전용 스레드(2개)에서 호출하고 `FOLLOW_UP_TIMEOUT_SEC`(기본 3초, 0.5~10)만 기다린다. 넘기거나 실패·두 스레드가 모두 바쁘면 규칙 문장(증상/일반 × 본인/대리 3개씩)
3. 라우터·MedGemma 프롬프트와 위기 응답은 바꾸지 않았다(안전 판정과 무관한 보조 값)

- 로그: `[CHAT-SUGGEST] source=model|rule count= cause= ms=` - 질문 문장·상담 내용 없음
- 대기를 넘긴 호출은 뒤에서 끝나도록 두고 결과를 버린다(사용자 응답은 최대 약 3초 늘어난다)

## 설정

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `FOLLOW_UP_ENABLED` | `true` | `false`면 항상 빈 목록 |
| `FOLLOW_UP_MODEL` | `claude-haiku-5-5` | 질문 생성 모델 |
| `FOLLOW_UP_TIMEOUT_SEC` | `3.0` | 기다리는 시간(범위 밖이면 기본값) |

## 테스트

- `tests/test_follow_up_questions.py` - 붙이지 않는 답 10종, 정리 규칙, 규칙 문장, 꺼짐, 모델 성공·실패·지연(0.5초 대기에서 1.5초 안에 응답), 로그에 문장 없음, 실제 챗 흐름(증상 답에 붙음·대리 상담 말투·자해 턴에 없음)
- 응답 키 고정 테스트 3곳(`test_chat_timing`·`test_chat_safety_rules`·`test_chat_crisis_followup`)에 새 키 추가
