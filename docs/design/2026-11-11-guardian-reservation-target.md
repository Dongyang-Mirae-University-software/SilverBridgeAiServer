# 보호자 챗봇 예약 대상 고르기 (2026-11-11)

보호자 챗봇에서 "상담할 피보호자" 선택 칸을 없애고 평소엔 보호자 본인 상담으로 시작한다. **보호자 챗봇 예약은 피보호자만 대상이다**(보호자 웹 예약 화면과 같다 - 사용자 결정). 예약을 시작할 때만 대상을 묻는다.

## 계약 (BE·AI·FE - 바꾸려면 세 쪽 모두에 알릴 것)

1. **BE → AI**: 보호자 경로 `context.guardianWards = [{wardId, name, relation}]`(ACTIVE 연결만, 이름·관계만). 연결이 없으면 `[]`. 클라이언트가 보낸 `guardianWards`는 버리고 서버 값으로 채운다. `context.wardId`는 지금처럼 ACTIVE 연결 검증 후에만 통과(아니면 403).
2. **AI**: 보호자 경로(`role=GUARDIAN`)에서 `wardId` 없이 예약 흐름(새 예약·취소·변경·목록)이 시작되면 예약을 진행하지 않고 선택지 답을 준다.
   - `type="ui"`, `ui = {kind:"select", field:"reservation_patient", label:"누구의 병원 예약인가요?", options:[{value: wardId, label: "이름(관계)"}]}`. 보호자 본인("나") 선택지는 없다. 피보호자가 1명이어도 묻는다.
   - `guardianWards == []` → "챗봇 병원 예약은 연결된 피보호자의 예약만 할 수 있어요…" 안내(`intent="general_chat"`), 예약하지 않는다.
   - `guardianWards` 칸이 없으면(옛 BE) 묻지 않는다 - 이전 동작 그대로(하위 호환).
   - "예약 취소는 어떻게 해요?" 같은 안내 질문은 묻지 않는다. 응급·자해 판정이 먼저다(대상 질문은 그 뒤).
3. **FE**: 선택지를 누르면 `uiSelection {field:"reservation_patient", value: wardId}`를 보내고 **그 요청부터** `context.wardId`에 그 값을 싣는다. 이후 싣는 규칙:
   - **버튼**: 그 버튼이 달린 응답을 받았던 요청의 wardId를 그대로 싣는다(완료 뒤 "예약 목록 보기"도 그 피보호자 목록).
   - **글 입력**: 직전 응답이 wardId를 실은 요청의 응답이고 `intent=="hospital_reservation"`이며 `tool`이 `make_appointment`·`cancel_reservation`이 아닐 때만 싣는다.
   - **새 대화**(`sessionId` 변경): 초기화.
4. **AI**: 선택 턴은 직전 기록(같은 세션)의 대상 질문(30분 안)과 맞고 **이 턴의 `wardId`가 고른 값과 같을 때만** 그 피보호자로 예약을 시작한다(시작 문장·시작 버튼을 다시 넘긴다). wardId가 없거나 다르면 같은 질문을 다시("아래에서 예약할 분을 골라 주세요."), 질문이 없거나 지났거나 목록에 없는 값이면 "다시 말씀해 주세요"(`intent="general_chat"`). 이후 초안은 그 wardId에 묶인다(기존 규칙).

## BE #472 와 맞춘 점 (2026-11-11 후속)

- 칸 이름은 BE가 먼저 정한 `reservation_patient`다(처음 AI #255는 `reservation_target`으로 내보냈다 - 후속 PR에서 맞춤). BE는 이 칸의 값이 `__self__`가 아니면 형식·ACTIVE 연결을 검증하고(남의 ID 403), 같은 턴 `context.wardId`와 다르면 400이다.
- BE 문서의 `"__self__"`(나)·"목록이 비면 본인 예약"은 원래 계약 문구다. 이후 사용자 결정(피보호자만)으로 AI는 `__self__`를 선택지로 내지 않고, 들어오면 지남 안내로 끝내며, 빈 목록이면 예약 불가 안내를 준다. BE는 `__self__`를 통과시켜도 해가 없어 그대로 둔다.

## 대상을 모를 때 나이·성별

`wardId` 없이 가족·옆 사람이 주어인 상담(`proxy_subject`, 예: "어머니가 어지럽대요")은 MedGemma에 나이·성별을 넘기지 않는다 - 그 context는 말하는 보호자 본인 값이다. 본인 상담·wardId 있는 상담은 그대로. 대리 말투 판정은 바뀌지 않았다.

## 저장·로그

- 기록 `upstreamMeta.reservationTargetAsk`: 허용 wardId 목록·시작 버튼·시각(이름 없음). 다시 묻는 기록만 앞 질문의 시작 문장·규격화 문장(`carry`)을 옮긴다.
- 선택지 라벨(이름)은 응답 `ui`로 챗 기록에 남는다(확인 카드 문구와 같은 수준). 앱 로그는 `[CHAT-RESERVATION] target_ask options=<개수>`·`target_selected outcome=<go|reask|stale>`, `[CHAT-QUALITY] profile_omitted=proxy_subject`만.
- 코드: `app/services/reservation_target.py`, `chat_service.py` 연결 4곳, 테스트 `tests/test_guardian_reservation_target.py`.
