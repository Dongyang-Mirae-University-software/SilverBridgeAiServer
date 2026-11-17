# 챗 기록 `context_json.userContext` 개인정보 저장 점검 (2026-11-17)

- 대상: 챗 기록 저장(`chat_logs.context_json`)에 들어가는 요청 프로필 사본 `userContext`와 같은 정보가 든 다른 칸
- 기준: AI main `21a8b99`(gosky 같은 커밋) / BE dev `04b49fb7` / FE develop `603de3d`
- 점검만 했다. 코드·설정·DB는 바꾸지 않았다. DB는 SELECT로 개수·존재 여부만 셌고 값·본문은 보지 않았다. API 키는 쓰지 않았다.
- 이 문서에는 이름·전화·주소·상담 본문을 적지 않는다.

## 1. 저장 경로 (PHASE 0)

| 위치 | 내용 |
|---|---|
| `app/services/chat_service.py:1336-1370` `_build_user_context` | 요청 `context`의 age·email·name·phone·wardId(대리 상담이면 name·phone을 wardName·wardPhone으로 바꿈)·gender·birthDate·postcode·address·addressDetail·guardianId·location·role을 그대로 담는다 |
| `chat_service.py:782-784` `context_payload["userContext"] = user_context` | 위 사본 전체를 **매 턴** 기록에 저장한다. 걸러내는 단계 없음 |
| `chat_service.py:85-88,163-166` `_UNSTORED_META_KEYS`·`_persistable_meta` | `upstreamMeta`만 거른다(예약 키·storedFlow·rawMessage·storedContact·awaitingField). `userContext`는 대상 아님 |
| BE `ChatRelayService.java:72-80,255-272` `putProfile` | 2026-11-04부터 BE가 환자(대리 상담이면 피보호자) DB 값으로 name·phone·birthDate·age·gender·address(시·도+시·군·구로 시작할 때)를 채워 보낸다 → 모든 사용자·모든 턴에 해당 |

### 저장된 `userContext`를 다시 읽는 곳

| 위치 | 읽는 키 | 용도 |
|---|---|---|
| `chat_service.py:974-981` `_row_ward_id` | `wardId` | 기록이 어느 피보호자 상담인지(이력·응급 직후·기록 응답 `wardId`) |
| `chat_service.py:1783-1799` `_proxy_ward_ids` | `wardId` | 탈퇴·연결 해제 때 대리 상담 기록 찾기 |
| `app/services/chat_stats_service.py:117-118` | `role` | 관리자 AI 통계 역할별 집계 |
| AI `tools/withdrawal_leftover_check.py` `_AI_SQL` | `wardId` | 탈퇴 잔여 대조(읽기 전용) |

- **name·phone·address·birthDate·age·gender·email·postcode·addressDetail·location·guardianId는 저장 뒤 아무 곳도 다시 읽지 않는다.** 이번 턴 처리(예약 계정 `chat_service.py:519-525`, MedGemma 나이·성별 `:1315-1329`, 예약 초안 `reservation_intake.py:298`)는 모두 **이번 요청의 값**을 쓴다.

### 같은 정보가 들어가는 다른 칸

| 칸 | 내용 | 필요성 |
|---|---|---|
| `upstreamMeta.flowState.draft`·`pendingAction.draft` (`reservation_confirm_flow.py:292-306,684-686`) | 예약 초안의 환자 이름·전화·생년월일·증상 요약 | 다음 턴 확인·고치기에 필요(`_last_flow_state` `chat_service.py:1275-1287`). 흐름이 끝난 뒤에도 그대로 남는다 |
| `data`(=기록 응답 `toolData`) (`reservation_orchestrator.py:757-761`) | 예약 완료 응답의 환자 이름·전화 | 본인(대리면 보호자) 예약 확인 표시. BE 허용 목록에 `toolData`가 있어 FE에 간다 - 요청자 본인 예약 정보라 의도된 노출 |
| `ui` | 대상 질문 선택지 이름·관계는 저장 전 "피보호자"로 바꿈(`reservation_target.py:65`) | PASS |
| `decisionTrace` | 프로필 값 없음(코드 확인) | PASS |

## 2. DB 실측 (gosky `silverbridge-ai-postgres`, 11/17 16:40 KST)

| 항목 | 값 |
|---|---|
| 전체 행 / JSON 파싱 불가 | 859 / 0 |
| 기간 | 2026-10-08 ~ 2026-11-17 (보관 기간 약 40일, 정리 스케줄 없음) |
| `userContext`가 비어 있지 않은 행 | 622 |
| name / phone / address / birthDate 있는 행 | 489 / 471 / 217 / 265 |
| email / postcode·addressDetail / location 있는 행 | 5 / 5 / 47 (email·postcode는 11/04 이전 FE가 보내던 값) |
| 대리 상담 행(`userContext.wardId`) / 그중 피보호자 이름 있음 | 98 / 98 |
| name 있는 행의 사용자 수 | 20명 |
| 11/04 이후 행 / 그중 phone 있음 | 227 / 207 (91%) |
| `upstreamMeta` 예약 초안 환자 이름 / 전화 | 96 / 95 |
| `data`에 환자 이름 칸 | 50 |
| 탈퇴 대조(`withdrawal_leftover_check.py`, BE dev 두 서버 54명 기준) | 챗 기록 사용자 22명 중 **11명이 BE dev 두 서버에 없음**(각 1행, 11행 모두 name·phone·address 있음, 마지막 행 11/13) / 대리 상담 wardId 잔여 0 / 예약 키 잔여 0 |

- 11명의 원인은 확정하지 못했다. gosky `dmu-prod-api`도 같은 AI(`AI_HTTP_BASE_URL` 기본값 `https://testai.gosky.kr`, `AI_API_KEY` 설정)를 부르므로 **prod 회원일 가능성**이 있으나, prod DB(`dmu-prod-db`) 대조는 이번 세션 자동 모드가 막아 하지 못했다(§5 확인 필요).

## 3. 노출·삭제 경로

| 경로 | 결과 | 근거 |
|---|---|---|
| AI `GET /api/v1/chat/logs`·`/logs/{id}` | 응답에 `contextJson`(userContext 포함) 전체를 싣는다(`chat_service.py:1820,1858`) | AI_API_KEY 필수(`app/main.py:261`), 6017 외부 차단(11/10) - 호출자는 BE뿐 |
| BE 기록 중계 → FE | **PASS** | `ChatLogProjection.ALLOWED_FIELDS`(`ChatLogProjection.java:28-31`)에 contextJson·upstreamMeta·decisionTrace 없음, `ChatLogProjectionTest` 고정 |
| 관리자 AI 통계 | **PASS** | AI는 role 개수만 집계, BE는 허용 목록 record |
| 로그(`[CHAT-*]` 등) | **PASS** | `app/` 전체에서 user_context·phone·address·birth 값을 로그에 넘기는 곳 없음(예약은 `patient_name_len`·`phone_last4`만) |
| MedGemma 입력 | **PASS** | 나이·성별만(10/29 #107) |
| 라우터·예약 추출(Claude) | **PASS** | userContext를 프롬프트에 넣지 않음, 지역은 시·구까지(S-6) |
| 탈퇴 정리 | **PASS** | `DELETE /logs?userId=` 본인 + 대리 상담 기록째(`chat_service.py:1755-1764`), BE `AiMemberDataPurgeListener` |
| 연결 해제 정리 | **PASS** | `DELETE /logs/proxy`(`chat_service.py:1768-1778`) |
| 보관 기간 정리 | **FAIL** | `chat_logs` 정리 스케줄 없음(`main.py`의 정리는 학습 샘플 `purge_expired`뿐) - 탈퇴·해제 전까지 무기한 |
| DB 백업·사본 | 확인 필요 | gosky `/root/backup/ai-db-qa-ai3-20261004.sql`(10/04, 약 460KB - 열람 안 함), `SilverBridgeAiServer/data/ai_server.db`(5/26 옛 SQLite, chat_logs 24행 모두 userContext 키 있음). 둘 다 탈퇴 정리 대상 밖 |

## 4. 정책 대조

| 정책 | 현재 | 판정 |
|---|---|---|
| BE policy-chat ③ "상담 내용은 로그에 남기지 않는다" | 로그에는 없음. DB 기록은 이 규칙 대상이 아님 | 일치 |
| policy-chat "MedGemma 입력은 나이·성별만"(10/29) | 일치. 다만 같은 취지(필요한 것만)가 기록 저장에는 적용되지 않았다 | 취지 불일치 |
| policy-chat 11/04 L-3 "주소는 AI 상담 기록 `contextJson`에 남고 탈퇴·연결 해제 정리로 지워진다" | 저장을 **알고 수용**한 기록. 다만 그 값을 다시 읽는 곳이 없다는 점은 검토되지 않았다 | 재검토 대상 |
| policy-chat 11/11 결정 B "`guardianWards`를 userContext에 저장하지 않는다" | 같은 논리(쓰지 않는 개인정보는 기록에 남기지 않는다)가 name·phone·address에는 아직 적용되지 않음 | 불일치 |
| domain-security-policy D-2 보관 기간(접속 로그 90일·감사 1년·말벗 대화 30일) | 챗 기록은 보관 기간 결정이 없다 | 결정 누락 |
| AI 10/30 탈퇴 정리 점검 | userContext에 피보호자 이름·전화·생년월일이 있다는 이유로 대리 기록까지 기록째 삭제 | 일치(삭제는 됨) |

## 5. 이슈

| 등급 | ID | 내용 |
|---|---|---|
| 🔴 | - | 밖으로 노출되는 경로 없음 |
| 🟠 | H-1 | `userContext`에 이름·전화·주소·생년월일 등이 매 턴 원문 저장(489행·20명)되지만 다시 읽는 키는 `wardId`·`role`뿐 - 쓰지 않는 개인정보를 대화 턴 수만큼 복제 보관 |
| 🟠 | H-2 | `chat_logs` 보관 기간 없음 - 탈퇴·연결 해제가 없으면 무기한(D-2 결정 누락) |
| 🟠 | H-3 (확인 필요) | BE dev에 없는 사용자 11명의 기록(각 1행, 이름·전화·주소 있음). prod 회원이면 정상, 아니면 탈퇴 정리 누락 - prod DB 대조 필요 |
| 🟡 | M-1 | 예약 초안(`flowState`·`pendingAction`)의 환자 이름·전화가 흐름이 끝난 뒤에도 남음(96행). 기능상 필요한 칸이라 H-2 보관 기간으로 함께 다룬다 |
| 🟡 | M-2 | 옛 사본: 10/04 AI DB 덤프·5월 SQLite가 탈퇴 정리 밖에 남음 |
| 🟡 | M-3 | `withdrawal_leftover_check.py`가 prod BE 회원을 대조하지 않아 prod 회원이 "잔여"로 잡힌다 |
| 🟢 | L-1 | 수정 후 policy-chat 11/04 L-3의 "주소는 contextJson에 남는다" 문장 갱신 필요 |

종합 판정: **조건부 PASS** - 밖으로 새는 길은 없고 삭제 경로도 동작한다. 다만 쓰지 않는 개인정보를 매 턴 저장(H-1)하고 보관 기한이 없다(H-2).

## 6. 수정안

| 안 | 내용 | 기존 기능 영향 | 판단 |
|---|---|---|---|
| **① 저장 허용 목록 (추천)** | 저장 직전 `userContext`를 `role`·`wardId`만 남긴다(`_STORED_USER_CONTEXT_KEYS`). 허용 목록 방식이라 새 키가 늘어도 자동 제외 | 다시 읽는 키가 이 둘뿐이라 이력·대리 판정·탈퇴/해제 정리·통계·잔여 대조 모두 그대로. 이번 턴 처리는 요청 값을 쓰므로 무관 | 추천 |
| ② 가려 저장 | 이름 길이·전화 끝 4자리 등 | 읽는 곳이 없어 가린 값도 쓸모가 없다 | 비추천 |
| ③ 기존 행 정리 | `userContext`를 `role`·`wardId`만 남기도록 UPDATE(622행 대상). 비가역·운영 DB → 백업 후 사용자 실행 | 같음 | ① 배포 뒤 실행 |
| (별도 결정) 보관 기간 | 예: 기록 N일 뒤 삭제 또는 예약 초안 칸만 비움(H-2·M-1) | 오래된 상담 기록이 FE 기록 목록에서 사라짐 | 사용자 결정 필요 |

## 7. 반영 (2026-11-17, H-1)

- `chat_service.py` `_STORED_USER_CONTEXT_KEYS = ("role", "wardId")`·`_persistable_user_context` - 저장할 때만 허용 목록으로 거른다. 이번 턴 처리(예약 계정·초안·MedGemma·지역)는 요청 값 그대로.
- 테스트 `tests/test_chat_log_user_context_strip.py`(대리·본인 상담 저장 키, 저장 행으로 대리 판정·탈퇴 정리, 허용 목록), `test_chat_profile_timeout.py`의 저장 단언을 새 기준으로 바꿈(초안 채우기는 저장 값이 아니라 요청 값을 쓴다 - `reservation_orchestrator.py:1083`).
- 전체 pytest(로컬 임시 venv): 3061 passed, 실패 7·수집 오류 56은 수정 전 main과 같은 목록(로컬에 없는 모델 의존성).
- gosky 배포 11/17 16:55 KST(`8f7f9e6`, `/health` 200, BE AI WS 16:56 재연결), 롤백 태그 `sb-ai-rollback:20261117-uc-strip`(1d0da47 이미지) 1개.
- 기존 행 정리(사용자 지시로 실행, 비가역): 백업 `/root/backup/ai-chatlogs-before-uc-strip-20261117.sql`(600, 11/24 이후 삭제) → 트랜잭션 UPDATE 489행. 정리 후 name·phone·address·birthDate 행 0, wardId 98·role 621·전체 859 그대로, 탈퇴 대조 결과도 정리 전과 같음.
- 남은 일: H-2 보관 기간은 별도 결정(이번 제외 - 사용자 결정), H-3 prod 대조, M-2 옛 사본(10/04 덤프·5월 SQLite) 처리, BE policy-chat 11/04 L-3 문장 갱신.
