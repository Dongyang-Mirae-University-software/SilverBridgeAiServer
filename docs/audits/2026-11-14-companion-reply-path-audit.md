# 말벗 대답 경로 기능 점검 (2026-11-14)

대상: AI main `b8de8e0`(코드는 `03c3bfd`와 같음) - #279(직전 말벗 문장을 질문으로·글자 되풀이 다시 생성·짧은 부정 대답·`labelKo`), #288(짧은 대답 화제 전환), #287(글 대답 되묻기 없음)과의 상호작용.
방법: 코드 읽기 + 화제 판정 정규식 재현. 기준 pytest 3998 passed(실패 1 `test_clip_encoder` - 컨테이너에 `pgrep` 없음). API 키 사용 없음, 운영 대화 원문 조회 없음. 코드 수정 없음.

## 턴 처리 순서 (`companion_router._run_turn`)

| 순서 | 단계 | 읽는 것 | 쓰는 것 |
|---|---|---|---|
| 1 | `pipeline.run_turn` (글 대답 → 일반 턴 / 음성 → 확신도·짧은 부정 대답 통과 / 낱말 규칙) | `lastQuestion or question`, `replies`, `lastSignals` | `lastSignals`, `riskSeen` |
| 2 | 되묻기 연속 종료 | `lowConfidenceStreak` | 대답 → 마무리 문장 |
| 3 | `_flat_topic_switch` | 대답 글, `replies` | `flatStreak`, 대답 → 화제 전환 질문(`TOPIC`) |
| 4 | `_apply_avoid_end` | `avoidCount` | 대답 → 마무리 문장, `endReason` |
| 5 | `_avoid_repeat` (글자가 같은 문장) | `replies` | 대답 → 대체 문장, `replies` 추가 |
| 6 | `_queue_for_review` | 출처 | 검수 목록 |
| 7 | `lastQuestion` 갱신 (되묻기 턴 제외) | 최종 대답 | `lastQuestion` |

## PASS

- 위험 우선: 위험 턴은 화제 전환(3)이 건너뛰고, 다시 생성·짧은 부정 대답 통과·글 대답 경로 모두 `_analyzed_turn`의 위험 판정을 거친다. 짧은 부정 대답 통과는 되묻기만 건너뛴다.
- 로그 원문 없음: `[COMPANION-REPLY-REDO] ok=`, `[COMPANION-TOPIC-SWITCH] streak=`, `[COMPANION-REPLY-FALLBACK] failures=`, `[COMPANION-REPLY-SOURCE]` 모두 이름·횟수·이유 코드만.
- 계약: 턴 `face.labelKo`·결과 `dominantFaceKo`는 더하기만, BE는 `face`를 Map으로 넘기고 결과는 아는 필드만 읽는다. FE #263은 `labelKo`를 먼저 쓴다. `replySource`는 응답에 나가지 않는다.
- 설정 끔: `COMPANION_REPLY_SIMILAR_RATIO=0`·`COMPANION_SHORT_ANSWER_PASS=false`·`COMPANION_FLAT_TOPIC_SWITCH=0` 각각 전 동작(시험 있음).
- 화제 전환 문장은 한 번 쓰면 그 낱말이 `replies`에 들어가 다시 고르지 않는다. 6개를 다 쓰면 모델 대답 그대로.

## 이슈

### 🟡 M-1 이어 가기(resume) 때 기기 질문과 AI 질문이 어긋난다
BE `resumeOrReject`는 여는 질문(`companion_question.text`)을 돌려주고 FE `startTalk`는 그 문장을 다시 보여 주고 읽는다. AI는 같은 대화를 이어 쓰므로 `lastQuestion`(마지막 말벗 문장)을 질문으로 본다. 어르신은 여는 질문에 답하는데 AI는 다른 질문의 대답으로 해석한다(#279 이전에는 둘 다 여는 질문이라 맞았다).
- 수정안 A: AI `GET /companion/dialogues/{id}`에 `lastQuestion`을 내고 BE 이어 가기 응답의 `question`을 그 값으로(동의와 무관, 저장 없음). 수정안 B: AI가 이어 가기 신호를 받으면 `lastQuestion`을 여는 질문으로 되돌림.

### 🟡 M-2 화제 판정 정규식 오탐 - 묻지 않은 화제를 "이미 물음"으로 본다
`topic_switch_reply`는 이 대화 말벗 문장 전체에서 낱말을 찾는다. 재현:
| 말벗 문장 | 잘못 걸린 화제 |
|---|---|
| 안전 안내 "많이 힘드셨겠어요 … 가까운 가족이나 …" | 식사(`드셨` ← 힘드셨), 가족 |
| "잠깐이라도 쉬셨군요" | 잠(`잠` ← 잠깐) |
| "저녁노을이 예뻤겠네요" | 식사(`저녁`) |
| "마음이 불편하셨군요" | 몸(`불편`) |
"힘드셨겠어요"는 모델 대답에 자주 나와 식사 화제가 거의 나가지 않을 수 있다. 안전과는 무관(다음 화제로 넘어감). 수정안: 물음 문장만 보고, 낱말 경계(`(?<![가-힣])잠(?!깐)`·`힘드셨` 제외 등) 또는 화제별 짧은 고정 구(식사는/잠은/몸은)로 판정.

### 🟡 M-3 위험·되묻기 턴도 짧은 대답 횟수에 들어간다
`flatStreak`는 가드보다 먼저 올린다. 위험 턴(안전 안내)에서 짧은 대답이 1로 세어진 뒤 다음 짧은 대답에 바로 화제 전환 질문("오늘 식사는…")이 나갈 수 있다 - 안전 안내 직후 화제를 가볍게 바꾸는 것은 11/11 결정(힘든 이야기에는 화제를 바꾸지 않는다)의 취지와 맞지 않는다. 확신도 낮아 되물은 턴(받아쓰기가 틀렸을 수 있음)도 센다. 수정안: 위험·안전 안내 턴은 0으로, 되묻기 턴은 세지 않음.

### 🟢 L-1 부정 기분 가드는 글 기분만 본다
표정이 슬픔이어도 글 기분이 평온·판단 불가면 화제를 바꾼다(짧은 대답은 글 기분 근거가 약하다). 표정 포함 턴 판정(`turnResult.label`)을 볼지 결정 필요.

### 🟢 L-2 무응답 턴은 횟수를 0으로 돌린다
"몰라" → (무응답) → "그냥" 은 화제를 바꾸지 않는다. 의도로 둘지 결정 필요.

### 🟢 L-3 짧은 대답 낱말 목록이 둘이다
`pipeline.SHORT_ANSWER_WORDS`(되묻기 통과, 부정 위주)와 `reply_service.FLAT_ANSWER_WORDS`(화제 전환). 목적이 달라 따로 두되, 한쪽에 낱말을 더할 때 다른 쪽도 볼 것(문서화).

### 🟢 L-4 `COMPANION_LLM_RETRY ≥ 1`이면 다시 생성이 곱해진다
재시도 반복 안에서 비슷하면 다시 생성하므로 최대 2×(1+재시도)회 생성. 기본 0이라 지금은 2회.

### 🟢 L-5 시험 빈곳
무응답·되묻기·위험 턴의 `flatStreak`, 글 대답(#287)에서 화제 전환, 화제 전환 다음 턴의 `질문:`이 전환 질문인지, 이어 가기 질문 어긋남, 정규식 오탐 표(M-2).

## 종합
조건부 통과 - 안전·로그·계약은 PASS. M-1~M-3은 대답 품질 문제로, M-2·M-3은 AI 안에서 고칠 수 있고 M-1은 BE·AI 계약 결정이 필요하다.

## 조치 (2026-11-14, 사용자 "전부 반영")
- M-1: AI `GET /companion/dialogues/{id}` - `{dialogueId, question, lastQuestion, turnNo, ended}`(원문 없음, 말벗 키 필요). BE 이어 가기 응답의 `question`을 `lastQuestion`으로(실패하면 여는 질문) - BE PR 별도.
- M-2: 화제 판정은 말벗의 물음 문장만 보고, `(?<!힘)드셨`·`잠(?!깐|시)`·`저녁(?!노을)`·`불편하신 (곳|데)`로 오탐을 막는다.
- M-3·L-2: 위험·안전 안내 턴은 짧은 대답 횟수 0, 무응답·되묻기·확신도 낮은 턴은 횟수를 바꾸지 않는다.
- L-1: 글 기분 또는 표정까지 합친 턴 판정(`turnResult.label`)이 부정이면 화제를 바꾸지 않는다.
- L-3: 두 낱말 목록에 서로를 가리키는 주석. L-4: 다시 생성은 한 대답에 한 번. L-5: `tests/test_companion_reply_audit_fixes.py` 19건.
