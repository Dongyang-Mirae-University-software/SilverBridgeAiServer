# 말벗 대체 문장 수정(#260) 기능 점검

- 날짜: 2026-11-12 / 대상: PR #260 (머지 `32ee839`) / 기준: `origin/main` `0b16224` (gosky 반영 커밋과 같음)
- 형식: 기능 점검(템플릿 B), 점검만 하고 코드는 고치지 않았다
- 판정: **조건부 PASS** · 🔴 0 · 🟠 1 · 🟡 1 · 🟢 3

## 사전 확인

| 항목 | 결과 |
|---|---|
| gosky AI | `0b16224`(#257 배포 때 #260 함께 반영), 측정 잠금 없음 |
| 전체 pytest (운영 이미지 일회용 컨테이너) | 3568 passed · 1 failed · 12 skipped - 실패는 `test_clip_encoder` 타임아웃(이미지에 `pgrep` 없음, 기존 환경 실패) |
| #257과 겹침 | 없음 (#257은 `chat_service`·`followup_steps`·`medical_llm_service`만) |

## 턴 종류별로 나가는 문장

| 턴 | 경로 | 나가는 문장 | #260 영향 |
|---|---|---|---|
| 일반, 모델 대답 통과 | `_analyzed_turn` | 모델 대답 | 없음 |
| 일반, 규칙 탈락·LLM 실패 | `generate_reply` → `_analyzed_turn` | 글 기분 부정이면 `FALLBACK_EMPATHY`, 아니면 `FALLBACK_FOLLOWUP` | 적용 |
| 일반, 대답 생성 시간 초과 | `no_response_reply` → `_analyzed_turn` | 위와 같음 | 적용 |
| 위험(낱말 규칙·모델) | `_analyzed_turn` 마지막 | `SAFETY_REPLY` (공감 문장 교체 뒤에 덮어써서 항상 이긴다) | 없음 |
| 되묻기(낱말 확인·확신도 낮음) | `_lexicon_checked_turn`·`_low_confidence_turn` | `CONFIRM` 문장 또는 위험이면 `SAFETY_REPLY` | 없음 |
| 확신도 낮음 연속 상한 | 라우터 `_run_turn` | `FALLBACK_LAST[0]` | 없음 |
| 마지막 턴 | `fallback_options(is_last_turn=True)` | `FALLBACK_LAST` (부정 기분이어도) | 없음 |
| **무응답(말이 비었음)** | `run_no_response_turn` | `FALLBACK_FOLLOWUP` | **H-1** |
| 반복 방지 | `_avoid_repeat` | 같은 기분 기준 목록에서 안 쓴 문장, `CONFIRM`·`SAFETY`는 그대로 | 적용 |

## 이슈

### 🟠 H-1 무응답 턴에 말을 들은 것처럼 답한다 (#260으로 넓어짐)

- `run_no_response_turn` → `no_response_reply` → `fallback_text`가 `FALLBACK_FOLLOWUP`을 쓴다. 새 3문장("그러셨군요. 조금 더 들려주실 수 있을까요?", "말씀해 주셔서 고맙습니다. 그 이야기…", "그랬군요. 그때 마음은…")이 모두 어르신이 말을 했다고 전제한다.
- 이전 목록도 2/3이 그랬지만 "오늘 하루는 어떻게 보내고 계세요?"처럼 대화를 다시 여는 문장이 있었다. 이번 변경으로 3/3이 됐다.
- 제안: 무응답 전용 목록(`FALLBACK_NO_RESPONSE`)을 두고 `run_no_response_turn`만 이 목록을 쓴다. 대화를 다시 여는 질문은 무응답 턴에는 화제 전환이 아니다. 시간 초과 경로는 글이 있으므로 지금처럼 `FALLBACK_FOLLOWUP`·`FALLBACK_EMPATHY`를 쓴다(`no_response_reply` 호출을 나눠야 함). 시험: 무응답 턴 문장이 "말씀해 주셔서"·"그랬군요"·"그러셨군요"로 시작하지 않는다.

### 🟡 M-1 "그때 마음은 어떠셨어요?"는 짧고 평범한 대답에 가리킬 것이 없다

- 대체가 많이 나는 대답은 "별일 없는데", "물론이죠"처럼 짧은 말이다(`reply_eval7` 탈락 표본). 여기에 "그때"는 가리킬 일이 없다.
- 제안: 셋째 문장을 "그렇군요. 그 말씀 조금 더 해 주시겠어요?"처럼 시점이 없는 문장으로 바꾼다. H-1과 같은 PR로.

### 🟢 L-1 검수 목록 버전이 대체 문장 교체를 구분하지 못한다

- `_queue_for_review`는 대체 문장을 모두 `fallback-v1`로 남긴다. 문장 자체가 달라 구분은 되지만, 버전으로 거르려면 `fallback-v2`로 올리는 편이 낫다. H-1 수정 때 함께.

### 🟢 L-2 공감 목록 3개를 다 쓰면 같은 공감 문장을 다시 쓴다

- 슬픈 대화에서 대체가 4번 넘게 나면 `_avoid_repeat`가 그대로 둔다(기존 규칙: 다 썼으면 그대로). 받아 주는 목록으로 넘기면 공감이 끊기므로 지금 동작을 유지한다. 기록만 남긴다.

### 🟢 L-3 시험이 덮지 않는 경우

- GGARA02 엔진의 `분노`(→화남)·`상처`, 대답 생성 시간 초과 경로, 공감 목록 소진. 동작은 코드로 확인했고(위 표), H-1 수정 때 시간 초과 경로 시험을 함께 넣는다.

## 확인한 것 (PASS)

- 안전: 공감 문장 교체가 `SAFETY_REPLY` 덮어쓰기보다 앞에 있어 위험 턴은 늘 안전 안내. 되묻기·안전 문장은 반복 방지에서 바뀌지 않는다(`companion_keep_fixed_replies`).
- 부정 기분 판정: `usable`이고 턴 라벨이 슬픔·화남·우울일 때만. e900(WORRY→슬픔, DEPRESSED→우울)·GGARA02(슬픔·분노·상처·불안) 모두 맞다. 짧은 말(`SHORT`)·모델 없음(`UNAVAILABLE`)은 `usable=False`라 받아 주는 목록.
- `risk_first` 꺼진 경로: 글 기분 계산이 반환 직전에서 공감 교체 직전으로 옮겨졌을 뿐, 대기 작업이 끝난 뒤라 턴 상한·결과 모양은 같다. gosky는 기본값(켜짐).
- 계약: 턴 응답 칸(`reply`·`replySource`·`textMood`·`riskFlag`)과 `replySource` 값은 그대로다. 공감 문장도 `FALLBACK`이다. BE 중계·저장에 영향 없음.
- 개인 정보: 새 로그·응답 칸 없음.
