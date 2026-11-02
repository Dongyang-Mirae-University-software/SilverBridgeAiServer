# 말벗 AI 계약 (구현 기준)

이 문서는 지금 코드가 실제로 하는 것을 적는다. 명세 원본은 `plan/API_명세_백엔드합의용.html`, 설계는 `docs/AI 말벗 정서 분석 아키텍처.html`이다. 둘과 다르면 이 문서가 아니라 코드를 기준으로 맞춘다.

## 인증

- 모든 `/companion/*` 경로는 헤더 `X-API-Key`가 필요하다. 주소에 키를 넣지 않는다 (명세 9-4).
- 서버의 `COMPANION_API_KEY`가 비어 있으면 모든 요청을 401로 막는다.
- 키 값은 문서, 로그, 응답 어디에도 쓰지 않는다.

## 응답 형식

성공은 `success_response` 형식이다. 실패는 아래 오류 본문이다.

```json
{ "success": false, "message": "…", "code": "DIALOGUE_NOT_FOUND", "errorCode": "DIALOGUE_NOT_FOUND", "data": null }
```

`code`와 `errorCode`는 같은 값이다 (사용자 결정, 둘 다 둔다).

## 경로

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/companion/status` | 말벗 설정 상태 (키 값은 없음) |
| POST | `/companion/dialogues` | 시각(09:00·12:00·15:00·18:00·21:00)의 질문으로 대화를 연다. 본문: `wardId`, `sessionId`, `slot` |
| POST | `/companion/dialogues/{id}/reply` | 글로 된 대답을 보낸다. 본문: `text` |
| POST | `/companion/dialogues/{id}/reply/audio` | 음성 대답을 보낸다. multipart: `audio` 1개, `frame` 최대 5개 |
| POST | `/companion/dialogues/{id}/finish` | 대화를 끝내고 최빈 결과를 받는다 |
| GET | `/companion/review/replies?status=` | 검수 목록 (AI가 만든 문장만, 대답 원문 없음). status: pending·approved·rejected |
| POST | `/companion/review/replies/{itemId}` | 검수 결정. 본문: `decision` = approved·rejected |
| DELETE | `/companion/wards/{wardId}/data` | 탈퇴 시 AI 쪽 말벗 데이터 삭제 (메모리 대화와 검수 목록) |

## 턴 응답 (reply, reply/audio)

```json
{
  "dialogueId": "…", "turnNo": 2,
  "turnResult": { "label": "기쁨", "source": "TEXT" },
  "reply": { "text": "…", "isFollowUp": true },
  "face": { "label": "Happiness", "confidence": 0.8, "frameCount": 3, "usable": true },
  "signals": { "emotion": "positive", "activityMotivation": "mid", "socialContact": "unknown", "physicalCondition": "unknown", "futureOrientation": "unknown", "responseEvasion": false, "riskFlag": false },
  "ended": false, "endReason": null
}
```

- `reply/audio`는 `transcript`(받아 적은 글)도 응답에 넣는다. 글은 저장하지 않는다.
- `signals`는 추출에 실패하면 `null`이다. 대화 응답은 그대로 나간다.
- `turnResult.label`: 기쁨·평온·슬픔·화남·우울·판단 보류 중 하나. 무응답이면 `null`.
- 음성에 글자가 없으면 422가 아니라 무응답 턴이다: 200, `turnResult` = `{label: null, source: NO_RESPONSE}`, `transcript` = `""`, `signals` = `null`. 답변은 미리 정한 문장을 쓴다 (아키텍처 3·4절).
- `turnResult.source`: `TEXT` · `FACE` · `CONFLICT` · `NO_RESPONSE`.
- `reply.isFollowUp`: 마지막 턴(8번째)이면 `false`이고 마무리 인사다.

## 음성 규칙

- 형식: WAV, webm, mp3만 받는다. 형식은 파일 앞부분(매직 바이트)으로 확인한다.
- 크기: 10MB 이하.
- 명세 Q12는 6개 형식을 적었지만 구현은 3개다. 백엔드와 맞추는 것이 남아 있다 (C-27).

## 표정 프레임 규칙

- 한 턴에 최대 5장, 장당 2MB 이하 (JPEG·PNG).
- 얼굴이 없거나 모델을 못 쓰면 `face.usable=false`이고, 대화는 계속된다.
- 영상은 저장하지 않는다 (명세 Q13).

## 오류 코드

| HTTP | code | 뜻 |
|---|---|---|
| 401 | AUTH_INVALID_KEY | 키가 없거나 틀림 |
| 404 | DIALOGUE_NOT_FOUND | 대화가 없음 |
| 404 | QUESTION_NOT_SET | 그 시각 질문이 아직 없음 |
| 404 | REVIEW_ITEM_NOT_FOUND | 검수 항목이 없음 |
| 409 | DIALOGUE_ENDED | 이미 끝난 대화 (9번째 답 포함, 명세 코드와 이름이 다름, 정리 필요) |
| 413 | AUDIO_TOO_LARGE | 음성이 너무 큼 |
| 413 | FRAMES_TOO_MANY | 프레임이 5장을 넘음 |
| 413 | FRAME_TOO_LARGE | 프레임 한 장이 2MB를 넘음 |
| 422 | SLOT_INVALID | 시각 값이 5개 중 하나가 아님 |
| 422 | ANSWER_EMPTY | 글 대답이 비어 있음 |
| 422 | AUDIO_MISSING | 음성 파일이 없음 |
| 422 | AUDIO_FORMAT_UNSUPPORTED | WAV·webm·mp3가 아님 |
| 422 | STATUS_INVALID | 검수 status 값이 잘못됨 |
| 503 | BACKEND_UNAVAILABLE | 질문을 백엔드에서 가져오지 못함 |
| 503 | STT_UNAVAILABLE | 음성 인식 모델을 못 씀 |

LLM 실패는 오류로 나가지 않는다. 1회 재시도 후 미리 정한 대체 문장을 낸다 (D20).

## 데이터 저장

- 턴 기록(대답 원문, 신호, 라벨)은 백엔드 DB에 저장한다 (명세 Q1·Q14). AI 서버는 턴 기록 DB를 두지 않는다.
- AI 서버에는 검수 목록(AI가 만든 문장과 상태)만 SQLite로 둔다. 30일 지나면 지운다 (D18).
- 대화 상태는 메모리에만 있다. 서버를 재시작하면 진행 중인 대화는 사라진다.
- 대화 결과 1건은 대화가 끝날 때 백엔드로 보낸다 (`POST {COMPANION_BACKEND_BASE_URL}/companion/dialogue-results`, 헤더 `X-API-Key`). 턴마다 보내지 않는다 (10/26 합의). 원문은 보내지 않는다. 기분(`mood`)은 판단할 수 없으면 `null`. 주소가 비어 있으면 보내지 않는다. 보내지 못해도 대화 응답은 그대로 나간다.
- 비식별 ID는 HMAC-SHA256으로 만든다 (`COMPANION_PSEUDONYM_SALT`가 비면 만들지 않는다).

## 대화 결과 1건 (AI → 백엔드)

- 보내는 때: 대화가 끝날 때 1번. 턴마다 보내지 않는다 (10/26 합의).
- 보내는 곳: `POST {COMPANION_BACKEND_BASE_URL}/companion/dialogue-results`, 헤더 `X-API-Key`.
- 원문 필드는 없다 (11/02 결정).

| 필드 | 뜻 |
|---|---|
| `dialogueId` | 대화 번호 (같은 값이면 한 번만 저장) |
| `wardId` | 피보호자 ID (이름은 백엔드가 붙임) |
| `questionId` | 백엔드가 준 질문 ID (그대로 되돌려 줌) |
| `slot` | 시간대 (09:00·12:00·15:00·18:00·21:00) |
| `endedAt` | 끝난 시각 |
| `endReason` | 종료 사유 (`최대 턴 수`·`연속 무응답`·`USER_STOP`) |
| `turnCount` | 전체 턴 수 (무응답 포함) |
| `noResponseCount` | 무응답 턴 수 |
| `answeredCount` | 답한 턴 수 (무응답 제외, 판단 보류는 포함). 응답률 = answeredCount ÷ 질문 수 |
| `mood` | 기쁨·평온·걱정·우울 또는 `null` (판단 불가. 0이나 빈 값으로 바꾸지 않는다) |
| `summary` | 한 줄 요약. 지금은 항상 `null` (LLM 필요) |
| `riskFlag` | 위험 신호 여부 |
| `alertLevel` | `URGENT` (위험 신호) 또는 `INFO`. 주의(NOTICE)는 백엔드가 계산 |
| `promptVersion` | 응답·신호 프롬프트 버전 |

## 대화 종료 이유 (endReason)

- `최대 턴 수`: 8번째 답 뒤.
- `연속 무응답`: 3번 연속 무응답 (D15, 사용자 결정 2026-11-02). 아키텍처 문서의 잠정값 2회와 다르다. 문서 정리 필요.
- `USER_STOP`: `finish`로 중간에 그만둠. 그때까지의 턴 수와 결과를 그대로 기록한다 (D16, 사용자 결정 2026-11-02). 아키텍처 문서의 "3턴 미만은 미완료"와 다르다. 문서 정리 필요.

## 검수 목록의 위험 표시

- 위험 플래그(`signals.riskFlag`)가 켜진 턴은 검수 항목에 `risk: true`로 들어간다 (S7-5).

## 아직 맞추지 않은 것

- 백엔드 테스트 서버 연결 실제 주소 (S12-4: 임시 서버로는 확인함, 실제 백엔드는 아직)
- 얼굴 사진 저장 (명세 Q13의 "확인 필요" 턴) 미구현
- main 서버 등록 (S1-3, S1-8) 보류

## 음성 인식 API (챗봇 공유)

- 경로: `POST /companion/stt` (multipart, 파일 필드 `audio`), 헤더 `X-API-Key` 필요
- 응답: `{"success": true, "data": {"text": "..."}}` (글자만. 인식 못 하면 빈 글자)
- 입력 형식: wav, webm, mp3 (머리 부분으로 판단). 크기 10MB 이하, 길이 30초 이하
- 음성과 글자는 저장하지 않고 로그에도 남기지 않는다. 로그에는 길이·처리 시간·상태만 남긴다
- 동시 처리는 2건 (`COMPANION_STT_API_CONCURRENCY`). 자리가 없으면 5초 기다린 뒤 503 `STT_BUSY`
- 오류: 413 `AUDIO_TOO_LARGE`, 413 `AUDIO_TOO_LONG`, 422 `AUDIO_MISSING`, 422 `AUDIO_FORMAT_UNSUPPORTED`, 422 `AUDIO_UNREADABLE`, 503 `STT_UNAVAILABLE`, 503 `STT_BUSY`
- 정서 분석은 호출하지 않는다 (글자 변환만)
