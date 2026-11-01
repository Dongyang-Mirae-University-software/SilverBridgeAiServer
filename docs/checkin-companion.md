# 안부 말벗 (checkin) - AI 서버 계약 v0

> 2026-11-01. 하루 다섯 번(09·12·15·18·21시) 어르신께 안부를 여쭙고, 대화가 끝나면 정서·생활 신호를 정리해 백엔드에 준다.
> 백엔드가 세션·결과를 갖고 평소 대비 판정을 한다. 이 서버는 대화 원문(받아쓴 글자·AI 문장)과 턴별 숫자만 갖는다.

## 모델

| 역할 | 모델 | 근거 (DGX) |
|---|---|---|
| 받아쓰기 | SenseVoiceSmall | 어르신 1,200문장 글자 오류 5.5%, 문장당 0.03초 (faster-whisper large-v3 CPU는 14초·자막 문구 덧붙임) |
| 공감·꼬리질문 | EXAONE 3.5 2.4B 4비트 (추가 가중치 끔) | |
| 생활 신호 5칸 | 같은 모델 + `summary` 추가 가중치 | 처음 보는 대답 20개 97/100칸·헛경보 0 |
| 정서 5가지 | 같은 모델 + `emotion` 추가 가중치 | 감성대화 노년 검증 500문장 정확도 70.4%·F1 0.73 (기본 모델 57.8%·0.57), 긍정을 부정으로 0.5% |
| 목소리 | 감정 판정 없이 측정값만 | 목소리 감정 모델은 일상 대화에서 평온을 부정으로 55~85% |
| 얼굴 찾기 | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) | 시험 사진 1,248장 모두 찾음 |
| 표정 (보조 신호) | HSEmotion `enet_b0_8_best_vgaf` 특징 + 한국인 표정으로 다시 학습한 3분류 층(웃음·평온·부정) | AI Hub 82 검증 2.6만 장, 사람 단위 7:3. 원래 8감정은 무표정→부정 32%(50대+ 약 50%). 다시 학습한 층 + 부정 0.8 이상만 부정: 서버 경로로 무표정→부정 2.1%(50대+ 3.6%)·부정 찾음 67%(74%)·웃음 99%, 장당 29ms |

정리와 정서를 한 추가 가중치로 함께 배우게 하면 정서는 좋아졌지만 정리가 97→94칸·헛경보 0→2로 나빠져 둘을 나눴다.

## API (`/api/v1/checkin`, `X-API-Key`, 응답 `{success, message, data}` / 실패 `errorCode`)

| API | 요청 | 응답 `data` |
|---|---|---|
| `POST /sessions` | `{sessionId, userId, slot, question}` - sessionId 는 백엔드가 만든 무작위 값(영숫자·`_-` 8~64자, 사용자 식별자 금지) | `{sessionId, startedAt}` 같은 요청 재전송은 같은 답, 다른 사람·시각이면 409 `SESSION_CONFLICT` |
| `POST /sessions/{id}/turns` | multipart `userId`, `turnNo`(1부터 순서대로), `audio`(webm·ogg·wav·m4a, 5MB·60초), `frames[]`(JPEG, 선택, 최대 120장·장당 300KB) | `{turnNo, transcript, reply, done, voice{answerSec, speechSec, pauseRatio, charsPerSec, loudnessDb}, face, risk{emergency, selfHarm}}` 같은 턴 번호 재전송은 저장된 결과 그대로 |
| `POST /sessions/{id}/finish` | `{userId, selfReport(5가지 중 하나 또는 null), reason}` | `{emotion(5가지 또는 "판단 보류"), selfReport, signals{sleep, meal, pain, mood, lonely}(모르면 null), risk, summary, evidence[{signal, turnNo}], voice(평균), face(턴들을 얼굴 사진 수로 가중 평균)}` 같은 요청은 같은 답 |
| `DELETE /logs?userId=` | 탈퇴 정리 | `{sessions, turns}` 킬 스위치가 꺼져도 동작 |

- 정서 5가지: 기쁨 · 편안 · 외로움 · 기운 없음 · 걱정. "확인 필요"는 백엔드가 평소 대비로 붙인다.
- `summary` 는 단정하지 않는 고정 문장("~라고 하셨어요")이고 대화 원문을 옮기지 않는다. `evidence` 는 어느 턴인지만 준다.
- 위험: 챗봇과 같은 응급·자해 키워드(`chat_keywords`)를 쓴다. 걸리면 그 턴은 고정 안내(119 / 109)로 답하고 `risk` 를 올린다 - 보호자 알림은 백엔드 몫.
- 남의 세션은 없는 것과 같게 404 `SESSION_NOT_FOUND`.
- `face`: `{frames(받은 장수), faceFrames(얼굴 찾은 장수), smile, calm, negative(평균 확률), smileRatio, negativeRatio(그렇게 판정된 사진 비율)}`.
  사진을 안 보냈거나 표정 모델 경로가 없으면 `null`, 얼굴을 못 찾으면 `faceFrames: 0`·나머지 `null`. 표정 분석이 실패해도 대화는 이어간다(그 턴 `null`).
  부정은 확률이 `CHECKIN_FACE_NEG_PCT`(기본 80)% 이상인 사진만 센다. 감정 판정이 아니라 **보조 신호**다 - 연기한 표정·20~40대 위주 데이터로 학습했고 50대+ 시험은 280장뿐이다. 단독으로 "확인 필요"를 만들지 말고 평소 대비로 쓴다.

### 오류 코드

`CHECKIN_DISABLED`(503) · `CHECKIN_BUSY`(429, 생성 대기 10초 초과) · `SESSION_NOT_FOUND`(404) · `SESSION_CONFLICT`·`SESSION_FINISHED`·`TURN_OUT_OF_ORDER`·`TURN_LIMIT`(409) · `AUDIO_EMPTY`·`AUDIO_UNREADABLE`·`SPEECH_NOT_RECOGNIZED`·`FRAME_NOT_JPEG`(422) · `AUDIO_TOO_LARGE`·`AUDIO_TOO_LONG`·`FRAME_TOO_LARGE`·`FRAMES_TOO_MANY`(413) · `CHECKIN_USER_ID_REQUIRED`(422)

## 개인정보

- 음성·사진은 저장하지 않는다. DB(`checkin_sessions`·`checkin_turns`)에는 받아쓴 글자·AI 문장·숫자만, 30일 뒤 자동 삭제(최대 30).
- 로그에는 세션 ID·턴 번호·결과·소요 시간만 남긴다(대화 내용 금지 - `tests/test_checkin.py` 가 고정).

## 설정

| 변수 | 기본 | 설명 |
|---|---|---|
| `CHECKIN_ENABLED` | false | 켜기 전에는 503, 모델을 올리지 않는다 |
| `CHECKIN_EXAONE_MODEL` | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` | transformers 5 에서는 remote code 를 고친 로컬 사본 경로 |
| `CHECKIN_SUMMARY_ADAPTER` · `CHECKIN_EMOTION_ADAPTER` | 비어 있음 | 추가 가중치 폴더 |
| `CHECKIN_SENSEVOICE_MODEL` | `FunAudioLLM/SenseVoiceSmall` | |
| `CHECKIN_FACE_MODEL` · `CHECKIN_FACE_HEAD` · `CHECKIN_FACE_DETECTOR` | 비어 있음 | 표정 모델(16MB)·3분류 층(28KB)·YuNet(230KB). 셋 다 있어야 `face` 를 낸다 |
| `CHECKIN_FACE_NEG_PCT` | 80 | 부정으로 셀 확률(%) |
| `CHECKIN_MAX_AUDIO_BYTES`·`_SEC`·`_FRAMES`·`_FRAME_BYTES`·`_TURNS`·`_RETENTION_DAYS`·`_BUSY_WAIT_SEC`·`_REPLY_MAX_TOKENS` | 5MB·60·120·300KB·8·30·10·80 | |

켜려면 `funasr`·`soundfile`·`av`·`peft`·`bitsandbytes`(표정은 `timm`·`opencv-python` 4.5.4 이상) 가 필요하다(이 PR 은 requirements 를 바꾸지 않는다 - gosky 이미지 크기·기동 영향은 메모리 측정과 함께 정한다).

## 실행 (개발용, 이 모듈만)

```bash
CHECKIN_ENABLED=true DATABASE_URL=sqlite:///./data/checkin_dev.db API_KEY=... \
  python -m uvicorn scripts.checkin_dev_app:app --host 127.0.0.1 --port 6020
```

2026-11-01 DGX 실측: 모델 적재 후 GPU 약 4.4GB, 턴 1.0~1.9초, 정리 1.5초. 표정은 사진 10장에 약 0.25초가 더 든다.
3분류 층(`head3_yunet.pt`)은 서버와 같은 자르기(YuNet + 넓히기)로 학습했다 - 라벨 상자로 학습한 층을 그대로 쓰면 무표정→부정이 3.3→5.5%로 나빠졌다.
