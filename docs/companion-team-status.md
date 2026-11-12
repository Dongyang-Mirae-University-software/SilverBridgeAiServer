# 말벗·정서 분석 AI 파트 현황 (팀 공유용)

Written for: 팀원 전체 (BE·FE·관리자 화면·AI 담당). 노션 요구사항, AI가 맡은 일, 지금까지 구현한 것, 아직 안 된 것과 그 이유를 한 문서에 정리했습니다.

- 기준 날짜: 2026-11-02
- 구현 위치: `companion-emotion2` 저장소, 브랜치 `feat/companion-emotion2-ai-part` (아직 커밋·push 전, 로컬 검증까지 완료)
- 지금 상태: **서버에 배포되지 않았습니다.** 모든 검증은 로컬 PC에서 했습니다.

---

## 1. 한눈에 보기

| 구분 | 항목 수 | 비율 |
|---|---|---|
| 완료 (구현 + 시험 통과) | 12 | 63% |
| 일부 (구현했지만 팀 결정 또는 다른 쪽 작업이 남음) | 3 | 16% |
| 미구현 | 4 | 21% |
| **합계 (AI 담당 요구사항)** | **19** | 100% |

---

## 2. 요구사항과 AI 담당 범위

노션 요구사항 중 **AI가 줄 데이터와 AI가 만들 기능**만 골랐습니다. 화면 표시, 집계, 알림 발송, 기준선 계산은 백엔드·FE 몫입니다.

| 요구사항 (노션) | AI 담당 여부 | 담당 이유 |
|---|---|---|
| 관리자 숫자 7칸 (질문 수, 응답률, 대화 없는 어르신, 알림 수, 기분 분포, 이어지는 날, 마지막 대화 시각) | AI는 **원자료만** 제공 | 집계는 백엔드가 함 (10/26 합의: 결과는 BE에 둠) |
| 보호자 화면 (지금 기분, 한 줄 설명, 확인할 일, 알림 설정) | 기분·한 줄 설명은 AI 제공 | 문구 생성과 알림 설정은 백엔드 |
| 지켜야 할 원칙 (원문 비노출, 이름은 BE가 붙임, 모르는 값은 null, 탈퇴 시 삭제) | AI가 지킬 부분 있음 | 원문·기분 값·삭제 기능은 AI 쪽 |
| 기분 값 4개 (기쁨·평온·걱정·우울) | AI가 턴 판정 후 묶음 | 판정은 AI |
| 판정 단위 (대화 1회 = 기분 1개) | AI | 대화 끝에 결과 1건 전송 |
| 질문 기록 (응답률 계산용) | AI는 질문 ID를 되돌려 줌 | 질문 발송 기록은 백엔드가 보관 |
| 전송 방식 (a: 대화 끝날 때 결과 1건) | AI | 전송은 AI가 함 |
| 한 줄 설명 (100자 안팎) | AI | LLM 필요 |
| 알림 조건 (위험 → 긴급, 걱정·우울 연속 → 알림) | AI는 긴급 판정, 연속 판정은 백엔드 | 연속 일수는 백엔드가 계산 |

---

## 3. 항목별 구현 현황 (19개)

| # | 항목 | 상태 | 구현 내용 / 근거 |
|---|---|---|---|
| 1 | 대화 1회 결과 전달 | ✅ 완료 | 백엔드가 `finish`를 부르면 응답에 결과 1건 (AI는 따로 보내지 않음, 2026-11-03 변경) |
| 2 | 원문 보내지 않기 | ✅ 완료 | 전송 기록에 원문 없음. 시험으로 확인 |
| 3 | 모르는 값은 null | ✅ 완료 | 판단 보류뿐이면 `mood: null` |
| 4 | 마지막 대화 시각 | ✅ 완료 | `endedAt` 포함 |
| 5 | 대화별 답 수·무응답 수 | ✅ 완료 | `turnCount`, `noResponseCount` |
| 6 | 대화별 기분 1개 | ✅ 완료 | 답한 턴의 최빈값 |
| 7 | 질문 ID 되돌려 주기 | ✅ 완료 | 백엔드 질문 응답의 `questionId`를 그대로 넣음 |
| 8 | 탈퇴 시 말벗 데이터 삭제 | ✅ 완료 (AI 쪽) | `DELETE /companion/wards/{wardId}/data` (대화 메모리, 검수 항목) |
| 9 | 이름은 백엔드가 붙임 (AI는 ID만) | ✅ 완료 | AI는 `wardId`만 다룸 |
| 10 | 표정 사진 저장하지 않기 | ✅ 완료 | 분석 후 바로 버림 |
| 11 | 목소리 감정 분석 안 하기 | ✅ 완료 | 팀 결정과 일치 (목소리 감정 중단) |
| 12 | 받아쓰기 SenseVoice (팀 합의) | ✅ 완료 | 기본 엔진. 측정: 글자 오류 **9.95%**, 0.62초/클립 |
| 13 | 알림 긴급 판정 (위험 플래그) | 🟡 일부 | URGENT·INFO만 결과에 포함. NOTICE는 백엔드가 계산 |
| 14 | 기분 값 4개 이름 | 🟡 일부 | 기쁨·평온·우울은 있음. 걱정은 슬픔·화남을 묶은 **제안 값**. 팀 확정 필요 |
| 15 | 집계용 원자료 (응답률 등) | 🟡 일부 | 대화별 원자료는 제공. 집계와 화면은 백엔드 몫 |
| 16 | 한 줄 요약 (LLM 생성) | ❌ 미구현 | EXAONE 생성이 이 PC에서 실패 (4번 참고) |
| 17 | 연속 일수 (며칠째 이어짐) | ❌ 미구현 | 설계상 백엔드가 계산 (AI는 기록을 쌓지 않음) |
| 18 | 표정 3분류 (웃음·평온·부정) | ❌ 미구현 | 팀 학습 층 파일이 없음 |
| 19 | 정리 모델 EXAONE + QLoRA | ❌ 미구현 | 기본 EXAONE만 로드 가능. QLoRA 가중치 없음 |

**합계:** ✅ 12 · 🟡 3 · ❌ 4

---

## 4. 미구현 항목의 이유와 근거

| 항목 | 이유 | 근거 |
|---|---|---|
| 한 줄 요약 | 요약 문장을 만들 모델이 이 PC에서 실행되지 않음 | EXAONE 3.5 2.4B는 내려받고 bfloat16으로 올리기까지 성공(37초). 하지만 생성 단계에서 원격 코드가 transformers 5.13과 맞지 않아 실패. 한 번은 프로세스가 강제 종료됨 |
| 연속 일수 | 날짜별 기록을 AI가 저장하지 않음. 설계상 백엔드가 맡기로 함 | 계약 문서 "대화 상태는 메모리에만 있다", 10/26 합의 ("판정은 대화가 끝날 때", 결과는 백엔드) |
| 표정 3분류 | 팀이 학습한 3분류 층 파일이 저장소와 이 PC에 없음 | 팀 학습 기록(11/01)의 "한국인 3분류 층"은 팀 쪽 DGX에만 있는 것으로 보임. 없는 층을 흉내 내면 근거 없는 판정이 됨 |
| 정리 모델 (QLoRA) | QLoRA 가중치를 받지 못함 | 팀 학습 기록에만 있음 |

---

## 5. 불일치 항목과 이유

| 항목 | 우리 구현 | 팀 기록 | 불일치 이유 (근거) |
|---|---|---|---|
| 기분 이름 평온·우울 | 평온, 우울 | 편안, 기운 없음 | 우리는 아키텍처 08의 턴 라벨을 따름 (2026-11-02 사용자 결정). 팀은 학습 기록에 다른 이름을 씀. 팀이 하나로 정해야 함 |
| 외로움 | 없음 | 있음 (학습 기록 11/01) | 아키텍처 08 라벨 5개에 외로움이 없음. 팀 결정 필요 |
| 걱정 | 슬픔·화남을 묶은 제안 값 | 있음 | 노션 화면은 "걱정"을 쓰지만 우리 라벨에는 없음. 묶는 규칙을 팀이 정해야 함 |
| 정서 판정 방식 | 글 감정 + 표정 8분류 결합 표 | 말 내용 5칸 정리(EXAONE+QLoRA) + 표정 3분류 + 목소리 행동 | 우리 방식은 아키텍처 08의 결합 표. 팀은 학습 기록(10/22~10/26)에서 글 감정 모델 GGARA02 F1이 낮아 EXAONE 판정으로 바꿈 |
| 표정 분류 | 8분류 → 턴 라벨 | 3분류 (웃음·평온·부정, 부정은 0.8 이상만) | 팀은 무표정을 부정으로 잘못 잡는 문제를 3분류 층으로 고침(부정 2.1%). 우리 층은 없음 |
| 알림 연속 조건 | 긴급은 위험 플래그, 알림은 부정 비율 2배 | 걱정·우울 3일 연속 | 우리 기준은 기준선 계산이 필요한데 AI가 기록을 쌓지 않음. 백엔드 계산으로 정함 |
| 서버 경로 | `/companion/*` | AI 쪽 `/api/v1/checkin`, BE `feature/checkin` | **팀 내부에서 구현이 둘로 나뉨.** 우리 구현은 팀 기록의 경로를 쓰지 않음. 어느 쪽으로 합칠지 정해야 함 |
| 세션·턴 관리 | AI 메모리 세션, 8턴을 AI가 셈 | BE가 세션 ID 생성, 2분·8턴을 BE가 셈 (10/26) | 우리는 설계 결정 D2 (AI 메모리). 팀은 10/26 합의에서 BE로 옮김. 팀 합의를 따르려면 AI 쪽을 고쳐야 함 |
| 받아쓰기 엔진 | SenseVoice (기본), faster-whisper (선택) | SenseVoice | **일치로 바뀜.** 처음에는 faster-whisper였으나 팀 합의에 맞춰 바꿈 |
| 원문 전송 | 전송 안 함 | 원문은 AI DB에만 | **일치로 바뀜.** 처음에는 원문을 보냈으나 10/26 합의에 맞춰 고침 |

---

## 6. 받아쓰기·판정 모델 준비 현황

| 모델 | 용도 | 준비 상태 | 측정 |
|---|---|---|---|
| SenseVoiceSmall | 받아쓰기 (기본) | ✅ 다운로드·로드·실행 | 글자 오류 9.95% (107 샘플 100개), 0.62초/클립 |
| faster-whisper large-v3 | 받아쓰기 (선택) | ✅ 실행 (CPU) | 글자 오류 46%, 17.3초/클립 |
| GGARA02 (KcELECTRA) | 글 감정 | ✅ 실행 | 453 대화 단위 67.8%, 문장 단위 53.1% |
| YuNet + HSEmotion | 표정 검출·8분류 | ✅ 실행 | 사진 한 장으로 확인 (Happiness, 얼굴 검출 성공). 실제 얼굴 평가 샘플은 없음 |
| EXAONE 3.5 2.4B | 요약 문장 | 🟡 로드 성공, 생성 실패 | 로드 37초. 생성은 원격 코드 호환 문제 |
| EXAONE 정리 (QLoRA) | 대답 정리 | ❌ 없음 | 팀 쪽에만 있음 |
| 표정 3분류 층 | 표정 보조 판정 | ❌ 없음 | 팀 쪽에만 있음 |

---

## 7. 백엔드에 전달해야 할 것

### 7-1. 우리가 백엔드에 요청하는 것

| 항목 | 내용 | 상태 |
|---|---|---|
| 질문 전달 | BE가 `POST /companion/dialogues` 본문에 `question`, `questionId`를 넣어 보냄 (AI는 BE를 호출하지 않음, 2026-11-03 변경). | 우리 쪽 수신 코드 구현. (이전 방식 `GET {BE}/companion/question` 호출은 제거) (임시 대역으로만 확인) |
| 대화 결과 전달 | `POST {AI}/companion/dialogues/{id}/finish` 응답 `data` (아래 7-2 필드) | AI 쪽 구현 완료. 백엔드가 finish를 호출해 받음 |
| 질문 발송 기록 | 질문을 보낸 시각과 질문 ID를 백엔드가 저장 | 백엔드 몫 |
| 인증 키 | 요청 헤더 `X-API-Key`. 키 값은 환경 변수로만 공유 | 키 발급 필요 |

### 7-2. 대화 결과 전송 필드 (AI → BE)

| 필드 | 뜻 | 예 |
|---|---|---|
| `dialogueId` | 대화 ID | 32자 문자열 |
| `wardId` | 피보호자 ID | `ward-demo` |
| `questionId` | 백엔드가 준 질문 ID | `q-0900` |
| `slot` | 시간대 | `09:00` |
| `endedAt` | 끝난 시각 (UTC ISO) | `2026-11-02T12:04:00+00:00` |
| `endReason` | 끝난 이유 | `최대 턴 수`, `연속 무응답`, `USER_STOP` |
| `turnCount` | 답한 턴 수 (무응답 포함) | 8 |
| `noResponseCount` | 무응답 턴 수 | 1 |
| `mood` | 기분 (기쁨·평온·걱정·우울) 또는 `null` | `걱정` |
| `summary` | 한 줄 요약 (지금은 항상 `null`) | `null` |
| `riskFlag` | 위험 플래그 여부 | `false` |
| `alertLevel` | 알림 등급 (`URGENT` 또는 `INFO`) | `INFO` |
| `promptVersion` | 판정 규칙 버전 | `{reply, signals}` |

원문은 보내지 않습니다.

### 7-3. AI가 제공하는 API (백엔드·FE가 호출할 수 있는 것)

| 메서드 | 경로 | 용도 | 상태 |
|---|---|---|---|
| GET | `/companion/status` | 말벗 설정 상태 (키 값은 없음) | ✅ |
| POST | `/companion/dialogues` | 대화 열기 | ✅ |
| POST | `/companion/dialogues/{id}/reply` | 글 대답 | ✅ |
| POST | `/companion/dialogues/{id}/reply/audio` | 음성 대답 (WAV·webm·mp3, 10MB 이하, 표정 프레임 최대 5장) | ✅ |
| POST | `/companion/dialogues/{id}/finish` | 대화 끝내기 | ✅ |
| GET | `/companion/review/replies?status=` | 검수 목록 (AI 응답 문장만) | ✅ |
| POST | `/companion/review/replies/{id}` | 검수 승인·반려 | ✅ |
| DELETE | `/companion/wards/{wardId}/data` | 탈퇴 시 AI 쪽 말벗 데이터 삭제 | ✅ |

모든 경로는 `X-API-Key` 헤더가 필요합니다.

### 7-4. 설정 값 (백엔드·운영이 알아야 할 것)

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `COMPANION_API_KEY` | (비움) | 말벗 API 키. 비어 있으면 모든 요청 거부 |
| `COMPANION_BACKEND_BASE_URL` | (비움) | 백엔드 주소. 비면 질문 조회와 결과 전송을 안 함 |
| `COMPANION_BACKEND_API_KEY` | (비움) | 백엔드를 부를 때 쓰는 키 |
| `COMPANION_MAX_TURNS` | 8 | 한 대화의 최대 답 수 |
| `COMPANION_NO_RESPONSE_LIMIT` | 3 | 연속 무응답 종료 기준 (팀 결정 2026-11-02) |
| `COMPANION_STT_ENGINE` | turbo | 말벗 턴 받아쓰기 엔진: turbo(Whisper large-v3-turbo GPU fp16, 11/09) · sensevoice(재희 버전) · whisper |
| `COMPANION_TURBO_SUBDIR` | companion/stt-turbo | turbo 모델 폴더 (`MODEL_BASE_PATH` 아래, `openai/whisper-large-v3-turbo` 파일). 없으면 턴이 503 → `sensevoice`로 되돌린다 |
| `COMPANION_TURBO_DEVICE` | cuda | turbo 장치 (cuda=fp16, cpu=fp32) |
| `COMPANION_STT_API_ENGINE` | sensevoice | 챗봇 받아쓰기 API(`/companion/stt`) 엔진. turbo로 켜면 말소리 감지·반복 정리도 함께 적용 |
| `COMPANION_VAD` | silero | 말소리 감지: silero(말소리 합계 `COMPANION_VAD_MIN_SPEECH_SEC`=0.3초 이상) · rms(재희: 소리 크기 0.01) |
| `COMPANION_STT_COLLAPSE_REPEATS` | true | turbo 결과에서 같은 1~4어절이 3번 이상 연달아 나오면 1번만 |
| `COMPANION_CONFIRM_MODE` | logprob | 되묻기: logprob(turbo 토큰 평균 로그확률 < `COMPANION_CONFIRM_LOGPROB`=-0.3) · lexicon(재희 형태소·어휘 규칙). 확률이 없는 턴은 lexicon |
| `COMPANION_CONFIRM_STREAK_LIMIT` | 3 | 확신도가 낮아 되묻는 턴(위험 아님)이 이 횟수만큼 연달아 나오면 마무리 문장으로 답하고 `NO_RESPONSE`로 끝냄. 0이면 끝내지 않음 |
| `COMPANION_KEEP_FIXED_REPLIES` | true | 되묻는 문장·안전 안내 문장은 같은 대화에서 반복돼도 다른 문장으로 바꾸지 않음. false면 재희 동작 |
| `COMPANION_TURBO_MAX_NEW_TOKENS` | 224 | turbo 30초 조각 하나의 최대 토큰 수(한국어 음절당 약 0.83토큰) |
| `COMPANION_TURBO_RETRY_SEC` | 60 | turbo 로드 실패 뒤 다시 시도할 때까지 초. 0이면 재시작 전까지 안 함 |
| `COMPANION_TEXT_MOOD_ENGINE` | e900 | 글 기분: e900(EXAONE + 기분 LoRA, 4코드 JOY·CALM·WORRY·DEPRESSED, 11/10) · ggara02(재희 KcELECTRA). 어댑터 폴더 `COMPANION_MOOD_LORA_SUBDIR`=companion/llm/lora-mood-e900 |
| `COMPANION_MIN_TEXT_CHARS` | (비움) | 기분 판단에 쓰는 최소 글자 수. 비우면 e900 2자 · ggara02 10자 |
| `COMPANION_FUSION_RULE` | prob_add | 말·표정 합치기: prob_add(글 4코드 확률 + 표정 코드 쪽으로 `COMPANION_FACE_WEIGHT`=0.2, 보류 없음) · plan41(재희: 방향 반대면 판단 보류) |
| `COMPANION_FACE_HEAD` | head3 | 표정: head3(vgaf 특징 + 한국인 3분류 층, 부정은 `COMPANION_FACE_NEG_MIN`=0.8 이상만, 파일 `face-emotion/enet_b0_8_best_vgaf.onnx`·`head3_yunet.pt`) · hsemotion8(재희 afew). 파일이 없으면 hsemotion8 |
| `COMPANION_RISK_PROMPT` | v9 | 위험 판정 프롬프트: v9(노인 자살 경고 신호) · v4(재희) |
| `COMPANION_RISK_THRESHOLD` | (비움) | 위험 '있' 확률 기준. 비우면 v9 0.99 · v4 0.5 |
| `COMPANION_RISK_KEYWORD_RULE` | kw4 | 위험 낱말 규칙: kw4('자살'·'자해' 앞이 한글이면 제외, '아파 죽겠' 같은 관용 제외) · jaehee |
| `COMPANION_REPLY_LORA` | true | 꼬리질문 대답을 대답 LoRA(`COMPANION_REPLY_LORA_SUBDIR`=companion/llm/lora-reply-cc5)로 생성. 프롬프트·규칙 검사는 그대로, 마지막 인사는 기본 모델. 폴더가 없으면 기본 모델 |
| `COMPANION_SUMMARY_STRIP_PREFIX` | true | 한 줄 요약 앞 "요약:"·"기록:" 머리말 지움 |
| `COMPANION_RISK_FIRST` | true | 위험·기분 확률(각 0.1초 안팎)을 생성(신호·요약·대답)보다 먼저 계산 (11/10 점검 H-1). false 면 넷을 함께 보냄 |
| `COMPANION_REPLY_RELAX_CALM` | true | 글 기분이 평온·기쁨인 턴은 '대답과 무관'(대답 낱말 이어 받기) 검사를 빼고, 권유 문장("~해 보세요"·"산책 어떠세요" 등)을 거르는 '권유' 검사를 꼬리질문 턴 전체에 더함 (11/12). 걱정·우울·판단 불가·기분 없음 턴과 `COMPANION_RISK_FIRST=false` 는 전 규칙. false 면 전과 같음 |
| `COMPANION_AVOID_REPLY` | true | 말을 피하는 대답("말하기 싫어"·"묻지 마"·"할 말 없어" 등, `avoidance.py`)이면 모델 대답을 만들지 않고 물러나는 문장(`FALLBACK_AVOID`)으로 답함 (11/12). 위험 턴은 안전 안내가 우선, 숨·가슴·아파 등 몸 불편 말은 감지하지 않음. 턴 응답 `avoided`. false 면 전과 같음 |
| `COMPANION_AVOID_END_COUNT` | 2 | 한 대화에서 이 횟수만큼 말을 피하면(위험 턴 제외) 마무리 문장으로 답하고 `endReason=USER_STOP` 으로 끝냄. 0이면 일찍 끝내지 않음 |
| `COMPANION_CONFIRM_RISK_MODEL` | true | 단어 확인(되묻기) 턴에서도 위험 모델을 돌리고 위험이면 안전 안내 (H-3). false 면 낱말 규칙만 |
| `COMPANION_LLM_GEN_CONCURRENCY` | 2 | 말벗 생성 동시 실행 상한(0 = 없음). 위험·기분 확률은 상한 밖. 마감 시각을 넘긴 작업은 계산하지 않음 (H-5, GPU) |
| `COMPANION_LLM_WARMUP` | true | 말벗 키가 있으면 서버 시작 때 EXAONE·어댑터를 백그라운드로 미리 올림 (M-1, 기동 직후 GPU 약 +2.9GB) |
| `LIVE_DETECT_OOM_RETRY` | true | 실시간 감지가 GPU 메모리 부족이면 캐시를 비우고 그 프레임을 한 번 더 봄 (`[LIVE-DETECT-OOM]`) |
| `COMPANION_FRAME_INTERVAL_SEC` | 2.0 | 표정 프레임 간격 (결정 D5) |
| `COMPANION_FRAME_MAX` | 5 | 한 대답의 최대 프레임 수 |
| `COMPANION_REPLY_MAX_CHARS` | 140 | 응답 최대 글자 수 (결정 D21) |
| `COMPANION_UPLOAD_MAX_MB` | 10 | 음성 파일 최대 크기 |
| `COMPANION_BASELINE_WINDOW_DAYS` | 30 | 기준선 기간 (결정 D22) |
| `COMPANION_BASELINE_MIN_DAYS` | 20 | 기준선 최소 기록 일수 (결정 D22) |
| `COMPANION_BASELINE_RATIO` | 2.0 | 변화 판정 배수 |
| `COMPANION_LOG_RETENTION_DAYS` | 30 | 검수 항목 보관 기간 (결정 D18) |
| `COMPANION_PSEUDONYM_SALT` | (비움, 비밀) | 가명 처리 키. 비우면 개선 DB 기능 끔 |
| `COMPANION_LLM_DTYPE` | bfloat16 | 요약 모델 메모리 설정 |

### 7-5. 오류 코드 (계약 문서 기준)

`docs/companion-contract.md`의 오류 표가 기준입니다. 주요 항목은 다음과 같습니다.

| HTTP | 코드 | 뜻 |
|---|---|---|
| 401 | AUTH_INVALID_KEY | 키가 없거나 틀림 |
| 404 | DIALOGUE_NOT_FOUND | 대화가 없음 |
| 404 | QUESTION_NOT_SET | 그 시각 질문이 아직 없음 |
| 409 | DIALOGUE_ENDED | 이미 끝난 대화 |
| 413 | AUDIO_TOO_LARGE, FRAMES_TOO_MANY, FRAME_TOO_LARGE | 크기 초과 |
| 422 | SLOT_INVALID, ANSWER_EMPTY, AUDIO_MISSING, AUDIO_FORMAT_UNSUPPORTED | 입력 오류 |
| 503 | BACKEND_UNAVAILABLE | 백엔드에서 질문을 가져오지 못함 |
| 503 | STT_UNAVAILABLE | 받아쓰기 모델을 못 씀 또는 시간 초과 |

LLM 실패는 오류로 나가지 않고 미리 정한 대체 문장을 씁니다.

### 7-6. 함께 공유할 문서

| 문서 | 위치 | 내용 |
|---|---|---|
| 계약 문서 | `docs/companion-contract.md` | 경로, 필드, 오류 코드, 설정 |
| 데이터 연동 제안 | `docs/companion-data-handoff.md` | 질문 1~6에 대한 답 |
| 평가 결과 안내 | `docs/eval-results/companion-README.md` | 측정 값과 측정 방법 |
| 노션 토글 문서 | `docs/notion-companion-toggle.md` | 노션에 붙여넣는 요약 |
| 이 문서 | `docs/companion-team-status.md` | 현황 전체 |

---

## 8. 팀이 결정해야 할 것

| 번호 | 항목 | 뭘 정해야 하나 | 영향 |
|---|---|---|---|
| 1 | 기분 이름 | 평온/편안, 우울/기운 없음, 걱정 묶음, 외로움 포함 여부 | 기분 값 4번 항목, 관리자 분포 칸 |
| 2 | 판정 단위 | 대화 1회 = 기분 1개 (현재 방식) 확정 | 분포·이어지는 날 계산 |
| 3 | 서버 경로 | `/companion/*`과 `/api/v1/checkin` 중 어느 쪽으로 합칠지 | 백엔드·FE 연결 전체 |
| 4 | 세션 관리 | 세션과 8턴 카운트를 AI가 할지 BE가 할지 | 우리 구현 수정 범위 |
| 5 | 알림 기준 | URGENT만 할지, 걱정·우울 3일 연속을 쓸지 | 알림 칸 |
| 6 | 한 줄 요약 | 요약 모델을 어디서 돌릴지 (팀 DGX 또는 우리 PC) | 보호자 화면 한 줄 설명 |
| 7 | 연속 일수 | 백엔드가 계산 (현재 방식) 확정 | 관리자 "이어지는 어르신" 칸 |

---

## 9. 다음 할 일

| 순서 | 작업 | 담당 | 조건 |
|---|---|---|---|
| 1 | 팀 결정 7개 확정 | 팀 전체 | — |
| 2 | 백엔드가 `finish`를 호출하고, 대화 열기 요청에 `question`·`questionId`를 넣는 구현 | BE | 1번 결정 후 |
| 3 | 요약 모델 실행 환경 결정 | 팀 | 6번 결정 |
| 4 | 우리 쪽 경로·세션 구조 수정 | AI | 3·4번 결정 후 |
| 5 | 실제 서버 연결 점검 | AI + BE | 2번 완료 후 |
| 6 | 전체 시험 결과 확인 및 커밋 | AI | 사용자 확인 후 |
