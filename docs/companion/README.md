# 말벗·정서 분석 연동 (AI 서버 측)

> 이 문서는 말벗과 정서 분석이 AI 서버와 어떻게 연결되는지 설명합니다.
> 메인 README 에서 분리했습니다. 기능 설명과 API 는 이 문서가 기준입니다.

### 10-1. 무엇을 하는 기능인가
- **말벗(AI 동반자)**: 보호자가 시간대(09·12·15·18·21시)별로 질문을 정하면, 그 시각에 어르신에게 질문을 보냅니다. 어르신은 한 질문에 대해 **최대 2분** 동안 대화하고, 대답할 때마다 AI가 공감과 꼬리질문을 합니다.
- **정서 분석**: 대답의 글(텍스트 감정)과 같은 순간의 표정을 함께 보고 턴마다 결과를 냅니다. 하루가 끝나면 그날의 결과를 보호자에게 보여 줍니다.
- **음성**: 말한 소리는 faster-whisper(large-v3)로 글자로 바꾼 뒤 텍스트 감정 분석을 합니다.

### 10-2. 코드가 있는 곳
- **이 저장소(AI 서버)**: 말벗이 읽는 **스트림(최신 프레임·상태)** 인증만 담당합니다. 이 브랜치에서 바뀌는 것도 이 부분입니다.
- **말벗 본체**: `companion-emotion` 폴더(이 저장소 밖)에 있습니다. 질문 저장, 대화, 턴 판단, 검수 목록, 하루 결과, 테스트 화면(`/ui/`, `/ui/product/`)이 들어 있습니다.

### 10-3. 이 브랜치에서 바뀐 것
- `COMPANION_STREAM_API_KEY`: 말벗 전용 읽기 키입니다. 비워 두면 말벗 키는 꺼져 있습니다.
- `require_stream_reader_key`: 메인 `API_KEY` 또는 말벗 키만 통과시킵니다. 두 키는 `hmac.compare_digest`로 비교합니다.
- 라우터 전체에 걸던 인증을 **경로별**로 바꿨습니다.
  - 말벗 키 허용: `GET /api/v1/live-streams/{session_id}/latest-frame`, `GET /api/v1/live-streams/{session_id}/status`
  - 메인 키만 허용: 나머지 스트림 경로(송출 생성·프레임 수신·종료, 목록, 최신 분석, MJPEG, 클립)
- 키는 **헤더(`X-API-Key`)로만** 받습니다. 쿼리스트링 키는 받지 않습니다.

### 10-4. 이 브랜치의 API (AI 서버)
| 메서드 | 경로 | 허용 키 | 설명 |
|---|---|---|---|
| GET | `/api/v1/live-streams/{session_id}/latest-frame` | 메인 또는 말벗 | 최신 JPEG 프레임 (저장하지 않고 읽기만) |
| GET | `/api/v1/live-streams/{session_id}/status` | 메인 또는 말벗 | 세션 상태 (`status`, `lastFrameAt`, `fps`, `viewerCount`, `isAnalyzing`) |
| 그 밖의 스트림 경로 | `/api/v1/stream-sessions...`, `/api/v1/live-streams...` | 메인만 | 기존 동작 그대로 |

### 10-5. 말벗 API (companion-emotion, 참고용)
모든 요청에 `X-API-Key` 헤더가 필요합니다. 응답은 `{success, message, data, code}` 형식입니다.

**질문과 예시**
- `GET /companion/question?wardId&slot`: 시각의 질문 조회
- `PUT /companion/questions/{slot}?wardId`: 질문 저장 (`{"question": "..."}`, 200자 이내)
- `GET /companion/examples`: 카테고리별 예시 질문

**대답 한 번 (대화 없이)**
- `POST /companion/answers`: 글 대답 `{wardId, sessionId, slot, text, faceCheckSec}`. `faceCheckSec`(0~10초)을 주면 그 시간 동안 스트림에서 얼굴을 확인합니다.
- `POST /companion/answers/audio`: 음성 대답 (multipart: `wardId, sessionId, slot, audio`)

**대화 (최대 2분, 최대 8턴)**
- `POST /companion/dialogues` `{wardId, sessionId, slot}`: 대화 시작. `dialogueId`, 질문, 제한 시간(120초)을 돌려줍니다.
- `POST /companion/dialogues/{id}/reply` `{text, face?, consentPhoto?, photo?}`: 글 대답. `face`는 같은 순간의 표정 `{label, score}`입니다. 응답에 `label`, `reason`, `reply`(AI 공감과 꼬리질문), `done`, `secondsLeft`, `photoForBackend`가 들어갑니다.
- `POST /companion/dialogues/{id}/reply/audio` (multipart: `audio, faceLabel, faceScore, consentPhoto, photo`): 음성 대답. 응답에 `transcript`가 추가됩니다.
- `POST /companion/dialogues/{id}/finish`: 대화 종료. `sessionLabel`, `summary`(전체 구성), `distribution`, `turns`를 돌려줍니다.

**표정과 결과**
- `POST /companion/expression-check` (multipart: `frame`, JPEG 2MB 이하): 사진 한 장의 얼굴과 표정 (`face`, `expression`, `top`, `topScore`)
- `GET /companion/emotions?wardId&date`: 하루 결과 (`dayResult`, `summary`, `distribution`, `items`)

**검수 (관리자)**
- `GET /companion/review/replies?status=pending|approved|rejected`: AI가 만든 공감·꼬리질문 목록 (어르신 대답 원문은 없음)
- `POST /companion/review/replies/{id}` `{decision: approved|rejected}`

### 10-6. 판단 규칙
**턴 결과 (말을 먼저 봄)**
- 말이 **기쁨**이면 기쁨, **평온**이면 평온. 표정은 보지 않습니다.
- 말이 **부정**이면 표정을 봅니다. 표정이 긍정이면 **판단 보류**, 아니면 **확인 필요**입니다.
- 말이 **애매**하면 표정이 부정이면 **확인 필요**, 긍정·중립이면 **판단 보류**, 표정도 애매하면 **확인 필요**입니다.
- 표정은 한 턴에 사진 3장을 1초 간격으로 읽어 가장 많이 나온 것을 씁니다.

**대화·하루 결과**
- 결과 라벨은 **1위**입니다. 동점 1위 중에 확인 필요가 있으면 확인 필요, 없으면 판단 보류입니다.
- 전체 구성(예: `기쁨 3회 (50%) · 확인 필요 1회 (17%) ...`)을 함께 보여 줍니다.
- "분석 실패"와 "판단 불가"는 결과 계산에서 뺍니다. 정해진 턴이 없으면 판단 불가입니다.

### 10-7. 얼굴 사진과 동의
- 확인 필요 턴에서 **동의(`consentPhoto: true`)가 켜져 있을 때만** 사진 한 장(JPEG, 60KB 이하)을 응답의 `photoForBackend`로 돌려줍니다.
- 서버 기본값은 **꺼짐**입니다. AI 서버는 사진을 **저장하지 않습니다.** 저장은 백엔드가 맡습니다.
- 테스트 화면의 체크 칸은 **테스트용으로 기본 켜짐**입니다. 실제 제품에서는 어르신이 직접 동의하는 화면이 있어야 합니다.

### 10-8. 음성
- 받는 형식: wav, webm, ogg, mp3, m4a, mp4 (10MB 이하). 기기에서 wav로 바꿀 필요가 없습니다.
- 글자 변환은 CPU에서 대략 13~35초 걸립니다. 이 시간은 대화 2분에 포함됩니다.
- 음성 모델은 서버를 켤 때 미리 읽습니다. 서버 시작에 1~3분 걸릴 수 있습니다.

### 10-9. 합칠 때 주의할 점
1. **경로별 인증을 꼭 확인하세요.** 라우터 전체 인증을 뺐기 때문에, 스트림 경로를 새로 추가할 때 `dependencies=_KEY`(또는 말벗 허용이면 `_READER_KEY`)를 붙이지 않으면 **인증 없이 열립니다.**
2. `COMPANION_STREAM_API_KEY`는 `.env`에만 넣습니다. `.env`는 Git에 올라가지 않습니다. `.env.example`에는 빈 값만 있습니다.
3. 말벗 키와 메인 키는 **서로 다른 값**으로 설정하세요. 같은 값이면 키를 나눈 의미가 없어집니다.
4. 말벗이 쓰는 키는 백엔드의 `AI_API_KEY`와 **별개**입니다. 백엔드 설정은 바꿀 필요가 없습니다.
5. 테스트 중 기본값 시험 일부가 실패할 수 있습니다. 개발 PC의 `.env`가 기본값을 켜 두기 때문이며, 코드 문제는 아닙니다. 깨끗한 환경에서 다시 확인하세요.
6. 말벗의 대화 상태와 검수 목록은 현재 **메모리·파일 기반 목업**입니다. 운영 전에 DB로 옮겨야 합니다.

### 10-10. 아직 안 된 것
- 웹 푸시 알림 (백엔드·프론트)
- 대화 기록의 백엔드 DB 저장 (나중에 구현)
- 피보호자·보호자 화면의 실제 제품 구현 (프론트)
- 실제 AI 서버 연결과 어르신 실사용 검증
- 개인정보 보관 기간 확정 (30일 이내 제안, 추후 결정)
