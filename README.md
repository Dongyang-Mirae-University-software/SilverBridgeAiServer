# SilverBridge AI Server

FastAPI 기반 AI 서버입니다. 아래 기능을 제공합니다.

- 모델 관리 CRUD
- 카메라 관리 CRUD 및 연결 테스트
- 단일 이미지 분석, 스트림 분석 시작/중지/상태/결과 조회
- iPad 송출 세션 수신 및 MJPEG 실시간 조회
- 의료 챗 API 및 챗 로그 조회
- API Key 기반 인증
- 예약 API 키 저장 API

## 1. 실행 방법

### 로컬 실행

1) 가상환경 생성 및 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) 환경변수 설정

```bash
cp .env.example .env
```

3) 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload
```

### Docker 실행

```bash
cp .env.example .env
docker compose up --build
```

## 2. Swagger

- Docs: `/api/docs`
- ReDoc: `/api/redoc`
- OpenAPI: `/api/openapi.json`

## 3. 인증 방식

- 보호된 API 요청 헤더:

```txt
X-API-Key: <API_KEY>
```

- 유효하지 않으면 401을 반환합니다.

## 4. 주요 API

### Health

- `GET /health`
  - `data.detectors`: 화재(`fire`)·흉기(`knife`)·낙상(`fall`) 감지기별 `enabled`·`loaded`(오류 문구·경로는 싣지 않음)

### Model

- `POST /api/v1/models`
- `GET /api/v1/models`
- `GET /api/v1/models/{id}`
- `GET /api/v1/models/by-identifier/{identifier}`
- `GET /api/v1/models/by-model-no/{model_no}`
- `PATCH /api/v1/models/{id}`
- `DELETE /api/v1/models/{id}`
- `PATCH /api/v1/models/{id}/activate`

### Camera

- `POST /api/v1/cameras`
- `GET /api/v1/cameras`
- `GET /api/v1/cameras/{id}`
- `GET /api/v1/cameras/by-identifier/{identifier}`
- `PATCH /api/v1/cameras/{id}`
- `DELETE /api/v1/cameras/{id}`
- `POST /api/v1/cameras/{id}/test-connection`

### Analysis

- `POST /api/v1/analysis/image`
- `POST /api/v1/analysis/start`
- `POST /api/v1/analysis/stop`
- `GET /api/v1/analysis/status/{camera_identifier}`
- `GET /api/v1/analysis/latest/{camera_identifier}`
- `GET /api/v1/analysis/results`
- `GET /api/v1/analysis/results/{id}`

### Chat

- `POST /api/v1/chat`
- `GET /api/v1/chat/logs?userId=` — `userId` 필수(없거나 공백이면 422 `CHAT_USER_ID_REQUIRED`). 본인 기록만 반환합니다.
- `GET /api/v1/chat/logs/{id}` — `userId`를 주면 소유자가 같을 때만 반환하고, 다르면 없는 기록과 같은 404 `CHAT_LOG_NOT_FOUND`입니다.
  `CHAT_REQUIRE_USER_ID=true`이면 `userId`가 필수입니다(없으면 422). 기본 `false` — FE가 상세 호출에 `userId`를 붙인 뒤 켭니다.
- 한계: 이 검사는 호출자가 보낸 `userId`를 믿습니다. FE 프록시(`/api/streams/**`)에 로그인 확인이 붙어야 본인 확인이 완성됩니다.

### Games

- `/api/v1/games/**`는 기본으로 API Key 없이 열려 있습니다(피보호자 게임 iframe과 그 안의 JS가 브라우저에서 직접 호출).
- `GAME_REQUIRE_API_KEY=true`이면 키가 필요합니다(없으면 401 `AUTH_INVALID_KEY`). 켜기 전에 FE가 바뀌어야 합니다:
  - 피보호자 게임 iframe 주소(`NEXT_PUBLIC_AI_API_DOMAIN/api/v1/games/embed`)를 키를 붙이는 서버 프록시 경유로 변경
  - embed 페이지 안의 JS 호출(`/state`·`/answer`·`/reset`, 기준 주소 `apiOrigin`)도 같은 프록시를 향하도록 변경
  - 보호자 화면(`/v1/games/progress`·`/activity`)은 이미 프록시 경유라 영향 없음

### Reservation Credentials

- `POST /api/v1/reservation-credentials`

### Medication OCR(약 봉투 인식)

- `POST /api/v1/medication/ocr` (X-API-Key) — `multipart/form-data`의 `image`(jpeg·png·webp, 최대 8MB) → 복약 정보 초안
- 성공(200)은 **표준 래퍼 없이** `{readable, reason, drugs[{name, perDose, timesPerDay, days, slots, timing, confidence}]}`를 그대로 내린다(백엔드 계약). 약 봉투가 아니면 `readable=false`, `drugs=[]`(오류 아님)
- 오류: 400 `OCR_INVALID_IMAGE` · 413 `OCR_IMAGE_TOO_LARGE` · 401 키 · 503 `OCR_DISABLED`/`OCR_NOT_CONFIGURED` · 502 `OCR_UPSTREAM_FAILED`(Claude 실패·거절·30초 초과)
- 사진은 EXIF 방향 반영 → 긴 변 1568px 이하 → EXIF 없는 JPEG로 바꿔 Claude(`MEDICATION_OCR_MODEL`, 기본 sonnet, haiku 금지)에 보낸다. 메모리에서만 다루고 저장·로그하지 않는다. 킬 스위치 `MEDICATION_OCR_ENABLED`
- `drugs[]`의 `inList`·`suggestions`는 서버가 **식약처 의약품 제품 허가정보 품목명**(`app/data/drug_names.txt`)과 대조해 채운다. 괄호 내용·용량 표기·구분 기호를 지우고 비교해 정확히 같거나 앞부분이 같으면(3자 이상) `inList=true`, 아니면 `false`와 비슷한 이름 최대 3개(괄호 내용 뺀 이름, 한글을 자모로 풀어 비교해 한 획 차이 오독도 찾는다. 용량만 다른 품목은 읽은 용량과 같은 것을 먼저 보여 준다). 대조는 스레드에서 돌린다(이름당 수십~수백 ms). 목록 파일을 못 읽으면 `inList=null`·`suggestions=[]`(OCR 응답은 정상). 이름 원문은 바꾸지 않는다(보호자가 고른다)

#### 약 이름 허가 목록 갱신
- 출처: 공공데이터포털 "식약처_의약품 제품 허가정보"(이용허락범위 제한 없음). 품목명만 한 줄에 하나씩 정렬·중복 제거해 둔다
- `DATA_GO_KR_KEY=<디코딩 키> python3 -I scripts/fetch_drug_names.py "<목록 오퍼레이션 요청 주소>"` → `app/data/drug_names.txt`를 덮어쓴다. 바뀐 파일로 PR을 올리고 배포하면 기동 때 다시 읽는다
- 키는 환경변수로만 넣는다 - 저장소·문서·로그에 남기지 말 것

### Live Stream(iPad Ingest + Viewer)

- `POST /api/v1/stream-sessions`
- `POST /api/v1/stream-sessions/{session_id}/frame`
- `POST /api/v1/stream-sessions/{session_id}/stop`
- `GET /api/v1/live-streams`
- `GET /api/v1/live-streams/{session_id}/mjpeg`
- `GET /api/v1/live-streams/{session_id}/latest-frame`
- `GET /api/v1/live-streams/{session_id}/status`
- `GET /api/v1/live-streams/{session_id}/latest-analysis`
- `POST /api/v1/live-streams/{session_id}/clips` — 이상감지 클립(WebM, 감지 앞 3초 + 뒤 2초). 계약: [docs/clip-endpoint-contract.md](docs/clip-endpoint-contract.md)

#### 라이브 이상감지(화재·흉기·낙상)

- 켜진 감지기(`FIRE_SMOKE_ENABLED`·`KNIFE_ENABLED`·`FALL_ENABLED`)를 같은 프레임에 차례로 돌린다(전용 1스레드). 모두 꺼져 있으면 분석하지 않는다.
- `latest_analysis.data`의 `detectedType`·`confidence`·`danger`·`detections`·`analyzedAt`는 백엔드 계약 그대로다. 대표 선택: danger인 것 중 **화재 > 흉기 > 낙상**, danger가 없으면 점수가 가장 높은 감지, 없으면 `normal`(전부 판정 불가면 `unknown`).
- 추가 필드 `results`: 종류별 `{detectedType, confidence, danger, available}`. `detections`에는 모든 종류의 박스가 담긴다.
- `detectedType` 값: `fire`/`smoke`(화재), `knife`(흉기, `knife_handle` 무시), `fall`(낙상 - 모델 클래스 `fallen`을 `fall`로 바꿔 보낸다. 백엔드는 `fallen`을 모른다).
- 낙상은 한 장으로 판정하지 않는다: 최근 `FALL_HOLD_SEC`초(기본 1.5) 동안 분석된 장면 중 `FALL_HOLD_RATIO`(0.7) 이상에서 점수가 `FALL_DANGER_THRESHOLD`(코드 기본 0.4, 서버 `.env`는 0.35) 이상일 때만 `danger=true`. 분석 간격이 `FALL_HOLD_SEC`보다 길면 판정되지 않는다(`[FALL-HOLD-GAP]` WARN).
- 모델이 없거나 로드에 실패하면 그 종류만 꺼진다(서버는 뜬다). 프레임당 추론 시간은 `STREAM_INFER_LOG_INTERVAL_SEC`마다 `[LIVE-INFER]` INFO로 남는다 - 느리면 `STREAM_SAMPLE_EVERY_N_FRAMES`를 늘린다.

#### 실시간 영상의 감지 박스

- `mjpeg`와 `latest-frame` 응답에는 감지 박스(파란 테두리 + `fire 0.73` 라벨)가 **그려질 수 있다**. 분석 JSON(`latest-analysis`·WS)의 형식은 변경 없다 - 백엔드 계약 그대로다.
- 보여주는 순간에만 복사본에 그린다. 분석 입력·클립 링버퍼·`frame_store`의 원본 JPEG는 건드리지 않아 **클립에는 박스가 없다**.
- 표시 기준 `LIVE_BBOX_MIN_CONFIDENCE=danger`(기본) = 종류별 위험 기준(화재·연기 `FIRE_SMOKE_DANGER_THRESHOLD`, 흉기 `KNIFE_DANGER_THRESHOLD`, 낙상 `FALL_DANGER_THRESHOLD`). 박스가 보이면 알림 기준을 넘은 것이다(낙상은 한 장 점수라 알림은 유지 조건까지 통과해야 나간다). 숫자를 주면 모든 종류에 그 값을 쓴다.
- 마지막 분석 결과를 `LIVE_BBOX_HOLD_SECONDS`(기본 1.0) 동안 유지한다. 감지 0건 결과가 오면 즉시 지운다.
- 감지가 없는 프레임은 디코딩·재인코딩 없이 원본 바이트를 그대로 내보낸다(평소 추가 비용 0). 같은 프레임을 여러 시청자가 봐도 그리기는 한 번이다. 그리기가 `LIVE_BBOX_DRAW_TIMEOUT_MS`를 넘거나 오류가 나면 박스 없이 원본을 내보낸다(`[BBOX-DRAW-TIMEOUT]`·`[BBOX-DRAW-FAILED]` WARN).
- **비상 스위치**: `LIVE_DRAW_BBOX=false` 후 재기동하면 이전과 100% 같게(원본 바이트 그대로) 동작한다.

## 6. 무저장 송출 모드

- 아래 설정으로 세션 상태를 DB에 저장하지 않고 메모리로만 운영할 수 있습니다.

```env
STREAM_STATE_BACKEND=memory
```

- 이 모드에서는 서버 재시작 시 라이브 세션 상태가 초기화됩니다.
- 카메라 송출만 필요한 경우 이 모드를 권장합니다.
- 이상감지 클립도 영구 저장하지 않습니다. 최근 10초 프레임을 메모리 링버퍼(JPEG 원본)에만 두고, 클립은 임시 파일로 인코딩해 응답한 뒤 바로 삭제합니다. 저장은 백엔드 몫입니다(`CLIP_*` 설정은 `.env.example`).
- 링버퍼 메모리 상한: 세션당 `CLIP_SESSION_MAX_BYTES`(기본 64MB), 전체 `CLIP_TOTAL_MAX_BYTES`(기본 512MB). 넘으면 오래된 프레임부터 지우고, 전체 상한은 가장 많이 쓰는 세션부터 줄입니다. 상한보다 큰 프레임 한 장은 버퍼에만 넣지 않습니다(분석·송출은 그대로). 로그는 세션 ID와 바이트 수만 남깁니다.

## 7. MediaMTX 운영 연동(권장)

- 운영에서는 iPad가 WebRTC로 MediaMTX에 송출하고, AI 서버는 스트림을 구독해 분석합니다.
- 아래 설정을 활성화하면 라이브 목록 응답에 MediaMTX 기반 URL이 함께 노출됩니다.

```env
MEDIAMTX_ENABLED=true
MEDIAMTX_WEBRTC_INGEST_BASE=https://mediamtx.example.com/publish
MEDIAMTX_WEBRTC_VIEW_BASE=https://mediamtx.example.com/play
MEDIAMTX_HLS_VIEW_BASE=https://mediamtx.example.com/hls
```

- `GET /api/v1/live-streams` 응답 필드:
  - `ingestUrl`: iPad 송출용 URL
  - `viewerUrl`: WebRTC 시청 URL
  - `hlsUrl`: HLS 시청 URL

## 8. iPad 송출 테스트 예시

1) 세션 생성

```bash
curl -X POST "http://localhost:6017/api/v1/stream-sessions" \
  -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId":"stream_001",
    "cameraIdentifier":"ipad-room-001",
    "deviceType":"ipad"
  }'
```

2) 프레임 업로드(JPEG)

```bash
curl -X POST "http://localhost:6017/api/v1/stream-sessions/stream_001/frame" \
  -H "X-API-Key: <API_KEY>" \
  -F "frame=@/path/to/frame.jpg;type=image/jpeg"
```

3) 라이브 조회

```bash
curl -H "X-API-Key: <API_KEY>" "http://localhost:6017/api/v1/live-streams"
```

4) 실시간 보기 URL

- 브라우저에서 아래 URL 열기(요청 헤더 인증이 가능한 클라이언트 권장)
- `http://localhost:6017/api/v1/live-streams/stream_001/mjpeg`

## 9. 표준 응답 형식

성공:

```json
{
  "success": true,
  "message": "요청 처리 완료",
  "data": {}
}
```

실패:

```json
{
  "success": false,
  "message": "에러 메시지",
  "errorCode": "ERROR_CODE",
  "data": null
}
```


## 10. 말벗·정서 분석 연동

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
