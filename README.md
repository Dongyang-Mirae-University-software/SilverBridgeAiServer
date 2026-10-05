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
- 낙상은 한 장으로 판정하지 않는다: 최근 `FALL_HOLD_SEC`초(기본 1.5) 동안 분석된 장면 중 `FALL_HOLD_RATIO`(0.7) 이상에서 점수가 `FALL_DANGER_THRESHOLD`(0.4) 이상일 때만 `danger=true`. 분석 간격이 `FALL_HOLD_SEC`보다 길면 판정되지 않는다(`[FALL-HOLD-GAP]` WARN).
- 모델이 없거나 로드에 실패하면 그 종류만 꺼진다(서버는 뜬다). 프레임당 추론 시간은 `STREAM_INFER_LOG_INTERVAL_SEC`마다 `[LIVE-INFER]` INFO로 남는다 - 느리면 `STREAM_SAMPLE_EVERY_N_FRAMES`를 늘린다.

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
