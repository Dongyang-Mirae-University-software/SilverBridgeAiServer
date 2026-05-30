# SilverBridge AI Server

FastAPI 기반 AI 서버입니다. 아래 기능을 제공합니다.

- 모델 관리 CRUD
- 카메라 관리 CRUD 및 연결 테스트
- 단일 이미지 분석, 스트림 분석 시작/중지/상태/결과 조회
- 의료 챗 API 및 챗 로그 조회
- API Key 기반 인증

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

- Docs: `/docs`
- ReDoc: `/redoc`
- OpenAPI: `/openapi.json`

## 3. 인증 방식

- 보호된 API 요청 헤더:

```txt
X-API-Key: change-me
```

- 유효하지 않으면 401을 반환합니다.

## 4. 주요 API

### Health

- `GET /health`

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
- `GET /api/v1/chat/logs`
- `GET /api/v1/chat/logs/{id}`

## 5. 표준 응답 형식

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
