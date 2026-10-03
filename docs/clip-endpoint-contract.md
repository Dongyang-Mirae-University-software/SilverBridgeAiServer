# AI 서버 클립 엔드포인트 계약 (SilverBridgeAiServer) — 최종본 v2

작성: 2026-10-04 (v1 최종본 → 구현 반영 v2) / 대상: AI 서버 개발(provider), 백엔드 개발(consumer)
변경 규칙: 한쪽이 계약과 다르게 구현하려면 **이 문서를 먼저 수정**하고 상대에게 알린다.

> **v1 → v2 변경점 (백엔드 확인 필요)** — 아래 본문에 ★로 표시
> 1. 503 `CLIP_DISABLED` 추가(킬 스위치 `CLIP_ENABLED=false`)
> 2. `preSeconds + postSeconds == 0`이면 422. 본문이 JSON 객체가 아니어도 422 `CLIP_INVALID_PARAMS`
> 3. 409 기준: 구간 프레임이 **2장 미만**(`CLIP_MIN_FRAMES`, v1은 "0장")
> 4. 429는 **요청 진입 시점**에 판단: 진행 중(뒤 구간 대기 + 큐 대기 + 인코딩) 요청이 12건(2 + 10) 이상이면 즉시 거절
> 5. `analyzedAt` 의미 정정: 분석 "완료" 시각이 아니라 **분석한 프레임의 수신 시각**(ms 단위 차이) → 보정 불필요
> 6. 응답 헤더 `Cache-Control: no-store` 추가, `/health`에 `ffmpeg` 필드 추가
> 7. 프레임이 초당 15장을 넘으면 고르게 솎아 길이를 유지(fps는 계속 1~15)

## 1. 목적
이상감지(danger=true) 시점의 **5초 영상 클립**(감지 앞 3초 + 뒤 2초)을 만들어 돌려준다.
저장은 백엔드가 한다. **AI 서버는 클립을 영구 저장하지 않는다**(임시 파일만 사용).
기존 `latest_analysis` WebSocket 페이로드와 송출 API는 변경 없음.

## 2. 엔드포인트
`POST /api/v1/live-streams/{sessionId}/clips`
- live_stream_router에 추가, 기존과 같이 `require_api_key` 적용(헤더 `X-API-Key`)

### 요청 (JSON, 본문 생략 가능 = 전부 기본값)
| 필드 | 타입 | 기본 | 설명 |
|---|---|---|---|
| detectedAt | string (ISO-8601) | 요청 수신 시각 | 감지 시각. `latest_analysis.analyzedAt`(naive UTC)을 UTC로 해석해 전달. `Z`·오프셋이 있으면 UTC로 바꾸고, 없으면 UTC로 본다 |
| preSeconds | number | 3 | 0~5. 범위 밖이면 422 |
| postSeconds | number | 2 | 0~3. 범위 밖이면 422 |

구간 = `[detectedAt - preSeconds, detectedAt + postSeconds]`(양 끝 포함), 합계 최대 8초. ★ 합계 0이면 422.

★ **시각 기준**: `analyzedAt`은 `detect_from_jpeg` 진입 직후(디코딩·추론 전)에 찍히고, 링버퍼 시각은 같은 요청에서 같은 시계(`datetime.utcnow()`)로 찍은 프레임 수신 시각이다. 둘의 차이는 ms 단위라 보정하지 않는다. 백엔드가 `detectedAt`을 ms로 잘라 보내면 구간 끝 경계에 정확히 걸린 프레임 1장이 빠질 수 있다(무시 가능).

### 성공 응답 (200)
- `Content-Type: video/webm`, 본문 = WebM(VP8) 바이트(임시 파일로 인코딩해 duration·cue 정보 포함)
- 헤더
  - `X-Clip-Duration-Ms`: 재생 길이(= 프레임 수 / fps × 1000, 정수)
  - `X-Clip-Frames`: 인코딩한 프레임 수
  - `X-Clip-Width`, `X-Clip-Height`: **출력** 해상도(첫 프레임 크기를 짝수로 내림)
  - `X-Clip-Started-At`: 첫 프레임 수신 시각, UTC ISO ms + `Z` (예: `2026-10-04T01:00:03.200Z`)
  - ★ `Cache-Control: no-store`

### 오류 응답
본문은 기존 API와 같은 형식: `{"success": false, "message": "...", "errorCode": "...", "data": null}`

검사 순서: 401 → 503 → 422 → 404 → 429 → (뒤 구간 대기) → 409 → 500

| 상태 | errorCode | 의미 |
|---|---|---|
| 401 | AUTH_INVALID_KEY | API 키 없음·불일치 (기존 의존성) |
| ★ 503 | CLIP_DISABLED | `CLIP_ENABLED=false` |
| 422 | CLIP_INVALID_PARAMS | 파라미터 범위 오류, 합계 0, JSON 파싱 실패·객체 아님, detectedAt 형식 오류 |
| 404 | STREAM_SESSION_NOT_FOUND | 세션 없음 |
| 429 | CLIP_BUSY | ★ 진행 중 요청 12건(동시 2 + 대기열 10) 이상 |
| 409 | CLIP_NOT_ENOUGH_FRAMES | ★ 구간 프레임 2장 미만(감지가 오래됨, 세션 종료(stop 시 버퍼 삭제), AI 재시작으로 버퍼 소실 등) |
| 500 | CLIP_ENCODE_FAILED | ffmpeg 실패·타임아웃, 큐 대기만으로 20초 소진, ffmpeg 없음 |

## 3. 동작 규칙 (AI 서버)
1. **링버퍼**(`app/services/clip_buffer.py`): 프레임 수신(`StreamFrameStore.set_frame`) 때 세션별 `(수신 시각, JPEG 바이트)` 보관.
   - JPEG 바이트 그대로(디코딩 금지), 최근 **10초**(시간 기준 제거) + 최대 **100프레임**
   - 세션 `stop`·같은 sessionId 재시작 시 해당 버퍼 삭제
   - stop 없이 사라진 세션: 프레임이 들어올 때 최대 5초에 한 번 전체를 훑어, 마지막 프레임이 10초보다 오래된 세션 버퍼를 지운다(백그라운드 작업 없음)
   - `CLIP_ENABLED=false`이면 버퍼에 넣지 않는다
2. **뒤 구간 대기**: 구간 끝 이후 프레임이 들어올 때까지 대기. 상한 `min(postSeconds + 1, 구간 끝까지 남은 시간 + 1)`초 — 구간 끝이 이미 1초 넘게 지났으면 기다리지 않는다.
3. **즉시 스냅샷**: 대기가 끝나면 바로 구간 프레임 목록(바이트 참조)을 뜬 뒤 인코딩 큐에 넣는다.
4. **인코딩**: `imageio_ffmpeg.get_ffmpeg_exe()`를 `nice -n 19` 서브프로세스로 실행(nice 없으면 생략), 입력은 stdin `image2pipe`.
   - 옵션: `-hide_banner -loglevel error -y -f image2pipe -framerate {fps} -c:v mjpeg -i - -vf scale={W}:{H} -c:v libvpx -b:v 3M -deadline realtime -cpu-used 8 -threads 2 -pix_fmt yuv420p -an -f webm {임시파일}`
   - `W×H` = 첫 프레임 크기를 짝수로 내린 값(홀수 해상도·클립 도중 회전을 함께 처리)
   - `fps = 프레임 수 / (pre+post)`, 1~15로 제한. ★ 15를 넘으면 프레임을 고르게 솎아 `15 × (pre+post)`장으로 맞춘다
   - 임시 파일(`CLIP_TMP_DIR`, 기본 시스템 tmp)로 출력 → 읽어서 응답 → 즉시 삭제(실패·타임아웃 포함)
5. **동시성**: 인코딩 전용 스레드 2개(이벤트 루프 비차단), 대기열 10. ★ 진입 시점에 진행 중 요청을 세어 12건 이상이면 429. 클라이언트가 끊겨도 인코딩은 끝까지 가고 그때 슬롯을 반환한다(ffmpeg 동시 실행이 2개를 넘지 않음).
6. **타임아웃**: 요청 수신부터 20초. 인코딩 시작 시 남은 시간을 ffmpeg 타임아웃으로 쓰고, 넘으면 강제 종료(kill + wait) 후 500. 큐에서 다 써 버리면 인코딩하지 않고 500.
7. **로그**: sessionId, 프레임 수, 소요 시간, 실패 사유 코드(timeout·returncode 등)만. API 키·프레임 내용·경로·ffmpeg stderr 금지.
8. **멱등성**: 같은 요청도 새로 만든다(캐시 없음). 호출 빈도는 백엔드 쿨다운이 제어한다.
9. ★ `/health` 응답 `data.ffmpeg`(boolean) — ffmpeg 바이너리 존재 여부(경로·버전 비노출).

## 4. 설정 (`.env`)
| 키 | 기본 | 설명 |
|---|---|---|
| CLIP_ENABLED | true | 킬 스위치 |
| CLIP_BUFFER_SECONDS | 10 | 링버퍼 보관 시간 |
| CLIP_MAX_FRAMES | 100 | 세션당 최대 프레임 |
| CLIP_MIN_FRAMES | 2 | 이보다 적으면 409 |
| CLIP_BITRATE | 3M | VP8 비트레이트 |
| CLIP_MAX_CONCURRENCY | 2 | 동시 인코딩 |
| CLIP_QUEUE_MAX | 10 | 대기열 |
| CLIP_TIMEOUT_SEC | 20 | 전체 처리 상한 |
| CLIP_TMP_DIR | (빈 값) | 임시 파일 위치 |

## 5. 의존성
- `requirements.txt`: `imageio-ffmpeg==0.6.0` (정적 ffmpeg 7.0.2, libvpx 포함)
- Dockerfile은 `pip install -r requirements.txt` 단계가 있어 재빌드만 하면 반영된다. 컨테이너에 `/usr/bin/nice` 있음(2026-10-04 확인).

## 6. 성능 기준
| 조건 | 결과 |
|---|---|
| gosky: FHD 25프레임, VP8 3Mbps | 클립 약 1.8~1.9MB, 인코딩 약 0.5~0.6초 (v1 측정) |
| 로컬(8코어, 합성 FHD 26프레임) | 단일 0.32초 / 5건 동시 1.23초 / 10건 동시 2.17초, 이벤트 루프 지연 최대 8ms |

## 7. 백엔드 쪽 동작 (참고)
- 이력 적재 후 클립 쿨다운(기본 5분)을 통과한 감지에만 호출. AI 오류(4xx·5xx 전부)가 나도 이력·알림은 영향 없음.
- 응답 검증: WebM(EBML) 시그니처 `1A 45 DF A3`, 크기 상한 10MB. 호출 타임아웃 20초(+ 네트워크 여유 권장).
- 호출자는 gosky·vkcs-linux 두 서버의 백엔드이며 같은 API 키를 쓴다(`testai.gosky.kr`).
- ★ 503 `CLIP_DISABLED`는 재시도해도 소용없다. 429는 잠시 후 재시도 가능하나 쿨다운 정책상 재시도 없이 실패로 처리해도 된다.

### 백엔드 확인 결과 (2026-10-04, `feature/anomaly-clip`의 `AiClipClient` 기준)
v2 변경점과 백엔드 클라이언트를 대조했다. 대부분 그대로 맞고, 아래 두 가지는 백엔드 쪽 조정을 권장한다.

| 항목 | 현재 백엔드 | 영향 | 권장 |
|---|---|---|---|
| 503 `CLIP_DISABLED` | `status >= 500` → `SERVER_ERROR`(retryable) | AI 킬 스위치가 꺼진 동안 감지 때마다 쿨다운을 풀고 다시 호출한다 | `503` + `CLIP_DISABLED`는 재시도하지 않는 결과로 분류(`REJECTED` 또는 전용 값) |
| 응답 제한 시간 | `request-timeout: 20s` | AI 상한도 요청 수신부터 20초라, 상한 직전에 끝난 응답은 전송 시간 때문에 백엔드에서 `UNAVAILABLE`이 될 수 있다 | 22~25초로 여유를 둔다 |

그 밖의 항목은 변경이 필요 없다: 422 확대(백엔드는 유효한 값만 보냄, `REJECTED`), 409·429·404 분류, `X-Clip-Started-At`(`Z` 포함, `OffsetDateTime.parse` 가능), EBML 시그니처 검사, `detectedAt` 직렬화(`ISO_INSTANT` - 소수점 3·6·9자리 모두 AI가 받으며 9자리는 마이크로초로 잘림, 테스트로 고정).
