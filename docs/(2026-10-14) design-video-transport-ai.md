# 영상 전송 방식 전환 - AI 서버 H.264 수신 PoC 설계 (2026-10-14)

> 상태: **PHASE 0 전제 검증 완료 / PHASE 1 설계안 - 승인 대기 (코드 변경·재시작·설치 없음)**
> 근거: FE "영상 전송 방식 전환 검토 요청 (2026-10-13)", BE `docs/(2026-10-13) design-video-transport-be.md`
> 대상 코드: `SilverBridgeAiServer` main `623a092` (gosky 컨테이너와 동일, 16시간 가동 중, 건드리지 않음)
> 범위: **안 B-lite** = WebSocket으로 H.264(Annex B) 수신 -> 서버에서 JPEG 변환 -> 기존 ingest/analyze 호출. 그 뒤 경로는 변경 없음.

## 0. 요약

- 프레임 소비 지점 9곳 중 **JPEG 바이트가 전제인 곳이 7곳**이다. 전부 `ingest_frame` 이후라서, 입구에서 JPEG로 바꿔 넣으면 **하나도 고칠 필요가 없다**(B-lite가 성립하는 이유).
- 컨테이너에 H.264 디코더가 이미 있다(`imageio_ffmpeg` 정적 ffmpeg 7.0.2, OpenCV 5.0.0 FFMPEG 빌드). **설치할 것이 없다.** PyAV는 없고, GPU 하드웨어 디코드(NVDEC)는 이 ffmpeg에 없다 -> CPU 디코드.
- 비용 예측(합성 장면, 1스레드): 디코드 + JPEG 변환 **640x480 약 1.6ms/장, 1280x720 약 4.6ms/장** -> 15fps 1대 기준 코어 1개의 **약 2~7%**. 현재 송출 1대 기준 AI 컨테이너가 코어 1개의 22~35%라서 증가분은 작다. 추론(19ms/장)보다 디코드가 훨씬 가볍다.
- 남는 위험은 성능보다 **(1) 업로드 자격 설계, (2) H.264 압축이 화재 판정(불꽃 질감·시간 기준 유지 조건)에 주는 영향, (3) 브라우저 -> AI 직결 가능 여부**다.

## 1. 전제 검증 (PHASE 0)

### 1-1. 프레임 소비 지점 분류

| # | 지점 | 위치 | JPEG 바이트 전제 | B-lite에서의 영향 |
|---|---|---|---|---|
| 1 | 수신(입구) | `live_stream_router.py:62` `content_type in {image/jpeg, image/jpg}` 검사 후 `frame.read()` | **입구 형식** | 이곳만 새 경로가 생긴다(WS). 기존 HTTP는 그대로 폴백 |
| 2 | 프레임 저장·fps | `stream_session_service.py` `set_frame` (바이트 보관, 순번, 도착 간격으로 fps) | 아니오 (바이트를 열지 않음) | 단 fps·`analyzedAt`·클립 구간이 **도착 시각**에 의존 - 디코드 완료 시각이 아니라 **WS 메시지 수신 시각**을 넘겨야 한다(6절 위험 R5) |
| 3 | 분석(화재·흉기·낙상) | `_detect_and_store` -> `detect_from_jpeg` (`fire_smoke_detection_service.py:116/149`, `yolo_class_detector.py:137/170`, 내부 `cv2.imdecode`) | **예** | 변환된 JPEG를 주므로 무변경 |
| 4 | 화면 품질 게이트 | `frame_quality.assess_jpeg` (`frame_quality.py:84`, `imdecode` 그레이) | **예** | 무변경. 단 H.264 압축으로 선명도 지표가 달라질 수 있음(R2) |
| 5 | 클립 링버퍼 | `clip_buffer.py` JPEG 그대로 보관, `stream_session_service.py:63` `add` | **예** | 무변경(JPEG로 변환한 값을 넣음) |
| 6 | 클립 인코딩 | `clip_service.py:147` `-f image2pipe -c:v mjpeg`, `parse_jpeg_size`(SOF 마커) | **예(강함)** | 무변경. 단 변환 JPEG이 baseline(SOF0)이어야 `parse_jpeg_size`가 읽는다 - PoC에서 확인 |
| 7 | latest-frame | `live_stream_router.py:125-136` -> `bbox_overlay.render` (`imdecode` -> 박스 -> `imencode` q85) | **예** | 무변경. 재인코딩 세대가 하나 늘어남(H.264 -> JPEG -> 박스 JPEG q85) |
| 8 | MJPEG 보기 | `live_stream_router.py:165-179` multipart `image/jpeg` + `bbox_overlay.render_async` | **예** | 무변경 |
| 9 | 학습 샘플 수집 | `training_sample_collector.offer` (`stream_session_service.py:275`, 원본 JPEG를 디스크 저장, 기본 `COLLECT_ENABLED=false`) | **예** | 무변경. 저장되는 것은 "변환된 JPEG"이며 H.264 원본이 아니다(학습 데이터 분포가 달라지므로 수집을 켤 때 표시 필요) |
| - | (참고) 별도 카메라 URL 분석 | `stream_service.py` `cv2.VideoCapture` 워커 | 아니오 | 송출 업로드 경로와 별개, 영향 없음 |
| - | (참고) 상태 WS | `live_ws_router.py` `/api/v1/ws/live` 텍스트 JSON(`x-api-key`) | 아니오 | 영상 아님. 새 수신 WS는 이것과 **별도 경로**로 둔다 |

### 1-2. 컨테이너 안 디코드 수단 (확인만, 설치·변경 없음)

| 수단 | 결과 |
|---|---|
| Python | 3.11.17 |
| OpenCV | 5.0.0, **FFMPEG 지원 YES**(avcodec 62.28), GStreamer NO, JPEG = libjpeg-turbo 3.1.2 |
| ffmpeg(`imageio_ffmpeg` 정적 바이너리) | 7.0.2. 디코더 `h264` 있음, 인코더 `libx264`·`mjpeg` 있음. **`-hwaccels` 비어 있음**(NVDEC/CUVID 없음) |
| PyAV(`av`) | **없음** |
| `websockets` | 17.2 있음(uvicorn[standard] 의존). `wsproto` 없음 |
| numpy / Pillow | 2.4.6 / 12.3.0 |
| simplejpeg / turbojpeg | 없음(필요 없음, cv2가 libjpeg-turbo) |
| 시스템 `ffmpeg` 명령 | 호스트에는 있고 컨테이너 PATH에는 없음 -> 정적 바이너리 경로 사용(클립 인코딩이 이미 같은 방식) |

결론: **디코드 수단은 `imageio_ffmpeg`의 ffmpeg를 서브프로세스로 쓰는 것이 현실적**이다(PyAV 설치 없이, 클립 서비스가 이미 쓰는 방식). `cv2.VideoCapture`는 메모리 바이트 입력이 불편해 부적합.

### 1-3. CPU·GPU 여유 (읽기 전용 측정, 2026-10-14 13:5x KST)

| 항목 | 값 |
|---|---|
| 호스트 코어 | 12 |
| 측정 시점 송출 | **1대 살아 있음**(최근 60초 `/frame` 요청 850건 = 약 14fps, FE 프록시 경유 `192.168.0.1`) |
| AI 컨테이너 CPU(`docker stats`, 100% = 코어 1개) | 5회 샘플 **22.5 / 30.2 / 33.8 / 34.7 / 35.8%** |
| 호스트 부하(loadavg) | 3.4 / 3.2 / 3.0 |
| AI 컨테이너 메모리 | 1.67GiB / 31GiB |
| GPU | RTX 4070 SUPER, 사용률 0%(샘플 시점), 메모리 8.9/12.3GB, **GPU를 쓰는 프로세스는 AI uvicorn 1개뿐**(공유 없음) |
| 환경값 | `STREAM_SAMPLE_EVERY_N_FRAMES=1`(전 프레임 분석), `CLIP_MAX_FRAMES=150`, `STREAM_STATE_BACKEND=memory` |

해석: CPU는 코어 1개 안팎만 쓰고 있어 **여유가 크다**. GPU는 추론 전용이고 디코드는 CPU라 서로 간섭하지 않는다. (송출이 1대라 3대 동시는 외삽이다.)

### 1-4. 디코드 비용 예측 (합성 장면 마이크로 벤치, 실제 영상 미사용)

방법: 가상 장면(그라디언트 + 노이즈 + 움직이는 사각형·원) 150장을 H.264(baseline, 키프레임 2초, B프레임 없음, 1.5Mbps)로 인코딩해 만든 Annex B 스트림을, **gosky AI 컨테이너 안에서 `nice -n 19`, ffmpeg `-threads 1`** 로 디코드하고 JPEG(q80)로 변환했다. 운영 송출과 같은 컨테이너라 측정 중 영향을 최소화했다(약 20초, 임시 파일은 삭제).

| 해상도 | H.264 평균 크기 | (a) 현재 경로 JPEG `imdecode` | (b) H.264 디코드 + JPEG q80 변환 | 15fps 환산 | 첫 프레임 지연 |
|---|---|---|---|---|---|
| 640x480 | 약 6.5KB/장 | 0.9ms | **1.6ms** (디코드 1.1 + 변환 0.4) | 코어의 **약 2.3%** | 14ms |
| 1280x720 | 약 10.7KB/장 | 2.7ms | **4.6ms** (디코드 3.3 + 변환 1.4) | 코어의 **약 6.9%** | 36ms |

- (c) ffmpeg 내장 mjpeg 인코더로 한 번에 변환하면 1.9ms / 5.0ms로 (b)와 거의 같다 -> **구현은 (c)가 단순**(JPEG가 SOI/EOI로 자기 경계를 가져 프레임 크기를 몰라도 되고, 해상도 파싱이 필요 없음).
- ⚠️ **측정 한계**: 출력 프레임 수가 150이 아니라 90장으로 잡혔다(원인 미확인 - 타임스탬프 없는 raw h264 입력의 ffmpeg 프레임 정리로 추정). 1장당 비용은 출력 프레임 수 기준이라 **과대 쪽(보수적)** 이다. 합성 장면은 실제 거실 영상보다 단순하고 FE의 실제 해상도는 확인하지 못했다(`latest-frame`이 AI 키가 필요하고 영상 열람을 하지 않기로 했다). **PoC에서 실제 송출 해상도로 재측정한다.**
- 입력 크기는 현재 JPEG(약 0.1MB/장, BE 측정) 대비 H.264가 훨씬 작다(대역폭 이득은 FE 몫).

### 1-5. 전제 vs 실제 표

| 전제(요청서) | 실제 |
|---|---|
| 수신은 `image/jpeg`만 허용 | 맞음(`live_stream_router.py:62`) |
| `ingest_frame` -> `analyze_stream_frame_async(detect_from_jpeg)` / 클립 버퍼 / MJPEG | 맞음. 추가로 **품질 게이트·latest-frame·학습 수집**도 JPEG 전제(위 표 4·7·9) |
| 컨테이너에 디코드 수단이 있을 것 | **있음**(ffmpeg 7.0.2 h264 디코더, OpenCV FFMPEG). PyAV·NVDEC는 없음 |
| CPU 여유 | 코어 1개 중 22~35% 사용(송출 1대). 추가 디코드는 15fps 1대당 코어 2~7% 예측 |
| GPU 공유 | 공유 없음. 단 디코드는 CPU |
| 업로드 인증 | **현재는 정적 `X-API-Key` 하나**(`security.py`)이고 FE 프록시가 키를 붙인다. 새 WS 경로에는 이것을 못 쓴다(키를 브라우저에 두면 안 됨) |
| 도착 시각 의미 | fps·`analyzedAt`·클립 구간이 모두 **수신 시각** 기준(`set_frame`). WS 경로에서 이 의미를 지켜야 함 |

## 2. 설계안 (PHASE 1 - 승인 대기)

### 2-1. 구조

```
브라우저(피보호자) --wss, Annex B--> [AI] /api/v1/stream-sessions/{id}/ingest-ws
   1) 연결 수립 시 BE 발급 업로드 자격 검증(키 아님)
   2) 메시지 = H.264 액세스 유닛(Annex B) 1개 이상
   3) 연결당 ffmpeg 서브프로세스(stdin: Annex B, stdout: image2pipe mjpeg, -threads 1)
   4) 나온 JPEG 1장마다 -> (기존) frame_store.set_frame / analyze_stream_frame_async / broadcast
기존 POST /api/v1/stream-sessions/{id}/frame (JPEG) 는 그대로(폴백)
```

- 라우터 본문에 있는 "ingest -> analyze -> session_status·latest_analysis 방송" 묶음을 **함수 하나로 뽑아** HTTP와 WS가 같이 쓴다(복제하면 두 경로가 어긋난다). 이 리팩터가 구현의 유일한 기존 코드 수정이다.
- 새 엔드포인트는 기존 라우터와 **별도 모듈**(예: `live_ingest_ws_router.py`)로 둔다. 기존 `require_api_key` 의존성 아래에 넣지 않는다(WS는 자체 자격으로 인증).

### 2-2. 결정 항목

| # | 항목 | 안 | 근거 |
|---|---|---|---|
| ① | 인증 | **BE가 서명한 1회용 업로드 자격**(`sid`·`exp`≤60초·`jti`·`typ=ingest`)을 AI가 **공유 비밀(HMAC)로 로컬 검증**하고, `jti`는 AI 메모리에 TTL 120초로 기억해 **재사용 거부**. AI 키는 어디에도 싣지 않음 | AI가 BE에 되묻는 방식(introspection)은 AI -> BE 의존·지연이 생기고 BE 장애가 송출을 막는다. 서명 검증은 AI 단독으로 결정되고 BE의 Redis `GETDEL` 1회용 모델과 의미가 같다. 단일 인스턴스 가정(BE도 동일) |
| | 서명 비밀 | 새 환경변수(예: `INGEST_TICKET_SECRET`), `API_KEY`와 **다른 값**. 미설정이면 WS 경로 비활성(기동은 막지 않음) | 키 재사용 시 한쪽 유출이 둘 다 뚫는다 |
| | 자격 위치 | 가능하면 `Sec-WebSocket-Protocol` 또는 첫 메시지. **쿼리 문자열은 로그·프록시에 남으므로 기본 금지**(기존 `/ws/live`의 `[WS-QUERY-KEY]` 경고와 같은 문제) | 브라우저 WS는 임의 헤더를 못 붙이므로 서브프로토콜/첫 메시지가 현실적. BE 합의 필요(7절) |
| ② | 세션 바인딩 | 자격의 `sid`와 URL의 `{session_id}`가 **정확히 같고**, 그 세션이 AI에 이미 등록(`require_session`)돼 있을 때만 허용. 불일치·미등록은 close 1008 | 남의 세션으로 프레임을 밀어 넣는 것(감지·화재 알림 위조)을 막는다. 무인증 연결은 만들지 않음 |
| | 세션당 연결 | 같은 `sid`에 **1연결만**. 새 연결이 오면 기존 연결을 닫고(1013) 교체 | 같은 카메라에 두 송출이 섞이면 fps·클립이 깨진다 |
| ③ | 방어 | 메시지 최대 **256KB**(키프레임 720p도 수십 KB) / 초당 프레임 **상한 = 권장 fps × 1.5**(토큰 버킷, 초과분 드롭) / 전체 동시 연결 **5**(카메라 등록 상한 3 + 여유) / **디코드 실패 연속 30회 또는 키프레임 없이 5초**면 close 1011 / 최대 연결 시간 **자격의 `maxSec`**(BE 정책 30분과 동일하게 재연결 유도) / 수신 유휴 15초면 close | 폭주·메모리 소진·좀비 ffmpeg 프로세스 방지. 임계값은 PoC 측정 후 확정 |
| ④ | JPEG 변환 품질 | 시작값 **ffmpeg `-q:v 4`**(q80 근처, 합성 측정에서 `-q:v 3`은 cv2 q80보다 약 15% 큼) + baseline JPEG(SOF0). PoC에서 FE 현행 JPEG(0.8)와 **같은 장면으로 바이트·감지 신뢰도 분포**를 비교해 값을 고정 | FE 0.8은 canvas `toBlob` 품질이라 ffmpeg 척도와 1:1이 아니다. 근거는 "검출 결과가 같아지는 값"이지 숫자 일치가 아니다 |
| ⑤ | 폴백 | 기존 `/frame` JPEG 경로 **무변경 유지**. FE가 WS 실패 시 기존 경로로 내려갈 수 있다 | 롤백 수단. WS 경로를 끄는 설정(`INGEST_WS_ENABLED=false`)도 둔다(수신만 막고 기존 경로 영향 없음) |
| ⑥ | 관측 | 로그: `sessionId`·**디코드 시간(ms)·프레임 간격·드롭 수·연속 실패·close 사유 코드**만. 프레임·자격·서명 원문 금지. 1초 단위 집계 로그(프레임마다 쓰지 않음) | BE 로그 원문 금지 정책과 같다. 상태 API(`/status`)에 `transport`(`jpeg-http`/`h264-ws`)를 더하는 것은 선택 |

### 2-3. 자격 흐름 (BE 합의 필요)

1. 피보호자 브라우저 -> BE: 업로드 자격 요청(본인 카메라, 현행 시청 티켓과 **분리**된 송출용 자격)
2. BE: 소유 확인 후 HMAC 서명 자격 발급(`sid`·`exp`·`jti`·`maxSec`) - **시청용 티켓과 섞지 말 것**(BE 설계 3-3 위험 표)
3. 브라우저 -> AI WS: 자격 제시 -> AI 로컬 검증
4. 중간 폐기: 연결 해제·정지·카메라 삭제 시 BE가 AI에 **세션 업로드 강제 종료**를 요청할 수단이 필요 -> `POST /api/v1/stream-sessions/{id}/stop`(기존, API 키)이 이미 있으므로 **WS 연결도 이 호출로 닫도록** 연결해 두면 새 API가 필요 없다(구현 시 확인)

## 3. PoC 성공 기준 (제안, 승인 필요)

| 기준 | 제안값 | 측정 방법 |
|---|---|---|
| 감지 일치 | 같은 합성 시험 영상을 **JPEG 경로와 H.264 경로로 각각** 보냈을 때, 분석된 프레임의 `detectedType`·`danger` 시퀀스가 **95% 이상 일치**, 신뢰도 차이 중앙값 ≤ 0.1 | 시험 영상 1세트(화재 유사 장면·정상·낙상·흉기 가상 장면)를 양쪽으로 재생, `latest_analysis` 로그 비교 |
| 화재 유지 조건 | 시간 기준 유지 조건(최근 2초 중 60% 이상 양성)이 **같은 시점에 켜지고 꺼짐**(±1초) | 위 시퀀스의 `danger` 전환 시각 |
| CPU 증가 | 15fps 기준 카메라당 **+10%p(코어 1개 기준) 이내**, 3대 동시 +30%p 이내(예측 +2~7%p/대) | `docker stats` 1초 샘플 5분, 송출 전·후 비교 |
| 프레임 간격 | 최대 프레임 간격 **1초 미만**(10분, 네트워크 중단 제외) | 서버 수신 시각 기록(프레임 내용 없이 시각만) |
| 변환 지연 | WS 수신 -> `set_frame`까지 p99 **100ms 이하** | 로그의 디코드 시간 |
| 안정성 | 30분 연속, ffmpeg 서브프로세스 누수 0(종료 후 프로세스 수 원복), 연결 끊김·재연결 5회 후에도 클립 정상 생성 | `ps`·클립 생성 확인 |
| 클립 | 클립이 정상 생성되고 `parse_jpeg_size`가 변환 JPEG에서 성공 | 클립 API(기존) |

시험 영상은 **가상 장면만** 사용한다(실제 가정 영상 금지). 가상 장면 생성기는 1-4절 벤치 스크립트의 합성 함수를 확장해 쓴다.

## 4. 위험

| # | 위험 | 영향 | 완화 |
|---|---|---|---|
| R1 | **업로드 자격 설계 미확정**(BE 몫) - 자격이 재사용·장수명이면 탈취 시 남의 집에 가짜 프레임을 넣어 화재 알림을 위조할 수 있다 | 높음 | 60초·1회·`sid` 바인딩·`maxSec`·서버 강제 종료 경로. BE 자격 설계 확정 전 구현 금지 |
| R2 | **H.264 압축이 화재 판정에 영향** - 불꽃·연기 질감은 압축에 약하고, 품질 게이트(선명도)·시간 기준 유지 조건(최근 2초 60%)이 JPEG 기준으로 튜닝됨 | 높음(오탐 증가·미탐) | 3절 감지 일치 기준을 **합격선으로**. 불일치 시 비트레이트·키프레임 간격·`-q:v` 조정 또는 B안 포기(C 유지) |
| R3 | 키프레임 대기 지연 - 연결 직후·손실 후 다음 키프레임까지 영상 없음(키프레임 2초면 최대 2초) | 중 | FE에 키프레임 간격 1초 요청, 연결 직후 첫 키프레임 강제 |
| R4 | ffmpeg 서브프로세스 관리(좀비·고아·파이프 막힘) - 연결당 1프로세스 | 중 | asyncio 서브프로세스, close 시 `kill`, 워치독, 동시 5 상한, 종료 시 정리 훅 |
| R5 | 도착 시각 왜곡 - 디코더가 버스트로 내보내면 `fps`·클립 구간 계산이 어긋남 | 중 | 시각은 **WS 메시지 수신 시각**을 쓰고, 디코드 출력에는 그 시각을 큐로 대응시킴 |
| R6 | 재인코딩 세대 증가(H.264 -> JPEG -> 박스 JPEG q85) - latest-frame·MJPEG 화질 저하, 클립 화질 저하 | 낮음 | 클립은 변환 JPEG를 쓰므로 PoC에서 육안 확인(가상 장면) |
| R7 | **브라우저 -> AI 직결 가능 여부 미확인** - AI가 외부에서 TLS(wss)로 열려 있는지 알 수 없다(현재 업로드는 FE 프록시가 대신 호출) | 높음(구조 결정) | 7절 결정 질문 1 |
| R8 | 단일 인스턴스 가정(`jti` 메모리·세션당 1연결·`memory` 상태 백엔드) | 낮음 | 현재 배포 구조와 같음. 확장 시 재설계 |
| R9 | 학습 수집 데이터가 "변환 JPEG"로 바뀜 | 낮음 | 수집 메타에 `transport` 표기(선택) |
| R10 | 측정 한계 - 합성 장면·1대·실제 해상도 미확인(1-4절) | 중 | PoC에서 실제 해상도·3대 동시로 재측정 |

## 5. 측정 방법 (PoC 때 사용)

1. 환경: gosky 동일 컨테이너가 아니라 **별도 포트의 시험 인스턴스**(또는 시험 시간대) - 운영 송출 중에는 측정 오염. 컨테이너 재시작 금지 원칙은 시험 인스턴스에는 해당 없음
2. 입력: 가상 장면 영상(해상도 640x480·1280x720, 15fps, 키프레임 1~2초) - JPEG 경로용과 H.264 경로용을 **같은 원본에서** 생성
3. 지표: CPU(`docker stats` 1초), 디코드 시간(서버 집계 로그), 프레임 간격(수신 시각), 분석 결과 시퀀스(`latest_analysis`)
4. 대조: 같은 영상으로 JPEG 경로 1회 -> H.264 경로 1회 -> 결과 diff
5. 반복: 해상도 2종 x 카메라 1·3대 x 5분

## 6. 이 문서가 확인하지 못한 것

- 로컬 단위 테스트는 **실행하지 못했다**(로컬에 pytest·의존성이 없고, 이미지는 `tests/`를 포함하지 않으며, 설치 금지). 이번 작업은 코드를 바꾸지 않아 영향은 없다.
- FE의 실제 송출 해상도·JPEG 품질(0.8)·WebCodecs 지원 단말(iPad Safari 등)은 확인하지 못했다(FE 몫).
- 3대 동시 부하는 외삽이다(현재 송출 1대).
- `frames_out=90/150` 현상(1-4절)의 원인.

## 7. 승인·결정이 필요한 질문

1. **브라우저가 AI에 직접 WS로 붙을 수 있는가?** (AI의 외부 노출·TLS 종단 여부). 불가하면 BE가 WS를 중계해야 하며 BE 스레드 점유·중계 CPU 문제가 생긴다(BE 설계 3-1의 B안 변형).
2. 업로드 자격을 **서명 검증(제안)** 으로 갈지 BE 확인 방식으로 갈지, 서명 비밀 공유 방식(`INGEST_TICKET_SECRET`).
3. 자격 전달 위치(서브프로토콜 vs 첫 메시지, 쿼리 금지 동의).
4. 3절 성공 기준 수치 승인(특히 CPU +10%p, 감지 일치 95%).
5. PoC를 **시험 인스턴스**에서 할지(권장), 운영 시간대 외 gosky에서 할지.

> 승인 전에는 구현하지 않는다. 구현(PHASE 2)은 [전환-BE] 자격 설계 확정 뒤 AI -> BE -> FE 순서.

---

## 9. 구현 확정 사항 (2026-10-14, 통합 계약 반영)

> 통합 계약: BE `docs/(2026-10-14) design-video-transport-contract.md`. 위 2절과 다르면 **이 절이 우선**한다.

### 9-1. 2절에서 바뀐 것

| 2절 | 확정 |
|---|---|
| ① BE 서명(HMAC) 자격을 AI가 로컬 검증, `INGEST_TICKET_SECRET` | **폐지.** 접속자는 BE 한 곳이고 인증은 기존 `X-API-Key` 헤더(상수 시간 비교). 쿼리 키는 받지 않는다. 자격 검증·소유 확인·60초 재확인은 BE가 한다 |
| ③ 최대 연결 시간 = `maxSec` | **없음.** 무수신 15초(`INGEST_WS_IDLE_TIMEOUT_SEC`)만 종료 조건 |
| 2-3 4 `/stop`으로 WS 닫기 | BE가 자기 WS를 닫으면 AI 연결도 닫힌다. `/stop`의 WS 종료 연계는 하지 않았다 |
| 7절 질문 1 | 브라우저 직결 없음(BE 중계)으로 확정 |

### 9-2. 엔드포인트·메시지

`/api/v1/stream-sessions/{session_id}/ingest-ws` (WebSocket, `app/routers/live_ingest_ws_router.py`)

- 첫 메시지(텍스트): `{"type":"config","codec":"avc1...","width":W,"height":H,"framerate":F,"format":"annexb"}`. 5초 안에 오지 않거나 형식이 틀리면 close 1003 `CONFIG_REQUIRED`. 이후 새 `config`가 오면 디코더를 새로 띄운다(회전·해상도 변경).
- 이후(바이너리): `[8B 타임스탬프(us)][1B 플래그, bit0=키프레임][H.264 Annex B 액세스 유닛]`. 타임스탬프는 쓰지 않는다(프레임 시각 = 서버 수신 시각).
- 서버 -> 클라이언트(텍스트): `{"type":"keyframe_request"}` (1초에 한 번 이하).

### 9-3. 종료 코드

| code | reason | 의미 |
|---|---|---|
| 1008 | `AUTH_INVALID_KEY` / `INGEST_DISABLED` / `SESSION_NOT_FOUND` | 인증 실패 / 플래그 꺼짐 / 세션 미등록 |
| 1003 | `CONFIG_REQUIRED` / `BAD_PAYLOAD` | 첫 메시지가 config 아님 / Annex B 아닌 payload 연속 |
| 1009 | `MESSAGE_TOO_LARGE` | 메시지 256KB 초과(텍스트는 4KB) |
| 1011 | `NO_KEYFRAME` / `DECODE_FAILED` / `DECODER_STALLED` / `DECODER_EXITED` / `DECODER_UNAVAILABLE` | 키프레임 5초 없음 / 출력 없이 30유닛 / stdin 막힘 / ffmpeg 종료 / 바이너리 없음 |
| 1013 | `REPLACED` / `BUSY` | 같은 세션의 새 연결에 교체됨 / 전체 동시 상한 |
| 1000 | `IDLE_TIMEOUT` | 무수신 |

### 9-4. 방어와 동작 규칙

- **세션당 1연결**: 새 연결이 이기고 기존은 1013 `REPLACED`. 전체 동시 상한 5(`INGEST_WS_MAX_CONNECTIONS`).
- **속도 제한**: `config.framerate x 1.5`(상한 `INGEST_WS_MAX_FPS`, 기본 30) 토큰 버킷. 키프레임은 버리지 않는다. **델타를 버리면 이후 델타를 모두 버리고 키프레임을 요청한다**(참조가 깨진 프레임을 디코더에 넣지 않으려는 규칙 - 단순히 초과분만 버리면 화면이 깨진다).
- 첫 키프레임 전의 델타는 버리고 키프레임을 요청한다.
- 디코드한 JPEG는 기존 `/frame`과 같은 함수(`app/services/frame_ingest.py::process_ingested_frame`)를 탄다. HTTP 경로는 이 함수를 쓰도록만 바뀌었고 동작은 같다.
- 분석이 진행 중이면 다음 프레임은 분석을 건너뛴다(기존 규칙 그대로, `_InFlightSessions`).
- 로그: `sessionId`·집계 수치(`INGEST_WS_LOG_INTERVAL_SEC`, 기본 5초)·종료 코드만. 프레임·키 원문 없음.

### 9-5. 환경변수 (모두 기본 꺼짐/보수적)

| 이름 | 기본 | 설명 |
|---|---|---|
| `INGEST_WS_ENABLED` | `false` | 꺼져 있으면 연결을 받자마자 1008 `INGEST_DISABLED`. 기존 `/frame` 경로는 무관 |
| `INGEST_WS_MAX_CONNECTIONS` | 5 | 전체 동시 연결(범위 1~20) |
| `INGEST_WS_MAX_MESSAGE_BYTES` | 262144 | 메시지 상한 |
| `INGEST_WS_IDLE_TIMEOUT_SEC` | 15 | 무수신 종료 |
| `INGEST_WS_KEYFRAME_TIMEOUT_SEC` | 5 | 키프레임 없음 종료 |
| `INGEST_WS_MAX_CONSECUTIVE_FAILURES` | 30 | 출력 없이 넣은 유닛 수 / 잘못된 payload 연속 수 |
| `INGEST_WS_MAX_FPS` | 30 | 속도 제한 상한 |
| `INGEST_WS_JPEG_QUALITY` | 4 | ffmpeg `-q:v`(PoC에서 같은 장면으로 비교해 고정) |
| `INGEST_WS_LOG_INTERVAL_SEC` | 5 | 집계 로그 간격 |

### 9-6. 구현 중 실측으로 확인한 것 (ffmpeg 7.0.2 번들, 합성 장면)

- **`-fflags nobuffer`를 쓰면 프레임을 잃는다**: 30장 중 0~15장만 출력(첫 GOP 유실). 넣지 말 것. `-flags low_delay -probesize 32 -analyzeduration 0`만 쓴다 - 이 조합이 없으면 첫 출력이 약 1.5초 늦다.
- **출력은 입력보다 1장 늦다**(raw H.264 파서가 다음 유닛의 시작을 봐야 프레임 경계를 안다). 15fps에서 약 67ms였고 gosky 실측 변환 지연 p50 71.5ms·창별 max 157ms로 "p99 100ms" 기준에 걸렸다. **해소(2026-10-14)**: `feed()`가 유닛 뒤에 AUD(NAL 9, `00 00 00 01 09 F0`)를 붙인다. 입력→출력 70ms -> 3ms, 프레임 손실 없음(FE 변경 불필요). 스트림 끝에서 ffmpeg가 `missing picture in access unit`을 내지만 마지막 AUD 뒤에 화면이 없어서이며 무해하다.
- 테스트용 합성 스트림은 `-threads 1`로 만든다. libx264의 슬라이스 스레딩은 프레임당 슬라이스 5개를 내서 "한 메시지 = 한 액세스 유닛" 가정을 깬다(WebCodecs 인코더는 보통 프레임당 슬라이스 1개 - FE 실측 필요).

### 9-7. 아직 하지 않은 것

- 합성 영상 기준 **감지 일치·CPU·지연 측정**은 PHASE 4(시험 인스턴스)에서 한다. 이 PR은 수신·변환 경로와 방어까지다.
- 운영 AI 컨테이너 재시작은 송출 세션을 끊는다 - 배포 전에 멈추고 확인한다. 먼저 별도 포트의 시험 인스턴스에서 검증한다.
