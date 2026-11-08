# 라우터 실패 보완 3건 - 누적 알림·크레딧 소진 구분 / 실패 경로 가슴압박 줄 / 엉뚱한 질문 고정 안내 (2026-11-08)

#187(라우터 실패 때 키워드 경로 응급 판정)의 남은 일(L-1 누적 알림, 실패 경로 심정지 줄)과 엉뚱한 질문 답이 생성마다 달라지는 문제를 처리했다.
API 키는 쓰지 않았다(gosky run_eval·실제 라우터 재생·시험 상담 없음) - pytest와 시험지 재분류, SDK 소스 확인만.

## A. 라우터 실패 누적 알림 + 크레딧 소진 구분

- 실패 종류(고정 코드, `router_failure_monitor.failure_kind`): `credit`(크레딧·결제) / `auth`(401·403) / `timeout`(SDK 시간 초과·마감·408·504) / `other` / `off`(키 없음).
  `StructuredCallError` 가 달고 있는 원래 SDK 예외(`__cause__`)로 나눈다 - `claude_structured_client.py`·원인 코드(`api_400` 등)는 그대로.
- 크레딧 판정(anthropic SDK 2.6.0 소스 기준): 본문 `error.type == "billing_error"`, `error.details.error_code` 가 `billing_frozen`·`enforced_spend_limit_reached`, 402,
  또는 400·403 + 크레딧 문구(11/04 실제 사례가 400 `invalid_request_error` + "credit balance is too low"). 문구는 판정에만 쓰고 로그·기록에 남기지 않는다.
- 로그(원인·종류 코드와 개수만, 본문·키 없음):
  - `[CHAT-ROUTER-FAILED] cause=<코드> fallback=keywords kind=<종류>` - 건마다 WARN(기존 줄 끝에 `kind=` 추가)
  - `[CHAT-ROUTER-FAILED-BURST] count= window_sec= causes=<종류/코드:개수>` - 창(기본 600초) 안 실패가 기준(기본 5회)에 닿을 때 ERROR, 창마다 1회
  - `[CHAT-ROUTER-CREDIT-EXHAUSTED] alert=first|repeat cause=` - 크레딧 오류 첫 건에 바로 ERROR, 이어지면 기본 3600초마다 1회
  - `[CHAT-ROUTER-CREDIT-RECOVERED]` - 소진 뒤 라우터가 다시 성공하면 WARN 한 번
- 키가 비어 있으면(`not_ready`) 실패로 세지 않는다(설정이지 장애가 아니다).
- 설정(코드 기본값, `.env` 무변경): `CHAT_ROUTER_FAILURE_ALERT_WINDOW_SEC=600`·`CHAT_ROUTER_FAILURE_ALERT_THRESHOLD=5`·`CHAT_ROUTER_CREDIT_ALERT_REPEAT_SEC=3600`.
- 기록·집계: 실패 턴 `upstreamMeta.routerFailed=true`·`routerFailureKind`. `GET /api/v1/chat/stats` 에 필드 추가
  - `routerFailures`(기간 안 실패 턴 수, 이 배포 뒤 기록만), `routerFailuresByKind`(credit·auth·timeout·other)
  - `routerStatus` = `{state: ok|failing|credit_exhausted|off, recentFailures, windowSec}` - 서버 프로세스의 지금 상태(기간과 무관, 재시작하면 초기화)
- 한계: 단일 프로세스 안 상태다(uvicorn 1개). 재시작하면 창·소진 상태가 비고, 다음 실패에서 다시 알린다.

## B. 실패 경로 심정지 가슴압박 줄

- 라우터가 있으면 `cardiac_arrest` 칸이 맡던 몫. `_keyword_route` 가 응급 여부와 상관없이 `chat_keywords.fallback_arrest_signal` 을 보고
  `fallback_arrest=true` 를 두면 `_router_hints` 가 라우터 칸과 같은 `arrest_hint` 를 넘긴다(MedGemma 생성·생성 실패 대체 응답 모두).
- 근거: #187 거리 패턴 중 `무반응:*`·`무호흡:*` + 줄여 쓴 꼴 2개(`숨 없음`·`흔들어도 … 꼼짝`). 의식 있다는 말(`arrest_conscious`)·걱정·가정 꼴은 제외.
  "몸이 축 처져요"는 피로 표현과 겹쳐 넣지 않았다. 청색증만으로는 붙이지 않는다(119·응급은 그대로).
- 기록 `meta.router.fallback_arrest`(그 밖의 턴 null). `meta.router.cardiac_arrest` 는 라우터가 본 값이라 계속 null.
- 시험지: `cpr_aed=true` 18문항 중 라우터 없이 줄이 붙는 문항 8 → 17(남은 `h2.A07` "꺽꺽 소리 내다 조용해짐"은 라우터 몫). 일반 221건·`cpr_aed=false` 58건 오탐 0.

## C. 엉뚱한 질문 고정 안내

- 최종 경로가 `general`(라우터 성공·실패 모두)이고 응급·자해가 아니며 위기·응급·되묻기 다음 턴이 아니고,
  `chat_keywords.is_off_topic_question` 이 참일 때만: 주제 낱말(날씨·주식·코인·정치·스포츠·연예·드라마·로또·레시피·번역·숙제·코딩·운세 등)
  + 묻거나 부탁하는 꼴 + 건강·몸·생활 관리 낱말·증상 낱말·응급·예약 낱말이 하나도 없음. 애매하면 지금처럼 MedGemma.
- 문구(고정): "저는 건강 상담과 병원 예약을 도와드리는 AI 도우미라서 그 질문에는 답해 드리기 어려워요. 몸이 불편한 곳이나 드시는 약, 병원 예약에 대해 물어봐 주세요."
- `engine=rule`·`riskLevel=low`·`intent=general_chat`·`meta.offTopic=true`. MedGemma·이어서 물어볼 질문 모델을 부르지 않는다(`chat_service` 에서 건너뜀).
- 시험지 326문항(중복 제외) 중 걸리는 문항 0(맞장구 "오늘 날씨가 참 좋네요"는 묻는 꼴이 아니라 MedGemma 그대로).

## 바뀌지 않는 것

라우터 성공 턴의 경로·응급·위기 판정, 자해·위기 고정 응답·해제·109/119 후속 줄, 라우터 지시문, 응답 필드(기록 meta·집계 필드 추가만),
`claude_structured_client.py`·`follow_up_questions.py`·`reservation_orchestrator.py`·`medication_ocr_service.py`(크레딧 소진 대비 세션 몫).

## 검증

- `tests/test_chat_router_failure_followups.py` 77건: 창 경계(4회 무알림 / 5회 1회 / 창 뒤 초기화), 창마다 1회, 로그에 본문·키·API 문구 없음,
  SDK 목 예외 11종 분류, 크레딧 첫 건 즉시·간격 반복·성공 시 회복, 키 없음 미집계, 상태 4종, 집계 필드,
  실패 경로 가슴압박 줄 양성 6·음성 8·시험지 221/58 오탐 0, 엉뚱한 질문 양성 8·음성 12·서비스 경로(모델 미호출) 및 판정 가드.
- 전체 pytest(임시 venv, 모델 패키지 제외): 2121 passed / 16 failed - 실패 16은 말벗 STT·얼굴·표정 환경 문제로 main 기준선과 같다.
- #187 지표 재분류: 응급 80 중 누락 2(`sh.09`·`sh.11`, 설계상 라우터 몫, 이전과 같음), 일반 221 과잉 0.
- 결함 주입 13종 모두 검출.

## 남은 것

- BE(별도 작업): `AdminAiChatStatsResponse`·`AdminAiChatStatsService` 에 `routerFailures`·`routerFailuresByKind`·`routerStatus` 매핑. 관리자 화면 표시는 FE 몫.
- 기존 `arrest_conscious` 가 "가슴이 안 움직여요"의 "움직여요"를 의식 있음으로 읽는다(부정이 낱말 사이에 없는 꼴). 이번 범위 밖 - 후속 점검 대상.
