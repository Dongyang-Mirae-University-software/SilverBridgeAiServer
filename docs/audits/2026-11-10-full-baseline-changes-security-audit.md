# AI 서버 전체 점검 1차 - 기준선·10/31 이후 변경분·보안 - 2026-11-10

- 대상: AI main `5de3e91`(#209) - 이상감지(카메라 분석·WS·클립), AI 의료 챗봇(라우터·MedGemma·응급/위기 판정·예약 연동·기록·통계·삭제 API), 공용 인증(X-API-Key)
- 제외: 안부 말벗·정서 분석. `/companion/stt`는 챗봇 마이크가 함께 쓰므로 인증·크기·동시 상한·무로그만 봤다.
- 방법: 코드 정독(sonnet 수집 3갈래 → 판정 재확인) + 합성 문장으로 판정 함수 직접 호출 + gosky 읽기 전용 확인(컨테이너 상태·nginx·열린 포트·무인증 GET 응답 코드).
- **API 키 사용 없음**(라우터 호출 0회, pytest + 함수 호출만). AI 서버 재시작·재빌드·`.env` 수정 없음. 코드 수정 없음.
- 문장은 모두 합성 문장이다. 상담 기록·답변 본문·비밀값은 남기지 않는다.

## 종합 판정: **조건부 PASS** (🔴 0 · 🟠 5 · 🟡 9 · 🟢 6)

기능 기준선은 깨끗하다(pytest 실패 0, 응답 형식·오류 코드가 BE와 일치, 상담 본문·받아쓴 글자 로그 없음). 조건은 🟠 5건이다 - 사용자 업로드 이미지를 푸는 Pillow의 알려진 취약점, AI 포트의 인터넷 직접 노출, 응급·위기 판정 빈틈 3건.

## PHASE -1. 기준선

| 항목 | 결과 |
|---|---|
| 날짜 | 2026-11-10 17:06 KST |
| origin/main | `5de3e91` (#209 docs). 로컬 `main` 체크아웃은 19커밋 뒤(`63a2217`) - 건드리지 않고 전용 worktree에서 점검 |
| gosky 배포본 | `f911155`(#206), 작업 트리 깨끗. main과의 차이는 compose `oom_score_adj` 3줄(#208) + 문서뿐 - 제품 코드는 같다 |
| gosky 컨테이너 | `silverbridge-ai-server` Up(11/10 15:50 KST 시작, 재시작 횟수 1), `silverbridge-ai-postgres` healthy(127.0.0.1:6019). `OomScoreAdj=0` - #208은 다음 배포 때 적용 |
| 운영 스위치(비밀 아님) | `CHAT_REQUIRE_USER_ID=true` · `GAME_REQUIRE_API_KEY=true` · `WS_ALLOW_QUERY_API_KEY=false` · `LOG_LEVEL=INFO` |
| 측정 잠금 | `/tmp/ai-measure.lock` 없음. 이번 세션은 만들지 않았다(서버 측정 없음) |
| pytest 전체 | **2534 passed · 4 skipped · 실패 0** (50.7초, gosky 호스트 venv + `git archive origin/main` 사본, 운영 `data/` 미접촉) |
| 의존성 취약점 | 운영 컨테이너 `pip freeze` 154개를 pip-audit + OSV 심각도로 대조 - 아래 표 |

| 패키지 | 알려진 취약점 | high 이상 | 고친 버전 | 판단 |
|---|---|---|---|---|
| pillow 12.3.0 | 39 | **CRITICAL 5 · HIGH 22** | 13.0.0 | 🟠 H-1 - 약 봉투 사진이 이 디코더로 들어간다 |
| torch 2.6.0+cu124 | 9 | HIGH 1(신뢰할 수 없는 체크포인트 로드 RCE) | 2.10.0 | 🟢 L-4 - 요청 경로에 `torch.load` 없음, 가중치는 로컬 파일 |
| accelerate 1.14.0 | 1 | 0(MODERATE) | 없음 | 조치 없음 |

## PHASE 0. 10/31 이후 main 머지 PR (67건, #142~#209)

문서 PR 22건(#143 #145 #147 #148 #150 #153 #155 #157 #158 #161 #162 #164 #166 #168 #170 #176 #179 #183 #190 #197 #202 #209)은 표에서 뺐다. "점검" 칸의 문서는 `docs/audits/` 파일 또는 BE `docs/audit-index.md` 행이다.

| PR | 내용 | 영역 | BE/FE 영향 | 점검 기록 |
|---|---|---|---|---|
| #142 | 연결 끝난 보호자의 대리 상담 기록 삭제 API | 챗 기록 | BE #420이 호출 | BE 전체 점검 L-13에서 나온 변경, 자체 점검 없음 → **PHASE 2에서 확인** |
| #144 #146 #149 | 고정 문장 표·대리 상담 화자 규칙·판정 범위 | 챗 판정 | 없음 | `2026-11-01-chat-proxy-table` |
| #151 #154 | 답 모양(layout)·이모지 | 챗 응답 | FE 표시 | `2026-11-01-chat-reply-layout` |
| #156 | 자해 경고 박스 가족용 문구 | 위기 | 없음 | **없음 → PHASE 1** |
| #159 #165 | 심정지 가슴압박·AED, 알약 오탐 제외 / 점검 반영 | 응급 | 없음 | `2026-11-03-emergency-overtriage` (#165 수정 자체는 미점검 → PHASE 1) |
| #160 #163 | 되묻기 답 칩 | 챗 응답 | FE(`suggestedKind`) | `2026-11-03-followup-answer-chips` |
| #167 #169 | 되묻기 다음 턴 종료 판단 | 챗 | 없음 | `2026-11-04-followup-close-words` |
| #171 #172 | 예약 선택지화 / 점검 반영 | 예약 | FE(`ui` 선택지) | BE audit-index 행 (#172 수정은 미점검 → PHASE 1) |
| #173 | 피보호자 본인 답 짧게 | 챗 | 없음 | **없음 → PHASE 1** |
| #174 #181 | 라우터 심정지·경계 질환 칸 / 대체 응답 가슴압박 줄 | 응급 | 없음 | `2026-11-04-router-emergency` (#181은 미점검 → PHASE 1) |
| #175 | 예약 날짜 선택지 | 예약 | FE(`ui` date) | **없음 → PHASE 1** |
| #178 #180 | 생활·집 안전 질문 상담 경로 / 점검 반영 | 챗 판정 | 없음 | BE audit-index 행 (#180 → PHASE 1) |
| #182 #184 | 프롬프트 캐싱·오프라인 재생 도구 | 도구 | 없음 | `2026-11-05-prompt-cache-replay-tool` |
| #185 #186 | 답변 품질 짧은 판 / 점검 반영 | 챗 | 없음 | BE audit-index 행 (#186 → PHASE 1) |
| #187 #188 #189 #191 | 라우터 실패 키워드 응급·누적 알림·크레딧 소진 대비 | 응급·운영 | #189 약 봉투 503 `OCR_UNAVAILABLE` | `2026-11-09-router-failure-followups` 외 |
| #192 | #188 점검 후속(고정 안내 응급 차단·가슴압박 빈틈) | 응급 | 없음 | **없음 → PHASE 1** |
| #193 #194 | 라우터 실패 때 애매한 자해 표현 | 위기 | 없음 | BE audit-index 행 (#194 수정 → PHASE 1) |
| #196 #199 | 단계 되묻기 / 점검 반영 | 챗 | FE 칩 표시 | `2026-11-09-followup-steps` (#199 → PHASE 1) |
| #198 #200 | `layout.urgent` 표시 칸 | 챗 응답 | FE #228·#230, BE 기록 허용 목록 | FE 점검 후속 (AI 쪽 → PHASE 1) |
| #177 #195 #201 #203 #204 #206 | 말벗 통합·음성·판단 단계 | 말벗 | - | 범위 밖(각자 점검 문서 있음) |
| #205 #207 #208 | 테스트 패키지 표시·검수 DB 임시 폴더·OOM 우선순위 | 테스트·인프라 | 없음 | 제품 코드 무변경, 점검 불요 |

## PHASE 1. 미점검 변경분 점검

| PR | 판정 | 근거 |
|---|---|---|
| #173 | PASS | 새 응답 칸 없음(`meta.wardConcise`는 기록용, BE 허용 목록 밖). `test_chat_ward_concise.py` |
| #175 | PASS | `ui.kind="select"`·`field="date"`·값 `YYYY-MM-DD`/`__calendar__` - BE `UiSelection`은 길이만 본다. `test_reservation_date_choices.py` |
| #181 | PASS | 대체 경로 `test_chat_fallback_emergency_actions.py`, 정상 경로 `test_chat_emergency_overtriage.py` 둘 다 고정 |
| #192 | PASS | `test_router_failure_audit_fixes.py`가 정상·실패 두 경로 고정 |
| #156 | 주의 | 대체 경로·채팅 경로 테스트는 있으나 라우터 `self_harm` 칸으로 들어온 턴의 가족용 박스를 따로 고정한 테스트는 확인하지 못함 |
| #165 #186 #194 | PASS | 정상·실패 두 경로 테스트 있음(#194 `test_router_ok_path_does_not_use_fallback_pattern` 등) |
| #199 | 주의 | 두 경로 테스트는 있으나 답 해석 경계에 빈틈 → 🟠 H-5 |
| #172 #180 | PASS | 응답 형식 변화 없음 |
| #198 #200 | PASS(주의) | `urgent`는 항상 `true`거나 칸 없음 - BE `ChatLogProjection.projectLayout` 규칙과 일치. 주석 불일치 → 🟢 L-2 |

- **응답 형식**: `suggestedQuestions`·`suggestedKind`("answer"만)·`layout`(dict|null)·`ui`가 BE와 맞는다. 전송 응답은 BE가 그대로 중계하고(`suggestedQuestions`만 정리), 기록 조회만 허용 목록으로 거른다.
- **오류 코드**: `/api/v1/chat/*`는 변화 없음. 새 코드는 약 봉투 OCR 503 `OCR_UNAVAILABLE`(#189) 하나, 삭제·변경된 코드 없음. `/chat/stats` 응답에 `routerStatus`가 추가됐다 - BE 관리자 집계가 이 칸을 어떻게 다루는지는 이번에 확인하지 못했다.

### 판정 경계 재확인 (함수 직접 호출, 합성 문장)

| 문장 | 경로 | 결과 | 판단 |
|---|---|---|---|
| "아까 깨셨는데 지금 다시 숨을 안 쉬어요" | 키워드 / 라우터 심정지 힌트 | 응급(119)은 유지, 응급 행동 줄 `[]` - **가슴압박 줄 빠짐**(둘 다) | 🟠 H-3 |
| "잠깐 깼다가 다시 쓰러져서 의식이 없어요" | 〃 | `collapse_fall`만, 가슴압박 줄 없음 | 🟠 H-3 |
| "아버지가 쓰러져서 의식이 없고 숨을 안 쉬어요" | 〃 | `cardiac_arrest` | 정상 |
| "숨은 쉬는데 불러도 반응이 없어요" | 〃 | 응급, 가슴압박 줄 없음 | 정상(숨을 쉬면 가슴압박 대상이 아님) |
| "아버지는 의식이 없어요. 어머니는 숨은 쉬세요" | 〃 | 응급, 가슴압박 줄 없음 | 🟡 M-9 (두 사람이 섞인 문장) |
| "숨 안 쉬는 사람은 어떻게 해야 해요?" | 라우터 실패 | 응급 키워드 없음, `fallback_arrest=True` | 🟡 M-9 (안전 쪽 과잉) |
| "다 끝내고 싶어요" / "힘들어서 다 끝내고 싶어" | 라우터 실패 | 위기 응답 | 정상 |
| "이제 정말 다 끝내고 싶은데요" | 라우터 실패 | 공통 키워드 `[]`·실패 전용 패턴 `[]` - **위기 응답 없음** | 🟠 H-4 |
| "이제 그만 살고 싶은데요"·"사라지고 싶은데요" | 공통 키워드 | 위기 응답 | 정상 |
| "일을 다 끝내고 싶어요"·"숙제 다 끝내고 싶어" | 라우터 실패 | 위기 아님 | 정상(오탐 없음) |
| 위험 신호 질문("숨이 차거나 가슴이 조이세요?")에 "안 답답해요"·"숨이 안 가빠요" | 단계 되묻기 | **`yes` → 119 안내** | 🟠 H-5 |
| 같은 질문에 "예전에도 가끔 그랬어요"·"응급실 가야 하나요?" | 〃 | `yes` → 119 안내 | 🟠 H-5 |
| 같은 질문에 "숨은 안 차요"·"아니요"·"그런 건 없는데요"·"네 괜찮아요" | 〃 | `no` | 정상 |
| "날씨가 더운데 물을 얼마나 마셔야 할까요?" | 생활 질문 판별 | `False`("날씨" 제외어) | 🟢 (라우터가 general로 보내면 짧은 일반 답) |

## PHASE 2. 보안

### 엔드포인트 × 인증

| 라우터 | 경로 | 인증 | 비고 |
|---|---|---|---|
| health | `GET /health` | 없음 | 탐지기 켜짐·로드 여부만 |
| 문서 | `/api/docs`·`/api/openapi.json`·`/api/redoc` | **없음** | 끄는 설정 없음. `testai.gosky.kr`·`:6017` 모두 200 (🟢 L-5) |
| model (8) | `/api/v1/models…` | 라우터 수준 X-API-Key | 삭제·활성화 포함 |
| camera (7) | `/api/v1/cameras…` | 라우터 수준 | `streamUrl` 검증은 스킴뿐 (🟡 M-8) |
| analysis (7) | `/api/v1/analysis/…` | 라우터 수준 | |
| chat (6) | `POST /api/v1/chat`, `GET /stats`, `GET·DELETE /logs`, `DELETE /logs/proxy`, `GET /logs/{chat_id}` | 라우터 수준 | userId는 BE 말을 신뢰(서버 간 신뢰 설계) |
| live_stream (9) | `/api/v1/stream-sessions…`, `/api/v1/live-streams…` | 라우터 수준 | 프레임 크기 상한 없음 (🟡 M-1) |
| live_ws | `WS /api/v1/ws/live` | 핸드셰이크에서 직접 검증(accept 전 1008) | 쿼리 키는 운영에서 꺼짐 |
| reservation_credential (2) | `POST·DELETE /api/v1/reservation-credentials` | 라우터 수준 | |
| medication_ocr (1) | `POST /api/v1/medication/ocr` | 라우터 수준 | 8MB·형식·화소·동시 2건 |
| medgemma_switch (3) | `/api/admin/medgemma/…` | 라우터 수준 **공용 키** | 관리자 전용 키 없음 (🟡 M-4) |
| game | `/api/v1/games`·`/start`·`/progress`·`/activity` | 경로 수준(스위치) | 코드 기본값은 꺼짐, 운영은 켜짐 (🟡 M-3) |
| game | `/embed`·`/{slug}/state`·`/answer`·`/reset` | **없음**(의도 - iframe 직접 호출) | userId 정수 (🟡 M-7) |
| companion (12) | `/companion/…` | 라우터 수준 `COMPANION_API_KEY`(`hmac.compare_digest`) | `/stt`만 점검 |

- **키 미설정 시 닫힘**: `API_KEY`·`COMPANION_API_KEY` 기본값은 빈 문자열이고 비면 전부 거부한다. 코드 안 기본 키·대체 키 없음. HTTP에서 쿼리로 키를 받는 경로 없음.
- **`/companion/stt`**: 말벗 키 인증, 10MB·30초·동시 2건(자리 없으면 5초 뒤 503 `STT_BUSY`)·처리 20초. 로그는 길이(초)·상태·처리 시간뿐이고 받아쓴 글자는 남기지 않는다. 크기·길이 검사 순서는 🟡 M-2.

### 외부에 열린 포트·경로 (gosky 실측)

| 항목 | 사실 |
|---|---|
| uvicorn | `--host 0.0.0.0 --port 6017`(`Dockerfile:27`), compose `network_mode: host` → 호스트 전 인터페이스에 직접 열림 |
| 호스트 방화벽 | `ufw` inactive |
| 외부에서 `http://<gosky>:6017` | `/health` 200 · `/api/docs` 200 · `/api/openapi.json` 200 · `/api/v1/games/embed` 422(인증 없이 인자만 요구) · `/api/v1/chat/logs/1` 401 · `/api/v1/games` 401 |
| nginx `testai.gosky.kr` | `/api/`만 6017로 전달, `client_max_body_size 100M`. BE는 `wss://testai.gosky.kr`로 붙는다 |
| nginx `reservation.dmu.gosky.kr` | `/admin/`을 6017로 전달 - 옛 관리 화면 포트 설정이 남아 지금은 AI 서버가 받는다 (🟢 L-3) |
| AI DB | `127.0.0.1:6019` 로컬 전용 |

### 남의 기록 접근·삭제 API

- 세션 ID만으로 남의 대화를 읽거나 이어 쓸 수 없다 - 이력은 `ChatLog.user_id == userId`로 먼저 거른다(`chat_service.py:702-704`). 요청 본문의 history는 읽지 않는다.
- `GET /logs/{chat_id}`는 `CHAT_REQUIRE_USER_ID`가 꺼져 있으면 userId 없이 열린다(`chat_router.py:98-102`). chat_id는 자동 증가 번호다. **운영은 켜져 있고** BE는 항상 userId를 붙인다(`AiChatClient.java:94-98`). 코드 기본값만 꺼짐 → 🟡 M-3.
- 삭제 범위는 맞다.

| API | 지우는 범위 | BE 호출 |
|---|---|---|
| `DELETE /api/v1/chat/logs?userId=` | 본인 기록 + `router.wardId`·`userContext.wardId`가 그 회원인 남의 대리 기록 (`chat_service.py:1313-1326`) | `AiMemberDataPurgeListener` - 인자 일치 |
| `DELETE /api/v1/chat/logs/proxy?guardianId=&wardId=` (#142) | `user_id == guardianId` 이면서 wardId가 정확히 같은 행만. 보호자 본인 상담·다른 피보호자 행은 남김 (`chat_service.py:1329-1340`) | `AiProxyChatLogPurgeListener` - 인자 일치 |
| `DELETE /api/v1/reservation-credentials?userId=` | 저장된 예약 키 행 + 예약 서비스 계정(8초, 실패해도 진행) | 탈퇴 리스너 - 인자 일치 |

- 조건절 삭제라 중복 호출·없는 사용자는 0건으로 200이다.
- **보류 시간**: 삭제 직후 150초 동안 같은 ID로 기록·예약 계정을 새로 만들지 않는다. 설정이 아니라 코드 상수(`reservation_credential_service.py:22`)이고 목록은 프로세스 메모리다 - 운영은 워커 1개라 동작하지만 재시작하면 사라진다 (🟢 L-1).
- BE가 삭제 호출에 실패하면 재시도가 없다 - BE `policy-chat.md`에 이미 **수용한 한계**로 적혀 있다(재확인만).

### 로그·비밀값·업로드·경로

| 항목 | 결과 |
|---|---|
| 상담 본문·답변 | 로그 없음. `[CHAT-SAFETY]`·`[CHAT-QUALITY]` 값은 개수·고정 이름(`medical_llm_service.py:1306-1355`, 반환형 `int` 확인) |
| 받아쓴 글자 | 로그 없음(`companion_router.py:421-433`) |
| API 오류 문구 | Claude 호출은 상태 코드·클래스명만(`claude_structured_client.py:33,119-122`), 예약 서비스는 상태 코드·마스킹 값만 |
| 키·토큰 | 로그·저장소에 없음. 추적되는 비밀 파일 없음(`.env.example`뿐), `git grep` 결과는 테스트 더미뿐 |
| 예외 원문이 남는 곳 | `fire_smoke_detection_service.py:114,164`·`yolo_class_detector.py:137,187`(`_LOG.exception`) - 이미지 입력이라 상담 내용은 없다 (🟡 M-6) |
| 예약 계정 비밀값 | `RESERVATION_ACCOUNT_SECRET` 기본값 없음, 비면 예약 기능만 꺼짐. 사용자별 예약 키는 DB 평문 (🟡 M-5) |
| 약 봉투 이미지 | 읽기 전 Content-Length 거절 + 누적 8MB 중단, JPEG·PNG·WEBP만, 화소 상한, 동시 2건. `Image.open`에 `formats=` 없음 (🟠 H-1) |
| 경로 조작 | 클립은 `mkstemp`, 스냅샷은 `uuid4`, 학습 샘플은 세션 ID를 `[^A-Za-z0-9_-]` 제거 뒤 사용 - 요청 값이 경로를 벗어날 수 없다 |
| 역직렬화 | 요청 경로에 `pickle`·`yaml.load`·`eval` 없음. 가중치는 로컬 파일 |

## 이슈

### 🟠 높음

**H-1. Pillow 12.3.0의 알려진 취약점 - 사용자 사진이 그 디코더로 들어간다**
- `requirements.txt`(pillow==12.3.0), `app/services/medication_ocr_image.py:53-67`
- OSV 기준 CRITICAL 5(JPEG 디코더 힙 오버플로 `GHSA-8g5g-8vmr-9782` 등)·HIGH 22, 모두 13.0.0에서 고쳐짐. 약 봉투 OCR은 회원이 올린 사진을 `Image.open` → `load()`로 푼다. `formats=` 제한이 없어 형식 식별 단계에서 모든 플러그인 파서가 돈다(허용 형식 검사는 연 뒤).
- 재현: 실제 공격 파일은 쓰지 않았다. 각 CVE가 이 호출 경로(JPEG·PNG·WEBP 읽기)에서 성립하는 조건은 확인하지 못했다 - 그래서 🔴이 아니라 🟠로 둔다.
- 수정안: pillow 13.0.0으로 올리고(ultralytics·torchvision·emotiefflib 호환 확인) `Image.open(..., formats=("JPEG","PNG","WEBP"))`로 여는 단계부터 제한. 말벗 `face_service.py:161`도 같은 줄 수정.

**H-2. AI 포트 6017이 nginx를 거치지 않고 인터넷에 직접 열려 있다**
- `Dockerfile:27`(`--host 0.0.0.0`), `docker-compose.yml:35`(`network_mode: host`), gosky `ufw` inactive
- 재현: 외부에서 `curl http://<gosky>:6017/health` → 200, `/api/docs` → 200.
- 영향: 암호화 없는 평문 HTTP로 전 API가 닿는다. 막는 것은 X-API-Key 하나이고 그 비교는 상수 시간이 아니다(M-4). nginx의 `/api/` 범위·본문 100M 제한이 우회된다(M-1·M-2와 겹침). 무인증 게임 경로(M-7)가 누구에게나 열린다.
- 수정안: `--host 127.0.0.1`로 묶는다(같은 호스트의 nginx가 넘기므로 BE·FE 경로는 그대로). 6017에 직접 붙는 쓰임이 없는지 먼저 확인. **AI 서버 재시작이 필요해 사용자 확인 대상.**

**H-3. "깼다가 다시" 꼴 심정지 문장에서 가슴압박 줄이 빠진다 (라우터 정상·실패 둘 다)**
- `app/services/chat_keywords.py:246`(`_ARREST_RECOVERED`), `:277-283`(`arrest_conscious`), `app/services/medical_llm_service.py:1005-1010`
- 재현: `emergency_action_sentences("아까 깨셨는데 지금 다시 숨을 안 쉬어요", arrest_hint=True)` → `[]`. "잠깐 깼다가 다시 쓰러져서 의식이 없어요" → `collapse_fall`만.
- 원인: 문장 어디든 "깼·깨셨"이 있으면 회복으로 본다. 119 안내는 나가지만 가슴압박·AED 줄이 없다.
- 수정안: 회복 낱말 **뒤**에 반응·호흡 없음 표현이 다시 나오면 회복으로 보지 않는다("다시·또" + 심정지 낱말). 두 경로 테스트 추가.

**H-4. 라우터 실패 때 "다 끝내고 싶은데요" 어미 변형이 위기 응답을 놓친다**
- `app/services/chat_keywords.py:152`(`_FALLBACK_SELF_HARM_BARE`, `fullmatch`), `app/services/chat_service.py:318`
- 재현: `fallback_self_harm_signals("이제 정말 다 끝내고 싶은데요")` → `[]`, 공통 키워드도 `[]`. "다 끝내고 싶어요"는 잡힌다.
- 영향: 라우터 실패 턴에 한정(정상 턴은 라우터가 판정). 크레딧 소진 때는 실패 경로가 계속 쓰이는 경로가 된다.
- 수정안: 허용 어미에 "-은데(요)·-네(요)·-다(고)"류를 더하고 "일을·숙제" 제외는 유지. #194 시험지에 어미 변형 문항 추가.

**H-5. 단계 되묻기 위험 신호 질문에서 부정 답이 "네"로 읽혀 119 안내가 나간다**
- `app/services/followup_steps.py:35`(`_YES`), `:48`(`_NEG`), `:239-`(`interpret`)
- 재현: 가슴·기침 묶음 첫 질문에 "안 답답해요"·"숨이 안 가빠요" → `yes`. "예전에도 가끔 그랬어요"(`^예`)·"응급실 가야 하나요?"(`^응`) → `yes`.
- 영향: 안전 쪽 오판이지만 사용자 답을 뒤집어 불필요한 119 안내를 낸다(#159에서 줄이려던 과잉 119).
- 수정안: `_NEG`를 "안" + 증상 용언 일반 꼴로 넓히고, `_YES`의 `예`·`응`은 뒤가 조사·끝·문장부호일 때만. 묻는 꼴은 다시 묻기로.

### 🟡 중간

| ID | 내용 | 위치 | 수정안 |
|---|---|---|---|
| M-1 | 프레임 업로드에 크기·해상도 상한이 없다. Content-Type 헤더만 보고 끝까지 읽어 `cv2.imdecode`에 넣는다. 분석 실행기가 1개라 큰 프레임이 화재 감지를 늦춘다(키 필요) | `live_stream_router.py:62-69` | 바이트 상한(예: 2MB) + JPEG 매직 바이트 + 디코딩 뒤 해상도 상한 |
| M-2 | `/companion/stt`가 본문을 다 읽은 뒤 크기를, 다 푼 뒤 길이를 본다. 디코딩은 동시 상한 밖 | `companion_router.py:397-416` | Content-Length 사전 거절 + 스트리밍 누적 중단(OCR과 같은 방식), 디코딩을 상한 안으로 |
| M-3 | 보안 스위치 3종의 코드 기본값이 열림 쪽(`CHAT_REQUIRE_USER_ID` false·`GAME_REQUIRE_API_KEY` false·`WS_ALLOW_QUERY_API_KEY` true). gosky는 안전한 값이지만 `.env`가 빠지면 열린다 | `config.py:270,273,276` | 기본값을 닫힘 쪽으로 뒤집기(BE는 이미 userId를 항상 보냄) |
| M-4 | 공용 키 비교가 상수 시간이 아니고(말벗은 `hmac.compare_digest`), 키 하나로 챗 조회·삭제·모델 삭제·MedGemma 끄기까지 된다 | `security.py:12`, `live_ws_router.py:29`, `main.py:149` | `hmac.compare_digest`로 통일. 관리 경로 키 분리는 선택 |
| M-5 | 사용자별 예약 서비스 키를 DB에 평문 저장. DB 비밀번호에 코드 기본값(`.env`가 덮어씀, 포트는 로컬 전용) | `models/reservation_credential.py:19`, `config.py:112` | 서버 비밀로 암호화 저장 또는 수용 기록. 기본 비밀번호 제거(미설정 시 기동 실패) |
| M-6 | 접속 로그에 `userId` 쿼리·세션 ID 경로가 남고, 일부 오류 응답·로그에 예외 원문이 실린다(`STT_UNAVAILABLE` 메시지의 모델 폴더 경로, 분석 결과 `loadError`) | `Dockerfile:27`, `stt_service.py:57`, `companion_router.py:343,425`, `yolo_class_detector.py:137,187,189`, `fire_smoke_detection_service.py:114,164,166` | 접속 로그 필터에 `userId=` 가림 추가, 응답·로그는 클래스명·고정 코드로 |
| M-7 | 게임 `/embed`·`/state`·`/answer`·`/reset`은 인증이 없고 userId가 정수라, 누구나 남의 진행을 보고 초기화할 수 있다(의도된 iframe 설계, H-2 때문에 인터넷 전체에 열림) | `game_router.py:32-80` | 수용 여부 결정. 막으려면 BE가 발급한 짧은 서명 토큰 |
| M-8 | 카메라 `streamUrl`은 스킴만 검사하고 서버가 그 주소로 접속한다(내부 주소 가능, 키 필요) | `camera_service.py:16-18`, `stream_service.py:68` | 사설·루프백·링크 로컬 주소 거절. 이 옛 경로를 안 쓰면 라우터 제거 |
| M-9 | 가슴압박 줄 경계 2건 - 두 사람이 섞인 문장에서 줄이 빠짐 / 라우터 실패 때 일반 질문 "숨 안 쉬는 사람은 어떻게…"에 줄이 붙음(안전 쪽) | `chat_keywords.py:180,253` | H-3 수정 때 시험지에 함께 넣어 판단 |

### 🟢 낮음

| ID | 내용 |
|---|---|
| L-1 | 삭제 보류 150초가 코드 상수·프로세스 메모리(`reservation_credential_service.py:22-24`). 삭제 API가 `reservation_token_claims`는 지우지 않는다(10분 뒤 자연 정리) |
| L-2 | `chat_schema.py`의 `layout` 주석("예약 흐름 답은 null")이 #200 이후 사실과 다르다 - 주석과 BE `policy-chat.md`에 예외 한 줄 |
| L-3 | nginx `reservation.dmu.gosky.kr`의 `/admin/` → 6017 전달이 남아 AI 서버로 간다(옛 관리 화면 포트) - 설정 정리 |
| L-4 | torch 2.6.0 HIGH 1건(신뢰할 수 없는 체크포인트 RCE) - 요청 경로에 해당 없음, CUDA 휠 상한 때문에 유지 |
| L-5 | `/api/docs`·`openapi.json` 무인증 공개 - BE Swagger와 같은 수용 여부를 AI 쪽에도 기록하거나 끄는 스위치 추가 |
| L-6 | #156 라우터 `self_harm` 칸 경로의 가족용 박스 전용 테스트 보강, `/chat/stats` `routerStatus`의 BE 처리 확인 |

## 확인하지 못한 것

- Pillow 각 CVE의 성립 조건(H-1), vkcs AI 서버의 스위치 값, 6017에 직접 붙는 다른 쓰임 유무(H-2 수정 전 확인 필요).
- `routerStatus`의 BE 관리자 집계 처리, FE 화면 동작. 운영 로그 실물은 보지 않았다(코드 기준).
- 이상감지 판정 로직 자체(화재 유지 조건·품질 게이트)는 10/31 이후 변경이 없어 이번 범위에서 다시 보지 않았다.

## 수정 PR 묶음 제안

| 묶음 | 내용 | 모델 | 비고 |
|---|---|---|---|
| A. 의존성·업로드 | H-1(pillow 13 + `formats=`), M-1, M-2 | sonnet | 이미지 재빌드 필요 → gosky 반영 시 재시작 |
| B. 노출·인증 | H-2(127.0.0.1 바인딩), M-3(기본값 닫힘), M-4(`compare_digest`), L-3·L-5 | fable(권한·토큰) | **재시작·nginx 변경은 사용자 확인 후** |
| C. 응급·위기 판정 | H-3, H-4, H-5, M-9 + 시험지 문항 | fable | 정상·실패 두 경로 테스트, stub·cache 재생만 |
| D. 로그·오류 문구 | M-6, L-2 | sonnet | 응답 형식 불변 |
| 결정 필요 | M-5(예약 키 암호화/수용), M-7(게임 무인증 수용), M-8(옛 카메라 경로 제거) | - | 사용자 결정 |

## 조치 결과 (2026-11-10, main `1d6e595`)

합친 최종 상태의 전체 pytest: **2668 passed · 6 skipped · 실패 0** (pillow 13.0.0, API 키 없음). gosky에는 아직 반영하지 않았다 - 이미지 재빌드와 AI 서버 재시작이 필요하다.

| 이슈 | 조치 | PR |
|---|---|---|
| H-1 | pillow 13.0.0, `Image.open`에 `formats=` 제한(약 봉투 JPEG·PNG·WEBP, 말벗 얼굴 JPEG·PNG) | #215 |
| H-2 | 바인딩 주소를 `APP_HOST`로 줄 수 있게 했다(기본 0.0.0.0 유지 - BE dev 컨테이너가 도커 브리지 주소로 직접 붙는다). **gosky에서 방화벽 규칙으로 차단**(아래 "gosky 반영") | #213 |
| H-3 | 회복 낱말 뒤에 다시 반응·호흡 없음이 오면 회복으로 보지 않는다. 뒤의 말이 지난 일이면 회복 유지 | #212 |
| H-4 | 라우터 실패 전용 자해 패턴에 말끝 변형(싶은데요·싶다고요·싶다는 생각이 들어요). "싶어서요"는 일상 표현과 겹쳐 넣지 않았다 | #212 |
| H-5 | "안 답답해요"·"숨이 안 가빠요" → 아니요, "예전에도"·"응급실"·"그래도" → 네 아님 | #212 |
| M-1 | 프레임 업로드 Content-Length 사전 거절·4MB(`STREAM_FRAME_MAX_BYTES`)·JPEG 매직 바이트·한 변 4096(`STREAM_FRAME_MAX_SIDE`). 새 코드 `STREAM_FRAME_TOO_LARGE`(413). 폼 파싱·누락 422·세션 404 순서는 그대로. Content-Length 없는 청크 전송은 파싱 뒤에 걸린다(남은 한계) | #215 |
| M-2 | `/companion/stt` 본문을 상한에서 읽기 중단. 디코딩을 동시 상한 안으로 옮기는 것은 하지 않았다(형식 오류 422가 503 `STT_BUSY`로 바뀐다) | #215 |
| M-3 | 스위치 3종 코드 기본값을 닫힘 쪽으로 | #213 |
| M-4 | 공용 키 비교 `hmac.compare_digest`(HTTP·WS·게임). 관리 경로 키 분리는 하지 않았다 | #213 |
| M-6 | 접속 로그 `userId`·`guardianId`·`wardId` 값 가림, `STT_UNAVAILABLE` 문구·탐지기 `loadError`에서 예외 원문 제거. 모델 파일 없음 `loadError`의 경로 문구는 그대로다 | #214 |
| M-9 | 바꾸지 않고 테스트로 고정(숨은 쉬는데 반응 없음 = 가슴압박 줄 없음, 라우터 실패 일반 질문 = 줄 붙임). 두 사람이 섞인 문장은 수용 | #212 |
| L-1 | 예약 키 삭제 때 `reservation_token_claims`도 삭제. 보류 150초 상수·메모리는 그대로 | #214 |
| L-2 · L-6 | `layout` 주석 정정, 라우터 `self_harm` 칸 턴의 가족용 박스 테스트 추가(결함 없음) | #214 |
| L-5 | `DOCS_ENABLED` 스위치 추가(기본 true = 지금과 같음) | #213 |
| M-5 | 예약 키를 `enc:v1:` + Fernet 암호문으로 저장(키는 `RESERVATION_ACCOUNT_SECRET`에서 용도 문자열로 갈라 만듦). 비밀값이 없으면 저장하지 않는다(평문 대체 없음). 풀 수 없는 행은 키 없음으로 보고 다시 발급. 기동 때 남은 평문 행을 일괄 암호화 | #217 |
| M-8 | 호출처가 없는(BE·FE 소스, gosky 7일 접속 로그 0건) `/api/v1/cameras`·`/api/v1/analysis`를 기본 미등록(`LEGACY_CAMERA_API_ENABLED=true`면 등록). 코드는 남겼다 | #217 |
| M-7 | **제외** - 게임은 담당 범위 밖(2026-11-10 사용자 결정) |  |
| L-3 | **미조치** - 공용 서버 nginx 설정이라 손대지 않았다 |  |
| L-4 | 조치 없음(요청 경로 해당 없음) |  |
| `routerStatus` | BE 처리 미확인 그대로 |  |

### gosky 반영 (2026-11-10 17:44 KST, `504c105`)

- 허용 배포 명령으로 재빌드·재기동. health 200(25초), 재시작 횟수 0, `OomScoreAdj=-500`(#208 적용), pillow 13.0.0·cryptography 50.0.2, 기동 로그 Traceback 0건, BE의 AI WS 재연결 17:44:47.
- 예약 키 7행이 기동 때 모두 암호화됐다(평문 0행). `/api/v1/cameras`·`/api/v1/analysis/*`는 404, 프레임·챗 경로는 그대로 401(키 필요).
- **H-2 차단**: `iptables -I INPUT -i eno1 -p tcp --dport 6017 -j DROP`(주석 `silverbridge-ai: 6017 nginx/docker only`). 외부 인터페이스로 들어오는 6017만 막고 루프백(nginx)·도커 브리지(BE 컨테이너)는 그대로다. 재부팅 뒤에도 남도록 `silverbridge-ai-6017-guard.service`(systemd oneshot, enabled)가 기동 때 같은 규칙을 넣는다. 되돌리기: `sudo systemctl disable --now silverbridge-ai-6017-guard.service`.
- 확인: 외부 `http://<gosky>:6017/health` 응답 없음(차단) · `https://testai.gosky.kr/api/docs` 200 · `https://testai.gosky.kr/api/v1/chat/logs/1` 401 · BE 컨테이너 → `http://10.0.22.1:6017/health` 정상.
- 실제 라우터를 부르는 상담 한 바퀴는 돌리지 않았다(API 키를 쓰는 검증 금지). 최종 전체 pytest 2674 passed · 6 skipped · 실패 0.
- 남은 것: L-3(nginx `reservation.dmu.gosky.kr`의 `/admin/` → 6017 전달 정리), M-2·M-4의 미조치 부분, `routerStatus`의 BE 처리 확인.

### 남은 항목 처리 (2026-11-10 저녁, main `cca5f2f` - gosky 11/10 22:35 `95ea2b7` 반영)

- M-2 남은 부분(#219): `/companion/stt` 디코딩을 받아쓰기 자리(동시 2건) 안에서 한다. 자리가 없을 때는 읽을 수 없는 음성·30초 초과 음성도 먼저 503 `STT_BUSY`를 받는다(다시 보내면 원래 422·413). 디코딩이 이벤트 루프 스레드를 막던 것도 없어졌다.
- M-4 남은 부분(#219): `ADMIN_API_KEY`(선택). 넣으면 `/api/admin/medgemma/*`는 그 키만 받고 공용 키는 거절한다. gosky `.env`에는 아직 넣지 않았다(비어 있으면 지금처럼 공용 키).
- `routerStatus`: BE `AdminAiChatStatsService`가 이미 읽고 형식이 어긋나면 null로 처리한다 - 확인 종결.
- L-3: Notion "DMU / 인프라 전달 내용"에 요청 페이지 작성(nginx `reservation.dmu.gosky.kr`의 `/admin/` → 6017 옛 설정 정리).
- 전체 pytest 2680 passed · 6 skipped · 실패 0. BE `docs/audit-index.md`·`progress.md` 기록.
- gosky 반영(22:35): health 200(20초), 재시작 횟수 0, Traceback 0건, BE의 AI WS 22:35:42 재연결, 6017 방화벽 규칙 유지. `ADMIN_API_KEY`는 gosky `.env`에 아직 없다(운영 `.env` 수정은 사용자 몫).
