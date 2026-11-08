# Claude 실패 대비 - 예약 추출·이어서 물어볼 질문·약 봉투 인식 (2026-11-08)

## 배경
- gosky AI의 Anthropic 키는 마지막 예산이다. 서비스는 지금처럼 쓰고, 다 쓰면 다시 채우지 않는다(2026-11-08 결정).
- Claude를 부르는 곳은 넷이다. 라우터(`chat_router_service`)는 다른 작업에서 다루고, 여기서는 나머지 셋을 다룬다.
- 목표: 크레딧이 떨어지거나 호출이 실패해도 사용자에게 엉뚱하거나 끊긴 화면을 보이지 않는다. **성공 경로는 바꾸지 않는다.**

## 바꾸기 전 동작 (코드·목 테스트로 확인)
| 기능 | 크레딧 소진(400) | 시간 초과 | 키 없음 |
|---|---|---|---|
| 예약 추출 | 규칙 초안으로 계속 진행(약 1초) | 60초 × 2회 시도, 챗 마감 안에서 최대 약 95초 | 규칙 초안으로 계속 진행 |
| 이어서 물어볼 질문 | 규칙 문장 | 3초 뒤 규칙 문장 | 규칙 문장 |
| 약 봉투 인식 | 502 `OCR_UPSTREAM_FAILED` "잠시 후 다시 시도" | 30초 뒤 502 | 503 `OCR_NOT_CONFIGURED` |

예약은 Claude가 실패해도 규칙 추출(날짜·시간·진료과·병원)과 단계별 버튼으로 예약 완료까지 이어진다. 화면이 끊기지는 않았다.

## 바꾼 것
1. **크레딧 소진 감지 + 건너뛰기** (`app/services/claude_availability.py`, 새 파일)
   - 판정은 라우터 실패 분류(#188 `router_failure_monitor.failure_kind == "credit"`)와 같은 기준이다: 402, 본문 `billing_error`·지출 한도 코드, 400·403 + `credit balance` 같은 문구. 문구는 판정에만 쓰고, 로그에는 고정 코드 `credit_exhausted`만 남긴다.
   - 소진을 감지하면 `CLAUDE_CREDIT_SKIP_SEC`(기본 1800초, 0~86400) 동안 **건너뛰기를 켠 호출자**는 Claude를 부르지 않는다. 시간이 지나면 다음 요청 1건이 다시 시도하고, 성공하면 바로 풀린다.
   - 로그: `[CLAUDE-CREDIT] cause=credit_exhausted skip_sec=…`(처음 감지할 때 한 번), `[CLAUDE-CREDIT] cause=recovered`.
2. **공용 클라이언트 선택 인자** (`claude_structured_client.generate_structured_text`)
   - `skip_when_exhausted=False`: 켜면 건너뛰는 동안 바로 `credit_exhausted`를 낸다. 크레딧 부족 응답도 `api_400` 대신 `credit_exhausted`로 받는다.
   - `retry_on_timeout=True`: 끄면 시간 초과는 다시 시도하지 않고 `timeout`을 낸다. 429·5xx·연결 실패 재시도는 그대로다.
   - 두 인자 모두 기본값이 예전 동작이다. **라우터는 인자를 넘기지 않으므로 동작이 같다.** 소진 감지 기록과 성공 시 해제만 함께 쓴다.
3. **예약 추출**: `skip_when_exhausted=True`, `retry_on_timeout=False`. 실패하면 예전처럼 규칙 초안으로 진행한다. 시간 초과 때 최악 대기가 약 95초에서 60초로 줄어든다.
4. **이어서 물어볼 질문**: 건너뛰는 동안에는 작업 스레드도 쓰지 않고 바로 규칙 문장을 쓴다(`[CHAT-SUGGEST] source=rule cause=credit_exhausted`).
5. **약 봉투 인식**: 크레딧 소진이거나 건너뛰는 중이면 **503 `OCR_UNAVAILABLE`** "사진 인식을 지금 쓸 수 없습니다. 직접 입력해 주세요."로 답한다(로그 `result=unavailable_credit_exhausted`). 일시적 실패는 그대로 502다.
   - 400을 쓰지 않은 이유: BE `AiMedicationOcrClient`가 AI의 400을 "JPG, PNG, WebP 사진만…" 안내로 바꾼다.
   - BE는 503을 `MEDICATION_OCR_UNAVAILABLE`로 바꾸고, FE는 "지금은 인식할 수 없어요. 직접 써 주세요"를 보인다. **BE·FE 변경은 필요 없다.**

## 바꾸지 않은 것
- 키가 있고 호출이 성공할 때의 요청·응답·프롬프트·모델.
- 라우터(`chat_router_service`)와 `chat_service`의 실패 처리.
- `.env`와 키. 새 환경변수 `CLAUDE_CREDIT_SKIP_SEC`는 코드 기본값으로 돈다.

## 남은 일 (이번 범위 밖)
- **FE 사진 버튼 숨기기**: AI `GET /api/v1/medication/ocr/status` → `{available}`, BE 중계 API, FE 모달에서 버튼 숨기기. 세 저장소를 함께 바꿔야 한다. FE 문구가 이미 "직접 써 주세요"라 급하지 않다.
- 라우터가 소진을 만났을 때의 동작은 라우터 작업에서 정한다. 이 모듈의 `skipping()`을 쓸 수 있다.

## 테스트
- `tests/test_claude_failure_fallbacks.py` (21건, 목만 사용, 실제 호출 없음): 세 기능 각각 크레딧 오류·시간 초과·키 없음·성공을 본다. 예약은 Claude 실패 상태(크레딧·시간 초과·키 없음)에서 버튼만으로 예약 완료까지 가는지, 문장 입력만으로 확인 카드까지 가는지 본다. 로그에 오류 문구가 없는지도 확인한다.
