# 기능 점검 - 프롬프트 캐싱·오프라인 재생 도구 (#182) (2026-11-05, 템플릿 B, API 없이)

- 대상: AI #182 `045df8b` - `ClaudeStructuredClient`(고정 지시문 캐시 블록·`[CLAUDE-USAGE]`·`CLAUDE_PROMPT_CACHE`), `tools/eval/replay_offline.py`(stub·cache·live·router-variance).
- 방법: 코드 읽기, DGX에서 가짜 키 `.env`로 키 재로딩 재현(네트워크 호출 없음), 운영 로그 읽기. **API 호출 없음.**
- **종합 판정: 조건부 PASS - 🔴 0 · 🟠 1 · 🟡 1 · 🟢 3**

## 통과

| 항목 | 결과 |
|---|---|
| 지시문 블록 | 라우터·이어서 물어볼 질문·예약 추출 모두 한 곳(`_request_kwargs`)을 거치고, 지시문은 모듈 상수라 요청마다 바이트가 같다(날짜·요청 값은 user 쪽) |
| 스위치 | `CLAUDE_PROMPT_CACHE=false`면 옛 문자열 그대로 |
| 사용량 로그 | 단계 이름과 토큰 수만, 본문·지시문 없음 |
| 도구 출력 | 본문(`--out`)·캐시 파일은 저장소 밖만 |
| live 상한 | 예상 호출이 상한을 넘으면 시작하지 않고, 실행 중에도 상한에서 멈춤. `router-variance`도 같음 |
| cache 모드 | 키 없는 라우터로 감싸 기록에 없으면 실제 라우터를 부르지 않고 누락으로 센다 |

## 이슈

- 🟠 **H-1 stub·cache 모드의 "API 0회"가 `.env`가 있는 저장소에서 깨진다** - 도구는 `ANTHROPIC_API_KEY`를 환경에서 지우지만, `app/core/config.py`가 가져올 때 `load_dotenv()`로 `.env`의 키를 다시 채운다(DGX 가짜 키 재현: 지운 뒤 False → config 뒤 True). 라우터는 막혀 있어도 예약 흐름 문항(v7 4개)의 Claude 예약 추출, 약 봉투 OCR 같은 라우터 밖 호출이 실제 키로 나갈 수 있다. DGX(`.env` 없음)에서는 0회였고, gosky 운영 저장소처럼 `.env`가 있는 곳에서 돌리면 위험하다. 수정안: stub·cache 모드에서 `ANTHROPIC_API_KEY`·`CLAUDE_API_KEY`를 빈 값으로 넣어(`load_dotenv`는 이미 있는 값을 덮지 않는다) 클라이언트가 준비되지 않게 하고, `ClaudeStructuredClient.generate_structured_text`를 막아 호출 시도를 세어 요약에 남긴다 + 테스트.
- 🟡 **M-1 live 상한이 라우터 호출만 센다** - 예약 추출·`--follow-up`일 때 이어서 물어볼 질문 호출은 예상 호출 수와 상한에 들어가지 않는다. 수정안: 상한을 Claude 호출 전체(단계별)로 세고 예상 수에 예약 문항을 더한다.
- 🟢 L-1 `--summary` 경로는 저장소 안도 허용(본문 없음이라 위험은 낮다 - 일관성 차원).
- 🟢 L-2 운영 확인 미완: 15:31 재시작 뒤 실제 상담 0건이라, 캐시 표시 + 서버 대체 모델(`fallbacks`) + 구조화 출력 조합을 API가 받아들이는지 아직 운영 로그로 확인되지 않았다. 첫 상담 뒤 `[CLAUDE-USAGE]`(cache_write·cache_read)와 `[CHAT-ROUTER-FAILED]`를 본다 - 거부되면 `CLAUDE_PROMPT_CACHE=false`로 끈다.
- 🟢 L-3 테스트 공백: `.env` 재로딩과 stub 전체 경로의 Claude 호출 0회를 고정하는 테스트가 없다.

## 조치 (2026-11-05, `fix/replay-tool-zero-api`, API 없이 검증)

- H-1: stub·cache 모드는 `ANTHROPIC_API_KEY`·`CLAUDE_API_KEY`를 지우지 않고 **빈 값으로** 넣는다(`load_dotenv`는 이미 있는 값을 덮지 않는다). 그리고 `ClaudeStructuredClient.generate_structured_text`를 감싸 모든 Claude 호출 시도를 막고 센다(요약 `blockedClaudeCalls`).
- M-1: live 상한은 Claude 호출 전체(라우터·예약 추출·이어서 물어볼 질문)를 세고, 예상 호출 수에 예약 경로 문항의 추출 몫을 더한다.
- L-1: `--summary`도 저장소 밖만. L-3: 테스트 5건(빈 키가 `.env`에 덮이지 않음·stub 차단 집계·live 상한이 단계 전체를 셈·예상 수·요약 경로).
- 확인: 관련 테스트 1375 통과, 결함 주입 5종 모두 검출. DGX에서 가짜 키 `.env`를 둔 복사본으로 stub 재생(v7 안전 + 예약 73문항): Claude 호출 0, 결과 이전과 같음(응급 27/27·자해 11/12·되묻기 30/30).
