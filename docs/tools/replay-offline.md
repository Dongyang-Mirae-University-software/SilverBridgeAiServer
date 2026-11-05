# 챗봇 오프라인 재생 도구 (`tools/eval/replay_offline.py`, 2026-11-05)

판정 재생이 운영과 같은 Anthropic 키로 라우터를 하루 수천 회 불러 크레딧이 바닥난 일(2026-11-04)이 있었다. 이 도구는 라우터 호출을 기본 0회로 두고, 실제 라우터가 필요할 때만 상한 안에서 부른다.

## 라우터 모드

| 모드 | API 호출 | 쓰는 때 |
|---|---|---|
| `stub` (기본) | 0 | 서버 쪽 판정·후처리·답 모양 회귀 확인. 라우터 결과 = 시험지 `expect`(route·emergency·crisis→self_harm·cpr_aed→cardiac_arrest·urgent_condition) |
| `cache` | 0 | 전에 `live`로 기록한 라우터 응답을 그대로 재사용. 기록에 없는 입력은 라우터 실패(키워드 분류)로 돌고 요약에 `routerCacheMiss`로 센다(누락이 있으면 종료 코드 1) |
| `live` | 상한 안에서 | 실제 라우터를 부르고 응답을 캐시 파일에 기록. 키 필수, `--max-router-calls`(기본 100) 필수, 시작 전에 예상 호출 수를 출력하고 넘으면 시작하지 않는다 |

캐시 키 = 모델·지시문 해시·스키마 해시·사용자 문장·이력·직전 예약 상태. 라우터 지시문이 바뀌면 옛 기록은 쓰이지 않는다(다시 `live`로 채운다).

## 기본값이 API를 줄이는 이유

- 회차 1: MedGemma는 같은 입력이면 같은 답이라, 회차를 늘리면 라우터 흔들림만 늘어난다. 흔들림은 아래 `router-variance`로 소수 문항만 잰다.
- 이어서 물어볼 질문(Haiku) 끔: 판정과 무관한 보조 값. 필요하면 `--follow-up`.
- `stub`·`cache` 모드는 실행 중 `ANTHROPIC_API_KEY`를 지워 실수로도 부를 수 없게 한다.

## 사용 예 (DGX, 저장소 밖에 출력)

```bash
# 서버 쪽 회귀 확인 - API 0회
python tools/eval/replay_offline.py run --set tests/eval/eval_set_v7.json --router stub \
  --out ~/replay/out/bodies.json --summary ~/replay/out/summary.json

# 라우터 응답을 한 번만 기록(상한 100) → 이후 main/수정본 비교는 cache 로 0회
python tools/eval/replay_offline.py run --set tests/eval/holdout_set_v3.json --router live --max-router-calls 100 \
  --key-file ~/replay/.router_key --cache-file ~/replay/router_cache.json --out ~/replay/out/v3_main.json
python tools/eval/replay_offline.py run --set tests/eval/holdout_set_v3.json --router cache \
  --cache-file ~/replay/router_cache.json --out ~/replay/out/v3_fix.json --summary ~/replay/out/v3_fix_summary.json

# 라우터 흔들림 - 응급·위기 소수 문항 × 5회(상한 안)
python tools/eval/replay_offline.py router-variance --set tests/eval/eval_set_v7.json --ids qa.C6,sh.09,em.04 --times 5 \
  --key-file ~/replay/.router_key --summary ~/replay/out/variance.json
```

- `cache` 재사용은 라우터 지시문·스키마가 같은 커밋끼리만 맞다. 라우터를 바꾸는 작업은 바뀐 쪽만 `live`로 새로 채운다.
- 키 파일은 저장소 밖, 권한 600, 끝나면 지운다. 키·본문을 출력하지 않는다.
- 운영 측정(`tests/eval/run_eval.py`)은 실서버를 부르므로 이 도구와 별개다 - 사용자 허락 없이 돌리지 않는다.

## 프롬프트 캐싱 (같은 날)

라우터·이어서 물어볼 질문·예약 추출의 고정 지시문(system)에 5분 캐시 표시를 붙였다(`CLAUDE_PROMPT_CACHE`, 기본 켬). 라우터 지시문은 약 2,000토큰이라 5분 안에 다시 부르면 입력 단가의 약 5%로 읽는다(첫 기록은 1.25배). 이어서 물어볼 질문(약 330토큰)·예약 추출(약 450토큰)은 Opus 5.5·Haiku 5.5의 최소 512토큰에 못 미쳐 표시가 있어도 캐시되지 않을 수 있다(오류 없음). 운영 로그 `[CLAUDE-USAGE] stage= in= out= cache_read= cache_write=`(숫자만)로 적중을 확인하고, 적중이 낮으면 `.env`에서 끈다.
