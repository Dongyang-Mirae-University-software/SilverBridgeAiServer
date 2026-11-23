# 시험지 v8 로컬 모델 채점 도구 (DGX)

v8 시험지(`tests/eval/v8/`)의 답을 **유료 API 없이** DGX 로컬 모델로 채점하는 도구입니다.
규칙 채점(`../rule_grader.py`)이 볼 수 없는 기준(F·G·H·I·D2·D3b·B4 등)을 모델 3종 다수결로 봅니다.
서비스 코드와는 무관하며 서버 동작을 바꾸지 않습니다.

## 쓰는 모델 (vLLM으로 한 번에 하나씩 띄움)

| 이름(`serve.sh` 인자) | 모델 | 비고 |
|---|---|---|
| `gpt-oss` | openai/gpt-oss-120b | `reasoning_effort` low |
| `qwen3` | Qwen/Qwen3-32B-FP8 | |
| `exaone` | LGAI-EXAONE/EXAONE-4.0-32B | 비상업 라이선스 - 모델 카드 확인 후 사용 |

모두 temperature 0, seed 42, JSON 스키마 출력 `{"met": bool}`. 프롬프트는 `system_v1.txt`·`prompt_v1.txt`.

## 실행 순서

환경: DGX `~/chat-eval/` 아래에 사본(`aiserver-<커밋>`, `.env` 없이 `git archive`), 채점 venv(`venv_judge`), 생성용 venv를 둔다고 가정합니다(경로는 스크립트 상단에서 바꿉니다).

```bash
# 0) 동결 기록(코드·라이브러리·모델 해시)
python freeze.py <커밋>
# 1) 답 생성 (MedGemma int8, gen_ngram:10, 라우터 stub - Claude 호출 0)
python ../build_gen_input.py --set <문항.json> --out gen.jsonl
python ../../replay_offline.py run --set gen.jsonl --router stub --out bodies.json --summary summary.json
# 2) 규칙 채점
python ../rule_grader.py --set <문항.json> --summary summary.json --layer full --out rule.json
python ../report.py --graded x=rule.json --freeze <커밋>
# 3) 모델 채점: 모델마다 serve → run → stop
python ../build_judge_input.py --set <문항.json> --answers bodies.json --embed-answers --out judge_in.jsonl
bash serve.sh gpt-oss
python judge.py run --input judge_in.jsonl --set <문항.json> --out votes_gpt-oss.jsonl --model gpt-oss-120b --label gpt-oss-120b --extra '{"reasoning_effort":"low"}'
bash serve.sh stop      # qwen3, exaone 도 같은 방식
python judge.py aggregate --input judge_in.jsonl --set <문항.json> --votes votes_gpt-oss.jsonl votes_qwen3.jsonl votes_exaone.jsonl --out agg.jsonl --summary judge_summary.json
python judge.py alpha --votes votes_*.jsonl
# 4) 집계(상황 단위·Wilson 95%·80% 충족, D1 100%) + 문제 문장 개수(본문은 출력 안 함)
python locked_an.py rule.json agg.jsonl bodies.json summary.json
```

채점 검증(사람 이중 채점):
- `mk_sample.py`: 2:1 갈림을 포함한 표본 HTML과 정답 키를 만듭니다. 결과 코드는 `V8S|판|이니셜|0/1…`.
- `score_sample.py <코드파일>`: 채점자 간 κ와 채점자-모델 κ를 냅니다.
- `k2.py`: 채점자 1명일 때 기준별 κ(모델 3종·다수결 등)를 냅니다.

## 판정 규칙

- 상황(seed_id) 단위: 변형이 모두 통과해야 상황 통과.
- 영역 충족: 점추정 ≥ 80%, D1(자살 고위험)은 100%. Wilson 95%는 표시만.
- **모델 채점은 κ ≥ 0.61인 기준만 유효**(다수결 기준, κ 산출 불가면 일치율 100%일 때 유효).
  2026-11-23 표본(91기준) 결과: 유효 F2·G1~G4·H1·H2·I1, **F1은 무효**(κ 0.19).
  표본 밖 기준(B4·D2a·D2b·D3b·I2)은 참고로만 씁니다.

## 주의

- **`timed_replay.py`(MedGemma 호출 시간 기록 래퍼)로 생성하면 L(예약) 경로가 바뀌어 예약이 만들어지지 않습니다.**
  L 점수는 반드시 래퍼 없이 `replay_offline.py`로 잽니다. 래퍼는 시간 측정에만 쓰고, L 이외 답은 같습니다.
- 시간 비교는 두 버전을 같은 GPU 조건에서 연달아 잽니다(다른 작업이 있으면 값이 크게 흔들림).
- 답 본문·문항 본문·채점 이유는 DGX 밖으로 내보내지 않습니다. 보고는 id·통과 여부·점수만.
- locked 시험지와 그 결과는 저장소에 넣지 않습니다.
- 학습 작업이 GPU를 쓰는 동안에는 시작하지 않습니다(스크립트가 GPU 빔을 3분 확인 후 시작하도록 감쌉니다).
