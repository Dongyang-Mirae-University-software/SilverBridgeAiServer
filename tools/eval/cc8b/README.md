# 대답 LoRA cc8b (2026-11-21) - 대화 흐름 형식 재학습

DGX 전용 스크립트다(경로가 `~/llmbench/...`로 고정). 외부 API 키를 쓰지 않는다 - 교사 EXAONE 3.5 7.8B, 요약·학습 EXAONE 3.5 2.4B 모두 로컬.

| 순서 | 스크립트 | 하는 일 | 걸린 시간 (11/21) |
|---|---|---|---|
| 1 | `teach_cc8b.py gen 600` | 교사가 말벗·어르신을 번갈아 맡아 대화 600개(6턴)를 만든다. 말벗 정답은 cc8 필터 + 운영 `check_reply` + 품질 채점기 규칙을 통과한 후보만 쓰고, 못 넘으면 운영처럼 대체 문장으로 이어 가되 학습에서 뺀다. 짧은 대답 어르신 역은 60%를 CareCall 실제 짧은 대답으로. 품질 세트 어르신 문장(같거나 유사도 0.8)은 쓰지 않는다 | 2.5시간 |
| 2 | `ft_reply8b.py data` | 운영 `summarize_answer`로 어르신 한 줄 요약을 만들고, 운영 `build_reply_prompt`(summary, 흐름 3쌍)로 입력을 만든다. cc7 자료(흐름 없음 = 첫 대답) + 새 자료, 검증은 대화 단위 분리 | 40분 |
| 3 | `ft_reply8b.py train` | cc7 과 같은 설정(nf4·r16·lr 2e-4·2 epoch) | 1.5시간 |
| 4 | `eval_cc8b.py <cc7\|cc8b> <off\|summary> <last>` | 짧은 대답 대화의 화제 겹침 (eval_cc8 topic 과 같은 정의) | 50분 |

품질 세트 비교는 `tools/eval/companion_quality_dgx_run.py` + `score_companion_quality.py`를 기준(cc7+off)과 나란히 3회.
결과: `docs/eval-results/2026-11-20-companion-quality-merge.md` "대답 LoRA cc8b" 절.

주의: 교사 정답을 품질 채점기 규칙으로 걸렀으므로, 같은 규칙으로 잰 점수는 실제 개선보다 크게 나올 수 있다. 규칙 밖 항목(공감·넘겨짚기·화제 끊기)은 대답 원문을 함께 비교한다.
