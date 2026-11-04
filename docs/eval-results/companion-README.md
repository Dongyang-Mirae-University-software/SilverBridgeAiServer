# 말벗 평가 결과 안내 (S11-1, S11-4)

이 폴더의 `2026-11-02-*` 파일이 말벗 평가 결과다. 다른 날짜 파일은 기존 안전성·성능 시험이라 여기서 다루지 않는다.

## 파일 구분

| 파일 | 내용 | 만든 도구 |
|---|---|---|
| `2026-11-02-text-emotion-453.json` | 텍스트 감정, 문장 단위 | `scripts/companion/eval_text_emotion.py` (기본 모드) |
| `2026-11-02-text-emotion-453-dialogue.json` | 텍스트 감정, 대화 단위 (최빈 라벨) | `scripts/companion/eval_text_emotion.py --mode dialogue` |
| `2026-11-02-stt-cer.json` | 음성 인식 글자 오류율 | `scripts/companion/eval_stt.py` |
| `2026-11-02-face-emotion.json` | 표정 정확도 (사진 샘플 필요) | `scripts/companion/eval_face.py` (아직 실행 안 함) |

원문(발화 글, 사진, 음성)은 여기에 두지 않는다. 점수와 혼동 집계만 둔다.
오판 문장 목록은 `data/misjudgment_samples.json`에 있고, `data/`는 Git에 올리지 않는다.
평가 입력 목록은 `data/eval_set_107.json` (`scripts/companion/build_eval_set.py`로 만든다).

## 측정과 추정 구분 (S11-4)

| 값 | 상태 | 근거 |
|---|---|---|
| 453 텍스트 감정 정확도 53.1% (문장) | **측정** | 2,300문장, 800대화 |
| 453 텍스트 감정 정확도 67.8% (대화) | **측정** | 같은 800대화, 최빈 라벨 |
| 가이드 이전 기록 "453 83.1%" | **확인 불가** | 근거 파일과 측정 방식이 이 저장소에 없음 |
| 음성 인식 CER 0.4603 | **측정** | 107 샘플 100개, large-v3, CPU |
| 음성 인식 평균 17.3초/클립 | **측정** | 같은 시험 (모델 적재 시간 제외) |
| 음성 한 턴 10.4초 | **측정 (1회)** | 점검 스크립트 한 번. 모델 적재 후 |
| 표정 정확도 | **미측정** | 얼굴 샘플 없음 (`data/`에 사진 없음) |
| EXAONE 응답 품질 | **미측정** | 가중치 없음. 현재 응답은 대체 문장 |
| 그래픽카드 메모리 (VRAM) 필요량 | **미측정** | 이 PC는 GPU 2GB (MX450). 서버 GPU 확인 필요 |
| 공개 데이터 점수 → 서비스 정확도 | **추정하지 않음** | 어르신 실제 대화와 다를 수 있음 |

## 쓰는 법

- 텍스트 감정 다시 재기: `.venv/Scripts/python.exe -m scripts.companion.eval_text_emotion --csv data/aihub_453/train.csv --dialogues 800 --mode dialogue`
- 오판 샘플 다시 모으기: `.venv/Scripts/python.exe -m scripts.companion.collect_misjudgments --csv data/aihub_453/train.csv`
- 음성 평가 입력 만들기: `.venv/Scripts/python.exe -m scripts.companion.build_eval_set --manifest data/eval_manifest_107.csv`
- 표정 평가 (사진 폴더 준비 후): `.venv/Scripts/python.exe -m scripts.companion.eval_face --root data/face_eval`
