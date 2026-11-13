# 말벗 목소리 구글 Gemini TTS(Kore) 지연 실측 (2026-11-14)

- 대상: BE V65 질문 19개 + AI 고정 문장 15개 = 34문장(회당 1,006자), 2회 순차 = 68건
- 방법: gosky 운영 이미지 일회용 컨테이너(`--memory 2g --cpus 3`, 같은 `.env`), 이번 브랜치 코드의 `load_google()`·`load_supertonic()` 직접 호출. 서비스 재시작 없음, 측정 잠금 사용
- 구글: `gemini-3.8-flash-tts`, 보이스 Kore, 말투 지시 `speechMetadata.style`, Agent Platform `generateContent`(프로젝트 없는 주소, 서비스 계정에 묶인 API 키). WAV 응답을 mp3(24kHz mono 48k)로 변환한 시간 포함

| 엔진 | 건수 | 실패 | p50 | p95 | 최대 | 평균 | mp3 평균 |
|---|---|---|---|---|---|---|---|
| 구글 Kore | 68 | 0 | 2,654ms | 3,661ms | 5,360ms | 2,716ms | 34.6KB |
| F1 (Supertonic 3, 3스레드) | 68 | 0 | 845ms | 1,693ms | 2,296ms | 932ms | 26.9KB |

- F1 모델 적재 911ms. 측정에 쓴 구글 글자 수 2,012자
- 확인한 전제 차이: 3.8 은 Cloud TTS(`text:synthesize`)에 없고 Agent Platform 전용, 일반 API 키는 401(`CREDENTIALS_MISSING`) - 서비스 계정에 묶인 키로 200, MP3 미지원(WAV), 언어 칸 없음(자동 감지)
