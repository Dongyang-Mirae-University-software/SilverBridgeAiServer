# 기능 점검 - 말벗 음성 단계 교체 (2026-11-09, 템플릿 B)

- 대상: PR #201 `63a2217`(기능 커밋 `eccadfa`, gosky 미배포) - `vad.py`·`turbo_service.py`·`repetition.py`·`stt_engine.py`·`pipeline.run_turn(stt_logprob)`·`companion_router`·`config` 스위치 7종
- 방법: 코드 읽기 + 가짜 모델 탐침(로컬 TestClient, 한 대화에 턴 2~3개 연달아) + DGX Whisper 토크나이저로 한국어 토큰 밀도 확인. Anthropic API 0회, gosky 접근 없음. BE(`SilverBridgeBe` dev)는 읽기만.
- 실제 모델 정확도는 PR 본문의 DGX 실측(어르신 200문장 clean 2.41%, 되묻기 0·25·72, 무음·생활 소음 20/20 거름)을 기준으로 삼고 다시 재지 않았다.
- 코드 수정 없음(점검만).

## 종합 판정: 조건부 PASS - 🟠 2 · 🟡 2 · 🟢 4

| 등급 | ID | 내용 | 근거 |
|---|---|---|---|
| 🟠 | H-1 | **확신도 낮은 턴이 두 번째부터 되묻지 않는다.** 되묻는 문장이 고정 한 문장이라 `_avoid_repeat`(같은 대화에서 이미 한 말은 미리 정한 문장으로 바꿈)가 2번째부터 "그러셨군요. 오늘 하루는 어떻게 보내고 계세요?"·"말씀해 주셔서 고맙습니다. …"로 바꾼다. 못 알아들은 말에 알아들은 것처럼 답하는 꼴이라, 되묻기의 목적("뜻을 추측해 공감하지 않는다")이 2번째부터 사라진다. 재희 버전도 후보 낱말이 없을 때 같은 문장이었지만 드물었고, 이번 PR로 확신도 낮은 턴은 늘 같은 문장이 됐다 | 탐침: lp -0.8 턴 3번 → 되묻기 1번, 대체 문장 2번 (`companion_router.py:126`) |
| 🟠 | H-2 | **같은 대화의 두 번째 위험 턴은 안전 안내 문장 대신 명랑한 대체 문장으로 답한다**(riskFlag·BE 위험 표시는 그대로 켜짐). 일반 턴에서도 생기는 **기존 동작**이고, 이번 PR은 되묻는 턴에도 안전 문장을 내게 해 같은 경로를 하나 더 만들었다. 어르신이 두 번째로 위험한 말을 하면 "그러셨군요. 오늘 하루는…"을 듣는다 | 탐침: 위험 확률 0.9 턴 2번 → 1번째 안전 문장, 2번째 대체 문장 (일반 턴·되묻는 턴 둘 다) |
| 🟡 | M-1 | **긴 대답이 잘릴 수 있다.** 30초 창마다 `max_new_tokens=128`. 한국어는 음절당 약 0.83토큰(Whisper 토크나이저 실측)이라 30초에 150음절 넘게 빨리 말하면 상한에 닿는다. 30초 나누기는 단어 중간을 자른다. DGX 시험 문장은 최대 38음절(10초 이내)이라 이 경로는 실제 모델로 시험된 적이 없다 | `turbo_service.py` WINDOW_SEC·max_new_tokens, 토크나이저 107음절=89토큰 |
| 🟡 | M-2 | **TV·옆 사람 말소리가 계속 들리면 대화가 최대 턴까지 이어진다.** 먼 웅성거림은 Silero를 통과하고(0/20 거름) 확신도로 되묻기가 된다(18/20). 되묻는 턴은 무응답이 아니라서 연속 무응답 3회 종료가 걸리지 않고, H-1과 겹치면 TV와 대화하는 꼴이 된다 | `dialogue_store.py:91`(NO_RESPONSE 아닐 때 0으로), DGX far 결과 |
| 🟢 | L-1 | 스위치 값 오타·대소문자("Silero"·"Turbo"·"Logprob")를 검사하지 않고 조용히 재희 동작(rms·sensevoice·lexicon)으로 떨어진다. 기존 `COMPANION_STT_ENGINE`과 같은 방식 | `vad.has_speech`·`stt_engine._engine`·`pipeline.run_turn` 비교식 |
| 🟢 | L-2 | turbo·Silero 실행 중 예외(CUDA 메모리 부족 등 `SttUnavailable`이 아닌 것)는 503이 아니라 처리되지 않은 500이 된다. SenseVoice도 같다(기존 방식) | `companion_router.py` 음성 턴 `except TimeoutError/SttUnavailable` |
| 🟢 | L-3 | turbo 로드 실패를 한 번 기록하면 재시작 전까지 계속 503(일시적 CUDA 오류도). SenseVoice·faster-whisper와 같은 방식이고, 앱 시작 때 예열이 없어 첫 턴이 GPU 로딩 시간을 낸다(받아쓰기 제한 20초 안으로 예상, 실측 없음) | `turbo_service._load` |
| 🟢 | L-4 | 반복 정리는 "네 네 네"(3번)도 "네"로 줄인다. 뜻은 같아 문제 없음 - 기록만 | `repetition.collapse_repeats` |

## PASS 항목
- **재희 동작 보존**: 스위치를 모두 끄면(`rms`·`sensevoice`·`false`·`lexicon`) 턴 흐름이 예전과 같다. SenseVoice 안의 RMS 기준도 그대로이며 테스트(`test_sensevoice_with_rms_matches_jaehee`·`test_lexicon_mode_is_jaehee_behaviour`)가 고정한다.
- **확신도 경계**: -0.3은 되묻지 않고 -0.3001은 되묻는다. 확률이 없는 턴(글 대답·SenseVoice)은 logprob 모드에서도 재희 규칙을 쓴다. 라우터에서 `avg_logprob`가 `run_turn`까지 전달된다(탐침 1번째 턴).
- **시간 예산**: 되묻는 턴 최악 = 받아쓰기 20초 + 표정 10초 + 위험 확률 25초(LLM 자체 제한) = 55초 < BE 60초.
- **잠금**: turbo 추론은 한 번에 하나(`_lock`), 한 번 0.2~1초에 `max_new_tokens`로 끝이 있어 잠금이 오래 묶일 가능성은 낮다. 챗봇 `/stt`는 기본 SenseVoice라 턴과 GPU 잠금을 다투지 않는다(`COMPANION_STT_API_ENGINE=turbo`일 때만 공유).
- **개인정보**: 받아 적은 글·음성을 로그에 남기지 않는다. Silero 로드 실패 경고는 예외 클래스명만, turbo 로드 실패 메시지에 경로·원문 없음. 가중치는 로컬 폴더에서만 읽는다(`trust_remote_code` 없음, 폴더에 `config.json`이 없으면 바로 503).
- **의존성**: `silero-vad==6.2.3`(MIT), 필수 의존성은 torch·packaging뿐, 모델이 패키지 안에 있어 실행 중 다운로드가 없다. 컨테이너 torch 2.6.0과 범위가 맞는다(실제 이미지 빌드·import는 배포 때 확인).
- **계약·BE**: 응답 필드·상태 코드 그대로. 새로 생길 수 있는 조합 `replySource=SAFETY` + `needsCheck=true`는 BE가 두 필드를 읽지 않아 영향 없다(BE는 `riskFlag`·`signals.riskFlag`·`reply.text`·`transcript`·`turnNo`·`ended`만 읽는다 - `CompanionService.java:235-247`). 위험 표시는 되묻는 턴에서도 `riskFlag`로 올라간다.
- `fuse_turn`·턴 라벨·무응답 경로·챗봇 `/stt` 기본 동작 변경 없음. `docs/companion-team-status.md` 스위치 표와 코드 기본값이 같다.

## 테스트(PHASE D)
`tests/test_companion_voice_stage.py` 28건이 말소리 감지(무음·생활 소음·0.3초 경계·Silero 실패 시 RMS 대체)·반복 정리·turbo 창 나누기·확신도 경계·되묻는 턴 위험·스위치 끔·챗봇 API 엔진을 고정한다.
빠진 것: **한 대화에서 되묻기·안전 문장이 반복되는 경우(H-1·H-2)**, 스위치 오타(L-1), 실행 중 예외(L-2), 긴 음성·토큰 상한(M-1).

## 수정 제안 (결정 필요한 것 표시)
1. H-1: `_avoid_repeat`에서 `replySource`가 `CONFIRM`인 턴은 바꾸지 않거나, 되묻는 문장을 2~3개 두고 돌려 쓴다. 테스트: 같은 대화에 lp -0.8 턴 3번 → 3번 모두 되묻는 문장.
2. H-2: `SAFETY` 턴도 `_avoid_repeat`에서 빼서 안전 문장을 그대로 둔다(같은 문장 반복을 허용). **결정 필요** - 안전 문장을 두 번 똑같이 듣는 것 vs 다른 안전 문장 하나 더 두기. 테스트: 위험 턴 2번 → 2번 모두 안전 문장.
3. M-1: `max_new_tokens`를 224 정도로 올리고(Whisper 디코더 한도 448 안), 30초 나누기를 Silero 말소리 경계에서 자르게 바꾼다. DGX에서 20~30초 긴 대답 몇 개로 확인.
4. M-2: 연속 되묻기 횟수를 세어 2~3번이면 무응답처럼 다루거나 대화를 끝낸다. **결정 필요** - 횟수와 끝낼 때의 문장.
5. L-1: 스위치 값을 소문자로 바꾸고, 모르는 값이면 기동 때 경고 로그를 남긴다.

수정 커밋 메시지 초안: `fix(companion): 되묻기·안전 문장이 같은 대화에서 다른 문장으로 바뀌지 않게 (#201 점검 H-1·H-2)`

## 조치 (2026-11-10, 사용자 "전부 반영해줘")
H-2·M-2의 결정은 안전한 쪽으로 정했다(같은 안전 문장 반복 허용 · 연속 3번). 모두 스위치로 끌 수 있고 끄면 지금(재희) 동작이다.

| ID | 조치 | 스위치(끄는 값) |
|---|---|---|
| H-1·H-2 | `_avoid_repeat`가 `replySource`가 `CONFIRM`·`SAFETY`인 턴은 바꾸지 않는다 - 되묻는 문장·안전 문장은 반복돼도 그대로 | `COMPANION_KEEP_FIXED_REPLIES=false` |
| M-1 | 토큰 상한 128 → 224, 30초를 넘으면 창 끝 5초 안 가장 조용한 20ms에서 나눈다 | `COMPANION_TURBO_MAX_NEW_TOKENS` |
| M-2 | 위험이 아닌 확신도 낮은 되묻기가 3번 연달아 나오면 마무리 문장("오늘도 이야기 나눠 주셔서 고맙습니다…")으로 답하고 `NO_RESPONSE`로 끝낸다. 알아들은 대답이 오면 0으로 | `COMPANION_CONFIRM_STREAK_LIMIT=0` |
| L-1 | 선택형 스위치 4종은 공백·대소문자를 무시하고, 모르는 값이면 이름만 경고 로그(`[CONFIG]`)를 남기고 기본값 | - |
| L-2 | 음성 턴·`/stt`에서 모델 실행 중 예외도 503 `STT_UNAVAILABLE`(로그는 클래스명만, 응답에 원문 없음) | - |
| L-3 | turbo 로드 실패는 60초 뒤 다시 시도 | `COMPANION_TURBO_RETRY_SEC=0` |
| L-4 | 수정 없음(뜻이 같음) | - |

검증: 새 테스트 `tests/test_companion_voice_stage_audit.py` 17건, 말벗 묶음 161 통과(main 144 + 17, 남은 실패 7건은 로컬에 없는 cv2·torch·faster_whisper·emotiefflib로 main과 같음). DGX 실제 turbo로 M-1 확인 - 어르신 문장을 이어 붙인 40~62초 대답 12개 글자 오류 옛 방식 11.96%(1개는 80% 미만으로 잘림) → 새 방식 2.64%(잘림 0), 짧은 문장 30개는 둘 다 2.65%로 같음.
