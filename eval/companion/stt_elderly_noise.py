"""실제 어르신 목소리(AI Hub 107 노인 남녀 대화)로 음성 인식을 잡음·말 속도 조건별로 시험한다.

- 음성: 공개 데이터의 실제 어르신 발화 (data\\aihub_107_sample). 정답은 데이터의 stt_ref.
- 주의: 이 음성들은 이전 글자 인식 평가(eval_stt_faster_107.py)에 이미 쓰였다. 새 음성은 아니다.
- 조건: 원래 소리 / 잡음 10dB / 잡음 5dB / 말 속도 느리게(0.85배)
- 잡음: 다른 어르신 발화들을 이어 붙인 소리를 섞는다 (사람 말소리 같은 잡음이 가장 어렵다)
- 측정: 글자 오류율(CER, 낮을수록 좋음)
- 결과: logs\\stt_elderly_noise.json (숫자만 기록)

실행 (저장소 루트에서, 오래 걸리므로 백그라운드 권장):
    venv\\Scripts\\python.exe -u eval\\stt_elderly_noise.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
sys.path.insert(0, str(ROOT / "app" / "companion"))
import api_mock  # noqa: E402
from stt_new_sentences import cer, mix_noise  # noqa: E402

DATA = ROOT / "data" / "aihub_107_sample"
N = 12  # 조건당 발화 수 (시간 제한 때문에 적게 잡음)
SLOW = 0.85  # 말 속도 배수 (1보다 작으면 느림)


def slow_down(audio: np.ndarray, factor: float) -> np.ndarray:
    idx = np.arange(0, len(audio), 1 / factor)  # 길이가 늘어난다 = 느린 말
    return np.interp(idx, np.arange(len(audio)), audio).astype("float32")


def main() -> None:
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))[:N]
    clips = [(sf.read(str(DATA / m["wav"]), dtype="float32")[0], m["stt_ref"]) for m in meta]
    babble = np.concatenate([c[0] for c in clips])  # 잡음은 다른 어르신 발화들을 이어 붙인 것

    conditions = {
        "원래 소리": lambda a: a,
        "잡음 10dB": lambda a: mix_noise(a, babble, 10),
        "잡음 5dB": lambda a: mix_noise(a, babble, 5),
        "느린 말 (0.85배)": lambda a: slow_down(a, SLOW),
    }
    results = {}
    for name, fn in conditions.items():
        cers, times = [], []
        for i, (audio, ref) in enumerate(clips):
            buf = io.BytesIO()
            sf.write(buf, fn(audio), 16000, format="WAV")
            t0 = time.time()
            hyp, _ = api_mock._transcribe(buf.getvalue())
            times.append(time.time() - t0)
            cers.append(cer(ref, hyp))
            print(f"[{name}] {i + 1}/{len(clips)} CER {cers[-1]:.2f}", flush=True)
        results[name] = {"n": len(clips), "mean_cer": round(float(np.mean(cers)), 3),
                         "median_cer": round(float(np.median(cers)), 3),
                         "share_under_10pct": round(float(np.mean([c <= 0.10 for c in cers])), 3),
                         "mean_seconds": round(float(np.mean(times)), 1)}
    out = {"voice": "AI Hub 107 노인 남녀 대화 (실제 어르신 발화, 이전 평가와 같은 음성)",
           "model": "faster-whisper large-v3", "results": results,
           "note": "잡음은 다른 어르신 발화를 섞은 것. 말 속도는 음을 늘려서 만든 것(음높이도 낮아짐)."}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "stt_elderly_noise.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
