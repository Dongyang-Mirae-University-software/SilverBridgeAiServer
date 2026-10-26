"""새 문장으로 음성 인식(faster-whisper large-v3)을 시험한다. 학습·평가에 쓰인 적 없는 문장이다.

- 음성: Windows 한국어 음성(Microsoft Heami)으로 만든 합성 음성. 실제 어르신 목소리가 아니므로 결과는 참고용이다.
- 조건 4가지: 깨끗함 / 잡음(다른 문장의 소리를 5dB로 섞음) / 느리고 망설이는 말투 / 잡음+망설임
- 측정: 글자 오류율(CER, 낮을수록 좋음), 핵심 단어 유지율(높을수록 좋음)
- 결과: logs/stt_new_sentences.json (문장과 숫자만 기록)

실행 (저장소 루트에서, 오래 걸리므로 백그라운드 권장):
    venv\\Scripts\\python.exe -u eval\\stt_new_sentences.py
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
sys.path.insert(0, str(ROOT / "app" / "companion"))
import api_mock  # noqa: E402

# (문장, 핵심 단어) — 학습·평가 데이터에 없는 새 문장
SENTENCES = [
    ("오늘은 날씨가 좋아서 마당에 나가 꽃에 물을 줬어요", ["날씨", "꽃"]),
    ("요즘 허리가 자꾸 아파서 병원에 가야 하나 고민이에요", ["허리", "병원"]),
    ("아들이 전화를 해서 아주 반가웠어요", ["아들", "전화"]),
    ("혼자 밥을 먹으니까 밥맛이 하나도 없네요", ["밥맛"]),
    ("경로당에서 친구들이랑 화투를 쳤는데 돈을 잃었어요", ["경로당", "화투"]),
    ("어젯밤에 잠을 설쳐서 오늘은 너무 피곤해요", ["잠", "피곤"]),
    ("손녀가 그림을 그려 줘서 방에 붙여 놨어요", ["손녀", "그림"]),
    ("그냥 그럭저럭 지냈어요 별일은 없었어요", ["별일"]),
    ("요즘 자꾸 깜빡깜빡해서 걱정이 많아요", ["깜빡", "걱정"]),
    ("시장에 가서 귤 한 봉지랑 두부를 샀어요", ["시장", "귤", "두부"]),
    ("남편 생각이 나서 오늘은 자꾸 눈물이 나요", ["남편", "눈물"]),
    ("텔레비전에서 옛날 노래가 나와서 같이 따라 불렀어요", ["노래"]),
]

# 망설임 말투 (단어 선택이 흔들리는 예): 군더더기 말을 앞에 넣는다
HESITANT = ["음... 그러니까", "아이구, 저기요", "글쎄... 뭐랄까"]

SAPI_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$text = Get-Content -Encoding UTF8 -Raw -LiteralPath '{txt}'
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft Heami Desktop')
$s.Rate = {rate}
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{out}', $fmt)
$s.Speak($text)
$s.Dispose()
"""


def synth(text: str, rate: int, out_wav: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        txt = f.name
    script = SAPI_SCRIPT.format(txt=txt.replace("'", "''"), rate=rate, out=str(out_wav).replace("'", "''"))
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)
    Path(txt).unlink(missing_ok=True)


def mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.resize(noise, speech.shape)
    p_s = float(np.mean(speech ** 2)) + 1e-12
    p_n = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    return (speech + scale * noise).astype("float32")


def cer(ref: str, hyp: str) -> float:
    a = "".join(ch for ch in ref if ch.isalnum())
    b = "".join(ch for ch in hyp if ch.isalnum())
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[len(b)] / max(len(a), 1)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="stt_new_"))
    # 잡음으로 쓸 소리: 다른 문장들을 이어 붙인 소리 (사람 말소리 같은 잡음)
    babble_src = []
    for i, (text, _) in enumerate(SENTENCES):
        p = work / f"babble_{i}.wav"
        synth(text, 0, p)
        babble_src.append(sf.read(str(p), dtype="float32")[0])
    babble = np.concatenate(babble_src)

    conditions = {"깨끗함": ("clean", 0, None), "잡음 5dB": ("noise", 0, 5),
                  "느리고 망설임": ("hesitant", -3, None), "잡음+망설임": ("both", -3, 5)}
    results = {}
    for cname, (_, rate, snr) in conditions.items():
        cers, recalls, times = [], [], []
        for i, (text, keys) in enumerate(SENTENCES):
            spoken = text
            if cname in ("느리고 망설임", "잡음+망설임"):
                spoken = HESITANT[i % len(HESITANT)] + ", " + text
            wav = work / f"{cname}_{i}.wav"
            synth(spoken, rate, wav)
            audio, sr = sf.read(str(wav), dtype="float32")
            if snr is not None:
                audio = mix_noise(audio, babble, snr)
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV")
            import time
            t0 = time.time()
            hyp, _ = api_mock._transcribe(buf.getvalue())
            times.append(time.time() - t0)
            cers.append(cer(text, hyp))
            recalls.append(sum(k in hyp for k in keys) / len(keys))
            print(f"[{cname}] {i + 1}/{len(SENTENCES)} CER {cers[-1]:.2f} | 핵심 {recalls[-1]:.2f}", flush=True)
        results[cname] = {"n": len(SENTENCES), "mean_cer": round(float(np.mean(cers)), 3),
                          "keyword_recall": round(float(np.mean(recalls)), 3),
                          "mean_seconds": round(float(np.mean(times)), 1)}
    out = {"voice": "Microsoft Heami (합성, 실제 어르신 목소리 아님)", "model": "faster-whisper large-v3",
           "results": results,
           "note": "학습·평가에 쓰이지 않은 새 문장. 합성 음성이므로 참고용."}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "stt_new_sentences.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
