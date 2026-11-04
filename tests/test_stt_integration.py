from __future__ import annotations

import io

import numpy as np
import pytest

from app.services.companion.stt_service import SttService

pytestmark = pytest.mark.skipif(
    not SttService().model_dir().is_dir(),
    reason="음성 인식 모델이 아직 없습니다 (models/companion/stt/large-v3). 내려받은 뒤 실행하세요.",
)


def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    data = b"\x00\x00" * n
    header = (
        b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        + rate.to_bytes(4, "little") + (rate * 2).to_bytes(4, "little") + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little") + b"data" + len(data).to_bytes(4, "little")
    )
    return header + data


def test_model_loads_and_silence_gives_empty_text():
    svc = SttService()
    result = svc.transcribe(_silence_wav())
    assert svc.is_loaded()
    assert result.text == ""  # VAD 가 무음을 버린다 → 무응답
    assert result.confidence == 0.0
    assert result.elapsed_ms >= 0


def _encoded_silence(container_format: str, codec: str) -> bytes | None:
    """PyAV 로 무음을 webm/mp3 로 인코딩한다. 인코더가 없으면 None (그 형식은 건너뜀)."""
    import av

    buf = io.BytesIO()
    try:
        out = av.open(buf, mode="w", format=container_format)
        stream = out.add_stream(codec, rate=16000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 16000), dtype=np.int16), format="s16", layout="mono")
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
        out.close()
    except Exception:
        return None
    return buf.getvalue()


@pytest.mark.parametrize("container_format,codec", [("webm", "libopus"), ("mp3", "libmp3lame")])
def test_webm_and_mp3_silence_decode_without_error(container_format, codec):
    data = _encoded_silence(container_format, codec)
    if not data:
        pytest.skip(f"{container_format} 인코더를 쓸 수 없어 건너뜀")
    result = SttService().transcribe(data)
    assert result.text == ""  # 무음은 글자가 없다 (디코딩이 되었다는 확인)
