from __future__ import annotations

import subprocess

import imageio_ffmpeg


def make_access_units(*, width: int = 640, height: int = 360, fps: int = 15, seconds: int = 3, gop: int = 15) -> list[tuple[bool, bytes]]:
    """합성 장면(testsrc2)을 libx264 Baseline 으로 인코딩해 (키프레임 여부, Annex B 액세스 유닛) 목록으로 돌려준다.

    가상 장면만 쓴다(실제 영상 금지). 반환 목록의 각 항목은 WebCodecs annexb 출력처럼 키프레임에 SPS/PPS 가 붙는다.
    """
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate={fps}", "-t", str(seconds),
        "-c:v", "libx264", "-profile:v", "baseline", "-preset", "ultrafast", "-tune", "zerolatency",
        "-threads", "1", "-g", str(gop), "-bf", "0", "-x264-params", "repeat-headers=1:sliced-threads=0", "-f", "h264", "-",
    ]  # fmt: skip
    data = subprocess.run(cmd, capture_output=True, check=True).stdout  # noqa: S603
    return split_access_units(data)


def split_access_units(stream: bytes) -> list[tuple[bool, bytes]]:
    """Annex B 스트림을 NAL 로 나눈 뒤, 슬라이스(1·5) 하나를 끝으로 하는 액세스 유닛으로 묶는다(앞의 SPS/PPS/SEI 는 같은 유닛)."""
    starts: list[int] = []
    i = 0
    n = len(stream)
    while i + 3 <= n:
        if stream[i] == 0 and stream[i + 1] == 0 and stream[i + 2] == 1:
            starts.append(i)
            i += 3
        else:
            i += 1
    units: list[tuple[bool, bytes]] = []
    current_start = 0
    key = False
    for idx, pos in enumerate(starts):
        nal_type = stream[pos + 3] & 0x1F
        end = starts[idx + 1] if idx + 1 < len(starts) else n
        if nal_type == 5:
            key = True
        if nal_type in (1, 5):
            # 다음 NAL 앞의 0 바이트(4바이트 시작 코드의 선행 0)는 다음 유닛 몫이다.
            cut = end
            while cut > pos and idx + 1 < len(starts) and stream[cut - 1] == 0:
                cut -= 1
            units.append((key, stream[current_start:cut]))
            current_start = cut
            key = False
    return units


def frame_message(is_key: bool, au: bytes, ts_us: int = 0) -> bytes:
    return ts_us.to_bytes(8, "big") + (b"\x01" if is_key else b"\x00") + au
