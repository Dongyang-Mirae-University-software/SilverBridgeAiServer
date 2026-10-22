from __future__ import annotations

import io
import math
import threading
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
# 긴 변 상한. API가 내부에서 줄이는 상한 근처 - 1024px로 줄이면 정확도가 5~6/10으로 떨어졌다(2026-10-22 측정).
MAX_LONG_EDGE = 1568
# 압축 폭탄 방어. 휴대폰 사진(5000만 화소급)까지 받는다. JPEG는 줄여 읽어(draft) 디코드 메모리가 수십 MB에 그친다.
MAX_PIXELS = 60_000_000
# PNG·WEBP는 줄여 읽을 수 없어 화소 수만큼 메모리를 쓴다(2,500만 화소 PNG = 파일 105KB, 디코드 232MB - 2026-10-22 측정).
# 화면 캡처·A4 300dpi 스캔(약 870만)은 들어온다.
MAX_PIXELS_NON_JPEG = 16_000_000
# 동시에 다듬는 사진 수. 위 상한과 함께 정규화 메모리를 몇백 MB 안으로 묶는다(한 장 0.1~1초라 대기는 짧다).
MAX_CONCURRENT_NORMALIZE = 2
JPEG_QUALITY = 90

_normalize_slots = threading.BoundedSemaphore(MAX_CONCURRENT_NORMALIZE)


class InvalidImageError(ValueError):
    """이미지가 아니거나, 디코드할 수 없거나, 허용하지 않는 형식."""


@dataclass(frozen=True)
class NormalizedImage:
    jpeg: bytes
    width: int
    height: int


def normalize_image(data: bytes) -> NormalizedImage:
    """EXIF 방향 반영 → 긴 변 MAX_LONG_EDGE 이하로 축소(확대 안 함) → JPEG 재인코딩.

    재인코딩 때 EXIF를 붙이지 않아 촬영 위치(GPS)·기기 정보가 Claude로 나가지 않는다. 메모리에서만 처리한다.
    """
    if not data:
        raise InvalidImageError("empty")
    with _normalize_slots:
        return _normalize(data)


def _normalize(data: bytes) -> NormalizedImage:
    try:
        with warnings.catch_warnings():
            # Pillow는 MAX_IMAGE_PIXELS 초과를 경고로만 알리고 2배부터 오류를 낸다 - 직접 픽셀 수로 거른다.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                width, height = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise InvalidImageError("format")
            limit = MAX_PIXELS if fmt == "JPEG" else MAX_PIXELS_NON_JPEG
            if width <= 0 or height <= 0 or width * height > limit:
                raise InvalidImageError("pixels")
            with Image.open(io.BytesIO(data)) as img:
                scale = MAX_LONG_EDGE / max(width, height)
                if fmt == "JPEG" and scale < 1:
                    # 디코드 단계에서 1/2·1/4·1/8로 줄여 읽는다. 결과는 목표 크기 이상이라 아래 축소 화질은 같다.
                    img.draft("RGB", (math.ceil(width * scale), math.ceil(height * scale)))
                img.load()
                img = ImageOps.exif_transpose(img)
                img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.Resampling.LANCZOS)
                if img.mode != "RGB":
                    img = _to_rgb(img)
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=JPEG_QUALITY)
                return NormalizedImage(jpeg=out.getvalue(), width=img.width, height=img.height)
    except InvalidImageError:
        raise
    except Exception as exc:  # 디코드 실패(잘린 파일·이미지 아님·압축 폭탄 오류 등)
        raise InvalidImageError("decode") from exc


def _to_rgb(img: Image.Image) -> Image.Image:
    # 투명 PNG·WEBP는 흰 바탕에 얹는다(검은 바탕이면 검은 글씨가 사라진다).
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")
