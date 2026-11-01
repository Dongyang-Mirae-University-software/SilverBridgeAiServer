"""표정 보조 신호 - 어르신 기기 앞면 카메라 정지 사진(JPEG)에서 웃음·평온·부정 확률을 낸다.

- 얼굴 찾기: OpenCV YuNet(face_detection_yunet_2023mar.onnx, OpenCV 4.5.4 이상). 얼굴이 없으면 그 사진은 뺀다.
  OpenCV 5 에는 예전 Haar 검출기가 없다.
- 표정: HSEmotion enet_b0_8_best_vgaf 의 특징(1280차원) + 한국인 표정(AI Hub 82)으로 다시 학습한 3분류 층.
  원래 8감정 출력은 한국인 무표정을 부정으로 32%(50대 이상 약 50%) 읽었다. 다시 학습한 층은 부정 확률
  0.8 이상만 부정으로 볼 때 무표정→부정 3.3%·부정 찾음 74% (사람 단위 시험, 2026-11-01 DGX).
- 감정 판정이 아니라 보조 신호다. 사진은 저장하지 않고 숫자만 남긴다.
"""

from __future__ import annotations

import io
import logging
import threading

from app.services.checkin.config import get_checkin_settings

log = logging.getLogger(__name__)

LABELS = ("smile", "calm", "negative")
MAX_SIDE = 640  # 얼굴 찾기 전에 줄이는 크기 (휴대폰 사진도 이 정도면 충분하다)
SIZE = 224


def _expand(box, w, h):
    """검출기 상자는 얼굴만 잡는다. 학습 데이터 상자(이마·머리 일부 포함)에 가깝게 넓힌다."""
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2 - 0.1 * bh
    nw, nh = bw * 1.3, bh * 1.6
    return (max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2)), min(w, int(cx + nw / 2)), min(h, int(cy + nh / 2)))


class FaceAnalyzer:
    def __init__(self) -> None:
        import cv2
        import torch

        s = get_checkin_settings()
        self.torch = torch
        self.device = s.device
        self.neg_threshold = s.face_neg_pct / 100
        model = torch.load(s.face_model, map_location=self.device, weights_only=False)  # 운영자가 둔 로컬 파일만
        model.classifier = torch.nn.Identity()
        self.body = model.eval()
        head = torch.load(s.face_head, map_location="cpu")
        self.head = torch.nn.Linear(head["mu"].numel(), 3)
        self.head.load_state_dict(head["state"])
        self.head = self.head.to(self.device).eval()
        self.mu, self.sd = head["mu"].to(self.device), head["sd"].to(self.device)
        self.cv2 = cv2
        self.detector_path = s.face_detector
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self._lock = threading.Lock()

    def _crop(self, data: bytes):
        import numpy as np
        from PIL import Image, ImageOps

        im = Image.open(io.BytesIO(data))
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
        # 검출기는 입력 크기를 상태로 가져 스레드마다 만든다(작고 빠르다)
        det = self.cv2.FaceDetectorYN.create(self.detector_path, "", (im.width, im.height), 0.7)
        _, found = det.detect(bgr)
        side = min(im.size)
        found = [f[:4] for f in (found if found is not None else []) if min(f[2], f[3]) >= side / 8]
        if not found:
            return None
        box = max(found, key=lambda b: b[2] * b[3])  # 가장 큰 얼굴 = 기기 앞의 본인
        face = im.crop(_expand(box, im.width, im.height)).resize((SIZE, SIZE))
        t = self.torch.from_numpy(np.asarray(face).copy()).permute(2, 0, 1).float() / 255
        return (t - self.mean) / self.std

    def probs(self, frames: list[bytes]) -> tuple[int, list[list[float]]]:
        """(얼굴을 찾은 사진 수, 사진별 [웃음, 평온, 부정] 확률)."""
        crops = []
        for data in frames:
            try:
                c = self._crop(data)
            except Exception as e:  # 깨진 사진은 그 장만 뺀다
                log.info("[CHECKIN] face frame skipped err=%s", type(e).__name__)
                continue
            if c is not None:
                crops.append(c)
        if not crops:
            return 0, []
        torch = self.torch
        with self._lock, torch.no_grad():
            feats = self.body(torch.stack(crops).to(self.device))
            p = torch.softmax(self.head((feats - self.mu) / self.sd), 1).cpu().tolist()
        return len(crops), p

    def label(self, p: list[float]) -> str:
        if p[2] >= self.neg_threshold:
            return "negative"
        return "smile" if p[0] >= p[1] else "calm"

    def analyze(self, frames: list[bytes]) -> dict | None:
        if not frames:
            return None
        _, ps = self.probs(frames)
        return summarize_frames(len(frames), ps, self.label)


def summarize_frames(received: int, ps: list[list[float]], label) -> dict:
    out = {"frames": received, "faceFrames": len(ps)}
    if not ps:
        return out | {k: None for k in ("smile", "calm", "negative", "smileRatio", "negativeRatio")}
    labels = [label(p) for p in ps]
    for i, k in enumerate(LABELS):
        out[k] = round(sum(p[i] for p in ps) / len(ps), 3)
    out["smileRatio"] = round(labels.count("smile") / len(ps), 3)
    out["negativeRatio"] = round(labels.count("negative") / len(ps), 3)
    return out


def merge(faces: list[dict | None]) -> dict | None:
    """턴별 결과를 얼굴 사진 수로 가중 평균한다. 사진을 한 장도 받지 않았으면 None."""
    faces = [f for f in faces if f]
    if not faces:
        return None
    total = sum(f["faceFrames"] for f in faces)
    out = {"frames": sum(f["frames"] for f in faces), "faceFrames": total}
    for k in ("smile", "calm", "negative", "smileRatio", "negativeRatio"):
        out[k] = round(sum(f[k] * f["faceFrames"] for f in faces if f["faceFrames"]) / total, 3) if total else None
    return out


_face: FaceAnalyzer | None = None
_load_lock = threading.Lock()


def face_enabled() -> bool:
    s = get_checkin_settings()
    return bool(s.face_model and s.face_head and s.face_detector)


def get_face() -> FaceAnalyzer | None:
    """표정 모델 경로를 두지 않으면 None - 표정 없이 대화는 그대로 된다."""
    global _face
    if not face_enabled():
        return None
    if _face is None:
        with _load_lock:
            if _face is None:
                _face = FaceAnalyzer()
                log.info("[CHECKIN] face model loaded neg_threshold=%.2f", _face.neg_threshold)
    return _face
