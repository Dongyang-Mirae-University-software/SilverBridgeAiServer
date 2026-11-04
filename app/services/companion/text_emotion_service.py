from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import get_settings

TEXT_MODEL_DIR = "kcelectra-korean-emotion"  # GGARA02/kcelectra-korean-emotion (C-07, D8). 모델 폴더 이름
TEXT_MODEL_NAME = "GGARA02/kcelectra-korean-emotion"


class TextEmotionUnavailable(Exception):
    """텍스트 감정 모델을 쓸 수 없음 (폴더 없음, 불러오기 실패). 대화는 계속되고 글자 감정만 빠진다."""


@dataclass
class TextEmotionResult:
    label: str  # 원래 6개 감정 중 하나 (분노·기쁨·불안·당황·슬픔·상처). 턴 라벨 매핑은 S6/S7에서 정한다
    confidence: float  # 0~1. 가장 높은 확률
    usable: bool  # confidence 가 companion_confidence_min 이상이면 True
    scores: dict[str, float] = field(default_factory=dict)  # 라벨별 확률 (S7 결합 때 쓴다)


class TextEmotionService:
    """KcELECTRA 감정 분류 래퍼 (S5-1). 모델은 처음 쓸 때 한 번만 올리고, 동시 요청은 lock 으로 막는다.

    GPU 메모리를 MedGemma 와 나누므로 CPU 에서 돌린다 (D8 이후 메모리 측정은 S12-3).
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    def model_dir(self) -> Path:
        s = get_settings()
        # D19 결정: MODEL_BASE_PATH 아래 모델별 폴더. 예: models/companion/text-emotion/kcelectra-korean-emotion
        return Path(s.model_base_path) / s.companion_text_emotion_subdir / TEXT_MODEL_DIR

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model
        if self._load_error is not None:
            raise TextEmotionUnavailable(self._load_error)
        with self._lock:
            if self._model is not None:
                return self._tokenizer, self._model
            path = self.model_dir()
            if not (path / "model.safetensors").is_file():
                self._load_error = f"텍스트 감정 모델 파일이 없습니다: {path}"
                raise TextEmotionUnavailable(self._load_error)
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                # local_files_only: 인터넷에서 다시 받지 않는다. 이미 내려받은 폴더만 쓴다 (D19)
                tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
                model.eval()
                self._tokenizer, self._model = tokenizer, model
            except Exception as exc:
                self._load_error = f"텍스트 감정 모델을 불러오지 못했습니다 ({type(exc).__name__})"
                raise TextEmotionUnavailable(self._load_error) from exc
            return self._tokenizer, self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def warm_up(self) -> None:
        """모델을 미리 올린다. 첫 분석이 시간 제한(S9-1)에 걸리지 않도록 서버 시작 때 부른다."""
        self._load()

    def classify(self, text: str) -> TextEmotionResult:
        """글을 감정 하나로 나눈다. 원문은 함수 밖으로 남기지 않는다 (D2)."""
        import torch

        tokenizer, model = self._load()
        s = get_settings()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        id2label = model.config.id2label
        scores = {str(id2label[i]): round(float(p), 4) for i, p in enumerate(probs)}
        label = max(scores, key=scores.get)
        confidence = round(scores[label], 4)
        return TextEmotionResult(
            label=label,
            confidence=confidence,
            usable=confidence >= s.companion_confidence_min,
            scores=scores,
        )


text_emotion_service = TextEmotionService()
