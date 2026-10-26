"""표정 모델 두 개를 같은 테스트 이미지로 비교한다.

- 모델 A: HSEmotion (EmotiEffLib, enet_b0_8_best_afew 가중치, AffectNet·AFEW 계열 학습)
- 모델 B: trpakov/vit-face-expression (FER2013 학습)
- 테스트 데이터: 다운로드 폴더의 Facial Emotion Recognition (Roboflow, CC BY 4.0) test 폴더 1,638장
  주의: 성인 얼굴 데이터다. 어르신 얼굴 정확도를 대신하지 않는다.

공정 비교 방법
- 데이터 라벨은 angry, fear, happy, neutral, sad, surprise 6개다.
- 두 모델 모두 이 6개 중에서만 답하도록 제한한다 (disgust 등 다른 라벨은 무시).

실행 (저장소 루트에서, 장시간 작업이므로 백그라운드 권장):
    venv\\Scripts\\python.exe -u eval\\eval_face_compare.py
결과: logs\\face_compare_roboflow_test.json
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
ZIP = Path(os.environ["FACE_EMOTION_ZIP"])  # 데이터 zip 경로는 실행할 때 환경 변수로 준다 (코드에 개인 경로를 두지 않음)
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
COMMON = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

# 각 모델 라벨 이름을 공통 소문자 이름으로 맞춘다
HSE_MAP = {"anger": "angry", "fear": "fear", "happiness": "happy", "neutral": "neutral",
           "sadness": "sad", "surprise": "surprise", "disgust": None}
TRP_MAP = {"angry": "angry", "fear": "fear", "happy": "happy", "neutral": "neutral",
           "sad": "sad", "surprise": "surprise", "disgust": None}


def load_test_items() -> list[tuple[str, str, bytes]]:
    z = zipfile.ZipFile(ZIP)
    items = []
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 3 and parts[0] == "test" and n.lower().endswith((".jpg", ".jpeg", ".png")):
            items.append((parts[1], n, z.read(n)))
    return items


def restricted_pick(scores: dict[str, float]) -> str:
    """공통 6개 라벨 중 가장 높은 것을 고른다."""
    return max(COMMON, key=lambda k: scores.get(k, float("-inf")))


def run_hsemotion(images: list[np.ndarray]) -> list[dict[str, float]]:
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    rec = EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_best_afew")
    names = rec.idx_to_emotion_class  # {0: 'Anger', ..., 7: 'Surprise'} (8개 확률)
    out = []
    for img in images:
        _, probs = rec.predict_emotions(img, logits=False)
        p = np.asarray(probs).reshape(-1)
        scores = {}
        for i, prob in enumerate(p):
            key = HSE_MAP.get(str(names[i]).lower())
            if key:
                scores[key] = float(prob)
        out.append(scores)
    return out


def run_trpakov(images: list[Image.Image]) -> list[dict[str, float]]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    proc = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression", cache_dir=str(CACHE))
    model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression", cache_dir=str(CACHE))
    model.eval()
    id2label = model.config.id2label
    out = []
    with torch.no_grad():
        for img in images:
            inputs = proc(images=img, return_tensors="pt")
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
            scores = {}
            for i, prob in enumerate(probs):
                key = TRP_MAP.get(str(id2label[i]).lower())
                if key:
                    scores[key] = prob
            out.append(scores)
    return out


def summarize(name: str, gold: list[str], pred: list[str], seconds: float) -> dict:
    return {
        "model": name,
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, labels=COMMON, average="macro", zero_division=0), 4),
        "per_class_recall": {
            c: round(sum(1 for g, p in zip(gold, pred) if g == c and p == c) / max(1, sum(1 for g in gold if g == c)), 4)
            for c in COMMON
        },
        "seconds": round(seconds, 1),
    }


def main() -> None:
    items = load_test_items()
    gold = [g for g, _, _ in items]
    print(f"테스트 이미지 {len(items)}장: {dict(Counter(gold))}", flush=True)
    pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for _, _, b in items]
    np_images = [np.asarray(im) for im in pil_images]

    t0 = time.time()
    print("HSEmotion 실행 중...", flush=True)
    hse_scores = run_hsemotion(np_images)
    hse_sec = time.time() - t0
    hse_pred = [restricted_pick(s) for s in hse_scores]
    print(f"  HSEmotion 완료 ({hse_sec:.0f}s)", flush=True)

    t1 = time.time()
    print("trpakov 실행 중...", flush=True)
    trp_scores = run_trpakov(pil_images)
    trp_sec = time.time() - t1
    trp_pred = [restricted_pick(s) for s in trp_scores]
    print(f"  trpakov 완료 ({trp_sec:.0f}s)", flush=True)

    result = {
        "test_set": "Facial Emotion Recognition (Roboflow, CC BY 4.0) test 폴더",
        "n": len(items),
        "gold_counts": dict(Counter(gold)),
        "labels_used": COMMON,
        "caveat": "성인 얼굴 데이터. 어르신 얼굴 정확도가 아님.",
        "results": [
            summarize("HSEmotion enet_b0_8_best_afew", gold, hse_pred, hse_sec),
            summarize("trpakov/vit-face-expression", gold, trp_pred, trp_sec),
        ],
    }
    LOGS.mkdir(exist_ok=True)
    out = LOGS / "face_compare_roboflow_test.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in result["results"]:
        print(f"{r['model']}: 정확도 {r['accuracy']} / macro F1 {r['macro_f1']}", flush=True)
    print(f"저장: {out}", flush=True)


if __name__ == "__main__":
    main()
