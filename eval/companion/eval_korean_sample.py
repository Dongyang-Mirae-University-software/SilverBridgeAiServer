"""한국인 감정인식 복합 영상(샘플)으로 표정 모델 두 개를 평가한다 (평가 전용, 학습에 쓰지 않음).

데이터: 다운로드 폴더의 '한국인 감정인식을 위한 복합 영상(샘플).zip' (AI Hub, 비상업 연구용)
- 라벨: 기쁨 / 분노 / 상처 / 당황 (4개). 슬픔·불안 라벨 없음.
- 평가 방식: 큰 분류 두 가지로만 본다. 기쁨 vs 부정(분노·상처·당황).
  모델 라벨 매핑
    HSEmotion: Happiness -> 기쁨 / Anger·Contempt·Disgust·Fear·Sadness -> 부정 / Neutral·Surprise -> 중립
    trpakov : happy -> 기쁨 / angry·disgust·fear·sad -> 부정 / neutral·surprise -> 중립
  '중립'은 맞힘도 틀림도 아닌 별도 칸으로 센다.
- 표본: 부정 전부(1,710) + 기쁨 무작위 1,800 + 노년(60세 이상) 기쁨 171. 시드 0.
- 얼굴은 데이터에 적힌 상자(annot_A)로 잘라서 쓴다.

결과: logs/eval_korean_sample.json (파일 이름·사람 정보는 기록하지 않고 숫자만 기록)
실행 (저장소 루트에서):
    venv\\Scripts\\python.exe -u eval\\eval_korean_sample.py
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
ZIP = Path(os.environ["KOREAN_SAMPLE_ZIP"])  # 데이터 zip 경로는 실행할 때 환경 변수로 준다 (코드에 개인 경로를 두지 않음)
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
NEG_LABELS = {"분노", "상처", "당황"}
SEED = 0
N_POS = 1800

HSE_MAP = {"Happiness": "기쁨", "Anger": "부정", "Contempt": "부정", "Disgust": "부정",
           "Fear": "부정", "Sadness": "부정", "Neutral": "중립", "Surprise": "중립"}
TRP_MAP = {"happy": "기쁨", "angry": "부정", "disgust": "부정", "fear": "부정", "sad": "부정",
           "neutral": "중립", "surprise": "중립"}


def crop(img: np.ndarray, box: dict) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box["minX"])), max(0, int(box["minY"]))
    x2, y2 = min(w, int(box["maxX"])), min(h, int(box["maxY"]))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return img[y1:y2, x1:x2]


def summarize(preds: list[str], golds: list[str], elder_mask: list[bool]) -> dict:
    out = {}
    for name, sel in (("all", [True] * len(golds)), ("elder_60plus", elder_mask)):
        idx = [i for i, s in enumerate(sel) if s]
        g = [golds[i] for i in idx]
        p = [preds[i] for i in idx]
        pos = [i for i, x in enumerate(g) if x == "기쁨"]
        neg = [i for i, x in enumerate(g) if x == "부정"]
        out[name] = {
            "n": len(idx),
            "recall_기쁨": round(sum(p[i] == "기쁨" for i in pos) / max(len(pos), 1), 4),
            "recall_부정": round(sum(p[i] == "부정" for i in neg) / max(len(neg), 1), 4),
            "중립_응답_비율": round(sum(x == "중립" for x in p) / max(len(p), 1), 4),
            "accuracy_기쁨vs부정(중립 제외)": round(
                sum(p[i] == g[i] for i in range(len(g)) if p[i] != "중립") / max(sum(x != "중립" for x in p), 1), 4),
        }
    return out


def main() -> None:
    rng = random.Random(SEED)
    z = zipfile.ZipFile(ZIP)
    meta = []
    for n in z.namelist():
        if n.lower().endswith(".json") and "라벨링데이터" in n:
            meta += json.loads(z.read(n).decode("utf-8", "ignore"))
    img_names = {Path(n).name: n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg"))}

    neg = [m for m in meta if m["faceExp_uploader"] in NEG_LABELS]
    pos_all = [m for m in meta if m["faceExp_uploader"] == "기쁨"]
    elder = [m for m in pos_all if m["age"] >= 60]
    pos = rng.sample([m for m in pos_all if m["age"] < 60], N_POS)
    sample = neg + pos + elder
    print(f"표본: 부정 {len(neg)}, 기쁨 {len(pos)}, 노년 기쁨 {len(elder)} (합계 {len(sample)})", flush=True)

    hse = EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_best_afew")
    hse_names = hse.idx_to_emotion_class
    trp_proc = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression", cache_dir=str(CACHE))
    trp = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression", cache_dir=str(CACHE)).eval()
    trp_names = trp.config.id2label

    golds, elder_mask = [], []
    hse_preds, trp_preds = [], []
    skipped = 0
    for k, m in enumerate(sample, 1):
        name = Path(m["filename"]).name
        if name not in img_names:
            skipped += 1
            continue
        img = cv2.imdecode(np.frombuffer(z.read(img_names[name]), np.uint8), cv2.IMREAD_COLOR)
        face = crop(img, m["annot_A"]["boxes"]) if img is not None else None
        if face is None:
            skipped += 1
            continue
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        _, probs = hse.predict_emotions(rgb, logits=False)
        pv = np.asarray(probs).reshape(-1)
        hse_preds.append(HSE_MAP.get(hse_names[int(np.argmax(pv))], "중립"))
        from PIL import Image

        inputs = trp_proc(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.no_grad():
            tp = torch.softmax(trp(**inputs).logits, dim=-1)[0]
        trp_preds.append(TRP_MAP.get(str(trp_names[int(torch.argmax(tp))]).lower(), "중립"))
        golds.append("기쁨" if m["faceExp_uploader"] == "기쁨" else "부정")
        elder_mask.append(m["age"] >= 60)
        if k % 200 == 0:
            print(f"  {k}/{len(sample)}", flush=True)

    result = {
        "n_used": len(golds), "skipped": skipped,
        "gold_counts": dict(Counter(golds)),
        "HSEmotion": summarize(hse_preds, golds, elder_mask),
        "trpakov": summarize(trp_preds, golds, elder_mask),
        "note": "평가 전용. 중립 응답은 정확도 계산에서 뺀 별도 비율. 얼굴 상자로 자른 이미지 기준.",
    }
    LOGS.mkdir(exist_ok=True)
    (LOGS / "eval_korean_sample.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
