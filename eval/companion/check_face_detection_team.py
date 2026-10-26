"""팀원 얼굴 검출 확인 (표정 정확도가 아니라 '얼굴이 잡히는지'만 본다).

목적: 실제 기기(핸드폰, 노트북 웹캠)와 조명에서 얼굴이 제대로 검출되는지 확인한다.
표정 정확도는 공개 데이터셋(Facial Emotion Recognition, Roboflow)으로 따로 측정한다.

입력: data/face_eval/detect/ 안의 영상이나 사진 (정답 라벨 없음, 파일 이름에 이름 넣지 않기)
처리: 0.5초마다 얼굴 검출 여부만 센다. 표정은 판단하지 않는다. 프레임은 저장하지 않는다.
결과: logs/face_detect_team.json (파일별 검출 비율만 기록)
실행 (저장소 루트에서):
    venv\\Scripts\\python.exe -u eval\\check_face_detection_team.py --delete
  --delete 를 붙이면 확인 후 영상을 지운다 (권장: 얼굴 데이터는 확인 후 바로 삭제).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
SRC = ROOT / "data" / "face_eval" / "detect"
LOGS = ROOT / "logs"
EXTS = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png"}
EVERY_SEC = 0.5


def load_detector() -> cv2.CascadeClassifier:
    src = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    tmp = Path(tempfile.gettempdir()) / "haarcascade_frontalface_default.xml"
    shutil.copyfile(src, tmp)
    return cv2.CascadeClassifier(str(tmp))


def frames_of(path: Path):
    if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        img = cv2.imread(str(path))
        if img is not None:
            yield img
        return
    cap = cv2.VideoCapture(str(path))
    step = max(1, int((cap.get(cv2.CAP_PROP_FPS) or 30) * EVERY_SEC))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield frame
        i += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="확인 후 영상 삭제")
    args = ap.parse_args()
    files = sorted(p for p in SRC.glob("*") if p.suffix.lower() in EXTS) if SRC.exists() else []
    if not files:
        raise SystemExit("확인할 파일이 없습니다. data\\face_eval\\detect\\ 에 영상을 넣어 주세요.")

    det = load_detector()
    rows = []
    for path in files:
        frames = with_face = 0
        for frame in frames_of(path):
            frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))) > 0:
                with_face += 1
        rows.append({"file_kind": path.suffix.lower(), "frames": frames,
                     "face_rate": round(with_face / max(frames, 1), 3)})

    rates = [r["face_rate"] for r in rows]
    summary = {
        "files": len(rows),
        "mean_face_rate": round(float(np.mean(rates)), 3),
        "min_face_rate": min(rates),
        "files_below_50pct": sum(r < 0.5 for r in rates),
        "per_file": rows,
        "note": "파일 이름과 얼굴 이미지는 기록하지 않음. 검출 비율만 기록.",
    }
    LOGS.mkdir(exist_ok=True)
    (LOGS / "face_detect_team.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"파일 {summary['files']}개, 평균 얼굴 검출 {summary['mean_face_rate']}, 최저 {summary['min_face_rate']}")
    print(f"검출 50% 미만 파일: {summary['files_below_50pct']}개")

    if args.delete:
        for path in files:
            path.unlink(missing_ok=True)
        print("확인에 쓴 영상 파일을 삭제했습니다.")


if __name__ == "__main__":
    main()
