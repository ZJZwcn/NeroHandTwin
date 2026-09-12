"""下载 MediaPipe 手势模型文件到 models/ 目录。"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
}


def main() -> int:
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    ok = True
    for name, url in URLS.items():
        path = out_dir / name
        if path.exists() and path.stat().st_size > 1e6:
            print(f"[skip] {name} 已存在")
            continue
        print(f"[download] {name} ...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  -> {path} ({path.stat().st_size} bytes)")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
