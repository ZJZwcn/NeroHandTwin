"""相机自检：打开相机源并显示手部追踪叠加，验证感知链路。

用法：
    python scripts/test_camera.py                # 合成画面（无摄像头）
    python scripts/test_camera.py --source webcam
    python scripts/test_camera.py --source realsense
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from nerohandtwin.perception.gesture_fsm import GestureFSM  # noqa: E402
from nerohandtwin.sources import CameraHandSource  # noqa: E402
from nerohandtwin.ui.visualizer import Visualizer  # noqa: E402
from nerohandtwin.interaction.interaction_fsm import InteractMode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "webcam", "realsense"], default="synthetic")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    model = str(ROOT / "models" / "gesture_recognizer.task")
    if args.source == "realsense":
        src = CameraHandSource.realsense(model, depth_mode="depth")
    elif args.source == "webcam":
        src = CameraHandSource.mock(model, source="webcam", index_or_path=args.index,
                                    depth_mode="estimate")
    else:
        src = CameraHandSource.mock(model, source="synthetic", depth_mode="estimate")

    fsm = GestureFSM()
    viz = Visualizer()
    print("按 q 退出")
    fps = 0.0
    try:
        while True:
            t0 = time.monotonic()
            frame = src.camera.read()
            if frame is None:
                break
            now_ms = frame.timestamp_ms
            obs = src.tracker.process(frame)
            src.last_obs = obs
            stable = fsm.update(obs.gesture, obs.present, now_ms)
            lines = []
            if fsm.pending:
                lines.append(f"confirming: {fsm.pending} {fsm.pending_progress:.0%}")
            img = viz.draw(frame.color, obs, InteractMode.IDLE, lines, fps)
            if not obs.present:
                cv2.putText(img, "No hand", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            cv2.imshow("camera test", img)
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.monotonic() - t0, 1e-4))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
