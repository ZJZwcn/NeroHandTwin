"""仿真模式入口：MuJoCo NERO + 可选相机源。

用法：
    python scripts/run_sim.py                     # 纯仿真（无相机，键盘兜底演示）
    python scripts/run_sim.py --camera webcam     # 笔记本摄像头（伪3D模式）
    python scripts/run_sim.py --camera video --video path.mp4
    python scripts/run_sim.py --headless          # 无窗口自动演示（联调用）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerohandtwin.app import NeroGestureApp  # noqa: E402
from nerohandtwin.control.arm_interface import create_arm  # noqa: E402
from nerohandtwin.sources import CameraHandSource  # noqa: E402


def make_arm(args, arm_cfg):
    if args.model:
        arm_cfg["model_path"] = str(ROOT / args.model)
    return create_arm(arm_cfg)


def make_source(args):
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "camera.yaml").read_text(encoding="utf-8"))
    model = str(ROOT / cfg["tracker"]["model_path"])
    if not args.camera:
        return None
    if args.camera == "realsense":
        return CameraHandSource.realsense(model, depth_mode="depth")
    if args.camera == "webcam":
        return CameraHandSource.mock(model, source="webcam", depth_mode="estimate")
    if args.camera == "video":
        if not args.video:
            sys.exit("--camera video 需要 --video 指定文件")
        return CameraHandSource.mock(model, source="video", index_or_path=args.video,
                                     depth_mode="estimate")
    sys.exit(f"未知相机类型: {args.camera}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NERO 手势交互仿真")
    parser.add_argument("--camera", choices=["realsense", "webcam", "video"], default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--headless", action="store_true", help="无窗口自动化演示")
    parser.add_argument("--model", default=None, help="覆盖 scene.xml 路径（相对项目根）")
    args = parser.parse_args()

    import yaml

    arm_cfg = yaml.safe_load((ROOT / "configs" / "arm.yaml").read_text(encoding="utf-8"))["arm"]
    # model_path 相对项目根
    if arm_cfg.get("model_path") and not Path(arm_cfg["model_path"]).is_absolute():
        arm_cfg["model_path"] = str(ROOT / arm_cfg["model_path"])

    arm = make_arm(args, arm_cfg)
    source = make_source(args)
    app = NeroGestureApp(ROOT / "configs", obs_source=source, arm=arm,
                         headless=args.headless, show=not args.headless)

    if args.headless:
        app.start()
        # 自动演示脚本：模拟张开手掌 -> 跟随 -> 急停 -> 解锁
        from nerohandtwin.perception.hand_tracker import HandObservation
        import numpy as np

        t0 = time.monotonic()
        print("[demo] headless 自检：跟随 -> 急停 -> 解锁")
        try:
            while time.monotonic() - t0 < 12.0:
                t = time.monotonic() - t0
                # 合成掌心点：绕圆运动（相机系占位值，标定矩阵为恒等）
                palm = np.array([0.3 + 0.1 * np.sin(2 * np.pi * t / 6), 0.0, 0.35 + 0.08 * np.cos(2 * np.pi * t / 6)])
                gesture = "Open_Palm" if t < 8 else ("Closed_Fist" if t < 10 else "Victory")
                obs = HandObservation(present=True, timestamp_ms=int(t * 1000),
                                      gesture=gesture, pinch=0.8, palm3d=palm,
                                      index3d=palm + [0, 0, 0.05])
                app.obs_source = _FixedSource(obs)
                app.run_once(time.monotonic() * 1000)
                time.sleep(0.033)
        finally:
            app.shutdown()
        return

    app.run()


class _FixedSource:
    def __init__(self, obs):
        self._obs = obs
        self.last_obs = obs
        self.last_frame = None

    def get(self, _now_ms):
        return self._obs


if __name__ == "__main__":
    main()
