"""快速方向标定（30 秒，无机械臂、无触碰）：三步手势求跟随方向映射。

原理：跟随需要"手部位移(相机系) → 机械臂位移(基座系)"的方向映射。
你只需按提示沿【自己视角】的三个方向各移动一次手（向你的右边、向上、
朝机械臂方向），脚本测量相机系位移并解出旋转矩阵——指令全部基于
你自己的身体视角，零歧义。

用法：
    python scripts/calibrate_quick.py            # 默认 RealSense
    python scripts/calibrate_quick.py --camera webcam
    python scripts/calibrate_quick.py --stand left   # 你站在机械臂左侧时

画面说明：
    黄色圆环 = 原点（第 1 步的手掌位置）；白点 = 手掌当前位置；
    白线 = 手掌相对原点的位移。按提示把白点沿指定方向移开再按空格。
窗口内按 q 中止。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

WIN = "calibrate_quick - 空格采样 / q 中止"


# 用户站位 -> 三个期望方向（基座系）：[向你的右, 向上, 朝机械臂]
# NERO 基座系：x+ = 机械臂正前（工作区/朝用户），y+ = 机械臂左，z+ = 上
_STAND_DIRS = {
    "front": [(0, 1, 0), (0, 0, 1), (-1, 0, 0)],   # 面对机械臂前方（默认）
    "left": [(-1, 0, 0), (0, 0, 1), (0, -1, 0)],   # 站在机械臂左侧
    "right": [(1, 0, 0), (0, 0, 1), (0, 1, 0)],    # 站在机械臂右侧
    "behind": [(0, -1, 0), (0, 0, 1), (1, 0, 0)],  # 站在机械臂后方
}


def _draw_hud(img: np.ndarray, instruction: str, progress: tuple[int, int] | None,
              origin_px: tuple | None, palm_px: tuple | None) -> np.ndarray:
    h, w = img.shape[:2]
    # 画面中心十字参考线
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 14, cy), (cx + 14, cy), (0, 255, 255), 1)
    cv2.line(img, (cx, cy - 14), (cx, cy + 14), (0, 255, 255), 1)
    if origin_px is not None:
        cv2.circle(img, origin_px, 14, (0, 255, 255), 2)          # 原点环
    if palm_px is not None and origin_px is not None:
        cv2.circle(img, palm_px, 6, (255, 255, 255), -1)          # 手掌点
        cv2.line(img, origin_px, palm_px, (255, 255, 255), 2)     # 位移线
    cv2.rectangle(img, (0, 0), (w, 52), (30, 30, 30), -1)
    cv2.putText(img, instruction, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (120, 255, 160), 1, cv2.LINE_AA)
    if progress:
        c, n = progress
        cv2.rectangle(img, (0, h - 12), (int(w * min(c / n, 1.0)), h), (80, 220, 90), -1)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["realsense", "webcam"], default="realsense")
    parser.add_argument("--stand", choices=list(_STAND_DIRS), default="front",
                        help="你站在机械臂的哪一侧（默认：面对前方工作区）")
    args = parser.parse_args()

    from nerohandtwin.geometry.calib_solver import solve_rotation_from_probes
    from nerohandtwin.perception.camera_realsense import MockCamera, RealSenseCamera
    from nerohandtwin.perception.hand_tracker import HandTracker

    model = str(ROOT / "models" / "gesture_recognizer.task")
    cam = RealSenseCamera() if args.camera == "realsense" else \
        MockCamera(source="webcam", index_or_path=0)
    tracker = HandTracker(model, depth_mode="depth" if args.camera == "realsense" else "estimate")

    dirs_base = np.asarray(_STAND_DIRS[args.stand], dtype=float)
    print("=" * 62)
    print(f"快速方向标定（站位={args.stand}）：3 步手势，全程约 30 秒")
    print("每步：听到/看到提示后【移动手并保持】，按空格采样")
    print("=" * 62)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def collect(instruction: str, need: int = 30, max_s: float = 15.0) -> np.ndarray:
        """空格触发后采集 need 帧有效掌心 3D，返回中位数位置。"""
        pts: list[np.ndarray] = []
        collecting = False
        t0: float | None = None
        origin_px = getattr(collect, "_origin_px", None)
        while True:
            frame = cam.read()
            if frame is None:
                img = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(img, "NO CAMERA FRAMES", (80, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow(WIN, img)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    raise KeyboardInterrupt
                continue
            obs = tracker.process(frame)
            img = frame.color.copy()
            palm_px = None
            if obs.present and obs.landmarks_px is not None:
                p9 = obs.landmarks_px.astype(int)[9]
                palm_px = (int(p9[0]), int(p9[1]))
            img = _draw_hud(img, instruction,
                            (len(pts), need) if collecting else None,
                            origin_px, palm_px)
            cv2.imshow(WIN, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if not collecting and key == ord(" "):
                collecting = True
                pts = []
                t0 = time.monotonic()
                if obs.present and obs.landmarks_px is not None:
                    p9 = obs.landmarks_px.astype(int)[9]
                    collect._origin_px = (int(p9[0]), int(p9[1]))
                origin_px = collect._origin_px
            if collecting and obs.present and obs.palm3d is not None:
                pts.append(obs.palm3d)
                if len(pts) >= need:
                    break
            if collecting and t0 is not None and time.monotonic() - t0 > max_s and len(pts) < 10:
                raise RuntimeError("有效帧不足（手离相机太近/太远或深度无效），请重试")
        collect._origin_px = None
        if len(pts) < need // 2:
            raise RuntimeError("有效帧不足，请重试")
        return np.median(np.array(pts), axis=0)

    deltas = []
    steps = [
        ("[1/3] 手移到舒适位置保持，按空格记原点", 30),
        ("[2/3] 手向你的【右边】移动约 15cm 并保持，按空格采样", 30),
        ("[3a/3] 手【向上】移动约 15cm 并保持，按空格采样", 30),
        ("[3b/3] 手向【机械臂方向】伸过去约 15cm 并保持，按空格采样", 30),
    ]
    origin = None
    try:
        for i, (instr, need) in enumerate(steps):
            pos = collect(instr, need)
            if i == 0:
                origin = pos
                print(f"[1] 原点(相机系): {np.round(origin, 3)}")
            else:
                deltas.append(pos - origin)
                print(f"   位移(相机系): {np.round(pos - origin, 3)}")
    except KeyboardInterrupt:
        print("已中止")
        cv2.destroyAllWindows()
        return 1
    except RuntimeError as exc:
        print(f"采样失败: {exc}")
        cv2.destroyAllWindows()
        return 1
    finally:
        tracker.close()
        cam.close()

    # 位移顺序与 steps 对齐：[向右, 向上, 朝机械臂] ↔ dirs_base 同序
    dirs = dirs_base.copy()
    try:
        rot, resid = solve_rotation_from_probes(deltas, dirs)
    except ValueError as exc:
        print(f"求解失败: {exc}（移动幅度大一些、方向明确一些后重试）")
        return 1
    if resid > 0.3:
        # 硬门禁：残差大 = 三步方向混了（曾把 0.937 的垃圾旋转写入）
        print(f"✗ 残差 {resid:.2f} > 0.3，拒绝写入。三次移动方向区分度不足，请重跑：")
        print("  每步沿提示方向移动约 15cm、方向明确、幅度拉开。")
        return 1

    print(f"求解完成，残差={resid:.3f}（<0.1 可用）")
    print("R_base<-camera =")
    print(np.round(rot, 4))

    cfg_path = ROOT / "configs" / "calibration.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["extrinsics"]["T_base_camera"] = [
        [round(float(rot[0, 0]), 5), round(float(rot[0, 1]), 5), round(float(rot[0, 2]), 5), 0.0],
        [round(float(rot[1, 0]), 5), round(float(rot[1, 1]), 5), round(float(rot[1, 2]), 5), 0.0],
        [round(float(rot[2, 0]), 5), round(float(rot[2, 1]), 5), round(float(rot[2, 2]), 5), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    cfg["extrinsics"]["calibrated"] = "axes_user"
    cfg["extrinsics"]["note"] = "快速方向标定（用户视角三步手势）- 相对跟随已校准"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已写入 {cfg_path} ✓")
    print("现在运行 python scripts/run_real.py，跟随方向即为你的视角方向")
    print("（如后续想要'手到哪臂到哪'的绝对跟随，可再跑 calibrate_touch.py 精标）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
