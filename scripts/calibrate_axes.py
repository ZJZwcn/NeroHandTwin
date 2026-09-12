"""引导式轴向标定（免标定板）：三次定向手势求相机->基座旋转。

带实时预览窗口：手部骨架 + 步骤提示 + 采样进度条，按【空格】采样。

用途：跟随方向不对（手右移臂左移/躲着走）时运行一次，全程约 1 分钟。
原理：你沿三个已知方向各移动一次手，脚本记录相机系位移并解出旋转矩阵
R_base<-camera，写回 configs/calibration.yaml。相对跟随只依赖该旋转，
与相机装在哪里、平移多少完全无关。

基座系方向约定（NERO）：
    +x = 机械臂正前方（工作可达方向，朝向你）
    +y = 机械臂的左方（你面对机械臂时 = 你的右手边）
    +z = 竖直向上

用法：
    python scripts/calibrate_axes.py          # 默认 RealSense
    python scripts/calibrate_axes.py --camera webcam
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

WIN = "calibrate_axes - 按空格采样 / q 退出"


def _draw_hud(img: np.ndarray, instruction: str, count: int, need: int) -> np.ndarray:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), (30, 30, 30), -1)
    cv2.putText(img, instruction, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (120, 255, 160), 1, cv2.LINE_AA)
    # 采样进度条
    if count > 0:
        cv2.rectangle(img, (0, h - 14), (int(w * min(count / need, 1.0)), h), (80, 220, 90), -1)
    return img


def _loop_until_space(cam, tracker, instruction: str, need: int,
                      collect: bool) -> tuple[np.ndarray, np.ndarray]:
    """预览循环：实时骨架 + 进度条；空格键触发/结束。

    collect=False: 等待空格（静止提示阶段），返回 (最后帧, None)
    collect=True:  空格按下后开始采集 need 帧有效掌心 3D，返回 (最后帧, 样本数组)
    """
    import mediapipe as _mp  # noqa: F401 - 确保 HandTracker 已初始化

    pts: list[np.ndarray] = []
    collecting = False
    last_frame = None
    while True:
        frame = cam.read()
        if frame is None:
            continue
        last_frame = frame
        obs = tracker.process(frame)
        img = frame.color.copy()
        if obs.present and obs.landmarks_px is not None:
            for p in obs.landmarks_px.astype(int):
                cv2.circle(img, tuple(p), 3, (90, 180, 255), -1, cv2.LINE_AA)
        if collecting:
            if obs.present and obs.palm3d is not None:
                pts.append(obs.palm3d)
            if len(pts) >= need:
                break
        img = _draw_hud(img, instruction, len(pts) if collecting else 0, need)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt
        if key == ord(" ") and not collecting:
            collecting = True
            pts = []
    cv2.destroyAllWindows()
    if not collect:
        return last_frame, None
    if len(pts) < need // 2:
        raise RuntimeError("有效帧不足：手距相机 >0.4m、光照充足后重试")
    return last_frame, np.array(pts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["realsense", "webcam"], default="realsense")
    parser.add_argument("--no-arm", action="store_true",
                        help="跳过机械臂连接（仅相机轴向标定，不回就绪位）")
    args = parser.parse_args()

    from nerohandtwin.geometry.calib_solver import solve_axes_rotation
    from nerohandtwin.perception.camera_realsense import MockCamera, RealSenseCamera
    from nerohandtwin.perception.hand_tracker import HandTracker

    model = str(ROOT / "models" / "gesture_recognizer.task")
    cam = RealSenseCamera() if args.camera == "realsense" else \
        MockCamera(source="webcam", index_or_path=0)
    tracker = HandTracker(model, depth_mode="depth" if args.camera == "realsense" else "estimate")

    # ---------- 前置：机械臂连接并回到初始姿态 ----------
    arm = None
    if not args.no_arm:
        from nerohandtwin.control.arm_interface import create_arm
        from nerohandtwin.control.safety import SafeStartup
        import yaml as _yaml

        cfg_arm = _yaml.safe_load((ROOT / "configs" / "arm.yaml").read_text(encoding="utf-8"))["arm"]
        try:
            print("[arm] 连接机械臂...")
            arm = create_arm(cfg_arm)
            arm.connect()
            print("[arm] 转移到就绪位（初始姿态）...")
            startup = SafeStartup(arm, cfg_arm.get("ready_joints"))
            if not startup.is_ready:
                startup.move_to_ready()
            # 仿真后端需要 pump 推进物理到位；真机固件已直接执行
            if hasattr(arm, "pump"):
                import time as _time

                for _ in range(120):
                    arm.pump()
                    _time.sleep(0.03)
            fp = arm.get_state().flange_pose
            print(f"[arm] 就绪位到位 ✓ TCP=[{fp[0, 3]:+.2f} {fp[1, 3]:+.2f} {fp[2, 3]:+.2f}]"
                  f"（标定期间保持使能）")
        except Exception as exc:  # noqa: BLE001
            print(f"[arm] 机械臂连接/就绪失败: {exc}")
            print("      如需纯相机标定请加 --no-arm；检查 CAN 接线后重试")
            tracker.close()
            cam.close()
            return 1

    print("=" * 62)
    print("轴向标定：窗口内按【空格】采样，共 4 步（约 1 分钟）；q 中止")
    print("=" * 62)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    deltas = []
    steps = [
        ("[1/4] 手停在机械臂前方30~40cm、与夹爪同高，按空格采样", 45, True),
        ("[2/4] 手向【远离机械臂】(朝自己)移约15cm停住，按空格采样", 30, True),
        ("[3/4] 手向【机械臂的左侧】(=你的右手边)移约15cm停住，按空格采样", 30, True),
        ("[4/4] 手【竖直向上】移约15cm停住，按空格采样", 30, True),
    ]
    try:
        origin = None
        for i, (instr, need, _) in enumerate(steps):
            _, samples = _loop_until_space(cam, tracker, instr, need, collect=True)
            pos = np.median(samples, axis=0)
            if i == 0:
                origin = pos
                print(f"[1/4] 原点(相机系): {np.round(origin, 3)}")
            else:
                d = pos - origin
                deltas.append(d)
                print(f"[{i+1}/4] 位移(相机系): {np.round(d, 3)}")
    except KeyboardInterrupt:
        print("已中止")
        cv2.destroyAllWindows()
        if arm is not None:
            print("[arm] 中止: 机械臂返回就绪位并保持使能")
            arm.close(go_home=True)
        return 1
    except RuntimeError as exc:
        print(f"采样失败: {exc}")
        cv2.destroyAllWindows()
        if arm is not None:
            print("[arm] 失败退出: 机械臂返回就绪位并保持使能")
            arm.close(go_home=True)
        return 1
    finally:
        tracker.close()
        cam.close()

    try:
        rot, resid = solve_axes_rotation(deltas)
    except ValueError as exc:
        print(f"求解失败: {exc}（重新运行，移动方向幅度大一些）")
        if arm is not None:
            print("[arm] 机械臂返回就绪位并保持使能")
            arm.close(go_home=True)
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
    cfg["extrinsics"]["calibrated"] = "axes"
    cfg["extrinsics"]["note"] = "轴向标定（免标定板）- 相对跟随方向已校准"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已写入 {cfg_path}")
    if arm is not None:
        print("[arm] 标定完成: 机械臂返回就绪位并保持使能")
        arm.close(go_home=True)
    print("现在运行 python scripts/run_real.py 即可正常跟随")
    return 0


if __name__ == "__main__":
    sys.exit(main())
