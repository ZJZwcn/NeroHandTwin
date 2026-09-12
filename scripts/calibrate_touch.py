"""对准标定（推荐）：机械臂走到已知位姿，手掌贴近夹爪尖 → 求完整外参。

全程实时预览：相机画面 + 手部骨架 + 状态提示，任何阶段（含机械臂移动中）
窗口都持续刷新。流程：
1. 预览确认门：确认相机画面正常后按空格，机械臂才开始移动；
2. 机械臂依次走到 6 个工作空间位姿（均已知基座系 TCP 坐标，fk 实测）；
3. 每个位姿：用户将手掌中心贴近夹爪尖（4~6cm），空格采样手掌 3D；
4. 收集 (手掌相机系坐标, 夹爪基座系坐标) 对，Umeyama 解出完整
   T_base<-camera（旋转+平移），RMSE<35mm 才写入 configs/calibration.yaml。

用法：
    python scripts/calibrate_touch.py            # 默认 RealSense
    python scripts/calibrate_touch.py --camera webcam
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

WIN = "calibrate_touch - 空格采样 / q 中止"


def _preview(cam, tracker, instruction: str, progress: tuple[int, int] | None = None,
             status_line: str = ""):
    """渲染一帧预览（相机无帧时显示占用警告），返回 (obs, key)。

    所有阶段（等待到位/静止校验/采样）都通过它刷新窗口。
    """
    frame = cam.read()
    cam._last_shape = frame.color.shape[:2] if frame is not None else (480, 640)
    if frame is None:
        obs = None
        img = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(img, "NO CAMERA FRAMES", (80, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.putText(img, "check: another app using the camera?", (55, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    else:
        obs = tracker.process(frame)
        img = frame.color.copy()
        if obs.present and obs.landmarks_px is not None:
            for p in obs.landmarks_px.astype(int):
                cv2.circle(img, tuple(p), 3, (90, 180, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(obs.landmarks_px.astype(int)[9]), 6,
                       (60, 60, 255), 2, cv2.LINE_AA)  # 高亮掌心

    h, w = img.shape[:2]
    # 画面中心十字参考线（构图参考；掌心仍须对准夹爪尖——
    # 标定要求掌心与夹爪尖是同一物理点，中心线只用于辅助取景）
    cx_img, cy_img = w // 2, h // 2
    cv2.line(img, (cx_img - 14, cy_img), (cx_img + 14, cy_img), (0, 255, 255), 1)
    cv2.line(img, (cx_img, cy_img - 14), (cx_img, cy_img + 14), (0, 255, 255), 1)
    cv2.circle(img, (cx_img, cy_img), 4, (0, 255, 255), 1)
    cv2.rectangle(img, (0, 0), (w, 52), (30, 30, 30), -1)
    cv2.putText(img, instruction, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (120, 255, 160), 1, cv2.LINE_AA)
    if status_line:
        cv2.putText(img, status_line, (8, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
    if progress:
        c, n = progress
        cv2.rectangle(img, (0, h - 12), (int(w * min(c / n, 1.0)), h), (80, 220, 90), -1)
    cv2.imshow(WIN, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        raise KeyboardInterrupt
    return obs, key


def _palm_valid(obs, cam) -> tuple[bool, str]:
    """采样有效性门禁：手掌必须在画面中央区域、深度在可靠区间。

    真机教训：相机距工作区太近时，贴夹爪的手只在画面角落露出边缘
    （landmark 归一化坐标 0.004~0.009），深度读数是背景固定值——
    6 个样本完全相同、RMSE 130mm。必须在源头拒绝。
    """
    if obs is None or not obs.present:
        return False, "未检测到手"
    if obs.landmarks_px is None:
        return False, "无关键点"
    h, w = (getattr(cam, "_last_shape", None) or (480, 640))
    u, v = obs.landmarks_px[9]  # 掌心(中指 MCP)
    if not (0.08 * w < u < 0.92 * w and 0.08 * h < v < 0.92 * h):
        return False, "手掌出框/在画面边缘——请调整相机位置覆盖工作区"
    if obs.palm3d is None:
        return False, "深度无效"
    z = float(obs.palm3d[2])
    if z < 0.25:
        return False, f"离相机太近({z*1000:.0f}mm)——相机应远离工作区 0.6~1.0m"
    if z > 1.2:
        return False, f"离相机太远({z*1000:.0f}mm)"
    return True, "OK"


def _sample_palm(cam, tracker, instruction: str, need: int = 30,
                 max_seconds: float = 12.0) -> np.ndarray:
    """空格触发后采集 need 帧【有效】掌心 3D，返回中位数位置。

    无效帧（出框/太近/无手）不计数并在 HUD 实时显示原因。
    """
    pts: list[np.ndarray] = []
    collecting = False
    reason = ""
    t_start: float | None = None
    while True:
        obs, key = _preview(cam, tracker, instruction,
                            progress=(len(pts), need) if collecting else None,
                            status_line=reason)
        cam._last_shape = getattr(cam, "_last_shape", None)
        if not collecting and key == ord(" "):
            collecting = True
            pts = []
            t_start = time.monotonic()
        if collecting:
            ok, reason = _palm_valid(obs, cam)
            if ok:
                pts.append(obs.palm3d)
                if len(pts) >= need:
                    break
            if t_start is not None and time.monotonic() - t_start > max_seconds and len(pts) == 0:
                raise RuntimeError(f"采集超时：{reason or '始终无有效手掌'}"
                                   "（请调整相机位置覆盖工作区后重试）")
    if len(pts) < need // 2:
        raise RuntimeError("有效帧不足：手距相机 >0.35m、光照充足后重试")
    return np.median(np.array(pts), axis=0)


def _wait_joints_arrived(arm, q_target: np.ndarray, cam, tracker, label: str,
                         timeout: float = 40.0, tol: float = 0.04) -> bool:
    """等待真机关节到达目标（move_j 流式重发，仿真恒 True），全程渲染预览。

    用关节收敛判据（与 close(go_home) 同源，真机实测可靠），而非
    move_p 的姿态约束到位——标定只需要 TCP 位置，姿态无关。
    """
    if not hasattr(arm, "_robot"):
        return True
    resent = False
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        q_now = np.asarray(arm.get_state().joint_angles, dtype=float)
        err = float(np.max(np.abs(q_now - q_target)))
        if err < tol:
            return True
        if not resent and time.monotonic() - t0 > timeout * 0.5:
            arm.move_joints(q_target.tolist())  # 半程重发一次
            resent = True
        _preview(cam, tracker, f"{label} 关节空间移动中...",
                 status_line=f"最大关节偏差={err:.3f}rad")
        time.sleep(0.15)
    return False


def _settle_check(arm, cam, tracker, attempts: int = 5, samples: int = 3,
                  interval: float = 0.3, tol: float = 0.003) -> bool:
    """静止校验：连续 samples 次 TCP 位移 < tol，最多 attempts 轮重试。

    容差 3mm 对标定无影响——采样记录的是 fk 实际 TCP（臂微动时读到的
    也是真实位置），校验只为避免用户贴手时臂还在大步移动。
    """
    if not hasattr(arm, "_robot"):
        return True
    for attempt in range(attempts):
        last = arm.get_state().flange_pose[:3, 3].copy()
        stable = True
        for _ in range(samples):
            time.sleep(interval)
            _preview(cam, tracker, "等待末端静止...")
            cur = arm.get_state().flange_pose[:3, 3]
            if float(np.linalg.norm(cur - last)) > tol:
                stable = False
                break
            last = cur.copy()
        if stable:
            return True
    return False


def _flange_rpy(fp: np.ndarray) -> tuple[float, float, float]:
    """4x4 法兰位姿 -> (roll, pitch, yaw)，与 pyAgxArm move_p 约定一致。"""
    import math

    rot = fp[:3, :3]
    sp = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        roll = math.atan2(-rot[1, 2], rot[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["realsense", "webcam"], default="realsense")
    args = parser.parse_args()

    from nerohandtwin.geometry.calib_solver import calibration_residuals, solve_rigid_transform
    from nerohandtwin.perception.camera_realsense import MockCamera, RealSenseCamera
    from nerohandtwin.perception.hand_tracker import HandTracker

    model = str(ROOT / "models" / "gesture_recognizer.task")
    cam = RealSenseCamera() if args.camera == "realsense" else \
        MockCamera(source="webcam", index_or_path=0)
    tracker = HandTracker(model, depth_mode="depth" if args.camera == "realsense" else "estimate")

    # ---------- 机械臂连接 + 就绪位（不移动，先过预览确认门） ----------
    import yaml as _yaml
    from nerohandtwin.control.arm_interface import create_arm
    from nerohandtwin.control.safety import SafeStartup

    cfg_arm = _yaml.safe_load((ROOT / "configs" / "arm.yaml").read_text(encoding="utf-8"))["arm"]
    try:
        print("[arm] 连接机械臂并转移到就绪位...")
        arm = create_arm(cfg_arm)
        arm.connect()
        startup = SafeStartup(arm, cfg_arm.get("ready_joints"))
        if not startup.is_ready:
            startup.move_to_ready()
        if arm.is_simulation:
            for _ in range(120):
                arm.pump()
                time.sleep(0.03)
        print("[arm] 就绪位到位 ✓")
    except Exception as exc:  # noqa: BLE001
        print(f"[arm] 机械臂连接失败: {exc}")
        tracker.close()
        cam.close()
        return 1

    fp = arm.get_state().flange_pose
    tcp_ready = fp[:3, 3].copy()
    print(f"[arm] 就绪 TCP={np.round(tcp_ready, 3)}")

    # ---------- 预览确认门：相机画面正常后再让机械臂移动 ----------
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("=" * 62)
    print("相机摆放要求（重要）：")
    print("  距机械臂工作区 0.6~1.0m、正对工作区，保证夹爪将要到达的")
    print("  整片区域都在画面中央（贴手时手掌不应出现在画面边缘）。")
    print("=" * 62)
    print("请确认窗口中的相机画面与手部骨架正常；按空格开始标定（机械臂将开始移动）")
    try:
        while True:
            _obs_preview, key = _preview(
                cam, tracker,
                "预览检查: 确认画面/骨架正常（黄色十字为构图参考），按空格开始 / q 退出")
            if _obs_preview is not None and _obs_preview.present:
                ok, reason = _palm_valid(_obs_preview, cam)
                if not ok and "出框" in reason or "太近" in reason or "太远" in reason:
                    print(f"[视野提示] {reason}")
            if key == ord(" "):
                break
    except KeyboardInterrupt:
        print("已中止")
        cv2.destroyAllWindows()
        arm.close(go_home=True)
        return 1

    # ---------- 6 点对准采样（前方工作区，全部 x>0） ----------
    # 关节空间走点：各标定位姿的关节角由 MuJoCo 仿真 IK 预先解出（姿态
    # 自由、必定可达，与 close(go_home) 的 move_j 机制同源——真机实测
    # move_p 带姿态约束对部分目标会拒绝执行，move_j 每次都成功）。
    # 仿真验证：顺序执行 6 点 IK 全部 0 误差、点云秩 3、最长单段 300mm。
    offsets = [
        (0.25, 0.00, 0.00),
        (0.35, 0.12, 0.05),
        (0.30, -0.15, 0.05),
        (0.35, 0.10, -0.08),
        (0.15, -0.10, 0.02),
        (0.28, 0.00, -0.10),
    ]
    q_ready = np.asarray(cfg_arm.get("ready_joints"), dtype=float)
    joint_targets: list[np.ndarray] = []
    try:
        print("[sim] 用 MuJoCo 仿真 IK 预解 6 个标定位姿的关节角 ...")
        from nerohandtwin.control.mujoco_arm import MujocoArmController

        sim = MujocoArmController(
            model_path=str(ROOT / "third_party/agx_arm_sim/mujoco/agilex_arm/agilex_nero/scene.xml"),
            render_width=0, render_height=0)
        sim.connect()
        sim.move_joints(q_ready)  # 预热：后续 solve_ik 从 ready 位形 warm-start
        sim_arm_ready = sim.forward_kinematics(q_ready)[:3, 3]
        for i, off in enumerate(offsets):
            target = tcp_ready + np.asarray(off, dtype=float)
            q = sim.solve_ik(target, q_init=q_ready)
            p = sim.forward_kinematics(q)[:3, 3]
            err = float(np.linalg.norm(p - target))
            if err > 0.02:
                raise RuntimeError(f"点 {i+1} 仿真 IK 误差 {err*1000:.1f}mm 过大，中止")
            joint_targets.append(q)
            print(f"[sim] 点 {i+1}: TCP={np.round(target,3)} 关节解误差={err*1000:.1f}mm ✓")
        sim.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[sim] 关节解算失败: {exc}")
        cv2.destroyAllWindows()
        arm.close(go_home=True)
        return 1

    pts_cam: list[np.ndarray] = []
    pts_base: list[np.ndarray] = []
    try:
        for i, q_target in enumerate(joint_targets):
            arm.move_joints(q_target.tolist())
            arrived = _wait_joints_arrived(arm, q_target, cam, tracker,
                                           f"[{i+1}/{len(joint_targets)}]")
            if not arrived:
                print(f"[{i+1}] 首次等待未确认到位，重发指令再等 ...")
                arm.move_joints(q_target.tolist())
                arrived = _wait_joints_arrived(arm, q_target, cam, tracker,
                                               f"[{i+1}/{len(joint_targets)}]")
            if not arm.is_simulation:
                if not arrived:
                    raise RuntimeError(
                        f"点 {i+1} 关节未收敛（偏差 >0.04rad），中止标定以避免坏数据")
                if not _settle_check(arm, cam, tracker):
                    raise RuntimeError(f"点 {i+1} 采样时臂仍在运动，中止标定")
            else:
                for _ in range(60):
                    arm.pump()
                    time.sleep(0.03)
            tcp_actual = arm.get_state().flange_pose[:3, 3].copy()
            print(f"[{i+1}/{len(joint_targets)}] 到位 ✓ TCP={np.round(tcp_actual, 3)}")
            palm = _sample_palm(
                cam, tracker,
                f"[{i+1}/{len(joint_targets)}] 用掌心【顶住】夹爪尖端（实体接触不悬空），"
                f"保持 2 秒后按空格采样")
            pts_cam.append(palm)
            pts_base.append(tcp_actual)
            print(f"[{i+1}/{len(joint_targets)}] 采样完成: palm_cam={np.round(palm,3)} "
                  f"tcp_base={np.round(tcp_actual,3)}")
    except KeyboardInterrupt:
        print("已中止")
        cv2.destroyAllWindows()
        arm.close(go_home=True)
        return 1
    except RuntimeError as exc:
        print(f"采样失败: {exc}")
        cv2.destroyAllWindows()
        arm.close(go_home=True)
        return 1
    finally:
        tracker.close()
        cam.close()

    # ---------- 求解完整外参 ----------
    result = solve_rigid_transform(pts_cam, pts_base)
    residuals = calibration_residuals(pts_cam, pts_base, result.transform)
    print(f"标定求解: RMSE={result.rmse*1000:.1f}mm 逐点残差(mm)="
          f"{[round(r*1000,1) for r in residuals]}")
    if result.rmse > 0.035:
        # 硬门禁：坏外参绝不能写入配置（曾把 147mm 垃圾数据写入导致跟随反向）
        print("✗ RMSE > 35mm，拒绝写入配置（保留原外参）。")
        print("  请重跑本脚本：每一步确保手掌中心对准夹爪尖 4~6cm、保持不动再按空格。")
        print("[arm] 机械臂返回就绪位并保持使能")
        arm.close(go_home=True)
        return 1

    cfg_path = ROOT / "configs" / "calibration.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    t = result.transform
    cfg["extrinsics"]["T_base_camera"] = [
        [round(float(t[0, 0]), 5), round(float(t[0, 1]), 5), round(float(t[0, 2]), 5),
         round(float(t[0, 3]), 5)],
        [round(float(t[1, 0]), 5), round(float(t[1, 1]), 5), round(float(t[1, 2]), 5),
         round(float(t[1, 3]), 5)],
        [round(float(t[2, 0]), 5), round(float(t[2, 1]), 5), round(float(t[2, 2]), 5),
         round(float(t[2, 3]), 5)],
        [0.0, 0.0, 0.0, 1.0],
    ]
    cfg["extrinsics"]["calibrated"] = "touch"
    cfg["extrinsics"]["note"] = "对准标定（手掌贴夹爪尖）- 完整外参含平移"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已写入 {cfg_path} ✓")

    print("[arm] 标定完成: 机械臂返回就绪位并保持使能")
    arm.close(go_home=True)
    print("现在运行 python scripts/run_real.py，跟随方向即为正确的手↔臂对应")
    return 0


if __name__ == "__main__":
    sys.exit(main())
