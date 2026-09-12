"""手眼标定工具（eye-to-hand）：采集 -> 求解 -> 验证。

流程（真机阶段）：
1. 在机械臂法兰/夹爪上固定 ArUco 标记板（标志板在基座系中的位姿 = FK 可算）；
2. 采集 N>=5 组：机械臂走不同位姿（非共线、覆盖视野），相机拍照；
3. 对每帧用 cv2.aruco 检测标记板位姿（相机系）；
4. solve_handeye.py 用 Umeyama 对齐相机系点集与基座系点集 -> T_base<-camera。

本脚本支持两种采集模式：
- interactive: 手动摆位（Web 上位机拖动或脚本来点），回车采集；
- scripted: 按给定位姿列表自动执行。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CALIB_DATA_DIR = ROOT / "data" / "calibration"


def detect_board_pose_camera(bgr: np.ndarray, board, camera_matrix, dist_coeffs):
    """检测标定板位姿（相机系 4x4）。返回 None 表示未检测到。"""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, board.dictionary)
    if ids is None or len(ids) < 1:
        return None
    ok, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix,
                                                 dist_coeffs, None, None)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = tvec.reshape(3)
    return pose


def interactive_collect(camera_matrix, dist_coeffs, board, arm=None) -> list[dict]:
    """手动摆位采集：终端回车触发拍照；q 结束。"""
    from nerohandtwin.perception.camera_realsense import RealSenseCamera

    cam = RealSenseCamera()
    samples: list[dict] = []
    print("摆好机械臂位姿后回车采集；输入 q 结束。")
    while True:
        cmd = input(f"[{len(samples)}] 回车=采集 / q=结束> ").strip().lower()
        if cmd == "q":
            break
        frame = cam.read()
        if frame is None:
            print("取帧失败")
            continue
        pose_cam = detect_board_pose_camera(frame.color, board, camera_matrix, dist_coeffs)
        if pose_cam is None:
            print("未检测到标定板，重试")
            continue
        if arm is None:
            flange = np.eye(4)
            print("（未连接机械臂，仅记录相机系位姿）")
            base_pt = np.zeros(3)
        else:
            fp = arm.get_state().flange_pose
            flange = fp
            base_pt = fp[:3, 3]  # 标定板原点即法兰原点的近似（贴装偏移可在后处理修正）
        samples.append(
            {"pose_camera": pose_cam.tolist(), "flange_pose": flange.tolist(),
             "board_offset_flange": [0, 0, 0], "point_base": base_pt.tolist()}
        )
        print(f"  已采集 {len(samples)} 组")
    cam.close()
    return samples


def solve(samples: list[dict], board_offset_flange=(0.0, 0.0, 0.0)):
    """由采集样本求解 T_base<-camera（Umeyama）。

    对应点选取：相机系取标定板原点；基座系取法兰原点 + 板贴装偏移。
    """
    from nerohandtwin.geometry.calib_solver import calibration_residuals, solve_rigid_transform
    from nerohandtwin.geometry.transforms import transform_point

    pts_cam, pts_base = [], []
    for s in samples:
        p_cam = np.asarray(s["pose_camera"])[:3, 3]
        p_base = np.asarray(s["flange_pose"])[:3, 3] + np.asarray(board_offset_flange)
        pts_cam.append(p_cam)
        pts_base.append(p_base)
    result = solve_rigid_transform(pts_cam, pts_base)
    residuals = calibration_residuals(pts_cam, pts_base, result.transform)
    return result, residuals


def main() -> int:
    import cv2  # noqa: F401 - 确认 aruco 可用

    CALIB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 默认内参：现场先用 rs-enumerate-devices 或 realsense-viewer 获取并填入
    camera_matrix = np.array([[615.0, 0, 319.5], [0, 615.0, 239.5], [0, 0, 1]])
    dist_coeffs = np.zeros(5)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.GridBoard_create(markersX=2, markersY=2, markerLength=0.04,
                                       markerSeparation=0.01, dictionary=aruco_dict)

    samples = interactive_collect(camera_matrix, dist_coeffs, board, arm=None)
    if len(samples) < 5:
        print(f"样本数 {len(samples)} < 5，不足以求解")
        return 1
    out = CALIB_DATA_DIR / f"calib_samples_{int(time.time())}.json"
    out.write_text(json.dumps(samples, indent=1), encoding="utf-8")
    print(f"样本已存 {out}")

    result, residuals = solve(samples)
    print(f"RMSE = {result.rmse*1000:.1f} mm；逐点残差(mm): "
          f"{[round(r*1000, 1) for r in residuals]}")
    print("T_base<-camera =")
    print(np.round(result.transform, 4))
    print("请把该矩阵与 calibrated: true 写回 configs/calibration.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
