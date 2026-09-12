"""Umeyama/Kabsch 刚体变换求解：eye-to-hand 外参标定核心。

给定 N 组对应点对（相机系点 P_cam[i] 与基座系点 P_base[i]），
求最小二乘最优刚体变换 T_base<-camera，使
    P_base[i] ≈ R @ P_cam[i] + t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CalibrationResult", "solve_rigid_transform", "calibration_residuals"]


@dataclass
class CalibrationResult:
    """标定结果。

    Attributes:
        transform: 4x4 齐次变换 T_base<-camera。
        rotation: 3x3 旋转矩阵。
        translation: 平移向量 (3,)，米。
        rmse: 均方根残差（米），逐点欧氏距离。
        num_points: 参与求解的点对数。
    """

    transform: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    num_points: int


def solve_rigid_transform(points_camera, points_base) -> CalibrationResult:
    """Umeyama（不含尺度）刚体对齐求解。

    Args:
        points_camera: (N,3) 相机坐标系点集，米。
        points_base: (N,3) 机械臂基座坐标系对应点集，米。

    Raises:
        ValueError: 点数 < 3 或形状不符（3 点共面情形仅对平面标定板场景受限，
            建议采集 5 组以上非共线位姿）。
    """
    src = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    dst = np.asarray(points_base, dtype=float).reshape(-1, 3)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("点对数量不一致")
    if src.shape[0] < 3:
        raise ValueError("至少需要 3 组点对才能求解刚体变换")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    # 协方差 H = sum(dst_c^T * src_c)；SVD 分解求旋转
    h = dst_c.T @ src_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    rot = u @ correction @ vt

    trans = mu_d - rot @ mu_s

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    residual = src @ rot.T + trans - dst
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    return CalibrationResult(
        transform=transform,
        rotation=rot,
        translation=trans,
        rmse=rmse,
        num_points=int(src.shape[0]),
    )


def calibration_residuals(points_camera, points_base, transform: np.ndarray) -> np.ndarray:
    """返回每个点对的欧氏残差 (N,)，用于逐点排查坏数据。"""
    src = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    dst = np.asarray(points_base, dtype=float).reshape(-1, 3)
    mapped = src @ transform[:3, :3].T + transform[:3, 3]
    return np.linalg.norm(mapped - dst, axis=1)


def solve_rotation_from_probes(deltas_camera, dirs_base) -> tuple[np.ndarray, float]:
    """由 N 组"相机系位移 ↔ 基座系期望方向"对应求解旋转（快速方向标定核心）。

    用户交互标定原理：让用户沿【自己视角】的三个方向各移动一次手
    （如：向你的右边、向上、朝机械臂方向），记录相机系位移 d_i。
    期望的基座系方向 u_i 由用户站位直接算出（零歧义，无需理解
    机械臂坐标系）。R @ d_i = u_i → R = D_u @ inv(D)。

    Args:
        deltas_camera: 3 组相机系位移 [(d1), (d2), (d3)]。
        dirs_base: 对应的 3 组基座系期望方向 [(u1), (u2), (u3)]。

    Returns:
        (R 3x3, 残差) —— 残差为 R@D_cam 与 D_base 的最大偏差。
    """
    d = np.asarray(deltas_camera, dtype=float).reshape(3, 3).T  # 列 = d_i
    u = np.asarray(dirs_base, dtype=float).reshape(3, 3).T      # 列 = u_i
    if np.linalg.matrix_rank(d, tol=1e-3) < 3:
        raise ValueError("三次移动方向区分度不足（接近共线/共面），请重新标定")
    rot = u @ np.linalg.inv(d)
    # SVD 投影到 SO(3)：消除测量噪声导致的非正交
    uu, _, vt = np.linalg.svd(rot)
    rot = uu @ vt
    if np.linalg.det(rot) < 0:
        uu[:, -1] *= -1.0
        rot = uu @ vt
    residual = float(np.max(np.abs(rot @ d - u)))
    return rot, residual


def solve_axes_rotation(deltas_camera) -> tuple[np.ndarray, float]:
    """由三次定向移动求旋转 R_base<-camera（基座轴版本，向后兼容入口）。"""
    return solve_rotation_from_probes(
        deltas_camera, np.eye(3))
