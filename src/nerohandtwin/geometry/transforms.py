"""齐次变换与旋转表示转换工具。

约定：
- 位姿统一用 4x4 齐次矩阵（米制）。
- 欧拉角约定与 pyAgxArm / NERO 驱动一致：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)，
  即 ZYX 内旋顺序（rpy）。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "rpy_to_matrix",
    "matrix_to_rpy",
    "make_pose",
    "pose_position",
    "pose_rotation",
    "transform_point",
    "invert_transform",
]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角 -> 3x3 旋转矩阵。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """3x3 旋转矩阵 -> (roll, pitch, yaw)，ZYX 约定；奇异时 pitch 取 pi/2。"""
    rot = np.asarray(rot, dtype=float)
    sp = -rot[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        # 万向锁：roll 与 yaw 耦合，约定 yaw=0
        roll = math.atan2(-rot[1, 2], rot[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def make_pose(position, rotation: np.ndarray) -> np.ndarray:
    """位置(3,) + 旋转(3,3) -> 4x4 齐次位姿。"""
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return pose


def pose_position(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=float)[:3, 3]


def pose_rotation(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=float)[:3, :3]


def transform_point(transform: np.ndarray, point) -> np.ndarray:
    """用 4x4 变换把点(3,)从一个坐标系转到另一坐标系。"""
    return (np.asarray(transform, dtype=float) @ np.append(np.asarray(point, dtype=float), 1.0))[:3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """刚体变换求逆（比通用矩阵求逆数值更稳）。"""
    t = np.asarray(transform, dtype=float)
    inv = np.eye(4)
    inv[:3, :3] = t[:3, :3].T
    inv[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return inv
