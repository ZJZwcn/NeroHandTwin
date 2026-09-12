"""几何模块单元测试：变换、Umeyama 标定求解、工作空间映射."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerohandtwin.geometry.transforms import (
    invert_transform,
    make_pose,
    matrix_to_rpy,
    rpy_to_matrix,
    transform_point,
)
from nerohandtwin.geometry.calib_solver import solve_rigid_transform, calibration_residuals
from nerohandtwin.geometry.workspace import WorkspaceMapper


def test_rpy_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r, p, y = rng.uniform(-2.5, 2.5, 3)
        p = np.clip(p, -1.4, 1.4)  # 避开万向锁
        R = rpy_to_matrix(r, p, y)
        r2, p2, y2 = matrix_to_rpy(R)
        assert np.allclose(rpy_to_matrix(r2, p2, y2), R, atol=1e-9)


def test_transform_point_and_inverse():
    T = make_pose([1, 2, 3], rpy_to_matrix(0.1, 0.2, 0.3))
    p = np.array([0.5, -0.2, 0.1])
    mapped = transform_point(T, p)
    back = transform_point(invert_transform(T), mapped)
    assert np.allclose(back, p, atol=1e-12)


def test_umeyama_recovers_rigid_transform():
    """随机刚体变换 + 噪声点集 -> 求解误差远小于噪声量级。"""
    rng = np.random.default_rng(42)
    T_true = make_pose([0.8, -0.3, 0.5], rpy_to_matrix(0.4, -0.3, 1.1))
    pts = rng.uniform(-0.3, 0.3, (30, 3))
    dst = pts @ T_true[:3, :3].T + T_true[:3, 3]
    dst_noisy = dst + rng.normal(0, 0.002, dst.shape)  # 2mm 噪声

    result = solve_rigid_transform(pts, dst_noisy)
    err_R = np.linalg.norm(result.rotation - T_true[:3, :3])
    err_t = np.linalg.norm(result.translation - T_true[:3, 3])
    assert err_R < 0.02
    assert err_t < 0.005
    # 残差 RMSE 应与噪声量级一致：3 轴独立噪声 -> RMSE ≈ sqrt(3)*sigma ≈ 3.5mm
    assert result.rmse < 0.005


def test_umeyama_requires_min_points():
    with pytest.raises(ValueError):
        solve_rigid_transform(np.zeros((2, 3)), np.zeros((2, 3)))


def test_workspace_absolute_mapping_and_clamp():
    ws = WorkspaceMapper(
        mode="absolute",
        box_min=(-0.2, -0.2, 0.2), box_max=(0.2, 0.2, 0.4),
        cam_min=(-0.4, -0.4, 0.4), cam_max=(0.4, 0.4, 0.8),
        shell_r_min=0.0, shell_z_min=0.2,  # 关闭低区壳层推离，纯盒裁剪语义
    )
    # 中心对中心
    out = ws.map(np.array([0.0, 0.0, 0.6]))
    assert np.allclose(out, [0, 0, 0.3], atol=1e-9)
    # 越界裁剪
    out = ws.map(np.array([10.0, -10.0, 10.0]))
    assert np.allclose(out, [0.2, -0.2, 0.4])


def test_workspace_shell_low_zone():
    """可达壳层：低矮区（z<split）目标被推离基座轴、托离桌面。"""
    ws = WorkspaceMapper(mode="absolute",
                         box_min=(-1, -1, -1), box_max=(1, 1, 1),
                         cam_min=(-1, -1, -1), cam_max=(1, 1, 1),
                         shell_r_min=0.22, shell_z_min=0.25, shell_z_split=0.35)
    # 基座轴附近低点 -> 推到半径 0.22、z 托到 0.25
    out = ws.map(np.array([0.0, 0.0, 0.10]), arm_position_base=np.zeros(3))
    assert np.hypot(out[0], out[1]) == pytest.approx(0.22, abs=1e-6)
    assert out[2] == pytest.approx(0.25)
    # 高处不受半径约束（如就绪位 r=0.09, z=0.47）
    out2 = ws.map(np.array([0.09, 0.0, 0.47]), arm_position_base=np.zeros(3))
    assert np.allclose(out2, [0.09, 0.0, 0.47], atol=1e-6)


def test_workspace_relative_mapping():
    ws = WorkspaceMapper(mode="relative", scale=0.5,
                         box_min=(-1, -1, -1), box_max=(1, 1, 1),
                         shell_r_min=0.0, shell_z_min=-1.0)  # 关壳层，纯相对语义
    ws.reset_relative_origin(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.4]))
    out = ws.map(np.array([0.1, 0.0, -0.1]))
    assert np.allclose(out, [0.3 + 0.05, 0.0, 0.4 - 0.05], atol=1e-9)
    # 未设零点：首次 map 自动以当前手/臂位置为原点 -> delta=0，目标=当前臂位
    ws2 = WorkspaceMapper(mode="relative", scale=1.0,
                          box_min=(-1, -1, -1), box_max=(1, 1, 1),
                          shell_r_min=0.0, shell_z_min=-1.0)
    out2 = ws2.map(np.array([0.05, 0, 0]), arm_position_base=np.array([0.2, 0, 0]))
    assert np.allclose(out2, [0.2, 0, 0], atol=1e-9)
    # 此后手部移动量按增益叠加到臂位上
    out3 = ws2.map(np.array([0.05 + 0.04, 0, 0]))
    assert np.allclose(out3, [0.2 + 0.04, 0, 0], atol=1e-9)
