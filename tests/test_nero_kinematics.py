"""NERO 本体运动学单元测试。

已用 pyAgxArm SDK fk() 与官方 MuJoCo 模型核对的基准值（零位/就绪位）
作为回归锚点；再验证：q3=q5=0 时的水平不变量、2R 逆解往返一致性。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerohandtwin.geometry.nero_kinematics import (
    D1, L1, L2,
    approach_angle, flange_inplane, fk_flange, opening_tilt,
    solve_level_pose,
)

Q_ZERO = np.zeros(7)
Q_READY = np.array([0.0, np.radians(-35), 0.0, np.radians(115), 0.0, 0.0, np.radians(15)])


def test_fk_zero_pose_matches_sdk():
    """零位法兰位姿 = SDK/MuJoCo 核对值：pos(0,-0.0235,0.718)，开口轴 -y。"""
    p, R = fk_flange(Q_ZERO)
    assert p == pytest.approx([0.0, -0.0235, 0.71801], abs=1e-4)
    assert R[:, 2] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)   # 开口轴
    assert R[:, 0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)    # 进给轴（竖直向上）


def test_fk_ready_pose_matches_sdk():
    """就绪位法兰位姿 = SDK 核对值。"""
    p, R = fk_flange(Q_READY)
    assert p == pytest.approx([-0.0881, -0.0235, 0.4388], abs=1e-3)
    assert R[:, 2] == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
    assert R[2, 2] == pytest.approx(0.0, abs=1e-9)  # 开口水平


def test_level_invariants():
    """q3=q5=q6=0 时：任意 q2/q4/q7 下开口恒水平（结构不变量）。"""
    rng = np.random.default_rng(7)
    for _ in range(50):
        q = Q_READY.copy()
        q[1], q[3], q[6] = rng.uniform(-1.6, 0.3), rng.uniform(-0.9, 2.0), rng.uniform(-1.4, 1.4)
        q[2] = q[4] = q[5] = 0.0
        assert opening_tilt(q) == pytest.approx(0.0, abs=1e-9)


def test_approach_angle_formula():
    """进给面内角 φ = π/2 + q2 + q4 + q7（q3=q5=0）。"""
    rng = np.random.default_rng(11)
    for _ in range(50):
        q = rng.uniform(-0.5, 0.5, 7)
        q[2] = q[4] = q[5] = 0.0
        q[1] = np.clip(q[1], -1.6, 0.3)
        expect = np.pi / 2 + q[1] + q[3] + q[6]
        assert approach_angle(q) == pytest.approx(float(expect), abs=1e-9)


def test_solve_level_pose_roundtrip():
    """臂展内 (s,z) 逆解往返：位置 <5mm；q7 未越限时进给水平 |φ|<1°。"""
    for s_t in np.linspace(0.24, 0.40, 5):
        for z_t in np.linspace(0.38, 0.52, 5):
            ik = solve_level_pose(s_t, z_t)
            q = np.array([0.0, ik.q2, 0.0, ik.q4, 0.0, 0.0, ik.q7])
            s, z = flange_inplane(q)
            assert (s, z) == pytest.approx((s_t, z_t), abs=5e-3), (s_t, z_t, ik.notes)
            if not ik.notes:  # 关节全部在限位内 -> 水平朝向必须达成
                assert abs(approach_angle(q)) < np.radians(1.0)
                assert ik.ok


def test_solve_level_pose_radius_clamp():
    """超伸展半径的目标沿射线收拢到可达域。"""
    ik = solve_level_pose(0.60, 0.60)  # r≈0.65 > L1+L2
    q = np.array([0.0, ik.q2, 0.0, ik.q4, 0.0, 0.0, ik.q7])
    s, z = flange_inplane(q)
    r = np.hypot(s, z - D1)
    assert r <= L1 + L2 - 1e-6
    assert any("半径" in n for n in ik.notes)


def test_elbow_forward_branch():
    """肘前支：肩-肘方向在竖直面前向半平面（q2 < 0 前伸区），肘位于肩前上方。"""
    ik = solve_level_pose(0.34, 0.47)
    assert ik.q2 < 0
    a1 = np.pi / 2 + ik.q2          # 上臂面内角
    e_s, e_z = L1 * np.cos(a1), D1 + L1 * np.sin(a1)
    assert e_s > 0.15 and e_z > D1  # 肘在肩的前上方
    # 法兰在肘基础上继续前伸
    s, z = flange_inplane(np.array([0.0, ik.q2, 0.0, ik.q4, 0.0, 0.0, ik.q7]))
    assert s > e_s - 1e-6
    assert z > e_z - 0.27 - 1e-6
