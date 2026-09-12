"""GimbalFollowController（图像空间伺服跟随）单元测试。

用户定义的跟随逻辑：手偏画面哪边，臂就朝哪边转/动，使手保持画面中心；
夹爪始终指向手。无需任何外参标定。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerohandtwin.interaction.follow_controller import GimbalFollowController
from nerohandtwin.perception.hand_tracker import HandObservation

Q_READY = np.array([0.0, -0.6109, 0.0, 2.0071, 0.0, 0.0, 0.2618])
HW = (480, 640)  # 画面 480 高 640 宽


def _obs(u: float, v: float, z: float = 0.40) -> HandObservation:
    return HandObservation(
        present=True, timestamp_ms=100, gesture="Open_Palm", pinch=0.8,
        landmarks_px=np.array([[u, v]] * 21),
        palm3d=np.array([0.0, 0.0, z]),
    )


@pytest.fixture()
def gimbal():
    g = GimbalFollowController(Q_READY)
    g.reset(Q_READY)
    return g


def test_hand_centered_no_motion(gimbal):
    """手在画面中心（死区内）→ 关节不动。"""
    q = gimbal.update(_obs(320, 240, 0.40), HW, 0.033)
    assert q == pytest.approx(Q_READY, abs=1e-9)


def test_hand_right_yaws_base(gimbal):
    """手偏画面右 → J1 减小（默认 sign_yaw=-1：转向画面右侧）。"""
    q = gimbal.update(_obs(320 + 640 * 0.3, 240, 0.40), HW, 0.033)  # 偏右 30% 宽
    assert q[0] < Q_READY[0]
    assert q[3] == pytest.approx(Q_READY[3])  # 垂直无误差不动


def test_hand_above_pitches_up(gimbal):
    """手偏画面上 → J4 减小（正=前下，负=上）。"""
    q = gimbal.update(_obs(320, 240 - 480 * 0.3, 0.40), HW, 0.033)
    assert q[3] < Q_READY[3]
    assert q[0] == pytest.approx(Q_READY[0])


def test_hand_farther_extends(gimbal):
    """手比参考深度远 → J2 减小（默认 sign_depth=-1：向前伸出）。"""
    q = gimbal.update(_obs(320, 240, 0.55), HW, 0.033)
    assert q[1] < Q_READY[1]


def test_hand_closer_retracts(gimbal):
    """手比参考深度近 → J2 增大（向后收，保持安全距离）。"""
    q = gimbal.update(_obs(320, 240, 0.30), HW, 0.033)
    assert q[1] > Q_READY[1]


def test_deadzone_no_hunting(gimbal):
    """死区内的小误差不产生运动（防抖动）。"""
    q = gimbal.update(_obs(320 + 640 * 0.02, 240, 0.40), HW, 0.033)
    assert q == pytest.approx(Q_READY, abs=1e-12)


def test_hand_lost_holds(gimbal):
    """手丢失 → 保持当前目标不动。"""
    gimbal.update(_obs(320 + 640 * 0.3, 240, 0.40), HW, 0.033)
    q_moved = gimbal.update(_obs(320 + 640 * 0.3, 240, 0.40), HW, 0.033)
    lost_obs = HandObservation(present=False, landmarks_px=None, palm3d=None)
    q_hold = gimbal.update(lost_obs, HW, 0.033)
    assert np.allclose(q_moved, q_hold)


def test_joint_limits_clamped(gimbal):
    """极端误差下关节目标不越出安全范围。"""
    for _ in range(200):
        gimbal.update(_obs(639, 0, 0.90), HW, 0.033)
    q = gimbal.update(_obs(639, 0, 0.90), HW, 0.033)
    limits = GimbalFollowController._Q_LIMITS
    assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])


def test_joint_rate_limited(gimbal):
    """全关节限速：单帧变化不超过 max_joint_rate*dt。"""
    g = GimbalFollowController(Q_READY, {"max_joint_rate": 0.5})
    g.reset(Q_READY)
    q = g.update(_obs(639, 0, 0.90), HW, 0.033)
    assert float(np.max(np.abs(q - Q_READY))) <= 0.5 * 0.033 + 1e-9
