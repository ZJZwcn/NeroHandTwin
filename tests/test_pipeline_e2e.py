"""端到端无头冒烟 v3：合成手部观测 -> 交互状态机 -> 切换/跟随/动作 全链路。

用户交互方案 v3：上电默认动作模式 -> OK 切换到跟随（免标定自动锚定，
相对映射）-> OK 切回动作 -> 点赞/碰拳夹爪动作；拇指向下复位回初始姿态。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MUJOCO_MODEL = ROOT / "third_party/agx_arm_sim/mujoco/agilex_arm/agilex_nero/scene.xml"


class _FixedSource:
    """固定/受控观测源（代替相机+MediaPipe）。"""

    def __init__(self):
        self.obs = None
        self.last_obs = None
        self.last_frame = None

    def get(self, _now_ms):
        return self.obs


@pytest.fixture()
def app():
    """每测试新建 app（无跨测试手势边沿/锚定状态污染）。"""
    if not MUJOCO_MODEL.exists():
        pytest.skip("缺少 MuJoCo NERO 模型")
    from nerohandtwin.app import NeroGestureApp
    from nerohandtwin.control.mujoco_arm import MujocoArmController

    src = _FixedSource()
    # 显式用仿真臂，不随 configs/arm.yaml 的 backend 切换而变
    arm = MujocoArmController(model_path=str(MUJOCO_MODEL), render_width=0, render_height=0)
    app = NeroGestureApp(ROOT / "configs", obs_source=src, arm=arm, headless=True)
    # 恒等外参：合成观测直接用基座系坐标
    app.t_base_cam = np.eye(4)
    app.follow.pipeline.t_base_cam = np.eye(4)
    app.point.pipeline.t_base_cam = np.eye(4)
    app.start()
    yield app
    app.shutdown()


def _obs(t, gesture, palm, index=None, pinch=0.8, gap_m=None, landmarks_px=None):
    from nerohandtwin.perception.hand_tracker import HandObservation

    return HandObservation(
        present=gesture is not None,
        timestamp_ms=int(t * 1000),
        gesture=gesture or "None",
        gesture_score=0.9,
        pinch=pinch,
        landmarks_px=landmarks_px,
        palm3d=None if palm is None else np.asarray(palm, dtype=float),
        index3d=None if index is None else np.asarray(index, dtype=float),
        gap_m=gap_m,
    )


def feed(src, app, gesture, palm, base_ms, n=10, step_ms=33):
    """喂手势帧；手势间插 8 帧"手在但无手势"（>去抖窗口），重置边沿状态。"""
    for k in range(n):
        src.obs = _obs(k * 0.033, gesture, palm)
        app.run_once(base_ms + k * step_ms)
    for g in range(8):
        src.obs = _obs((n + g) * 0.033, "None", palm)
        app.run_once(base_ms + (n + g) * step_ms)


def settle(src, app, base_ms, palm, n=90):
    """喂无手势帧：让启动序列的关节滑行完成、末端静止（锚定前置）。"""
    for k in range(n):
        src.obs = _obs(k * 0.033, "None", palm)
        app.run_once(base_ms + k * 33)


def test_ok_toggle_follow_gimbal(app):
    """v4 跟随 = 协同多关节伺服：进入 FOLLOW 即锚定，J2/J4/J7 收敛到工作位。"""
    src: _FixedSource = app.obs_source
    palm = np.array([0.30, 0.0, 0.45])

    settle(src, app, 500, palm)                  # 启动序列滑行完成
    feed(src, app, "Okay", palm, 1000)
    assert app.interact.mode.value == "FOLLOW"
    assert app._follow_anchored                  # 云台模式即刻锚定（无需外参）

    # 手居锚定列、指距 120mm（> 满行程 → 夹爪全开）
    q0 = np.asarray(app.arm.get_state().joint_angles, dtype=float).copy()
    for k in range(300):                          # 10s：限速下收敛到工作位
        src.obs = _obs(k * 0.033, "Open_Palm", palm, gap_m=0.12,
                       landmarks_px=np.full((21, 2), [320.0, 240.0]))
        app.run_once(1500 + k * 33)
    q1 = np.asarray(app.arm.get_state().joint_angles, dtype=float)
    assert not np.allclose(q1, q0), "协同伺服应驱动 J2/J4/J7 到工作位"
    # 夹爪同步：指距 120mm > 满行程 100mm → 全开
    assert app._gripper_opening > 0.9


def test_ok_toggle_back_to_idle(app):
    """跟随中 OK 切回待机：臂保持，点赞/碰拳不再有动作事件。"""
    src: _FixedSource = app.obs_source
    palm = np.array([0.30, 0.0, 0.45])

    settle(src, app, 500, palm)
    feed(src, app, "Okay", palm, 10000)          # -> FOLLOW
    assert app.interact.mode.value == "FOLLOW"
    feed(src, app, "Okay", palm, 11000)          # -> IDLE
    assert app.interact.mode.value == "IDLE"
    # 点赞/碰拳在 v5 已移除动作编排：无事件、无 sequencer
    feed(src, app, "Thumb_Up", palm, 12000)      # 点赞 -> 复位请求（IDLE 保持）
    assert app.interact.mode.value == "IDLE"
    feed(src, app, "Closed_Fist", palm, 13000)
    assert app.interact.mode.value == "IDLE"


def test_reset_gesture_returns_home(app):
    """拇指向上（跟随中或待机中）-> 关节空间回初始姿态，停在待机。"""
    src: _FixedSource = app.obs_source
    palm = np.array([0.30, 0.0, 0.45])

    settle(src, app, 500, palm)
    feed(src, app, "Okay", palm, 40000)          # -> FOLLOW
    settle(src, app, 41000, palm)                # 锚定
    assert app._follow_anchored

    feed(src, app, "Thumb_Up", palm, 42000)      # 复位
    assert app.interact.mode.value == "IDLE"
    assert not app._follow_anchored
    # 回位完成：关节角接近就绪位（重力下垂容差 8°）
    for k in range(200):                          # 6.6s 等待回位
        src.obs = _obs(k * 0.033, "None", palm)
        app.run_once(42500 + k * 33)
    q = np.degrees(np.asarray(app.arm.get_state().joint_angles, dtype=float))
    ready = np.degrees(np.asarray(app.startup.ready_joints, dtype=float))
    assert np.max(np.abs(q - ready)) < 8.0, f"复位不到位: {np.round(q,1)} vs {np.round(ready,1)}"


def test_ok_transition_thumb_up_not_reset(app):
    """OK 刚切入跟随后，松手过渡姿态误识别成点赞不应触发复位踢出。

    真机故障复现：OK 进入跟随 -> 手松开过程被识别成 Thumb_Up（3 票即
    确认）-> reset_request 把模式切回 IDLE，且旧同步复位冻结主循环 3s+
    （画面卡死）。修复：OK 切换后 RESET_GRACE_S 宽限期内吞掉 Thumb_Up
    边沿；复位改为随主循环分步推进（不阻塞、UI 不冻结）。
    """
    src: _FixedSource = app.obs_source
    palm = np.array([0.30, 0.0, 0.45])

    settle(src, app, 500, palm)
    feed(src, app, "Okay", palm, 1000)           # -> FOLLOW（此刻开启宽限窗）
    assert app.interact.mode.value == "FOLLOW"

    feed(src, app, "Thumb_Up", palm, 1500)       # 宽限期内(<1.2s)的点赞
    assert app.interact.mode.value == "FOLLOW", "切换过渡期的点赞不应触发复位"
    assert not app._resetting

    feed(src, app, "Thumb_Up", palm, 2500)       # 宽限期外再点赞 -> 正常复位
    assert app.interact.mode.value == "IDLE"
    assert app._resetting                        # 复位异步进行中（不阻塞）
    for k in range(200):                          # 6.6s 等待分步回位完成
        src.obs = _obs(k * 0.033, "None", palm)
        app.run_once(3000 + k * 33)
    assert not app._resetting, "复位应在 RESET_DURATION_S 内分步完成"
