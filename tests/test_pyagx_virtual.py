"""pyAgxArm 虚拟 CAN 集成测试 v3：真实后端跑通 app 主循环（无需硬件）。

用官方 NeroCanSlave 从站模拟 NERO 应答帧，验证：
- PyAgxArmController.connect（臂使能 + 夹爪独立使能）
- app 就绪序列（关节空间离零位）
- OK 切换跟随 move_p 点位流 + get_state + 夹爪动作
- 拇指向下复位
"""

import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_TP = str(ROOT / "third_party" / "pyAgxArm")
if _TP not in sys.path:
    sys.path.insert(0, _TP)

pytest.importorskip("pyAgxArm", reason="third_party/pyAgxArm 未安装（真机后端未装）")

try:
    from tests.slaves.nero_can_slave import NeroCanSlave
except ImportError:
    pytest.skip("需要 third_party/pyAgxArm 的官方从站模拟器", allow_module_level=True)

from nerohandtwin.app import NeroGestureApp  # noqa: E402
from nerohandtwin.control.pyagx_arm import PyAgxArmController  # noqa: E402
from nerohandtwin.perception.hand_tracker import HandObservation  # noqa: E402


class _FixedSource:
    def __init__(self):
        self.obs = None
        self.last_obs = None
        self.last_frame = None

    def get(self, _now_ms):
        return self.obs


def _obs(t, gesture, palm, pinch=0.8, gap_m=None):
    return HandObservation(
        present=gesture is not None,
        timestamp_ms=int(t * 1000),
        gesture=gesture or "None",
        gesture_score=0.9,
        pinch=pinch,
        palm3d=None if palm is None else np.asarray(palm, dtype=float),
        index3d=None,
        gap_m=gap_m,
    )


def feed(src, app, gesture, palm, base_ms, n=10, step_ms=33):
    """喂手势帧；手势间插 8 帧"手在但无手势"（>去抖窗口）重置边沿状态。"""
    for k in range(n):
        src.obs = _obs(k * 0.033, gesture, palm)
        app.run_once(base_ms + k * step_ms)
    for g in range(8):
        src.obs = _obs((n + g) * 0.033, "None", palm)
        app.run_once(base_ms + (n + g) * step_ms)


def settle(src, app, base_ms, palm, n=90):
    """喂无手势帧：让启动序列滑行完成、末端静止（锚定前置）。"""
    for k in range(n):
        src.obs = _obs(k * 0.033, "None", palm)
        app.run_once(base_ms + k * 33)


@pytest.fixture()
def slave_and_app():
    """每测试新建 app+从站（module 作用域会跨测试污染手势边沿状态）。"""
    ch = f"nero_vcan_itest_{uuid.uuid4().hex[:8]}"
    slave = NeroCanSlave(ch)
    slave.start()
    arm = PyAgxArmController(interface="virtual", channel=ch,
                             firmware="default", speed_percent=10)
    app = NeroGestureApp(ROOT / "configs", obs_source=_FixedSource(),
                         arm=arm, headless=True)
    app.t_base_cam = np.eye(4)
    app.follow.pipeline.t_base_cam = np.eye(4)
    app.point.pipeline.t_base_cam = np.eye(4)
    app.start()
    yield app
    app.shutdown()
    slave.stop()


def test_real_backend_ok_toggle_follow(slave_and_app):
    """真机后端全指令流：就绪 -> OK->跟随 -> OK->待机 -> 拇指向上复位。"""
    app = slave_and_app
    src: _FixedSource = app.obs_source
    palm = np.array([0.30, 0.0, 0.45])

    settle(src, app, 500, palm)
    assert app.interact.mode.value == "IDLE"     # 上电默认

    feed(src, app, "Okay", palm, 1000)           # OK -> 跟随
    assert app.interact.mode.value == "FOLLOW"
    for k in range(60):                          # 60 帧关节流无异常
        src.obs = _obs(k * 0.033, "Open_Palm", palm, gap_m=0.10)
        app.run_once(1500 + k * 33)
    st = app.arm.get_state()
    assert len(np.atleast_1d(st.joint_angles)) == 7

    feed(src, app, "Okay", palm, 4500)           # OK -> 待机
    assert app.interact.mode.value == "IDLE"

    # 拇指向上复位（回就绪位，保持使能）
    feed(src, app, "Thumb_Up", palm, 5500)
    assert app.interact.mode.value == "IDLE"


def test_real_backend_gripper_gap_mirror(slave_and_app):
    """真机后端跟随中夹爪镜像指距：捏拢 8mm→闭合，张开 120mm→全开。"""
    app = slave_and_app
    src: _FixedSource = app.obs_source
    palm = np.array([0.30, 0.0, 0.45])

    settle(src, app, 5000, palm)                 # 90帧=2970ms，结束~7970
    feed(src, app, "Okay", palm, 8100)           # -> FOLLOW
    for k in range(12):
        src.obs = _obs(k * 0.033, "Open_Palm", palm, gap_m=0.008)
        app.run_once(8500 + k * 33)
    assert app._gripper_opening < 0.2
    for k in range(20):
        src.obs = _obs(k * 0.033, "Open_Palm", palm, gap_m=0.12)
        app.run_once(9100 + k * 33)
    assert app._gripper_opening > 0.9
    app.arm.hold()
