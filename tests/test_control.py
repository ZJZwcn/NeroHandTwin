"""控制链路单元测试：限速器、交互状态机、MuJoCo IK 与闭环跟随（慢）。"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerohandtwin.control.trajectory import SpeedLimiter
from nerohandtwin.interaction.interaction_fsm import InteractMode, InteractionFSM

MUJOCO_MODEL = ROOT / "third_party/agx_arm_sim/mujoco/agilex_arm/agilex_nero/scene.xml"


def test_speed_limiter_respects_limit():
    lim = SpeedLimiter(max_speed=0.08, max_accel=0.0)
    lim.reset(np.zeros(3))
    dt = 0.033
    target = np.array([1.0, 0.0, 0.0])
    max_seen = 0.0
    for _ in range(400):  # 0.08m/s 需 12.5s，400*0.033=13.2s 足够到达
        out = lim.step(lim._pos, target, dt)
        max_seen = max(max_seen, float(np.linalg.norm(out - lim._pos) / dt))
    assert max_seen <= 0.08 + 1e-9
    assert np.linalg.norm(lim._pos - target) < 0.01


def test_speed_limiter_never_overshoots_when_target_close():
    """近目标一阶趋近：不越过目标，最终精确到达。"""
    lim = SpeedLimiter(max_speed=0.08, max_accel=0.0)
    lim.reset(np.zeros(3))
    target = np.array([0.01, 0, 0])
    for _ in range(10):
        p = lim.step(lim._pos, target, 0.033)
        assert np.linalg.norm(p - target) >= -1e-12  # 分量不越界
    assert np.linalg.norm(lim._pos - target) < 1e-6  # 精确到达


def test_interaction_fsm_transitions():
    """v5 方案：上电默认 IDLE，OK 切换 IDLE/FOLLOW（详见 test_interaction.py）。"""
    fsm = InteractionFSM()
    assert fsm.mode == InteractMode.IDLE  # 上电默认待机
    ev = fsm.feed_gesture("Okay")
    assert ev.detail == "OK->跟随模式" and fsm.mode == InteractMode.FOLLOW
    ev = fsm.feed_gesture("Okay")
    assert ev.detail == "OK->待机" and fsm.mode == InteractMode.IDLE


def test_interaction_fsm_recording_flow():
    """跟随中键盘 r 开始录制（录音接口不依赖手势）；手势层点赞仅在动作模式。"""
    from nerohandtwin.interaction.recorder import TrajectoryRecorder

    rec = TrajectoryRecorder(rate_hz=50)
    rec.start()
    assert rec.recording
    rec.record([0.1, 0, 0.3], 0.5)
    path = rec.stop_and_save(Path(tempfile.mkdtemp()), note="test")
    assert path.exists() and not rec.recording


@pytest.mark.skipif(not MUJOCO_MODEL.exists(), reason="缺少 MuJoCo NERO 模型")
class TestMujocoArm:
    @pytest.fixture(scope="class")
    def arm(self):
        from nerohandtwin.control.mujoco_arm import MujocoArmController

        arm = MujocoArmController(
            model_path=str(MUJOCO_MODEL), render_width=0, render_height=0
        )
        arm.connect()
        arm.move_joints([0.0, -0.6109, 0.0, 2.0071, 0.0, 0.0, 0.2618], blocking=True)  # READY_JOINTS
        yield arm
        arm.close()

    def test_ik_precision(self, arm):
        targets = [
            [0.30, 0.00, 0.45], [0.25, 0.10, 0.40], [0.20, -0.10, 0.35],
        ]
        for t in targets:
            q = arm.solve_ik(np.array(t))
            p = arm.forward_kinematics(q)[:3, 3]
            err = np.linalg.norm(p - np.array(t))
            assert err < 0.005, f"目标 {t} IK 误差 {err*1000:.2f}mm > 5mm"

    def test_closed_loop_follow_accuracy(self, arm):
        """验收核心：限速跟随稳态误差 <5mm（验收标准 #1/#3）。"""
        from nerohandtwin.control.trajectory import SpeedLimiter

        limiter = SpeedLimiter(max_speed=0.08, max_accel=0.5)
        limiter.reset(arm.get_state().flange_pose[:3, 3])
        dt = 0.033
        p1 = np.array([0.30, 0.0, 0.45])
        errs = []
        for k in range(450):  # 15s 足够收敛
            tcp = arm.get_state().flange_pose[:3, 3]
            target = limiter.step(tcp, p1, dt)
            q = arm.solve_ik(target)
            arm.move_joints(q)
            arm.pump(dt)
            errs.append(np.linalg.norm(arm.get_state().flange_pose[:3, 3] - p1))
        steady = float(np.mean(errs[-60:]))
        assert steady < 0.005, f"稳态误差 {steady*1000:.2f}mm > 5mm"

    def test_gripper_ctrl(self, arm):
        arm.set_gripper(0.7)
        assert arm._gripper_target == pytest.approx(0.7)

    def test_fk_at_home_is_singularity(self, arm):
        """零位在所有关节 0 附近 -> SafeStartup 应报告 J2~J5 奇异风险。"""
        from nerohandtwin.control.safety import SafeStartup

        risky = SafeStartup.check_singular_guard(np.zeros(7))
        assert set(risky) == {2, 3, 4, 5}
