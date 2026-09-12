"""常驻 mock 控制台服务（视觉验收用）：真实 server + 假状态，http://127.0.0.1:8099

用法：python scripts/serve_mock.py [joint1 joint2 joint3 joint4 joint5 joint6 joint7]
不传参数用就绪位。
"""
import asyncio
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from aiohttp import web

from nerohandtwin.web import server as web_server
from nerohandtwin.perception.hand_tracker import HandObservation

POSE = [float(x) for x in sys.argv[1:8]] if len(sys.argv) >= 8 \
    else [0.0, -0.6109, 0.0, 2.0071, 0.0, 0.0, 0.2618]
GRIP_OPEN = float(sys.argv[8]) if len(sys.argv) >= 9 else 0.5


class _FakeFrame:
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    shape = (480, 640)
    timestamp_ms = 12345


class _FakeSource:
    last_frame = _FakeFrame()
    last_obs = HandObservation(present=False, landmarks_px=None, palm3d=None)


class _FakeRecorder:
    recording = False


class _FakeGimbal:
    err = (0.12, -0.05, 0.08)
    max_joint_rate = 0.5
    k_yaw = 0.9
    k_pitch = 0.7
    k_depth = 0.6
    sign_yaw = -1
    sign_pitch = 1
    sign_depth = -1
    dz_u = 0.04
    dz_v = 0.05
    dz_z = 0.04
    z_ref = 0.4

    def reset(self, q):
        pass


class _FakeFSM:
    current = "None"
    pending = None
    pending_progress = 0.0


class _FakeInteract:
    from nerohandtwin.interaction.interaction_fsm import InteractMode as _M

    mode = _M.IDLE

    @property
    def in_follow(self):
        return True


class _FakeState:
    joint_angles = np.array(POSE)
    flange_pose = np.eye(4)


class _FakeArm:
    def get_state(self):
        return _FakeState()


def make_fake_app():
    obj = type("FakeApp", (), {})()
    obj.obs_source = _FakeSource()
    obj.interact = _FakeInteract()
    obj.gesture_fsm = _FakeFSM()
    obj.recorder = _FakeRecorder()
    obj.gimbal = _FakeGimbal()
    obj.arm = _FakeArm()
    obj._gripper_opening = GRIP_OPEN
    obj._follow_anchored = True
    obj._state = _FakeState()
    obj.fps = 12.3
    obj.cmd_queue = queue.Queue()
    obj.web_clients = set()
    return obj


async def main():
    app = web_server._make_app(make_fake_app())
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8099)
    await site.start()
    print(f"mock server up: http://127.0.0.1:8099  pose={POSE}", flush=True)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.new_event_loop().run_until_complete(main())
