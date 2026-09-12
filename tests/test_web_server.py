"""网页控制台服务端测试：状态帧/命令入队/MJPEG 端点。

依赖 aiohttp 自带 pytest 插件（pytest_plugins 声明）提供 aiohttp_client fixture。
"""

import sys
from pathlib import Path

import asyncio

import numpy as np
import pytest

pytest.importorskip("aiohttp")
pytest_plugins = ["aiohttp.pytest_plugin"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerohandtwin.perception.hand_tracker import HandObservation  # noqa: E402
from nerohandtwin.web import server as web_server  # noqa: E402


class _FakeFrame:
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    shape = (480, 640)


class _FakeSource:
    last_frame = _FakeFrame()
    last_obs = HandObservation(present=False, landmarks_px=None, palm3d=None)


class _FakeRecorder:
    recording = False


class _FakeGimbal:
    err = (0.0, 0.0, 0.0)

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
        return False

    def add_listener(self, cb):
        pass


class _FakeState:
    joint_angles = np.zeros(7)
    flange_pose = np.eye(4)


class _FakeArm:
    def get_state(self):
        return _FakeState()


@pytest.fixture()
def fake_app():
    """最小主循环替身（只读属性 + cmd_queue）。"""
    import queue

    obj = type("FakeApp", (), {})()
    obj.obs_source = _FakeSource()
    obj.interact = _FakeInteract()
    obj.gesture_fsm = _FakeFSM()
    obj.recorder = _FakeRecorder()
    obj.gimbal = _FakeGimbal()
    obj.arm = _FakeArm()
    obj._gripper_opening = 0.5
    obj._follow_anchored = False
    obj._state = _FakeState()
    obj.fps = 12.3
    obj.cmd_queue = queue.Queue()
    obj.web_clients = set()
    return obj


@pytest.fixture()
async def client(aiohttp_client, fake_app):
    return await aiohttp_client(web_server._make_app(fake_app))


async def test_index_page(client):
    resp = await client.get("/")
    assert resp.status == 200
    body = await resp.text()
    assert "NERO" in body and "cam-wrap" in body and "settings-panel" in body
    # 缓存根因回归：HTML 自身 no-cache，且 app.css/app.js 注入 ?v= 版本号，
    # 换版本即换 URL，强制浏览器绕开旧启发式缓存拉新资源。
    assert resp.headers.get("Cache-Control") == "no-cache"
    assert "app.js?v=" in body and "app.css?v=" in body


async def test_static_js_served(client):
    resp = await client.get("/static/js/app.js")
    assert resp.status == 200
    assert "neroFK" in await resp.text()
    # 静态资源必须带 no-cache，否则浏览器可能长期用旧 STL/脚本（模型改了不生效）
    assert resp.headers.get("Cache-Control") == "no-cache"


async def test_static_path_traversal_blocked(client):
    """静态路由必须拒绝越界路径（../ 解析到 STATIC_DIR 外）。"""
    resp = await client.get("/static/../../../etc/passwd")
    assert resp.status in (400, 404)


async def test_ws_state_frame_and_command(client, fake_app):
    ws = await client.ws_connect("/ws")
    from aiohttp import WSMsgType

    msg = None
    while True:  # 推送流含 PING/PONG：收到 TEXT 才解析
        m = await asyncio.wait_for(ws.receive(), timeout=4.0)
        if m.type == WSMsgType.TEXT:
            import json as _json

            msg = _json.loads(m.data)
            break
        if m.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
            pytest.fail(f"WS 异常关闭: {m}")
    # 状态帧字段完整性
    assert msg["mode"] == "IDLE"
    assert msg["hand"]["present"] is False
    assert msg["gripper_mm"] == 50.0
    assert msg["fps"] == pytest.approx(12.3, abs=0.1)
    assert isinstance(msg["arm"]["joint_angles"], list)
    assert "params" in msg and "speed_percent" in msg["params"]
    # 发命令 → 主循环 cmd_queue 收到。
    # 注意：pusher 每 33ms 推状态帧，receive 会在 TEXT/PING 间交错——
    # 这里只验证命令进入队列（异步时序下轮询确认）。
    await ws.send_json({"cmd": "follow_toggle"})
    got = None
    for _ in range(30):
        await asyncio.sleep(0.05)
        if not fake_app.cmd_queue.empty():
            got = fake_app.cmd_queue.get_nowait()
            break
    assert got == {"cmd": "follow_toggle"}, "命令未进入主循环 cmd_queue"
    # dict 命令（set_params）应原样入队
    await ws.send_json({"cmd": "set_params", "params": {"speed_percent": 45}})
    got_dict = None
    for _ in range(30):
        await asyncio.sleep(0.05)
        if not fake_app.cmd_queue.empty():
            got_dict = fake_app.cmd_queue.get_nowait()
            break
    assert got_dict == {"cmd": "set_params", "params": {"speed_percent": 45}}
    await ws.close()


def test_apply_params_runtime():
    """set_params 在 app._apply_params 中即时生效（速度裁剪 1~100）。"""
    from nerohandtwin.app import NeroGestureApp

    class FakeArm:
        def __init__(self):
            self.sp = None

        def set_speed_limit(self, p):
            self.sp = int(p)

    class FakeGimbal:
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

    obj = type("X", (), {})()
    obj.arm = FakeArm()
    obj.gimbal = FakeGimbal()
    NeroGestureApp._apply_params(obj, {
        "speed_percent": 45, "max_joint_rate": 1.0,
        "k_yaw": 1.4, "sign_pitch": 0,
    })
    assert obj.arm.sp == 45
    assert obj.gimbal.max_joint_rate == 1.0
    assert obj.gimbal.k_yaw == 1.4
    assert obj.gimbal.sign_pitch == 0
    # 越界值裁剪到安全范围
    NeroGestureApp._apply_params(obj, {"speed_percent": 999, "speed_only": 1})
    assert obj.arm.sp == 100
    # 未知键忽略
    NeroGestureApp._apply_params(obj, {"bogus": 3})


async def test_mjpeg_endpoint(client):
    resp = await client.get("/video")
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("multipart/x-mixed-replace")
    chunk = await resp.content.read(200)
    assert chunk.startswith(b"--frame")
    assert b"Content-Type: image/jpeg" in chunk
    resp.close()  # 无限流：必须显式关闭，否则 client 收尾会挂起
