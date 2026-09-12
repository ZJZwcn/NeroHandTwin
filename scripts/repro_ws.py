"""独立复现 WS 测试卡死（无 pytest 捕获干扰，faulthandler 超时硬退并打印堆栈）。"""
import asyncio
import faulthandler
import json
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from aiohttp import ClientSession, WSMsgType, web

from nerohandtwin.web import server as web_server
from nerohandtwin.perception.hand_tracker import HandObservation


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


class _FakeState:
    joint_angles = np.array([0.0, -0.6109, 0.0, 2.0071, 0.0, 0.0, 0.2618])
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
    obj._gripper_opening = 0.5
    obj._follow_anchored = False
    obj._state = _FakeState()
    obj.fps = 12.3
    obj.cmd_queue = queue.Queue()
    obj.web_clients = set()
    return obj


async def main():
    fake_app = make_fake_app()
    built = web_server._make_app(fake_app)

    runner = web.AppRunner(built, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8099)
    await site.start()
    print("server up")
    async with ClientSession() as sess:
        ws = await sess.ws_connect("http://127.0.0.1:8099/ws", heartbeat=15.0)
        print("ws connected")
        for i in range(100):
            try:
                m = await asyncio.wait_for(ws.receive(), timeout=5.0)
            except asyncio.TimeoutError:
                print("TIMEOUT — dumping all task stacks", flush=True)
                for t in asyncio.all_tasks():
                    print("TASK:", t, "done=", t.done(), flush=True)
                    if t.done():
                        try:
                            print("   exc:", t.exception(), flush=True)
                        except Exception as e:
                            print("   exc?", e, flush=True)
                    else:
                        for f in t.get_stack():
                            print("   stack:", f, flush=True)
                raise
            print("frame", i, m.type)
            if m.type == WSMsgType.TEXT:
                print("text ok:", json.loads(m.data)["fps"])
                break
        print("sending command")
        await ws.send_json({"cmd": "follow_toggle"})
        await asyncio.sleep(0.3)
        print("queue size:", fake_app.cmd_queue.qsize())
        await ws.send_json({"cmd": "set_params", "params": {"speed_percent": 45}})
        await asyncio.sleep(0.3)
        while not fake_app.cmd_queue.empty():
            print("cmd:", fake_app.cmd_queue.get_nowait())
        print("closing ws")
        await ws.close()
        print("ws closed")
    await runner.cleanup()
    print("done")


if __name__ == "__main__":
    faulthandler.dump_traceback_later(15, exit=True)
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    loop.set_exception_handler(lambda _l, ctx: print("LOOP EXC:", ctx.get("message"), ctx.get("exception")))
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    print("ALL OK")
