"""aiohttp 网页控制台服务：独立 event loop 跑在守护线程，只读主循环状态。

端点：
    GET  /          控制台页面
    GET  /static/*  前端资源（vendored Three.js / P5.js / D3.js）
    WS   /ws        ~30Hz JSON 状态帧（推送）+ 前端命令（接收 → cmd_queue）
    GET  /video     MJPEG 实时视频流

命令：前端发 {"cmd": "..."}，服务端放入 app.cmd_queue，由主循环在
run_once 开头 drain 执行（follow_toggle/reset/anchor/record/playback/shutdown）——
主循环控制逻辑与线程安全边界不变。
"""

from __future__ import annotations

import asyncio
import threading
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 前端资源版本号：每次改 app.js/app.css 时递增。index 处理器把它注入到
# app.css/app.js 的 URL query（?v=...），换版本即换 URL，强制浏览器首次就拉新，
# 绕开旧版服务端遗留的无校验头启发式缓存。
ASSET_VERSION = "2026-09-11-7"

# 手部骨架拓扑（与 ui/visualizer._HAND_EDGES 一致，前端自绘用）
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

def _json_safe(obj):
    """状态帧 JSON 兜底转换：枚举→值、方法→调用结果、numpy→原生标量，
    无法序列化的对象降级为字符串——单个字段异常不拖垮整个 30Hz 推送。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, Enum):
        return _json_safe(obj.value)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if callable(obj):
        try:
            return _json_safe(obj())
        except Exception:  # noqa: BLE001
            return None
    return str(obj)


def build_state_frame(app) -> dict:
    """从主循环对象提取一帧 JSON 安全的状态快照（只读，无阻塞调用）。"""
    obs = getattr(app.obs_source, "last_obs", None)
    frame = getattr(app.obs_source, "last_frame", None)
    state = getattr(app, "_state", None)  # run_once 缓存的 ArmState

    landmarks = None
    if obs is not None and obs.landmarks_px is not None:
        landmarks = [[round(float(u), 1), round(float(v), 1)]
                     for u, v in obs.landmarks_px]

    palm3d = None
    if obs is not None and obs.palm3d is not None:
        palm3d = [round(float(x), 4) for x in obs.palm3d]

    gap_m = None
    if obs is not None and obs.gap_m is not None:
        gap_m = round(float(obs.gap_m), 4)

    joint_angles = None
    tcp = None
    if state is not None:
        joint_angles = [round(float(a), 4) for a in np.asarray(state.joint_angles, float)]
        tcp = [round(float(x), 4) for x in state.flange_pose[:3, 3]]

    mode = app.interact.mode.value
    gimbal_err = list(app.gimbal.err) if hasattr(app, "gimbal") else [0, 0, 0]

    # 运行参数（机械臂参数设置面板回显）
    gimbal_params = {}
    gimbal = getattr(app, "gimbal", None)
    if gimbal is not None:
        for key in ("max_joint_rate", "k_yaw", "k_pitch", "k_depth",
                    "sign_yaw", "sign_pitch", "sign_depth",
                    "dz_u", "dz_v", "dz_z", "z_ref"):
            val = getattr(gimbal, key, None)
            if val is not None:
                gimbal_params[key] = round(float(val), 4)
    speed_percent = None
    get_sp = getattr(app.arm, "get_speed_percent", None)
    if callable(get_sp):
        try:
            speed_percent = get_sp()
        except Exception:  # noqa: BLE001 - 后端不支持回读
            speed_percent = None

    # 夹爪开度：优先硬件回读（真机手动/示教改动也能同步到孪生），
    # 后端不支持回读时退回指令值
    gripper_opening = getattr(app, "_gripper_opening_fb", None)
    if gripper_opening is None:
        gripper_opening = app._gripper_opening

    return {
        "ts": int(time.time() * 1000),
        "mode": mode,
        "in_follow": app.interact.in_follow,
        "hand": {
            "present": bool(obs is not None and obs.present),
            "landmarks": landmarks,
            "palm3d": palm3d,
            "gesture_raw": obs.gesture if obs is not None else "None",
            "gesture_score": round(float(obs.gesture_score), 2) if obs is not None else 0.0,
            "gap_m": gap_m,
        },
        "gesture_stable": getattr(app.gesture_fsm, "current", "None"),
        "gesture_pending": getattr(app.gesture_fsm, "pending", None),
        "gesture_pending_progress": round(float(getattr(app.gesture_fsm, "pending_progress", 0.0)), 2),
        "gripper_opening": round(float(gripper_opening), 3),
        "gripper_mm": round(float(gripper_opening) * 100, 1),
        "gimbal_err": [round(float(x), 3) for x in gimbal_err],
        "follow_anchored": bool(app._follow_anchored),
        "recording": bool(app.recorder.recording) if hasattr(app, "recorder") else False,
        "arm": {"joint_angles": joint_angles, "tcp": tcp},
        "frame_shape": list(frame.color.shape[:2]) if frame is not None else [480, 640],
        "fps": round(float(getattr(app, "fps", 0.0)), 1),
        "params": {"speed_percent": speed_percent, "gimbal": gimbal_params},
    }


async def _mjpeg_boundary_iter(app, quality: int = 70):
    """MJPEG 帧生成器：异步轮询主循环最近帧（不阻塞 event loop/WS 推送）。

    时间戳去重：主循环未产生新帧时不重复编码（同一帧只发一次）。
    """
    last_ts = None
    while True:
        frame = getattr(app.obs_source, "last_frame", None)
        ts = getattr(frame, "timestamp_ms", None)
        if frame is not None and frame.color is not None \
                and (ts is None or ts != last_ts):
            last_ts = ts
            ok, buf = cv2.imencode(".jpg", frame.color,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        await asyncio.sleep(1.0 / 60.0)


def _make_app(app) -> web.Application:
    """构建 aiohttp 应用（传入主循环 NeroGestureApp 对象）。"""

    routes = web.RouteTableDef()

    async def index(_req):
        """返回控制台页面，并把 ASSET_VERSION 注入 app.css/app.js 的 URL，
        使资源版本号随构建变化 → 浏览器换 URL 强制拉新（配合静态 no-cache）。"""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = (html
                .replace('href="/static/css/app.css"',
                         f'href="/static/css/app.css?v={ASSET_VERSION}"')
                .replace('src="/static/js/app.js"',
                         f'src="/static/js/app.js?v={ASSET_VERSION}"'))
        return web.Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    async def mjpeg(_req):
        resp = web.StreamResponse(
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"})
        await resp.prepare(_req)
        try:
            async for chunk in _mjpeg_boundary_iter(app):
                await resp.write(chunk)
                # 强制刷出当前帧：aiohttp 默认 64KB 缓冲会让画面攒批、卡顿
                await resp.drain()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def ws_handler(req):
        ws = web.WebSocketResponse(heartbeat=15.0)
        await ws.prepare(req)
        clients = getattr(app, "web_clients", None)
        if clients is None:  # 宿主未预置集合（测试替身/旧版 app）时兜底
            clients = set()
            try:
                app.web_clients = clients
            except Exception:  # noqa: BLE001
                pass
        clients.add(ws)
        try:
            # 推送循环：~30Hz 状态帧
            pusher = asyncio.create_task(_push_state(ws, app))
            # 接收前端命令
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        import json as _json

                        data = _json.loads(msg.data)
                        # {"cmd": "..."} 字符串命令；带 params 的 dict 原样入队
                        if isinstance(data, dict) and data.get("cmd"):
                            app.cmd_queue.put(data)
                    except Exception:  # noqa: BLE001 - 坏帧忽略
                        pass
                elif msg.type == WSMsgType.ERROR:
                    break
            pusher.cancel()
        finally:
            clients.discard(ws)
        return ws

    async def _push_state(ws, app_obj):
        logged_error = False
        try:
            while True:
                try:
                    await ws.send_json(_json_safe(build_state_frame(app_obj)))
                except Exception as exc:  # noqa: BLE001 - 单帧失败仅告警一次，推送不中断
                    if not logged_error:
                        print(f"[web] 状态帧失败(仅告警一次): {exc}")
                        logged_error = True
                await asyncio.sleep(1.0 / 30.0)
        except (ConnectionResetError, asyncio.CancelledError):
            pass

    routes.get("/")(index)
    routes.get("/ws")(ws_handler)
    routes.get("/video")(mjpeg)

    async def static_handler(req):
        """/static/* 文件响应 + no-cache：强制每次协商校验，杜绝浏览器
        启发式缓存拿旧 STL/app.js（模型修正后刷新仍不生效的根因）。
        路径穿越防护：解析后必须仍在 STATIC_DIR 内。"""
        rel = req.match_info["path"]
        target = (STATIC_DIR / rel).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()):
            raise web.HTTPNotFound()
        if not target.is_file():
            raise web.HTTPNotFound()
        resp = web.FileResponse(target)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    routes.get("/static/{path:.*}")(static_handler)
    app_router = web.Application()
    app_router.add_routes(routes)
    return app_router


def _run_loop(app, host: str, port: int) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runner = web.AppRunner(_make_app(app), access_log=None)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, host, port)
    loop.run_until_complete(site.start())
    print(f"[web] 控制台已启动: http://{host}:{port}")
    loop.run_forever()


def start_web_server(app, host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
    """启动网页控制台守护线程（独立 event loop）。"""
    th = threading.Thread(target=_run_loop, args=(app, host, port),
                          daemon=True, name="nero-web-console")
    th.start()
    return th
