"""相机后端：RealSense D435i（真机）与 Mock（开发/仿真）统一封装。

统一输出 CameraFrame：
- color: BGR uint8 (H,W,3)
- depth: 米制深度 (H,W) float32，无深度能力时为 None
- intrinsics: (fx, fy, cx, cy) 像素
- timestamp_ms: 单调递增毫秒（MediaPipe VIDEO 模式要求）

3D 定位链路（真机）：像素坐标 -> aligned depth 采样 -> deproject -> 相机系 (米)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["CameraFrame", "RealSenseCamera", "MockCamera"]


@dataclass
class CameraFrame:
    color: np.ndarray
    depth: Optional[np.ndarray]
    fx: float
    fy: float
    cx: float
    cy: float
    timestamp_ms: int
    shape: tuple = ()  # (H, W)，供帧级校验使用（不随 copy 丢失）

    def has_depth(self) -> bool:
        return self.depth is not None

    def deproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """像素 + 深度 -> 相机系 3D 点（米，RealSense 约定：x右 y下 z前）。"""
        x = (float(u) - self.cx) * depth_m / self.fx
        y = (float(v) - self.cy) * depth_m / self.fy
        return np.array([x, y, depth_m])

    def depth_at(self, u: float, v: float, radius: int = 6) -> Optional[float]:
        """取 (u,v) 附近 radius 邻域内有效深度的中位数，抗噪。"""
        if self.depth is None:
            return None
        h, w = self.depth.shape
        x0 = max(0, int(u) - radius)
        x1 = min(w, int(u) + radius + 1)
        y0 = max(0, int(v) - radius)
        y1 = min(h, int(v) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        patch = self.depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 1e-3) & (patch < 10.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))


class RealSenseCamera:
    """Intel RealSense D435i：RGB 与深度对齐后输出。"""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
    ):
        import pyrealsense2 as rs  # 延迟导入：无 D435i 的环境可运行 Mock

        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._cfg = cfg  # 保存供掉线自动重连
        profile = self._pipeline.start(cfg)

        # 深度对齐到彩色帧：后续可直接用彩色图上检测到的像素查深度
        self._align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()  # 米 / z16 单位

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self._fx, self._fy = float(intr.fx), float(intr.fy)
        self._cx, self._cy = float(intr.ppx), float(intr.ppy)

    def read(self) -> Optional[CameraFrame]:
        """读取一帧；流式异常（掉线/被抢占）时自动重建设备，最多重试 3 次。"""
        for attempt in range(3):
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                frames = self._align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color:
                    return None
                color_img = np.asanyarray(color.get_data())
                depth_img = (
                    np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale
                    if depth else None
                )
                return CameraFrame(
                    color=color_img,
                    depth=depth_img,
                    fx=self._fx,
                    fy=self._fy,
                    cx=self._cx,
                    cy=self._cy,
                    timestamp_ms=int(round(time.monotonic() * 1000)),
                    shape=color_img.shape[:2],
                )
            except RuntimeError as exc:
                # wait_for_frames 超时 / 设备掉线：重建 pipeline 再试
                print(f"[camera] 帧获取失败({attempt + 1}/3): {exc}")
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    pass
                time.sleep(0.5)
                try:
                    self._pipeline.start(self._cfg)
                except RuntimeError as exc2:
                    print(f"[camera] 流重建失败: {exc2}")
                    return None
        return None

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass


class MockCamera:
    """开发用相机：笔记本内置摄像头 / 视频文件 / 纯合成画面。

    - 无深度能力（depth=None），上层 HandTracker 走 2D/伪 3D 模式。
    - source="synthetic" 时生成带运动方块的画面，用于无摄像头环境自检渲染链路。
    """

    def __init__(self, source: str = "webcam", index_or_path=None, size=(640, 480)):
        import cv2

        self._cv2 = cv2
        self._size = size
        self._t0 = time.monotonic()
        self._intrinsics = (615.0, 615.0, size[0] / 2, size[1] / 2)
        self._cap = None
        if source == "webcam":
            self._cap = cv2.VideoCapture(int(index_or_path or 0))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            if not self._cap.isOpened():
                raise RuntimeError("无法打开 webcam，请检查索引或改用 video/synthetic 源")
        elif source == "video":
            self._cap = cv2.VideoCapture(str(index_or_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {index_or_path}")
        elif source != "synthetic":
            raise ValueError(f"未知 MockCamera source: {source}")
        self._source = source

    def read(self) -> Optional[CameraFrame]:
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:  # 视频循环播放
                if self._source == "video":
                    self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                if not ok:
                    return None
        else:
            # 合成画面：一个来回移动的圆块，仅用于验证渲染与叠加链路
            import time as _t

            w, h = self._size
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            t = _t.monotonic() - self._t0
            cx = int(w / 2 + w * 0.3 * np.sin(2 * np.pi * t / 6.0))
            cy = int(h / 2 + h * 0.2 * np.cos(2 * np.pi * t / 8.0))
            self._cv2.circle(canvas, (cx, cy), 24, (80, 170, 255), -1)
            frame = canvas
        fx, fy, cx, cy = self._intrinsics
        return CameraFrame(
            color=frame,
            depth=None,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            timestamp_ms=int(round(time.monotonic() * 1000)),
            shape=frame.shape[:2],
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
