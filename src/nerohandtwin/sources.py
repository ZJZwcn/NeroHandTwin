"""感知数据源：把相机+手部追踪包装成 app 可用的 get(now_ms) 接口。"""

from __future__ import annotations

from typing import Optional

from .perception.camera_realsense import MockCamera, RealSenseCamera
from .perception.hand_tracker import HandObservation, HandTracker

__all__ = ["CameraHandSource"]


class CameraHandSource:
    """相机帧 -> 手部观测 的流水线（app 主循环每帧调用 get）。"""

    def __init__(self, camera, tracker: HandTracker):
        self.camera = camera
        self.tracker = tracker
        self.last_obs = HandObservation()
        self.last_frame = None
        self._log_ffwd_ms = 0  # MediaPipe 时间戳回退保护

    @classmethod
    def realsense(cls, model_path: str, depth_mode: str = "depth") -> "CameraHandSource":
        return cls(RealSenseCamera(), HandTracker(model_path, depth_mode=depth_mode))

    @classmethod
    def mock(cls, model_path: str, source: str = "synthetic", index_or_path=None,
             depth_mode: str = "estimate") -> "CameraHandSource":
        return cls(
            MockCamera(source=source, index_or_path=index_or_path),
            HandTracker(model_path, depth_mode=depth_mode),
        )

    def get(self, now_ms: float) -> HandObservation:
        frame = self.camera.read()
        if frame is None:
            return self.last_obs
        self.last_frame = frame
        try:
            self.last_obs = self.tracker.process(frame)
        except Exception as exc:  # noqa: BLE001 - 单帧识别失败不阻塞主循环
            print(f"[perception] 手部识别失败: {exc}")
        return self.last_obs

    def close(self) -> None:
        self.tracker.close()
        self.camera.close()
