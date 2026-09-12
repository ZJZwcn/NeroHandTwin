"""手部追踪：MediaPipe Tasks（GestureRecognizer，一次前向同时输出关键点+手势）。

输出统一的 HandObservation：
- landmarks_px: 21 个关键点像素坐标
- point3d_camera: 指定关键点（默认掌心/食指尖）的相机系 3D 坐标（米）
  - 有深度（D435i）：像素 -> aligned depth -> deproject（真 3D）
  - 无深度（webcam/视频）：按手宽估距的伪 3D（仅用于逻辑验证）
- gesture: 识别的预置手势类别名
- pinch: 拇指尖(4)与食指尖(8)归一化开合度 0(捏紧)~1(张开)
- gap_m: 拇指尖(4)与中指尖(12)的实测距离（米）——夹爪跟随的输入，
  满行程 100mm 夹爪按此距离 1:1 镜像开合
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["HandObservation", "HandTracker"]

# MediaPipe 手部关键点索引
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
PINKY_MCP = 17


@dataclass
class HandObservation:
    """单帧手部观测结果。"""

    present: bool = False
    timestamp_ms: int = 0
    gesture: str = "None"
    gesture_score: float = 0.0
    handedness: str = "Unknown"
    pinch: float = 1.0
    landmarks_px: Optional[np.ndarray] = None  # (21,2) 像素
    palm3d: Optional[np.ndarray] = None  # 相机系 (3,) 米
    index3d: Optional[np.ndarray] = None  # 相机系 (3,) 米
    gap_m: Optional[float] = None  # 拇指尖-中指尖实测距离（米）
    meta: dict = field(default_factory=dict)


class HandTracker:
    """MediaPipe 手部追踪封装。

    Args:
        model_path: gesture_recognizer.task 模型文件路径。
        depth_mode: "depth" 使用相机深度反投影（D435i）；"estimate" 用
            手宽估计距离的伪 3D（webcam 调试用）。
        tracking_keypoints: 需要 3D 化的关键点索引，默认掌心(9)与食指尖(8)。
    """

    def __init__(
        self,
        model_path: str | Path,
        depth_mode: str = "depth",
        assumed_hand_width_m: float = 0.085,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        self._vision = vision
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"缺少手势模型文件: {model_path}\n"
                "请先运行 scripts/download_models.py 下载。"
            )
        options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._mp = mp
        self.depth_mode = depth_mode
        self.assumed_hand_width_m = float(assumed_hand_width_m)
        self._last_ts_ms: Optional[int] = None

    def close(self) -> None:
        self._recognizer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, frame) -> HandObservation:
        """处理一帧相机数据（BGR ndarray 或 CameraFrame）。"""
        obs = HandObservation(timestamp_ms=frame.timestamp_ms)
        rgb = frame.color[:, :, ::-1]  # BGR -> RGB（零拷贝视图）
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        ts = int(frame.timestamp_ms)
        if self._last_ts_ms is not None and ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1  # VIDEO 模式要求单调递增
        self._last_ts_ms = ts

        result = self._recognizer.recognize_for_video(mp_image, ts)

        if not result.hand_landmarks:
            return obs

        lm = result.hand_landmarks[0]
        pts_px = np.array([[p.x * frame.color.shape[1], p.y * frame.color.shape[0]] for p in lm])
        world = None
        if result.hand_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])

        obs.present = True
        obs.landmarks_px = pts_px
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            obs.gesture = top.category_name
            obs.gesture_score = float(top.score)
        if result.handedness and result.handedness[0]:
            obs.handedness = result.handedness[0][0].category_name
        obs.pinch = self._pinch_ratio(pts_px)
        # 自定义 OK 手势（拇指-食指捏圈 + 其余三指伸直）：识别器无此预置，
        # 用几何判据覆盖；OK 与 Victory/Closed_Fist 的关键区分是三指伸直
        if self._ok_sign(pts_px):
            obs.gesture = "Okay"
            obs.gesture_score = 0.90

        self._fill_3d(obs, frame, pts_px, world)
        return obs

    @staticmethod
    def _ok_sign(pts_px: np.ndarray) -> bool:
        """OK 手势判据：拇指尖-食指尖捏合 + 中/无名/小指三指伸直。

        伸直判定：指尖距腕 > 指节(PIP)距腕 × 1.05。
        """
        wrist = pts_px[WRIST]
        span = float(np.linalg.norm(pts_px[MIDDLE_MCP] - wrist))
        if span < 1e-3:
            return False
        pinch = float(np.linalg.norm(pts_px[THUMB_TIP] - pts_px[INDEX_TIP])) / span
        if pinch > 0.5:
            return False
        for tip, pip in ((12, 10), (16, 14), (20, 18)):
            tip_d = float(np.linalg.norm(pts_px[tip] - wrist))
            pip_d = float(np.linalg.norm(pts_px[pip] - wrist))
            if tip_d < pip_d * 1.05:
                return False
        return True

    @staticmethod
    def _pinch_ratio(pts_px: np.ndarray) -> float:
        """拇指尖-食指尖距离 / 掌宽，映射到 0~1 开合度。"""
        hand_span = np.linalg.norm(pts_px[WRIST] - pts_px[MIDDLE_MCP])
        if hand_span < 1e-3:
            return 1.0
        pinch = np.linalg.norm(pts_px[THUMB_TIP] - pts_px[INDEX_TIP]) / hand_span
        # 经验范围约 0.2(捏紧)~1.4(张开)，线性压缩到 0~1
        return float(np.clip((pinch - 0.2) / 1.2, 0.0, 1.0))

    def _fill_3d(self, obs: HandObservation, frame, pts_px: np.ndarray, world) -> None:
        """按深度能力填充 palm3d / index3d / gap_m（拇指-中指指尖距离）。"""
        targets = {"palm3d": MIDDLE_MCP, "index3d": INDEX_TIP}
        if self.depth_mode == "depth" and frame.has_depth():
            for name, idx in targets.items():
                u, v = pts_px[idx]
                d = frame.depth_at(u, v)
                if d is not None:
                    setattr(obs, name, frame.deproject(u, v, d))
            # 拇指尖-中指尖实测距离：两指尖各自反投影后求欧氏距离（米）。
            # 指尖较细，采样半径收紧到 4，避免中位数混入背景深度。
            p_thumb = p_middle = None
            for idx, slot in ((THUMB_TIP, "t"), (MIDDLE_TIP, "m")):
                u, v = pts_px[idx]
                d = frame.depth_at(u, v, radius=4)
                if d is not None:
                    p = frame.deproject(u, v, d)
                    if slot == "t":
                        p_thumb = p
                    else:
                        p_middle = p
            if p_thumb is not None and p_middle is not None:
                obs.gap_m = float(np.linalg.norm(p_thumb - p_middle))
            elif obs.palm3d is not None:
                # 降级 1：单指尖深度缺失时，用掌心深度按焦距换算像素距离
                dist_px = float(np.linalg.norm(pts_px[THUMB_TIP] - pts_px[MIDDLE_TIP]))
                obs.gap_m = dist_px * float(obs.palm3d[2]) / frame.fx
            return

        if world is None:
            return
        # 伪 3D：用世界坐标手宽（米）与像素手宽之比估计距离 z = f * W_real / W_px
        w_real = float(np.linalg.norm(world[WRIST] - world[MIDDLE_MCP]))
        w_px = float(np.linalg.norm(pts_px[WRIST] - pts_px[MIDDLE_MCP]))
        if w_px < 1e-3 or w_real < 1e-4:
            return
        z = frame.fx * w_real / w_px
        for name, idx in targets.items():
            u, v = pts_px[idx]
            x = (u - frame.cx) * z / frame.fx
            y = (v - frame.cy) * z / frame.fy
            setattr(obs, name, np.array([x, y, z]))
        # world 坐标本身即米制：直接给出拇指-中指指尖距离
        obs.gap_m = float(np.linalg.norm(world[THUMB_TIP] - world[MIDDLE_TIP]))
        obs.meta["depth_mode"] = "estimate"
