"""可视化：相机帧 + 手部关键点/手势/状态叠加 + 仿真画面拼接。"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..perception.hand_tracker import HandObservation
from ..interaction.interaction_fsm import InteractMode

__all__ = ["Visualizer"]

_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

_MODE_COLOR = {
    InteractMode.FOLLOW: (80, 220, 100),
    InteractMode.REC: (60, 60, 240),
    InteractMode.PLAY: (240, 180, 60),
}


class Visualizer:
    """在相机帧上叠加手部/状态信息；可拼接仿真渲染画面。"""

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress

    def draw(
        self,
        frame: np.ndarray,
        obs: Optional[HandObservation],
        mode: InteractMode,
        status_lines: Optional[list[str]] = None,
        fps: float = 0.0,
        follow_anchor: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]

        if follow_anchor is not None:
            # 协同跟随视图：锚定十字（方位/水平线参考）+ 掌心连线
            ax, ay = int(follow_anchor[0]), int(follow_anchor[1])
            cv2.line(img, (ax - 16, ay), (ax + 16, ay), (0, 255, 255), 2)
            cv2.line(img, (ax, ay - 16), (ax, ay + 16), (0, 255, 255), 2)
            cv2.circle(img, (ax, ay), 22, (0, 255, 255), 1)

        if obs is not None and obs.present and obs.landmarks_px is not None:
            pts = obs.landmarks_px.astype(int)
            for a, b in _HAND_EDGES:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), (200, 220, 255), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(p), 3, (90, 180, 255), -1, cv2.LINE_AA)
            # 高亮：掌心(9) 拇指尖(4) 中指尖(12)——夹爪镜像后两者指距
            cv2.circle(img, tuple(pts[9]), 5, (255, 120, 90), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[4]), 4, (200, 120, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[12]), 4, (200, 120, 255), -1, cv2.LINE_AA)
            # 拇指尖-中指尖 距离连线（夹爪开合的镜像源）
            cv2.line(img, tuple(pts[4]), tuple(pts[12]), (80, 255, 180), 2, cv2.LINE_AA)
            if follow_anchor is not None:
                # 掌心 -> 锚定十字 连线（跟踪视觉反馈）
                cv2.line(img, tuple(pts[9]), (ax, ay), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                img, f"{obs.gesture} {obs.gesture_score:.2f}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
            gap_txt = "--" if obs.gap_m is None else f"{obs.gap_m * 1000:.0f}mm"
            cv2.putText(
                img, f"gap(4-12)={gap_txt} [{obs.handedness}]", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

        color = _MODE_COLOR.get(mode, (255, 255, 255))
        cv2.rectangle(img, (0, 0), (w, 34), (30, 30, 30), -1)
        cv2.putText(img, f"[{mode.value}]", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2, cv2.LINE_AA)
        if fps > 0:
            cv2.putText(img, f"{fps:.0f} FPS", (w - 90, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1, cv2.LINE_AA)
        if status_lines:
            for i, line in enumerate(status_lines):
                cv2.putText(img, line, (10, 58 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def compose(camera_img: Optional[np.ndarray], sim_img: Optional[np.ndarray]) -> np.ndarray:
        """把相机画面与仿真画面左右拼接（二者可为 None）。"""
        if camera_img is None and sim_img is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        if camera_img is None:
            return sim_img
        if sim_img is None:
            return camera_img
        target_h = max(camera_img.shape[0], sim_img.shape[0])

        def _fit(img: Optional[np.ndarray]) -> np.ndarray:
            if img is None:
                return np.zeros((target_h, 320, 3), dtype=np.uint8)
            scale = target_h / img.shape[0]
            new_w = max(1, int(img.shape[1] * scale))
            return cv2.resize(img, (new_w, target_h))

        left = _fit(camera_img)
        right = _fit(sim_img)
        return np.concatenate([left, right], axis=1)
