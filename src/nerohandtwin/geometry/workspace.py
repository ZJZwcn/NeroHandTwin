"""工作空间映射：手部坐标 -> 机械臂末端目标点。

流程：相机系点 --T_base<-camera--> 基座系点 --缩放/偏移--> 意图点
      --安全盒裁剪--> 末端目标点。

支持两种映射模式（config: mapping.mode）：
- absolute: 手掌在相机工作区中的位置按比例映射到机械臂安全盒（绝对对应）。
- relative: 以进入跟随模式时的手掌位置为零点，位移增量乘增益后叠加到
  当时的末端位置（遥操作风格），零点跟随模式启动时刷新。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .transforms import transform_point

__all__ = ["WorkspaceMapper"]


@dataclass
class WorkspaceMapper:
    """把感知层输出的基座系意图点映射为受限末端目标点。

    Attributes:
        box_min: 安全盒下界 (3,)（基座系，米）。
        box_max: 安全盒上界 (3,)（基座系，米）。
        cam_min: 手部交互区在基座系中的参考下界 (3,)，用于 absolute 归一化。
        cam_max: 手部交互区参考上界 (3,)。
        scale: 位移增益，relative 模式的位移缩放。
        mode: "absolute" 或 "relative"。
        enabled: False 时直通（仍做安全盒裁剪），用于标定/调试。
    """

    box_min: tuple = (-0.45, -0.35, 0.05)
    box_max: tuple = (0.45, 0.35, 0.55)
    cam_min: tuple = (-0.50, -0.35, 0.05)
    cam_max: tuple = (0.50, 0.35, 0.65)
    # 可达壳层（仅低矮区生效）：z < shell_z_split 时强制 xy 半径 >= shell_r_min，
    # 且目标 z >= shell_z_min。基座轴附近低处是肘部限位下的不可达区；
    # 高处（z >= split）臂近垂直，允许靠近轴线（如就绪位 r=0.09）。
    shell_r_min: float = 0.22
    shell_z_min: float = 0.25
    shell_z_split: float = 0.35
    scale: float = 1.0
    mode: str = "relative"
    enabled: bool = True

    _origin: Optional[np.ndarray] = field(default=None, repr=False)
    _arm_origin: Optional[np.ndarray] = field(default=None, repr=False)

    def reset_relative_origin(self, hand_point_base, arm_point_base) -> None:
        """进入跟随模式时调用：记录手与末端各自的参考零点。"""
        self._origin = np.asarray(hand_point_base, dtype=float).copy()
        self._arm_origin = np.asarray(arm_point_base, dtype=float).copy()

    def anchor(self, hand_point_base) -> np.ndarray:
        """免标定一键校准：以当前手部位置为中心重设手部交互区。

        交互区跨度沿用现有 cam_min/cam_max 的尺寸；这样无论相机装在哪、
        外参准不准，进入跟随时手所在的位置都会映射到工作区中心。
        返回新的 (cam_min, cam_max)。
        """
        center = np.asarray(hand_point_base, dtype=float).reshape(3)
        span = np.asarray(self.cam_max, dtype=float) - np.asarray(self.cam_min, dtype=float)
        if not np.all(span > 1e-3):
            span = np.asarray(self.box_max, dtype=float) - np.asarray(self.box_min, dtype=float)
        self.cam_min = tuple(center - span / 2.0)
        self.cam_max = tuple(center + span / 2.0)
        return self.cam_min, self.cam_max

    def clamp_point(self, p) -> np.ndarray:
        """对任意基座系点做安全盒+可达壳层约束（公开接口）。"""
        return self._clamp(np.asarray(p, dtype=float).reshape(3))

    def map(self, point_base, arm_position_base=None) -> np.ndarray:
        """把基座系手部点映射为末端目标点（已裁剪到安全盒）。

        Args:
            point_base: 手部/指尖点，基座系 (3,)，米。
            arm_position_base: 当前末端位置 (3,)，relative 模式需要。
        """
        p = np.asarray(point_base, dtype=float).reshape(3)
        if not self.enabled:
            return self._clamp(p)
        if self.mode == "relative":
            target = self._map_relative(p, arm_position_base)
        else:
            target = self._map_absolute(p)
        return self._clamp(target)

    def _map_absolute(self, p: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.cam_min, dtype=float)
        hi = np.asarray(self.cam_max, dtype=float)
        span = np.where(np.abs(hi - lo) < 1e-6, 1.0, hi - lo)
        alpha = np.clip((p - lo) / span, 0.0, 1.0)
        box_lo = np.asarray(self.box_min, dtype=float)
        box_hi = np.asarray(self.box_max, dtype=float)
        return box_lo + alpha * (box_hi - box_lo)

    def _map_relative(self, p: np.ndarray, arm_position_base) -> np.ndarray:
        if self._origin is None or self._arm_origin is None:
            if arm_position_base is None:
                raise ValueError("relative 模式需要先 reset_relative_origin() 或提供 arm_position_base")
            self.reset_relative_origin(p, arm_position_base)
        delta = (p - self._origin) * float(self.scale)
        return self._arm_origin + delta

    def _clamp(self, p: np.ndarray) -> np.ndarray:
        """安全盒裁剪 + 可达壳层约束（仅低矮区）。"""
        lo = np.asarray(self.box_min, dtype=float)
        hi = np.asarray(self.box_max, dtype=float)
        out = np.clip(p, lo, hi)
        out[2] = max(out[2], float(self.shell_z_min))
        if out[2] < float(self.shell_z_split):
            r = float(np.hypot(out[0], out[1]))
            r_min = float(self.shell_r_min)
            if r < r_min:
                if r < 1e-6:
                    out[0], out[1] = r_min, 0.0
                else:
                    out[0] *= r_min / r
                    out[1] *= r_min / r
        return out
