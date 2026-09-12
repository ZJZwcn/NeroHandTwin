"""限速与插值：末端笛卡尔目标点的速度限制器。

跟随控制中，手部观测 ~30Hz、且可能跳变。把"意图目标"变成
"限速目标流"是安全核心：无论输入怎么跳，输出以不超过
max_speed 的速度趋近意图点。
"""

from __future__ import annotations

import numpy as np

__all__ = ["SpeedLimiter"]


class SpeedLimiter:
    """一阶速度限幅（可选加速度限幅）。

    Attributes:
        max_speed: 线速度上限 (m/s)。
        max_accel: 加速度上限 (m/s^2)；0 表示不限制。
        v_current: 内部记忆的速度向量。
    """

    def __init__(self, max_speed: float = 0.08, max_accel: float = 0.5):
        if max_speed <= 0:
            raise ValueError("max_speed 必须为正")
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.v_current = np.zeros(3)
        self._pos: np.ndarray | None = None

    def reset(self, position) -> None:
        """同步到给定位置（模式切换/急停恢复时调用）。"""
        self._pos = np.asarray(position, dtype=float).copy()
        self.v_current = np.zeros(3)

    def step(self, current_position, target, dt: float) -> np.ndarray:
        """从内部设定点向 target 限速推进一步，返回新设定点。

        ⚠️ 内部 _pos 是开环设定点积分器，独立于实际末端位置推进——
        伺服滞后/重力下垂必然使实际位置落后于设定点，若每帧重同步到
        实际位置，限速流会退化为"从实际位置爬向目标"，与重力下垂形成
        正反馈。仅通过 reset() 在模式切换时显式同步。

        Args:
            current_position: 当前实际末端位置 (3,)（仅用于异常检测）。
            target: 意图目标点 (3,)。
            dt: 步长时间 (s)。
        """
        cur = np.asarray(current_position, dtype=float).reshape(3)
        tgt = np.asarray(target, dtype=float).reshape(3)

        if self._pos is None:
            self._pos = cur.copy()

        dt = max(dt, 1e-4)
        desired_v = (tgt - self._pos) / dt
        speed = float(np.linalg.norm(desired_v))

        if speed > self.max_speed:
            desired_v = desired_v * (self.max_speed / speed)

        if self.max_accel > 0:
            dv = desired_v - self.v_current
            dv_norm = float(np.linalg.norm(dv))
            max_dv = self.max_accel * dt
            if dv_norm > max_dv:
                desired_v = self.v_current + dv * (max_dv / dv_norm)

        self.v_current = desired_v
        self._pos = self._pos + desired_v * dt
        return self._pos.copy()
