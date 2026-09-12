"""One Euro Filter：低延迟手部抖动平滑（自实现，无第三方依赖）。

参考 Casiez et al. 2012。特点：慢速移动时强滤波（去抖），
快速移动时弱滤波（低延迟），适合手势跟随。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["OneEuroFilter"]


class _LowPass:
    """一阶低通（状态持有人）。"""

    __slots__ = ("value", "initialized")

    def __init__(self):
        self.value: float | None = None
        self.initialized = False

    def filter(self, x: float, alpha: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """One Euro 滤波器，支持标量或定长向量。

    Attributes:
        freq: 采样频率估计值（Hz），无时间戳时使用。
        min_cutoff: 最小截止频率，越小慢速越平滑。
        beta: 速度增益，越大快速移动延迟越低。
        d_cutoff: 导数通道截止频率。
    """

    def __init__(
        self,
        size: int = 3,
        freq: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.size = int(size)
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_lpf = [_LowPass() for _ in range(self.size)]
        self._dx_lpf = [_LowPass() for _ in range(self.size)]
        self._prev_x: np.ndarray | None = None
        self._prev_t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        """清除历史状态（例如手丢失后重新进入）。"""
        self._x_lpf = [_LowPass() for _ in range(self.size)]
        self._dx_lpf = [_LowPass() for _ in range(self.size)]
        self._prev_x = None
        self._prev_t = None

    def filter(self, x, timestamp: float | None = None) -> np.ndarray:
        """输入一帧数据，返回滤波结果。

        Args:
            x: 长度 size 的数组（如 3D 坐标，单位米）。
            timestamp: 秒；与上一帧的时间差用于估计实际采样频率。
        """
        x = np.asarray(x, dtype=float).reshape(self.size)

        if self._prev_x is None:
            self._prev_x = x.copy()
            self._prev_t = timestamp
            out = np.array(
                [lpf.filter(v, 1.0) for v, lpf in zip(x, self._x_lpf)]
            )
            return out

        # 估计实际采样频率
        freq = self.freq
        if timestamp is not None and self._prev_t is not None:
            dt = timestamp - self._prev_t
            if dt > 1e-4:
                freq = 1.0 / dt
        self._prev_t = timestamp

        # 速度估计（原始差分）-> 低通 -> 调整主通道截止频率
        dx = (x - self._prev_x) * freq
        self._prev_x = x.copy()
        a_d = self._alpha(self.d_cutoff, freq)
        dx_hat = np.array(
            [lpf.filter(v, a_d) for v, lpf in zip(dx, self._dx_lpf)]
        )

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        out = np.array(
            [
                lpf.filter(xv, self._alpha(c, freq))
                for xv, c, lpf in zip(x, cutoff, self._x_lpf)
            ]
        )
        return out
