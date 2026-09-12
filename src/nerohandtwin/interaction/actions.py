"""动作编排器：点赞/碰拳触发的小幅编排动作。

两个动作（相对触发时 TCP 位置的偏移航点序列，smoothstep 平滑插值）：
- nod   点头（点赞·动作1）：上-回-上-回，竖直面内
- shake 摇头（碰拳·动作2）：左-回-右-回，水平面内

安全设计：
- 返回的绝对目标点由 app 层经 WorkspaceMapper.clamp_point 约束（安全盒+可达壳层）
- 经专用 SpeedLimiter 限速执行，与跟随同一速度上限
- 锁定(OK)时 app 层调用 abort() 立即停止；播放中忽略新的动作触发
- 结束前有 hold 尾段：回到触发点停留片刻，保证干净停住
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ActionSequencer"]

_PATTERNS = {
    # 航点偏移（不含触发点自身；末段自动回零）。
    # 极点重复一次 = 在峰值短暂停留（人观感 + 物理可跟踪，瞬态峰值会欠达）
    "nod": [(0, 0, 1), (0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), (0, 0, 0)],
    "shake": [(0, 1, 0), (0, 1, 0), (0, 0, 0), (0, -1, 0), (0, -1, 0), (0, 0, 0)],
}


@dataclass
class ActionSequencer:
    """小幅动作编排播放器。

    Attributes:
        amplitude_m: 动作幅度（米，偏移峰值）。
        segment_s: 每段航点时长（秒）。
        hold_tail_s: 结束时回到触发点的保持时长（秒）。
    """

    amplitude_m: float = 0.04
    segment_s: float = 0.6
    hold_tail_s: float = 0.4
    name: str | None = None
    _waypoints: list = field(default_factory=list, repr=False)
    _t: float = 0.0
    _total: float = 0.0
    _start: np.ndarray | None = None

    @property
    def playing(self) -> bool:
        return self._start is not None

    @property
    def progress(self) -> float:
        if not self.playing or self._total <= 0:
            return 0.0
        return min(1.0, self._t / self._total)

    def start(self, name: str, start_pos) -> bool:
        """开始播放动作（正在播放时忽略）。"""
        if self.playing or name not in _PATTERNS:
            return False
        self.name = name
        self._start = np.asarray(start_pos, dtype=float).copy()
        a = float(self.amplitude_m)
        seg = float(self.segment_s)
        # 航点时间轴：0 起点 + 各偏移点，最后回到 0（触发点）
        self._waypoints = [(0.0, np.zeros(3))] + [
            ((i + 1) * seg, np.asarray(off, dtype=float) * a)
            for i, off in enumerate(_PATTERNS[name])
        ]
        self._total = self._waypoints[-1][0] + float(self.hold_tail_s)
        self._t = 0.0
        return True

    def abort(self) -> None:
        self._start = None
        self._waypoints = []
        self.name = None

    def step(self, dt: float) -> np.ndarray | None:
        """推进 dt 秒，返回当前绝对目标点；播放结束返回 None。"""
        if not self.playing:
            return None
        self._t += max(dt, 0.0)
        seg = float(self.segment_s)

        if self._t >= self._total:
            # hold 尾段结束：收尾，最后目标为触发点（由最后一次 step 给出）
            self.abort()
            return None

        # hold 尾段：停在触发点
        if self._t >= self._waypoints[-1][0]:
            return self._start.copy()

        # 找当前区间 [t_i, t_i+1)
        for i in range(len(self._waypoints) - 1):
            t0, off0 = self._waypoints[i]
            t1, off1 = self._waypoints[i + 1]
            if t0 <= self._t < t1:
                span = max(t1 - t0, 1e-6)
                alpha = (self._t - t0) / span
                a = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
                offset = off0 + (off1 - off0) * a
                return self._start + offset
        return self._start.copy()
