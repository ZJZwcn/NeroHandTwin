"""安全层：看门狗 + 交互状态机安全约束。

安全设计（对应 NERO 手册约束）：
- Watchdog: 控制指令流中断超过 timeout 时触发回调（缓停），恢复后 clear。
- 从原点/零位直接下点位指令是禁区（奇异点）：SafeStartup 负责
  启动时先用关节运动把臂带到就绪位（J2~J5 远离 0），之后才允许笛卡尔模式。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = ["Watchdog", "SafeStartup", "READY_JOINTS"]

# 就绪位姿（弧度，对应关节角度 0, -35, 0, 115, 0, 0, 15）：
# J2=-35° 抬臂、J4=115° 收肘，J2~J5 均远离 0 位（手册奇异点约束）
# 各关节限位核对：J2 ±102° ✓  J4 -60~125° ✓  J7 ±97° ✓
READY_JOINTS = (0.0, np.radians(-35.0), 0.0, np.radians(115.0), 0.0, 0.0, np.radians(15.0))


class Watchdog:
    """指令流看门狗：feed() 停止超过 timeout 秒触发 on_timeout（一次）。

    用法：
        wd = Watchdog(timeout=0.5, on_timeout=arm.hold)
        wd.start()
        # 主循环每帧: wd.feed()
        wd.stop()
    """

    def __init__(self, timeout: float = 0.5, on_timeout: Optional[Callable[[], None]] = None):
        self.timeout = float(timeout)
        self.on_timeout = on_timeout
        self._lock = threading.Lock()
        self._last_feed: float = 0.0
        self._fired = False
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop_evt.clear()
        self._last_feed = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name="safety-watchdog")
        self._thread.start()

    def feed(self) -> None:
        with self._lock:
            self._last_feed = time.monotonic()
            if self._fired:
                self._fired = False

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._fired

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_evt.wait(0.05):
            with self._lock:
                expired = time.monotonic() - self._last_feed > self.timeout
                if expired and not self._fired:
                    self._fired = True
                    cb = self.on_timeout
                else:
                    cb = None
            if cb is not None:
                try:
                    cb()
                except Exception:  # noqa: BLE001 - 安全回调不允许让线程死掉
                    pass


class SafeStartup:
    """启动序列与奇异点守卫。"""

    def __init__(self, arm, ready_joints: Sequence[float] = READY_JOINTS):
        self.arm = arm
        self.ready_joints = np.asarray(ready_joints, dtype=float)
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """是否已离开原点、允许笛卡尔指令。"""
        return self._ready

    def move_to_ready(self, duration: float = 3.0, steps: int = 100) -> None:
        """关节空间慢速移动到就绪位（阻塞）。

        用关节插值而不是一次性 move_joints，保证从零位出发的过程
        可控（零位是奇异点，此处不涉及 IK）。
        """
        state = self.arm.get_state()
        start = np.asarray(state.joint_angles, dtype=float)
        for i in range(1, steps + 1):
            alpha = i / steps
            q = start + (self.ready_joints - start) * alpha
            self.arm.move_joints(q)
            time.sleep(duration / steps)
        self._ready = True

    @staticmethod
    def check_singular_guard(joint_angles: Sequence[float], zero_tol: float = 0.08) -> list[int]:
        """检查 J2~J5 中处于 0 位附近（±zero_tol）的关节，返回索引列表。

        手册要求运动学控制时避免 J2~J5 长时间处于 0 位；该函数用于
        事后诊断/日志，而不是实时约束。
        """
        q = np.asarray(joint_angles, dtype=float)
        return [int(i) + 1 for i in range(1, 5) if abs(q[i]) < zero_tol]
