"""末端执行器：夹爪开合度模型。

NERO 两指夹爪行程 0~100mm；MuJoCo nero.xml 中 gripper_joint1 滑块行程
0~0.05m（两指对称，总开度 0~0.1m）。抽象层统一用 0~1 开合度，
与机械臂后端的 set_gripper() 对接。

跟随模式下夹爪按「拇指尖-中指尖实测距离」1:1 镜像（gap_to_opening）：
指尖张开 60mm -> 夹爪张开 60mm，100mm（满行程）-> 完全张开。
"""

from __future__ import annotations

from typing import Optional

__all__ = ["GripperModel"]


class GripperModel:
    """手指开度 -> 夹爪开合度 的换算模型。

    跟随模式：gap_to_opening（拇指-中指指尖实测距离，米制 1:1 镜像）。
    兼容旧捏合度映射：pinch_to_opening（拇指-食指归一化捏合度）。

    Attributes:
        stroke_m: 总行程（米），真机默认 0.1（两指夹爪 100mm 满行程）。
        pinch_min/pinch_max: 手部 pinch 值的饱和区间。
        invert: True 时 pinch 越大夹爪越紧（用户习惯可调）。
    """

    def __init__(
        self,
        stroke_m: float = 0.1,
        pinch_min: float = 0.15,
        pinch_max: float = 0.85,
        invert: bool = False,
    ):
        self.stroke_m = float(stroke_m)
        self.pinch_min = float(pinch_min)
        self.pinch_max = float(pinch_max)
        self.invert = bool(invert)

    def pinch_to_opening(self, pinch: float) -> float:
        """pinch(0捏紧~1张开) -> 开合度 0~1，区间外饱和。"""
        if self.pinch_max <= self.pinch_min:
            return 0.0
        opening = (pinch - self.pinch_min) / (self.pinch_max - self.pinch_min)
        opening = min(1.0, max(0.0, opening))
        return 1.0 - opening if self.invert else opening

    def gap_to_opening(self, gap_m: Optional[float]) -> Optional[float]:
        """拇指尖-中指尖实测距离(米) -> 开合度 0~1。

        满行程 stroke_m（100mm）与手指开度 1:1 镜像：指尖张开多少毫米，
        夹爪就张开多少毫米，超过满行程饱和全开。gap_m 无效时返回 None
        （上层保持上次开度）。
        """
        if gap_m is None:
            return None
        return min(1.0, max(0.0, float(gap_m) / self.stroke_m))

    def opening_to_meters(self, opening: float) -> float:
        """开合度 0~1 -> 实际开口宽度（米）。"""
        opening = min(1.0, max(0.0, opening))
        return opening * self.stroke_m
