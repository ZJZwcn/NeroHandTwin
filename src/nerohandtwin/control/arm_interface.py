"""机械臂控制抽象接口：仿真/真机后端共用同一契约。

所有位姿均为 米 / 弧度；position 指末端法兰位姿 [x,y,z,roll,pitch,yaw]
（ZYX 欧拉角，与 pyAgxArm 驱动一致）。

安全设计贯穿接口层：
- 所有后端必须实现 safety_stop()（软急停）与 hold()（保持）。
- move_to_* 由上层调用前应已通过安全盒/限速处理；后端自身再兜底限幅。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["ArmState", "ArmInterface", "create_arm"]


@dataclass
class ArmState:
    """机械臂状态快照。"""

    joint_angles: np.ndarray  # (n,) 弧度
    flange_pose: np.ndarray  # (6,) [x,y,z,r,p,y]
    moving: bool = False
    enabled: bool = False


class ArmInterface(abc.ABC):
    """七轴机械臂统一控制接口。"""

    n_joints: int = 7

    # ---------- 生命周期 ----------
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def close(self, go_home: bool = True) -> None:
        """关闭连接。go_home=True 时先回就绪位并保持使能（退出安全姿态）。"""

    # ---------- 状态 ----------
    @abc.abstractmethod
    def get_state(self) -> ArmState: ...

    @abc.abstractmethod
    def forward_kinematics(self, joint_angles: Sequence[float]) -> np.ndarray:
        """关节角 -> 法兰 4x4 位姿。"""

    # ---------- 运动 ----------
    @abc.abstractmethod
    def move_joints(self, joint_angles: Sequence[float], blocking: bool = False) -> None:
        """关节空间运动（用于离奇异点的就绪位转移）。"""

    @abc.abstractmethod
    def move_pose(self, pose: Sequence[float], blocking: bool = False) -> None:
        """笛卡尔点位运动 [x,y,z,r,p,y]。"""

    @abc.abstractmethod
    def move_cartesian(self, target_pos: Sequence[float]) -> None:
        """跟随模式专用：向基座系目标点 (3,) 迈进一小步。

        实现方式因后端而异：仿真后端本地 IK + 关节限速；
        真机后端保持当前姿态，直接下 move_p 点位（板载 IK）。
        上层每帧调用的目标点已经过限速，间距为毫米级。
        """

    def pump(self, dt: float | None = None) -> None:
        """推进控制/物理（仿真后端覆盖；真机后端为 no-op）。"""

    @abc.abstractmethod
    def hold(self) -> None:
        """保持当前位姿（停止接收新目标）。"""

    @abc.abstractmethod
    def safety_stop(self) -> None:
        """软急停：立即停止运动并禁用（上层在握拳等手势触发时调用）。"""

    # ---------- 末端速度 ----------
    def set_speed_limit(self, percent: int) -> None:
        """设置全局速度百分比 0~100（后端不支持时忽略）。"""

    def get_speed_percent(self) -> Optional[int]:
        """回读全局速度百分比（后端不支持时返回 None）。"""
        return None

    # ---------- 夹爪（使能与运动独立于机械臂使能） ----------
    def enable_gripper(self, timeout: float = 3.0) -> bool:
        """独立使能夹爪驱动并回读确认（仿真恒 True；无夹爪 False）。"""
        return True

    def gripper_enabled(self) -> bool:
        """回读夹爪驱动使能状态。"""
        return True

    def set_gripper(self, opening: float) -> None:
        """开合度 0(完全闭合)~1(满行程全开)。"""

    def get_gripper_opening(self) -> Optional[float]:
        """硬件回读夹爪开合度 0~1；后端不支持回读时返回 None。"""
        return None


def create_arm(config: dict) -> ArmInterface:
    """按配置创建机械臂后端。

    config 示例：
        {"backend": "mujoco", "model": ".../scene.xml", ...}
        {"backend": "pyagx", "interface": "agx_cando", "channel": "0", ...}
    """
    backend = str(config.get("backend", "mujoco")).lower()
    if backend == "mujoco":
        from .mujoco_arm import MujocoArmController

        return MujocoArmController(**config)
    if backend == "pyagx":
        from .pyagx_arm import PyAgxArmController

        return PyAgxArmController(**config)
    raise ValueError(f"未知机械臂后端: {backend}")
