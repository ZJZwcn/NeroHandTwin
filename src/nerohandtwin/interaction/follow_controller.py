"""跟随控制器。

FollowController / PointController：笛卡尔跟随（需外参）。
GimbalFollowController：协同跟随（免标定）——J1 对方位、J2/J4 追手的
水平线与前后安全距离（深度测距）、J7 保持夹爪水平朝前、J3/J5/J6 配合
保持，夹爪开口永远与手对准、与手同一水平线。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..geometry.workspace import WorkspaceMapper
from ..geometry.transforms import transform_point
from ..control.trajectory import SpeedLimiter
from ..perception.filters import OneEuroFilter
from ..perception.hand_tracker import HandObservation

__all__ = ["FollowController", "PointController", "GimbalFollowController"]


class _TargetPipeline:
    """相机系点 -> 基座系 -> 映射 -> 滤波 -> 限速 的公共管线。"""

    def __init__(self, workspace: WorkspaceMapper, t_base_cam: np.ndarray,
                 max_speed: float, max_accel: float, filter_cfg: dict):
        self.workspace = workspace
        self.t_base_cam = np.asarray(t_base_cam, dtype=float)
        self.limiter = SpeedLimiter(max_speed=max_speed, max_accel=max_accel)
        self.filter = OneEuroFilter(size=3, **filter_cfg)
        self._synced = False

    def sync(self, arm_position) -> None:
        """与机械臂当前位姿同步（启动/模式切换时）。"""
        self.limiter.reset(arm_position)
        self.filter.reset()
        self._synced = True

    def process(self, point_camera, arm_position, dt: float) -> np.ndarray:
        p_base = transform_point(self.t_base_cam, point_camera)
        mapped = self.workspace.map(p_base, arm_position)
        smoothed = self.filter.filter(mapped)
        return self.limiter.step(arm_position, smoothed, dt)


class FollowController:
    """掌心跟随控制器。"""

    def __init__(self, workspace: WorkspaceMapper, t_base_cam, speed_cfg: dict, filter_cfg: dict):
        self.pipeline = _TargetPipeline(workspace, t_base_cam, **speed_cfg, filter_cfg=filter_cfg)
        self.gripper_opening: Optional[float] = None

    def update(self, obs: HandObservation, arm_position, dt: float,
               interact_locked: bool) -> Optional[np.ndarray]:
        """返回限速后的目标点；手丢失或急停锁定时返回 None（保持）。"""
        if interact_locked or not obs.present or obs.palm3d is None:
            return None
        if not self.pipeline._synced:
            self.pipeline.sync(arm_position)
        target = self.pipeline.process(obs.palm3d, arm_position, dt)
        self.gripper_opening = obs.pinch  # 由 app 层经 GripperModel 换算
        return target


class PointController:
    """食指指点控制器：末端趋近指尖映射点，支持到位判定。"""

    def __init__(self, workspace: WorkspaceMapper, t_base_cam, speed_cfg: dict,
                 filter_cfg: dict, arrive_tol_m: float = 0.02, confirm_s: float = 0.5):
        self.pipeline = _TargetPipeline(workspace, t_base_cam, **speed_cfg, filter_cfg=filter_cfg)
        self.arrive_tol_m = float(arrive_tol_m)
        self.confirm_s = float(confirm_s)
        self._in_tol_since: Optional[float] = None
        self.arrived: bool = False

    def reset(self) -> None:
        """进入 POINT 模式时重置到位状态。"""
        self._in_tol_since = None
        self.arrived = False
        self.pipeline._synced = False

    def update(self, obs: HandObservation, arm_position, dt: float,
               interact_locked: bool) -> Optional[np.ndarray]:
        """返回目标点；到位后返回 None 让末端停在原地。"""
        if interact_locked or not obs.present or obs.index3d is None:
            return None
        if not self.pipeline._synced:
            self.pipeline.sync(arm_position)
        target = self.pipeline.process(obs.index3d, arm_position, dt)

        dist = float(np.linalg.norm(np.asarray(arm_position) - target))
        now = time.monotonic()
        if dist < self.arrive_tol_m:
            if self._in_tol_since is None:
                self._in_tol_since = now
            elif now - self._in_tol_since >= self.confirm_s:
                self.arrived = True
                return None  # 到位：保持
        else:
            self._in_tol_since = None
            self.arrived = False
        return target


class GimbalFollowController:
    """图像空间伺服（云台式追踪）：手保持画面中心十字，夹爪始终指向手。

    控制分工（纯像素误差增量伺服，无逆解、无锚定、无需标定）：
    - J1 底座旋转：手偏画面哪边，臂就朝哪边转（手左→臂左）；
    - J4 肘俯仰：手偏画面上/下，臂抬/落跟手（4舵 负=上/正=前下）；
    - J2 肩前后：手-相机深度 vs 参考距离，保持安全距离（2舵 前伸/后缩）；
    - J3/J5/J6/J7 保持不动（夹爪姿态维持）。

    全部关节目标流经 max_joint_rate 限速（rad/s）；手丢失时保持不动；
    各轴死区防抖、增益与方向符号可配（方向不对翻 sign_* 即可）。
    """

    # NERO 关节安全范围（跟随伺服用，比固件限位更保守）
    _Q_LIMITS = np.array([
        [-1.2, 1.2],     # J1 底座旋转
        [-1.60, 0.40],   # J2 肩前后（固件 ±1.745；跟随只用前伸区）
        [-2.6, 2.6],     # J3 回中保持
        [-0.95, 2.05],   # J4 肘俯仰（固件 [-1.012, 2.147]）
        [-2.6, 2.6],     # J5 回中保持
        [-0.7, 0.9],     # J6 开口水平保持
        [-1.5, 1.5],     # J7 腕转（固件 ±1.5708）
    ])

    def __init__(self, ready_joints, cfg: dict | None = None):
        cfg = cfg or {}
        self.ready = np.asarray(ready_joints, dtype=float)
        # --- 水平：J1 底座旋转（手偏画面哪边臂往哪转） ---
        self.k_yaw = float(cfg.get("k_yaw", 0.9))
        self.sign_yaw = float(cfg.get("sign_yaw", -1))
        self.dz_u = float(cfg.get("deadzone_u", 0.04))
        # --- 垂直：J4 肘俯仰（手偏画面上/下，臂抬/落跟手） ---
        self.k_pitch = float(cfg.get("k_pitch", 0.7))
        self.sign_pitch = float(cfg.get("sign_pitch", 1))
        self.dz_v = float(cfg.get("deadzone_v", 0.05))
        # --- 深度：J2 肩前后（手-相机距离 vs 参考距离，保持安全距离） ---
        self.k_depth = float(cfg.get("k_depth", 0.6))
        self.sign_depth = float(cfg.get("sign_depth", -1))
        self.dz_z = float(cfg.get("deadzone_z", 0.04))
        self.z_ref = float(cfg.get("z_ref", 0.40))
        # --- 限速 ---
        self.max_joint_rate = float(cfg.get("max_joint_rate", 0.5))
        # 诊断输出（eu/ev/z_err）
        self.err = (0.0, 0.0, 0.0)
        # 伺服参考十字（画面中心像素，供可视化叠加）
        self.anchor_uv: Optional[tuple[float, float]] = None
        self._q: Optional[np.ndarray] = None

    def reset(self, q_now) -> None:
        """进入跟随（或按 c）时：以当前位姿为追踪起点。"""
        self._q = np.clip(np.asarray(q_now, dtype=float).copy(),
                          self._Q_LIMITS[:, 0], self._Q_LIMITS[:, 1])

    @staticmethod
    def _dead(v: float, dz: float) -> float:
        return 0.0 if abs(v) < dz else (v - dz if v > 0 else v + dz)

    def update(self, obs: HandObservation, frame_hw: tuple[int, int],
               dt: float) -> Optional[np.ndarray]:
        """输入一帧手部观测，返回增量调整后的关节目标；手丢失时保持。"""
        if self._q is None:
            return None
        h, w = frame_hw
        # 参考十字 = 画面中心（手丢失时也刷新，供跟随视图持续显示）
        self.anchor_uv = (w / 2.0, h / 2.0)
        if not obs.present or obs.landmarks_px is None:
            return self._q.copy()  # 保持（安全默认）

        u, v = obs.landmarks_px[9]  # 掌心像素
        eu = (float(u) - w / 2.0) / w      # 手偏画面右 = 正
        ev = (float(v) - h / 2.0) / h      # 手偏画面下 = 正
        z_err = (float(obs.palm3d[2]) - self.z_ref)             if obs.palm3d is not None else 0.0   # 手比参考远 = 正
        self.err = (eu, ev, z_err)

        q = self._q.copy()
        q[0] += self.sign_yaw * self.k_yaw * self._dead(eu, self.dz_u) * dt
        q[3] += self.sign_pitch * self.k_pitch * self._dead(ev, self.dz_v) * dt
        if obs.palm3d is not None:
            q[1] += self.sign_depth * self.k_depth * self._dead(z_err, self.dz_z) * dt

        q = np.clip(q, self._Q_LIMITS[:, 0], self._Q_LIMITS[:, 1])
        # 全关节限速（rad/s）
        max_dq = self.max_joint_rate * max(dt, 0.0)
        self._q = self._q + np.clip(q - self._q, -max_dq, max_dq)
        return self._q.copy()
