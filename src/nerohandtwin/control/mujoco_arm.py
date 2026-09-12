"""MuJoCo 仿真后端：NERO 七轴臂 + DLS 逆运动学。

模型来源：third_party/agx_arm_sim/mujoco/agilex_arm/agilex_nero/scene.xml
（官方 NERO 模型，7 个旋转关节 + 平行夹爪 2 滑块（equality 联动））。

控制点约定：以 link7 body 原点为 IK 参考点（法兰中心在其前方 ~32mm，
跟随任务对这一差异不敏感；真机侧对应法兰 pose）。

关键实现：
- DLS（阻尼最小二乘）IK，阻尼 λ 抗奇异；
- 零空间偏置把 J2~J5 从 0 位拉离（手册奇异点要求）；
- 关节速度限幅的 ctrl 插值，末端 SpeedLimiter（上层）+ 关节限速（本层）双层保护；
- pump() 推进物理仿真，与真实时间对齐。
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np

from ..geometry.transforms import matrix_to_rpy
from .arm_interface import ArmInterface, ArmState
from .safety import READY_JOINTS

__all__ = ["MujocoArmController"]

_GRIPPER_STROKE = 0.05  # nero.xml 中 gripper_joint1 滑块行程 (m)


class MujocoArmController(ArmInterface):
    """MuJoCo 仿真机械臂后端。

    Args:
        model_path: scene.xml 路径。
        joint_rate: 关节速度限幅 (rad/s)。
        ik_damping: DLS 阻尼系数 λ。
        ik_iterations: 单次 IK 最大迭代数。
        nullspace_gain: 零空间回偏增益（把关节拉向 ready 姿态、避开奇异）。
        render_height/render_width: offscreen 渲染分辨率（0 表示不渲染）。
    """

    def __init__(
        self,
        model_path: str = "",
        joint_rate: float = 1.2,
        ik_damping: float = 0.08,
        ik_iterations: int = 120,
        nullspace_gain: float = 0.3,
        gravity_comp: bool = True,
        render_height: int = 360,
        render_width: int = 480,
        **_ignored,
    ):
        if not model_path:
            raise ValueError("必须提供 model_path（scene.xml）")
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        # 关节/执行器索引缓存
        self._joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        if any(j < 0 for j in self._joint_ids):
            raise ValueError("模型缺少 joint1~joint7，请确认传入的是 NERO scene.xml")
        self._qadr = [self.model.jnt_qposadr[j] for j in self._joint_ids]
        self._vadr = [self.model.jnt_dofadr[j] for j in self._joint_ids]
        self._ctrl_adr = [self.model.actuator_ctrladr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}")
        ] for i in range(1, 8)]
        self._gripper_ctrl = self.model.actuator_ctrladr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        ]
        self._link7_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self._jrange = self.model.jnt_range[self._joint_ids]  # (7,2)

        self._joint_rate = float(joint_rate)
        self._ik_damping = float(ik_damping)
        self._ik_iterations = int(ik_iterations)
        self._nullspace_gain = float(nullspace_gain)
        self._retry_attempts = 6      # 首解差时的随机种子重试次数
        self._retry_err_thresh = 0.01 # 重试触发阈值（米）
        self.gravity_comp = bool(gravity_comp)
        self.last_ik_error: float = 0.0
        self.ready_joints = np.asarray(READY_JOINTS, dtype=float)

        self._q_start = np.zeros(7)
        self._q_target = np.zeros(7)
        self._ctrl = np.zeros(7)
        self._move_t0 = 0.0
        self._move_duration = 0.0
        self._gripper_target = 0.0
        self._moving = False
        self._enabled = False
        self._sim_time: float = 0.0
        self._last_pump: float | None = None

        self._renderer = None
        if render_height > 0 and render_width > 0:
            try:
                self._renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
            except Exception as exc:  # noqa: BLE001 - 渲染不可用不阻塞仿真
                print(f"[mujoco] offscreen 渲染不可用，将只输出物理状态: {exc}")

        mujoco.mj_forward(self.model, self.data)

    # ---------- 生命周期 ----------
    def connect(self) -> None:
        self._enabled = True

    def close(self, go_home: bool = True) -> None:
        self._enabled = False
        if go_home:
            self.move_joints(self.ready_joints)  # 仿真同步到就绪位姿态
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None

    # ---------- 状态 ----------
    def get_joints(self) -> np.ndarray:
        return self.data.qpos[self._qadr].copy()

    def get_state(self) -> ArmState:
        pose = self.forward_kinematics(self.get_joints())
        return ArmState(
            joint_angles=self.get_joints(),
            flange_pose=pose,
            moving=self._moving,
            enabled=self._enabled,
        )

    def forward_kinematics(self, joint_angles: Sequence[float]) -> np.ndarray:
        """关节角 -> link7 位姿 4x4（临时数据，不污染仿真状态）。"""
        mujoco = self._mj
        tmp = mujoco.MjData(self.model)
        tmp.qpos[self._qadr] = np.asarray(joint_angles, dtype=float)
        mujoco.mj_forward(self.model, tmp)
        rot = tmp.xmat[self._link7_id].reshape(3, 3).copy()
        pos = tmp.xpos[self._link7_id].copy()
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        return pose

    # ---------- 运动 ----------
    def move_joints(self, joint_angles: Sequence[float], blocking: bool = False) -> None:
        """设定关节目标：ctrl 以关节限速向目标连续推进（积分式）。

        逐帧流式目标（跟随模式）：ctrl 向目标以 joint_rate 逼近，目标每帧
        更新自然形成平滑轨迹，与 SpeedLimiter 配合使用。
        离散动作（启动序列）：blocking=True 内部推进到到位。

        注意：ctrl 基于「上一次 ctrl」推进而非实际 q——重力下垂大于
        伺服增益时，跟随实际 q 的策略会塌向限位死锁。
        """
        q = np.clip(np.asarray(joint_angles, dtype=float), self._jrange[:, 0], self._jrange[:, 1])
        self._q_target = q
        self._moving = True
        if blocking:
            while self._moving:
                self.pump()
                time.sleep(self.dt)

    def move_pose(self, pose: Sequence[float], blocking: bool = False) -> None:
        """笛卡尔点位 [x,y,z,r,p,y]：IK 求解后走 move_joints。"""
        x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
        q = self.solve_ik(np.array([x, y, z]))
        self.move_joints(q, blocking=blocking)

    def move_to_ready(self, blocking: bool = False) -> None:
        """关节空间到就绪位（供上层启动序列使用）。"""
        self.move_joints(self.ready_joints, blocking=blocking)

    def move_cartesian(self, target_pos: Sequence[float]) -> None:
        """跟随模式：目标点(3,) -> 本地 IK -> 关节限速执行。"""
        q = self.solve_ik(np.asarray(target_pos, dtype=float).reshape(3))
        self.move_joints(q)

    def hold(self) -> None:
        """保持：冻结当前 ctrl，不再趋近目标。"""
        self._ctrl = self.get_joints()
        self._q_target = self._ctrl.copy()
        self._moving = False

    def safety_stop(self) -> None:
        self.hold()

    def set_speed_limit(self, percent: int) -> None:
        self._joint_rate = 1.2 * float(np.clip(percent, 1, 100)) / 50.0

    # ---------- 夹爪 ----------
    def set_gripper(self, opening: float) -> None:
        """开合度 0(闭)~1(开)。move 指令在仿真中即代表使能+运动。"""
        self._gripper_target = float(np.clip(opening, 0.0, 1.0))

    def get_gripper_opening(self) -> float:
        """仿真无独立硬件反馈，返回最近指令值。"""
        return float(self._gripper_target)

    def gripper_enabled(self) -> bool:
        return True

    # ---------- IK ----------
    def solve_ik(self, target_pos: np.ndarray, q_init: Optional[np.ndarray] = None) -> np.ndarray:
        """DLS + 零空间姿态偏置逆解（解析雅可比 + 精修阶段数值雅可比）。

        首解（从 q_init 出发）残差超过 retry_err_thresh 时，自动换随机
        种子重解（最多 retry_attempts 次），保留最优——应对关节限位/奇异
        附近的局部极小（DLS 单路径容易卡在肘部限位）。

        Args:
            target_pos: 目标点 (3,)（link7 原点，基座系）。
            q_init: 初始关节角，默认上次指令角。
        Returns:
            求解后的 7 关节角（已裁剪到限位）。
        """
        q_best = self._solve_ik_once(target_pos, q_init)
        err_best = self.last_ik_error
        if err_best > self._retry_err_thresh:
            rng = np.random.default_rng()
            for _ in range(self._retry_attempts):
                q_try = np.asarray(q_best if q_best is not None else q_init, dtype=float)
                q_try = q_try + rng.uniform(-0.6, 0.6, size=7)
                q_cand = self._solve_ik_once(target_pos, q_try)
                if self.last_ik_error < err_best:
                    q_best, err_best = q_cand, self.last_ik_error
                if err_best <= self._retry_err_thresh:
                    break
        return q_best

    def _solve_ik_once(self, target_pos: np.ndarray, q_init: Optional[np.ndarray]) -> np.ndarray:
        mujoco = self._mj
        q = (self._ctrl.copy() if q_init is None else np.asarray(q_init, dtype=float)).copy()
        target = np.asarray(target_pos, dtype=float).reshape(3)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        n_arm = 7
        eye3 = np.eye(3)
        eye7 = np.eye(n_arm)
        ns_scale = 0.003  # 米：零空间项权重尺度（3mm）——必须远小于目标容差，
        # 否则毫米级残差处零空间项与位置项抵消，IK 停摆
        numer_eps = 1e-6

        def fk_pos(qv: np.ndarray) -> np.ndarray:
            self.data.qpos[self._qadr] = qv
            mujoco.mj_kinematics(self.model, self.data)
            return self.data.xpos[self._link7_id].copy()

        # 阶段 A：解析雅可比 DLS + 零空间姿态偏置（粗收敛 + 姿态清理）
        for _ in range(self._ik_iterations):
            self.data.qpos[self._qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            error = target - self.data.xpos[self._link7_id]
            err_norm = float(np.linalg.norm(error))
            if err_norm < 1e-4:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self._link7_id)
            j = jacp[:, self._vadr].copy()
            jt = j.T
            dq = jt @ np.linalg.solve(j @ jt + self._ik_damping**2 * eye3, error)

            # 关节限位回避：接近限位的关节受恒定推离力（始终生效，不受
            # 零空间权重衰减影响），避免解在限位边界卡死
            margin = 0.12  # rad
            push = np.zeros(n_arm)
            for i in range(n_arm):
                lo_i, hi_i = self._jrange[i]
                if q[i] > hi_i - margin:
                    push[i] -= (q[i] - (hi_i - margin)) / margin
                elif q[i] < lo_i + margin:
                    push[i] += ((lo_i + margin) - q[i]) / margin
            dq += 0.12 * push

            ns_weight = self._nullspace_gain / (1.0 + (err_norm / ns_scale) ** 2)
            if ns_weight > 1e-4:
                j_pinv = jt @ np.linalg.solve(j @ jt + self._ik_damping**2 * eye3, eye3)
                null_proj = eye7 - j_pinv @ j
                dq += ns_weight * (null_proj @ (self.ready_joints - q))

            dq = np.clip(dq, -0.35, 0.35)
            q = np.clip(q + dq, self._jrange[:, 0], self._jrange[:, 1])

        # 阶段 B：数值雅可比纯 DLS 精修（毫米 -> 亚毫米，不带零空间项）
        for _ in range(60):
            p0 = fk_pos(q)
            error = target - p0
            err_norm = float(np.linalg.norm(error))
            if err_norm < 5e-5:
                break
            j = np.empty((3, n_arm))
            for jx in range(n_arm):
                qp = q.copy()
                qp[jx] += numer_eps
                j[:, jx] = (fk_pos(qp) - p0) / numer_eps
            jt = j.T
            dq = jt @ np.linalg.solve(j @ jt + (0.02**2) * eye3, error)
            dq = np.clip(dq, -0.35, 0.35)
            q = np.clip(q + dq, self._jrange[:, 0], self._jrange[:, 1])

        self.last_ik_error = err_norm
        return q

    # ---------- 物理推进 ----------
    def pump(self, wall_dt: Optional[float] = None) -> None:
        """推进仿真至与真实时间同步（或推进指定时长）。

        位置执行器 ctrl 按关节速度限幅向 q_target 插值；夹爪 ctrl 直达目标。
        """
        mujoco = self._mj
        now = time.monotonic()
        if wall_dt is None:
            wall_dt = 0.0 if self._last_pump is None else now - self._last_pump
        self._last_pump = now
        wall_dt = float(np.clip(wall_dt, 0.0, 0.2))

        if self._moving:
            q_prev = self._ctrl.copy()
            delta = self._q_target - q_prev
            max_step = self._joint_rate * wall_dt
            step = np.clip(delta, -max_step, max_step)
            self._ctrl = np.clip(q_prev + step, self._jrange[:, 0], self._jrange[:, 1])
            if np.all(np.abs(self._q_target - self._ctrl) < 1e-3):
                self._moving = False

        n_steps = int(round(wall_dt / self.dt))
        if self.gravity_comp:
            # 重力补偿：按当前 q 估算 qfrc_bias 前馈到广义力（冻结一次，本批
            # 步内近似有效）。消除位置伺服的重力稳态下垂（官方 kp 下 ~15mm）。
            mujoco.mj_forward(self.model, self.data)
            self.data.qfrc_applied[self._vadr] = self.data.qfrc_bias[self._vadr]
        for _ in range(max(n_steps, 1)):
            for i, adr in enumerate(self._ctrl_adr):
                self.data.ctrl[adr] = self._ctrl[i]
            self.data.ctrl[self._gripper_ctrl] = self._gripper_target * _GRIPPER_STROKE
            mujoco.mj_step(self.model, self.data)
        self._sim_time += max(n_steps, 1) * self.dt

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def render(self) -> Optional[np.ndarray]:
        """offscreen 渲染当前画面 (H,W,3) RGB；渲染不可用时返回 None。"""
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    @property
    def is_simulation(self) -> bool:
        return True
