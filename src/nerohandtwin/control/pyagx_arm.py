"""pyAgxArm 真机后端（松灵 NERO，CAN 通讯）。

⚠️ 安全须知（务必先读 README「真机安全清单」）：
- 首次上电后必须先执行就绪序列（关节空间离开零位奇异点）；
- speed_percent 从 10 开始逐级上调；
- 任何异常立即按键 Ctrl+C / 断电急停。

依赖：pip install -e third_party/pyAgxArm（或把该目录加入 sys.path）。

接口依据（third_party/pyAgxArm 官方 demo / 驱动源码）：
    create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=..., interface=..., channel=...)
    AgxArmFactory.create_arm(cfg) -> robot.connect() -> robot.enable()
    robot.set_speed_percent(p) / set_motion_mode('j'|'p') / move_j([7]) / move_p([x,y,z,r,p,y])
    robot.get_joint_angles().msg / robot.fk(ja) -> [x,y,z,roll,pitch,yaw]
    end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    end_effector.move_gripper_m(meters)
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .arm_interface import ArmInterface, ArmState
from .safety import READY_JOINTS

__all__ = ["PyAgxArmController"]


class PyAgxArmController(ArmInterface):
    """NERO 真机后端。

    Args:
        interface: 通讯接口（Windows: agx_cando；Linux: socketcan）。
        channel: CAN 通道。
        firmware: 固件版本常量（NeroFW.DEFAULT / V111 / V112 / V120 / V121）。
        speed_percent: 全局速度百分比（首次联调建议 10）。
        stroke_m: 夹爪行程（两指夹爪默认 0.1m）。
    """

    def __init__(self, interface: str = "agx_cando", channel: str = "0",
                 firmware: str = "default", speed_percent: int = 10,
                 stroke_m: float = 0.1,
                 gripper_open_width_m: float = 0.055, **_ignored):
        self._interface = interface
        self._channel = channel
        self._firmware = firmware
        self._speed_percent = int(speed_percent)
        self._stroke = float(stroke_m)
        # 满开时的实际指间距（传感器回读归一化分母）。100mm 行程夹爪满开
        # 指间距 ≈55mm（真机实测：满开回读 100 时传感器报 ≈0.055m），
        # 丝杠行程 ≠ 指间距，若用行程归一满开只显示 55%。
        self._gripper_open_width_m = float(gripper_open_width_m)
        self.ready_joints = np.asarray(READY_JOINTS, dtype=float)

        self._robot = None
        self._effector = None
        self._enabled = False
        self._gripper_last_cmd: float | None = None
        self._last_q: np.ndarray | None = None
        self._last_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])

    # ---------- 生命周期 ----------
    def connect(self) -> None:
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

        # 冻结打包（PyInstaller）后 python-can 的插件入口点元数据丢失，
        # 手动把 agx_cando 后端注册进 BACKENDS，保证 exe 内也能打开 CAN
        try:
            import can.interfaces

            can.interfaces.BACKENDS.setdefault("agx_cando", "agx_cando.bus")
        except Exception:  # noqa: BLE001 - 源码运行时入口点本就可用
            pass

        fw = getattr(NeroFW, self._firmware.upper(), None) or NeroFW.DEFAULT
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=fw,
            interface=self._interface,
            channel=self._channel,
        )
        self._robot = AgxArmFactory.create_arm(cfg)
        self._robot.connect()

        deadline = time.monotonic() + 5.0
        while not self._robot.enable():
            if time.monotonic() > deadline:
                raise RuntimeError("NERO 使能失败（enable 超时）")
            time.sleep(0.02)
        self._robot.set_speed_percent(self._speed_percent)
        try:
            self._effector = self._robot.init_effector(self._robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            # 夹爪使能与机械臂使能是两条独立链路，显式使能并回读确认
            self.enable_gripper()
        except Exception as exc:  # noqa: BLE001 - 未接夹爪时继续
            print(f"[pyagx] 夹爪初始化跳过: {exc}")
        self._enabled = True

    def close(self, go_home: bool = True) -> None:
        """关闭：默认先回就绪位并**保持使能**，再断开总线。

        用户要求：程序退出后机械臂回到初始姿态且不可失能——
        因此这里绝不调用 disable()，仅断开通讯（固件保持使能与目标）。
        回位循环屏蔽 Ctrl+C：二次中断也坚持回位完成后再断开。
        """
        if self._robot is None:
            return
        if go_home and self._enabled:
            print("[pyagx] 关闭流程: 返回就绪位（保持使能，再按 Ctrl+C 不跳过）...")
            q_ready = self.ready_joints.tolist()
            t0 = time.monotonic()
            interrupted = False
            while time.monotonic() - t0 < 10.0:
                try:
                    self._robot.move_j(q_ready)
                    time.sleep(0.1)
                    st = self._robot.get_arm_status()
                    ja = self._robot.get_joint_angles()
                    motion_done = st is not None and \
                        getattr(st.msg, "motion_status", None) == 0
                    at_ready = ja is not None and float(np.max(np.abs(
                        np.asarray(ja.msg, dtype=float) - self.ready_joints))) < 0.08
                    if motion_done and at_ready:
                        print("[pyagx] 已回到就绪位")
                        break
                except KeyboardInterrupt:
                    interrupted = True  # 忽略，继续回位
                except Exception as exc:  # noqa: BLE001 - 回就绪失败也要断开
                    print(f"[pyagx] 回就绪异常（保持当前位姿）: {exc}")
                    break
            else:
                if not interrupted:
                    print("[pyagx] 回就绪超时(10s)，以最后位置保持使能")
            # 吃掉排队的第二次 Ctrl+C
            try:
                import signal

                orig = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                self._robot.disconnect()  # 不 disable：保持使能
                signal.signal(signal.SIGINT, orig)
            except Exception:  # noqa: BLE001
                try:
                    self._robot.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            self._enabled = False
            return
        try:
            self._robot.disconnect()  # 不 disable：保持使能
        except Exception:  # noqa: BLE001
            pass
        self._enabled = False

    # ---------- 状态 ----------
    def get_state(self) -> ArmState:
        ja = self._robot.get_joint_angles()
        if ja is not None:
            q = np.asarray(ja.msg, dtype=float)
            self._last_q = q
        else:
            q = self._last_q if self._last_q is not None else np.zeros(7)
        fp = self._robot.fk(q.tolist())
        pose = self._pose6_to_matrix(fp)
        return ArmState(joint_angles=q, flange_pose=pose, enabled=self._enabled)

    def forward_kinematics(self, joint_angles: Sequence[float]) -> np.ndarray:
        fp = self._robot.fk(list(joint_angles))
        return self._pose6_to_matrix(fp)

    @staticmethod
    def _pose6_to_matrix(fp) -> np.ndarray:
        """SDK fk() 返回 [x,y,z,roll,pitch,yaw]（米/弧度）-> 4x4 位姿。"""
        arr = np.asarray(fp, dtype=float).reshape(6)
        from ..geometry.transforms import make_pose, rpy_to_matrix

        return make_pose(arr[:3], rpy_to_matrix(*arr[3:]))

    # ---------- 运动 ----------
    def move_joints(self, joint_angles: Sequence[float], blocking: bool = False) -> None:
        """关节空间运动。move_* 自动切换运动模式（SDK 默认开启），勿手动重发。"""
        self._robot.move_j(list(map(float, joint_angles)))
        if blocking:
            time.sleep(0.1)
            while self._robot.get_arm_status() is not None and \
                    getattr(self._robot.get_arm_status().msg, "motion_status", 0) != 0:
                time.sleep(0.05)

    def move_pose(self, pose: Sequence[float], blocking: bool = False) -> None:
        """笛卡尔点位 [x,y,z,r,p,y]（米/弧度）。

        blocking=True：等待到位（0.5s settle 等状态翻转 + 60s 超时上限，
        防止目标不可达时死循环）。
        """
        self._robot.move_p(list(map(float, pose)))
        if blocking:
            time.sleep(0.5)  # motion_status 残留 0 的竞态防护
            t0 = time.monotonic()
            while time.monotonic() - t0 < 60.0:
                st = self._robot.get_arm_status()
                if st is not None and \
                        getattr(st.msg, "motion_status", None) == 0:
                    return
                time.sleep(0.05)
            print("[pyagx] move_p 等待到位超时(60s)")

    def move_cartesian(self, target_pos: Sequence[float]) -> None:
        """跟随模式：向基座系目标点(3,)迈进一小步（P 模式点位流）。

        姿态保持当前法兰姿态不变；目标点由上层限速，帧间距毫米级。
        官方语义：连续 move_p 会覆盖上一个目标，控制器内部平滑衔接
        （position-velocity 模式），适合 30Hz 目标流。
        """
        from ..geometry.transforms import matrix_to_rpy

        pose = self.get_state().flange_pose
        rpy = matrix_to_rpy(pose[:3, :3])
        tgt = np.asarray(target_pos, dtype=float).reshape(3)
        self._robot.move_p([float(tgt[0]), float(tgt[1]), float(tgt[2]),
                            rpy[0], rpy[1], rpy[2]])

    def hold(self) -> None:
        """保持：停止下发新指令即可（控制器停在最后目标点）。"""

    def safety_stop(self) -> None:
        """软急停（电子急停；恢复需重新 enable）。"""
        try:
            self._robot.electronic_emergency_stop()
        finally:
            self._enabled = False

    def set_speed_limit(self, percent: int) -> None:
        self._speed_percent = int(np.clip(percent, 1, 100))
        self._robot.set_speed_percent(self._speed_percent)

    def get_speed_percent(self) -> int:
        return self._speed_percent

    # ---------- 夹爪 ----------
    # ArmMsgGripperCtrl status_code（官方码表，can_protocol/msgs/.../arm_gripper_ctrl.py）：
    #   0x00 disable/width  0x01 enable/width  0x02 disable+clear/width
    #   0x03 enable+clear/width
    _GRIPPER_CODE_DISABLE = 0x00
    _GRIPPER_CODE_ENABLE = 0x01
    _GRIPPER_CODE_ENABLE_CLEAR = 0x03

    def _gripper_send(self, status_code: int) -> None:
        """直接发送夹爪控制指令（SDK 未封装使能，按官方码表下发）。"""
        from pyAgxArm.protocols.can_protocol.msgs.effector.agx_gripper.default.transmit.arm_gripper_ctrl import (
            ArmMsgGripperCtrl,
        )

        self._effector._send_msg(ArmMsgGripperCtrl(status_code=status_code))

    def enable_gripper(self, timeout: float = 3.0) -> bool:
        """独立使能夹爪（与机械臂使能是两条链路），回读确认。

        驱动报错/未回零时自动 clear+enable；再失败补一次回零（homing，
        首次上电未回零的夹爪会拒绝使能）后重试。
        """
        if self._effector is None:
            print("[pyagx] 未接夹爪，跳过使能")
            return False
        # 直接使能
        self._gripper_send(self._GRIPPER_CODE_ENABLE)
        if self._wait_gripper_enabled(timeout):
            print("[pyagx] 夹爪已使能 ✓")
            return True
        # clear + enable
        print("[pyagx] 夹爪使能未确认，尝试 enable+clear ...")
        self._gripper_send(self._GRIPPER_CODE_ENABLE_CLEAR)
        if self._wait_gripper_enabled(timeout):
            print("[pyagx] 夹爪已使能（clear 后）✓")
            return True
        # 补回零后重试（未 homing 的夹爪驱动拒绝使能）
        print("[pyagx] 尝试夹爪回零（homing）后重新使能 ...")
        try:
            self._effector.reset_gripper()
        except Exception as exc:  # noqa: BLE001
            print(f"[pyagx] 回零指令异常: {exc}")
        time.sleep(1.0)
        self._gripper_send(self._GRIPPER_CODE_ENABLE_CLEAR)
        if self._wait_gripper_enabled(timeout + 2.0):
            print("[pyagx] 夹爪已使能（回零后）✓")
            return True
        print("[pyagx] ⚠ 夹爪使能失败：请检查夹爪供电/CAN 接线（不影响机械臂运行）")
        return False

    def _wait_gripper_enabled(self, timeout: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            gs = self._effector.get_gripper_status()
            if gs is not None and gs.msg.foc_status.driver_enable_status:
                return True
            time.sleep(0.05)
        return False

    def gripper_enabled(self) -> bool:
        """回读夹爪驱动使能状态。"""
        if self._effector is None:
            return False
        gs = self._effector.get_gripper_status()
        return bool(gs is not None and gs.msg.foc_status.driver_enable_status)

    def set_gripper(self, opening: float) -> None:
        """开合度 0(闭)~1(开) -> 宽度指令。

        官方语义：move_gripper_m 自带 enable/width 码（status_code=1），
        每次下发同时保证驱动处于使能状态——与 enable_gripper() 配合
        构成完整的使能-运动链路。
        指令满开用 _gripper_open_width_m（满开实际指间距 ≈55mm），
        与回读归一化同一标定，捏合镜像映射线性一致。
        """
        if self._effector is None:
            return
        opening = float(np.clip(opening, 0.0, 1.0))
        try:
            self._effector.move_gripper_m(opening * self._gripper_open_width_m)
            self._gripper_last_cmd = opening
        except Exception as exc:  # noqa: BLE001
            print(f"[pyagx] 夹爪指令失败: {exc}")

    def get_gripper_opening(self) -> Optional[float]:
        """硬件回读：夹爪当前开口宽度 → 开合度 0~1。

        SDK get_gripper_status 的 value 在 width 模式下为开口宽度（米，
        解析层已从 µm 换算）。归一化分母用 *_open_width_m（满开时的实际
        指间距）而非 stroke_m：100mm 行程夹爪满开时指间距 ≈55mm，丝杠
        行程 ≠ 指间距，用行程归一会让满开只显示 55%。个别固件以毫米为
        单位（正常应 ≤ 满开间距 2 倍），按量纲自动判别换算。
        角度模式/未接夹爪/回读失败返回 None（上层保持指令值显示）。
        """
        if self._effector is None:
            return None
        try:
            gs = self._effector.get_gripper_status()
        except Exception:  # noqa: BLE001
            return None
        if gs is None or getattr(gs.msg, "mode", "width") != "width":
            return None
        try:
            full_open = self._gripper_open_width_m
            width = float(gs.msg.value)
            if width > full_open * 2.0:   # 毫米单位兼容（55mm 满开下 >0.11 即为 mm）
                width /= 1000.0
            return float(np.clip(width / full_open, 0.0, 1.0))
        except Exception:  # noqa: BLE001
            return None

    @property
    def is_simulation(self) -> bool:
        return False
