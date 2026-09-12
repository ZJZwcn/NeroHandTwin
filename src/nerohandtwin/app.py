"""主程序：感知 -> 交互状态机 -> 映射 -> 限速 -> 机械臂 的主循环编排。

数据流（每帧）：
    CameraFrame -> HandTracker -> HandObservation -> GestureFSM(去抖)
      -> InteractionFSM(模式/动作事件) -> Follow 控制器 -> 限速目标点
      -> ArmInterface(pump 限速执行) + 夹爪(动作事件驱动)

交互方案（用户定义 v5）：
    OK               -> 跟随模式 <-> 待机 切换（上电默认待机）
    跟随模式中：臂协同追随手（方位/水平线/前后安全距离），
                 夹爪 1:1 镜像拇指-中指指尖距离（满行程 100mm）
    拇指向上         -> 复位回初始姿态
    键盘兜底: o=切换 c=重新锚定 h=复位 g=点赞动作 f=碰拳动作
              r=录制 y=回放 q=退出
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .control.arm_interface import ArmInterface, create_arm
from .control.safety import SafeStartup, Watchdog
from .endeffector.gripper import GripperModel
from .geometry.transforms import transform_point
from .geometry.workspace import WorkspaceMapper
from .interaction.follow_controller import (
    FollowController,
    GimbalFollowController,
    PointController,
)
from .interaction.interaction_fsm import InteractMode, InteractionEvent, InteractionFSM
from .interaction.recorder import TrajectoryPlayer, TrajectoryRecorder
from .perception.gesture_fsm import GestureFSM
from .perception.hand_tracker import HandObservation
from .ui.visualizer import Visualizer

__all__ = ["NeroGestureApp"]

WIN_NAME = "NeroHand - gesture control"

# 冻结（PyInstaller）模式下：资源根 = 解包目录（只读），可写目录 = exe 所在目录
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
    USER_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    USER_ROOT = PROJECT_ROOT

RESET_DURATION_S = 3.0   # 复位回初始姿态的总时长（随主循环分步插值）
RESET_GRACE_S = 1.2      # OK 切换后吞掉过渡姿态误触发的宽限期


class NeroGestureApp:
    """手势交互主应用。

    Args:
        config_dir: configs 目录路径。
        obs_source: 提供 get(now_ms)->HandObservation 的来源
            （CameraHandSource 或合成源）；None 时只用键盘兜底。
        arm: ArmInterface 实例；None 时按 configs/arm.yaml 创建。
        headless: True 不开窗口不渲染（自动化测试）。
    """

    def __init__(self, config_dir: str | Path, obs_source=None, arm: ArmInterface | None = None,
                 headless: bool = False, show: bool = True):
        config_dir = Path(config_dir)
        self.cfg_arm = yaml.safe_load((config_dir / "arm.yaml").read_text(encoding="utf-8"))
        self.cfg_map = yaml.safe_load((config_dir / "mapping.yaml").read_text(encoding="utf-8"))
        self.cfg_gest = yaml.safe_load((config_dir / "gestures.yaml").read_text(encoding="utf-8"))

        self.headless = headless
        self.show = show and not headless
        self.obs_source = obs_source
        self.arm = arm if arm is not None else create_arm(self.cfg_arm["arm"])
        self.startup = SafeStartup(self.arm, self.cfg_arm["arm"].get("ready_joints"))

        ws_cfg = dict(self.cfg_map["workspace"])
        self.follow_mode = str(ws_cfg.pop("follow_mode", "gimbal"))
        self.workspace = WorkspaceMapper(**ws_cfg)
        calib = yaml.safe_load((config_dir / "calibration.yaml").read_text(encoding="utf-8"))
        self.t_base_cam = np.asarray(calib["extrinsics"]["T_base_camera"], dtype=float)

        speed_cfg = self.cfg_map["speed"]
        filter_cfg = self.cfg_map["filter"]
        self.follow = FollowController(self.workspace, self.t_base_cam, speed_cfg, filter_cfg)
        self.gimbal = GimbalFollowController(
            self.cfg_arm["arm"].get("ready_joints"), self.cfg_map.get("gimbal"))
        self.point = PointController(
            self.workspace, self.t_base_cam, speed_cfg, filter_cfg,
            arrive_tol_m=self.cfg_map["point"]["arrive_tol_m"],
            confirm_s=self.cfg_map["point"]["confirm_s"],
        )
        self.gripper_model = GripperModel(**self.cfg_map["gripper"])

        self.gesture_fsm = GestureFSM(
            **self.cfg_gest["debounce"]
        )
        self.interact = InteractionFSM()
        self.interact.add_listener(self._on_interaction_event)

        self.recorder = TrajectoryRecorder(rate_hz=self.cfg_gest["record"]["rate_hz"])
        self.player: Optional[TrajectoryPlayer] = None
        self.play_speed = float(self.cfg_gest["record"]["play_speed"])
        self.data_dir = USER_ROOT / "data"
        self.last_recording: Optional[Path] = None

        self.viz = Visualizer()
        self.last_frame = None  # 最近相机帧（可视化）
        self.status: list[str] = []
        self._watchdog: Optional[Watchdog] = None
        self._running = False
        self._shutdown_done = False
        self._last_tick: Optional[float] = None
        self._follow_anchored = False   # 进入 FOLLOW 后是否已自动锚定
        self._settle_frames = 0         # 末端静止帧计数（锚定前置条件）
        self._settle_need = 10          # 静止判定帧数（~0.33s @30fps）
        self._last_tcp: Optional[np.ndarray] = None
        self._gripper_opening = 1.0     # 夹爪当前开合度（真机默认全开 100mm）
        self._gripper_opening_fb: Optional[float] = None  # 硬件回读开合度（网页显示优先用）
        self._gripper_ema = 0.5         # 指距 EMA 状态
        self._diag_last = 0.0           # 协同诊断周期打印
        self._now_ms = 0.0              # 最近 run_once 时间戳（宽限期基准）
        self._cmd_grace_until_ms = -1.0  # OK 切换后的手势宽限截止（吞过渡姿态）
        self._resetting = False         # 复位分步进行中
        self._reset_q0: Optional[np.ndarray] = None   # 复位起点关节角
        self._reset_q1: Optional[np.ndarray] = None   # 复位终点（就绪位）
        self._reset_progress = 0.0      # 复位插值进度 0~1
        # ---------- 网页控制台挂钩 ----------
        import queue as _queue

        self.cmd_queue: "_queue.Queue[str]" = _queue.Queue()  # web 命令 → 主循环
        self.web_clients: set = set()                     # 活跃 WS 连接
        self._state = None                                # 每帧 ArmState 缓存（web 只读）
        self.fps = 0.0                                    # FPS EMA（实例属性，web 读取）

    # ---------- 生命周期 ----------
    def start(self) -> None:
        """连接机械臂并执行就绪序列（离奇异点）。"""
        self.arm.connect()
        if not self.startup.is_ready:
            print("[app] 关节空间转移到就绪位（避开零位奇异点）...")
            self.startup.move_to_ready()
        state = self.arm.get_state()
        self.follow.pipeline.sync(state.flange_pose[:3, 3])
        self.point.pipeline.sync(state.flange_pose[:3, 3])
        self._watchdog = Watchdog(
            timeout=float(self.cfg_arm["arm"].get("watchdog_s", 0.5)),
            on_timeout=self.arm.hold,
        )
        self._watchdog.start()
        self._running = True
        print("[app] 就绪（待机）。OK=进入/退出跟随 | 拇指向上=复位回初始姿态 | "
              "键盘: o/c/h/g/f/r/y/q")

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._running = False
        if self._watchdog is not None:
            self._watchdog.stop()
        if self.recorder.recording:
            self.recorder.discard()
        print("[app] 关闭: 机械臂返回就绪位并保持使能（不 disable）")
        self.arm.close(go_home=True)

    # ---------- 主循环 ----------
    def run_once(self, now_ms: float) -> None:
        """单帧处理（可被外部循环驱动，便于测试）。"""
        dt = 0.0 if self._last_tick is None else now_ms / 1000.0 - self._last_tick
        self._last_tick = now_ms / 1000.0
        dt = min(max(dt, 0.0), 0.2)
        self._now_ms = float(now_ms)

        self._drain_commands()  # 网页命令（主循环内执行，控制逻辑线程边界不变）

        obs = self._get_observation(now_ms)
        if self.obs_source is not None:
            stable = self.gesture_fsm.update(obs.gesture, obs.present, now_ms)
            # 仅在手势边沿（切换/经间隔重现）时喂交互层——
            # 投票制下 stable 会持续多帧输出同一手势，边沿语义由 FSM 维护
            if self.gesture_fsm.edge and not self._resetting:
                if stable == "Thumb_Up" and now_ms < self._cmd_grace_until_ms:
                    # OK 切换后松手的过渡姿态易被识别成点赞（手型相近），
                    # 宽限期内不触发复位，防止"一进跟随就被复位踢出"
                    print("[app] 忽略切换过渡期的点赞（防误触发复位）")
                else:
                    self.interact.feed_gesture(stable)
        else:
            stable = "None"  # 纯键盘模式

        state = self.arm.get_state()
        self._state = state  # 缓存供 web 线程只读（避免重复 CAN 查询）
        arm_pos = state.flange_pose[:3, 3]

        self._update_control(obs, stable, arm_pos, dt)
        self._refresh_status(obs, stable, arm_pos)
        self._sync_gripper_feedback()

    def _sync_gripper_feedback(self) -> None:
        """夹爪开度显示值同步硬件回读（真机手动/网页/示教改动都能反映到孪生）。

        与指令值 _gripper_opening 分离：不污染捏合镜像的 EMA 链路；
        后端不支持回读时保持 None（显示退回指令值）。
        """
        fb = self.arm.get_gripper_opening()
        if fb is not None:
            self._gripper_opening_fb = float(np.clip(fb, 0.0, 1.0))

    def _drain_commands(self) -> None:
        """执行网页控制台命令（与 _handle_key 同款动作映射）。

        命令可为字符串（如 "follow_toggle"）或带参数的 dict
        （{"cmd": "set_params", "params": {...}}）。
        """
        while not self.cmd_queue.empty():
            try:
                cmd = self.cmd_queue.get_nowait()
            except Exception:  # noqa: BLE001
                return
            params = None
            if isinstance(cmd, dict):
                params = cmd.get("params")
                cmd = cmd.get("cmd")
            if cmd == "set_params" and isinstance(params, dict):
                self._apply_params(params)
                continue
            if cmd == "follow_toggle":
                self.interact.feed_gesture("Okay")
            elif cmd == "reset":
                self.interact.feed_gesture("Thumb_Up")
            elif cmd == "anchor":
                if self.follow_mode == "gimbal":
                    q_now = np.asarray(self.arm.get_state().joint_angles, dtype=float)
                    self.gimbal.reset(q_now)
                    print("[app] 协同跟随重新锚定（网页触发）")
                else:
                    obs = getattr(self.obs_source, "last_obs", None)
                    if obs is not None and obs.present and obs.palm3d is not None:
                        self._anchor_follow(obs)
            elif cmd == "record":
                self._toggle_record()
            elif cmd == "playback":
                path = self._latest_recording()
                if path is not None:
                    self.player = TrajectoryPlayer.from_file(path, speed=self.play_speed)
                    self.follow.pipeline.limiter.reset(
                        self.arm.get_state().flange_pose[:3, 3])
                    print(f"[app] 回放 {path.name} (x{self.play_speed})")
            elif cmd == "shutdown":
                self._running = False

    def _apply_params(self, params: dict) -> None:
        """网页下发的运行参数（速度/增益），运行时即时生效（不落盘）。

        安全：仅允许数值，速度百分比裁剪到 1~100。
        """
        speed = params.get("speed_percent")
        if speed is not None:
            pct = int(np.clip(float(speed), 1, 100))
            self.arm.set_speed_limit(pct)
            print(f"[app] 全局速度 → {pct}%")
        gimbal = getattr(self, "gimbal", None)
        if gimbal is None:
            return
        applied = []
        for key in ("max_joint_rate", "k_yaw", "k_pitch", "k_depth",
                    "sign_yaw", "sign_pitch", "sign_depth",
                    "dz_u", "dz_v", "dz_z", "z_ref"):
            val = params.get(key)
            if val is None or not hasattr(gimbal, key):
                continue
            cast = int if key.startswith("sign_") else float
            setattr(gimbal, key, cast(val))
            applied.append(f"{key}={getattr(gimbal, key):g}")
        if applied:
            print(f"[app] 跟随参数已更新: {', '.join(applied)}")

    def _get_observation(self, now_ms: float) -> HandObservation:
        if self.obs_source is None:
            return HandObservation(timestamp_ms=int(now_ms))
        return self.obs_source.get(now_ms)

    def _update_control(self, obs: HandObservation, stable: Optional[str],
                        arm_pos: np.ndarray, dt: float) -> None:
        mode = self.interact.mode

        # 复位进行中：每帧推进一小段回位插值（不阻塞主循环），
        # 并屏蔽跟随/回放控制，防止与回位目标打架；结束后自动恢复
        if self._resetting:
            self._step_reset(dt)
            self.arm.pump(dt)
            return

        # 回放（键盘触发）：占用控制权
        if self.player is not None:
            frame = self.player.step(dt)
            if frame is None:
                self.player = None
            else:
                self.arm.set_gripper(frame["gripper"])
                target = self.follow.pipeline.limiter.step(arm_pos, frame["xyz"], dt)
                self.arm.move_cartesian(target)
                self._watchdog.feed()
            self.arm.pump(dt)
            return

        if mode in (InteractMode.FOLLOW, InteractMode.REC) and obs.present:
            if self.follow_mode == "gimbal":
                # 协同跟随（免标定）：J1 对方位、J2/J4 追手的水平线与前后
                # 安全距离（深度测距）、J7 保持夹爪水平朝前、J3/J5/J6 配合
                if not self._follow_anchored:
                    q_now = np.asarray(self.arm.get_state().joint_angles, dtype=float)
                    self.gimbal.reset(q_now)
                    self._follow_anchored = True
                    print("[app] 协同跟随已就位：夹爪水平朝前，对准并追随手的水平线")
                if self.last_frame is not None:
                    hw = self.last_frame.color.shape[:2]
                else:
                    hw = (480, 640)
                q_target = self.gimbal.update(obs, hw, dt)
                if q_target is not None:
                    self.arm.move_joints(q_target)
                    self._watchdog.feed()
                if mode == InteractMode.REC:
                    self.recorder.record(
                        self.arm.get_state().flange_pose[:3, 3], self._gripper_opening)
            else:
                # 笛卡尔跟随（需外参标定）
                if not self._follow_anchored:
                    settled = self._track_settle(arm_pos)
                    if obs.palm3d is not None and settled:
                        self._anchor_follow(obs)
                    elif obs.palm3d is None:
                        self.status = ["跟随待锚定: 深度无效, 手距>0.4m或改善光照"]
                    else:
                        self.status = ["跟随待锚定: 等待末端就位..."]
                if self._follow_anchored:
                    target = self.follow.update(obs, arm_pos, dt, interact_locked=False)
                    if target is not None:
                        self.arm.move_cartesian(target)
                        self._watchdog.feed()
                    if mode == InteractMode.REC:
                        self.recorder.record(arm_pos, self._gripper_opening)
            # 夹爪同步：镜像 拇指尖-中指尖 实测距离（满行程 100mm 1:1）
            # EMA 平滑防深度/识别抖动；距离无效时保持上次开度
            if self._follow_anchored:
                self._gripper_opening = self._gripper_from_hand(obs)
                self.arm.set_gripper(self._gripper_opening)

        # IDLE / 锚定前：臂保持，无运动指令
        self.arm.pump(dt)

    def _track_settle(self, arm_pos: np.ndarray) -> bool:
        """末端静止检测：TCP 连续 _settle_need 帧位移 <2mm 视为就位。"""
        if self._last_tcp is not None and float(np.linalg.norm(arm_pos - self._last_tcp)) < 0.002:
            self._settle_frames += 1
        else:
            self._settle_frames = 0
        self._last_tcp = arm_pos.copy()
        return self._settle_frames >= self._settle_need

    def _gripper_from_hand(self, obs: HandObservation) -> float:
        """跟随模式的夹爪映射：镜像拇指尖-中指尖实测距离。

        拇指与中指指尖张开多少毫米，夹爪就张开多少毫米（0~100mm 满行程
        饱和）；EMA 平滑防抖，深度无效（gap_m=None）时保持上次开度。
        """
        opening = self.gripper_model.gap_to_opening(obs.gap_m)
        if opening is None:
            return self._gripper_ema
        self._gripper_ema = 0.7 * self._gripper_ema + 0.3 * opening
        return self._gripper_ema

    def _anchor_follow(self, obs: HandObservation) -> None:
        """免标定校准：以当前手掌位置为跟随舞台中心。

        absolute 模式：手掌位置 = 舞台中心 → 映射到输出盒中心，
        之后手在舞台内移动、末端 1:1 对应（手到哪臂到哪）。
        relative 模式：以手/臂当前位置为相对零点。
        按 c 可随时重新锚定。
        """
        p_base = transform_point(self.t_base_cam, obs.palm3d)
        arm_pos = self.arm.get_state().flange_pose[:3, 3]
        if self.workspace.mode == "absolute":
            self.workspace.anchor(p_base)
        else:
            self.workspace.reset_relative_origin(p_base, arm_pos)
        self.follow.pipeline.sync(arm_pos)
        self._follow_anchored = True
        print(f"[app] 跟随锚定: 手掌基座系 {np.round(p_base, 2)} 为舞台中心"
              f"（可按 c 重新锚定）")

    def _on_interaction_event(self, ev: InteractionEvent) -> None:
        if ev.kind == "reset_request":
            self._begin_reset()
        elif ev.kind == "mode_change" and ev.detail == "OK->跟随模式":
            self._follow_anchored = False  # 进入跟随：等待末端静止后自动锚定
            self._arm_transition_grace()
        elif ev.kind == "mode_change" and ev.detail == "OK->待机":
            self._arm_transition_grace()
            print("[app] 已切换到待机（臂保持当前位姿）")
        elif ev.detail == "结束录制":
            self._finish_recording("手势结束录制")

    def _arm_transition_grace(self) -> None:
        """OK 切换后开短暂宽限窗：吞掉松手过渡姿态触发的点赞复位。"""
        self._cmd_grace_until_ms = self._now_ms + RESET_GRACE_S * 1000.0

    def _begin_reset(self) -> None:
        """复位手势（拇指向上）：中止一切运动，分步回初始姿态。

        旧实现是同步 for 循环（60 步 × sleep 0.05s），主循环整段冻结
        3s+，相机/识别/渲染全部停摆，用户观感即"一切换就卡一下"。
        改为记录起止关节角，随 run_once 每帧推进一小段插值。
        """
        if self._resetting:
            return
        self.player = None
        self._follow_anchored = False
        self._settle_frames = 0
        state = self.arm.get_state()
        self._reset_q0 = np.asarray(state.joint_angles, dtype=float)
        self._reset_q1 = np.asarray(self.startup.ready_joints, dtype=float)
        self._reset_progress = 0.0
        self._resetting = True
        print("[app] 复位: 返回初始姿态...")

    def _step_reset(self, dt: float) -> None:
        """复位插值推进：约 RESET_DURATION_S 秒匀速回到初始姿态。"""
        self._reset_progress = min(1.0, self._reset_progress + dt / RESET_DURATION_S)
        q = self._reset_q0 + (self._reset_q1 - self._reset_q0) * self._reset_progress
        self.arm.move_joints(q)
        if self._reset_progress >= 1.0:
            self._resetting = False
            state = self.arm.get_state()
            self.follow.pipeline.sync(state.flange_pose[:3, 3])
            self.follow.pipeline.limiter.reset(state.flange_pose[:3, 3])
            print("[app] 复位完成：已回初始姿态（保持使能，比 OK 后继续跟随）")

    def _toggle_record(self) -> None:
        """键盘录制开关（跟随过程中记录末端位姿流）。"""
        if self.recorder.recording:
            self._finish_recording("键盘结束录制")
        else:
            self.recorder.start()
            print("[app] 开始录制轨迹")

    # ---------- 状态显示 ----------
    def _refresh_status(self, obs: HandObservation, stable: Optional[str],
                        arm_pos: np.ndarray) -> None:
        mode = self.interact.mode
        gap_mm = "--" if obs.gap_m is None else f"{obs.gap_m * 1000:.0f}"
        lines = [f"[{mode.value}] tcp=[{arm_pos[0]:+.2f} {arm_pos[1]:+.2f} {arm_pos[2]:+.2f}]m "
                 f"夹爪={self._gripper_opening:.0%}({self._gripper_opening * 100:.0f}mm) "
                 f"指距={gap_mm}mm"]
        if self.obs_source is not None:
            if obs.present:
                if self.follow_mode == "gimbal" and self.interact.in_follow:
                    eu, ev, z_err = self.gimbal.err
                    lines.append(f"伺服: eu={eu:+.2f} ev={ev:+.2f} 深度误差={z_err:+.2f}m")
                if obs.palm3d is not None:
                    lines.append(f"手(相机)=[{obs.palm3d[0]:+.2f} {obs.palm3d[1]:+.2f} {obs.palm3d[2]:+.2f}]")
                else:
                    lines.append("深度无效: 手离远些(>0.4m)/改善光照")
                pend = self.gesture_fsm.pending
                if pend:
                    lines.append(f"确认中: {pend} {self.gesture_fsm.pending_progress:.0%}")
            else:
                lines.append("无手")
            if stable:
                lines.append(f"手势={stable}")
        lines.append("OK=跟随/待机切换 拇指向上=复位 C=重新锚定")
        self.status = lines

        # 协同诊断：每 2s 打印感知链路状态（定位跟随/锚定问题的环节）
        now_wall = time.monotonic()
        if self.obs_source is not None and now_wall - self._diag_last > 2.0:
            self._diag_last = now_wall
            if obs.present and obs.landmarks_px is not None:
                extra = ""
                if self.follow_mode == "gimbal" and self.interact.in_follow:
                    eu, ev, z_err = self.gimbal.err
                    extra = f" eu={eu:+.2f} ev={ev:+.2f} z_err={z_err:+.2f}"
                palm = [round(float(x), 2) for x in obs.palm3d] \
                    if obs.palm3d is not None else [0.0, 0.0, 0.0]
                gap = "--" if obs.gap_m is None else f"{obs.gap_m * 1000:.0f}mm"
                print(f"[diag] 手(相机系)={palm} 指距={gap} raw={obs.gesture} stable={stable} "
                      f"模式={self.interact.mode.value} "
                      f"锚定={'是' if self._follow_anchored else '否'}{extra}")
            elif obs.present:
                print(f"[diag] 手在视野但深度无效 raw={obs.gesture} "
                      f"（手距>0.4m/光照/背景检查）模式={self.interact.mode.value}")
            else:
                print(f"[diag] 未检测到手 模式={self.interact.mode.value}")

    def _finish_recording(self, reason: str) -> None:
        if not self.recorder.recording:
            return
        if len(self.recorder.frames) < 5:
            self.recorder.discard()
            print(f"[app] ({reason}) 轨迹太短，已丢弃")
            return
        self.last_recording = self.recorder.stop_and_save(self.data_dir, note=reason)
        print(f"[app] 已保存 {self.last_recording.name} "
              f"({self.recorder.duration_s:.1f}s / {len(self.recorder.frames)} 帧)")

    def _latest_recording(self) -> Optional[Path]:
        if self.last_recording is not None and self.last_recording.exists():
            return self.last_recording
        files = sorted(self.data_dir.glob("trajectory_*.json"))
        return files[-1] if files else None

    # ---------- 交互式运行 ----------
    def run(self) -> None:
        """交互式主循环（OpenCV 窗口可选 + 键盘兜底）。"""
        import cv2

        self.start()
        try:
            while self._running:
                t0 = time.monotonic()
                frame = None
                if self.obs_source is not None and hasattr(self.obs_source, "last_frame"):
                    frame = self.obs_source.last_frame

                self.run_once(time.monotonic() * 1000)

                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(time.monotonic() - t0, 1e-4))
                if self.show:
                    self._render_frame(frame, self.fps)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)
                    # 点窗口 X 关闭也要走完整的退出流程（回就绪位）
                    try:
                        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            print("[app] 窗口已关闭")
                            self._running = False
                    except cv2.error:
                        self._running = False
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _render_frame(self, frame, fps: float) -> None:
        import cv2

        obs = getattr(self.obs_source, "last_obs", None)
        cam_img = None
        follow_anchor = None
        if self.follow_mode == "gimbal" and self.interact.in_follow:
            follow_anchor = self.gimbal.anchor_uv
        if frame is not None:
            img = frame.color if hasattr(frame, "color") else frame  # CameraFrame | ndarray
            cam_img = self.viz.draw(
                img, obs, self.interact.mode, self.status, fps,
                follow_anchor=follow_anchor,
            )
        sim_img = self.arm.render() if hasattr(self.arm, "render") else None
        if sim_img is not None:
            sim_img = sim_img[:, :, ::-1]  # RGB->BGR
        combined = self.viz.compose(cam_img, sim_img)
        cv2.imshow(WIN_NAME, combined)

    def _handle_key(self, key: int) -> None:
        import cv2

        if key == ord("q"):
            self._running = False
            cv2.destroyAllWindows()
        elif key == ord("o"):               # OK：跟随/待机模式切换
            self.interact.feed_gesture("Okay")
        elif key == ord("c"):               # 手动重新锚定
            if self.follow_mode == "gimbal":
                q_now = np.asarray(self.arm.get_state().joint_angles, dtype=float)
                self.gimbal.reset(q_now)
                print("[app] 协同跟随重新锚定：以当前手部为参考（下一帧生效）")
            else:
                obs = getattr(self.obs_source, "last_obs", None)
                if obs is not None and obs.present and obs.palm3d is not None:
                    self._anchor_follow(obs)
                else:
                    print("[app] 锚定失败: 无有效手部 3D")
        elif key == ord("h"):               # 复位回初始姿态
            self.interact.feed_gesture("Thumb_Up")
        elif key == ord("r"):               # 录制开关
            self._toggle_record()
        elif key == ord("y"):               # 回放
            path = self._latest_recording()
            if path is None:
                print("[app] 没有可回放的轨迹")
            else:
                self.player = TrajectoryPlayer.from_file(path, speed=self.play_speed)
                self.follow.pipeline.limiter.reset(self.arm.get_state().flange_pose[:3, 3])
                print(f"[app] 回放 {path.name} (x{self.play_speed})")
