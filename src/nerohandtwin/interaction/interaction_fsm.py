"""交互模式与手势→动作的顶层状态机 v5。

模式（用户最终方案）：
    FOLLOW  跟随模式：协同多关节跟随（手到哪臂到哪，夹爪镜像指距）
    IDLE    待机：臂保持当前位姿（回初始姿态后停在此模式）

手势映射：
    Okay        -> FOLLOW <-> IDLE 切换开关（上电默认 IDLE）
    Thumb_Up    -> 复位：回初始姿态（保持使能，停在 IDLE）
    ILoveYou    -> 回放最近一次录制
    手部丢失    -> 保持当前位姿

设计要点：
- OK 为边沿触发的二态开关；
- 点赞/碰拳动作编排（点头/摇头）与 ACTION 模式已按用户要求移除；
- 上电默认 IDLE（臂保持就绪位，OK 进入跟随）；
- 进入/退出跟随时 app 层负责锚定重置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

__all__ = ["InteractMode", "InteractionFSM", "InteractionEvent"]

LOST_G = "Lost"
NONE_G = "None"


class InteractMode(Enum):
    IDLE = "IDLE"          # 待机：臂保持当前位姿
    FOLLOW = "FOLLOW"      # 跟随模式：协同多关节跟随 + 夹爪镜像指距
    REC = "REC"            # 跟随 + 录制中
    PLAY = "PLAY"          # 回放中


@dataclass
class InteractionEvent:
    """状态切换/动作触发事件，供 app 层执行副作用。"""

    kind: str  # "mode_change" | "play_request" | "reset_request"
    from_mode: InteractMode
    to_mode: InteractMode
    detail: str = ""


@dataclass
class InteractionFSM:
    """手势驱动的交互模式状态机（去抖由 GestureFSM 在上游完成）。"""

    mode: InteractMode = InteractMode.IDLE
    _listeners: list = field(default_factory=list, repr=False)

    def add_listener(self, cb: Callable[[InteractionEvent], None]) -> None:
        self._listeners.append(cb)

    @property
    def in_follow(self) -> bool:
        """是否处于跟随（含录制）。"""
        return self.mode in (InteractMode.FOLLOW, InteractMode.REC)

    def feed_gesture(self, gesture: str) -> Optional[InteractionEvent]:
        """输入稳定手势（app 层已在 FSM 边沿时调用），返回触发的事件。"""
        ev = self._transition(gesture)
        if ev is not None:
            for cb in self._listeners:
                try:
                    cb(ev)
                except Exception:  # noqa: BLE001 - 监听器异常不影响状态机
                    pass
        return ev

    def _transition(self, gesture: str) -> Optional[InteractionEvent]:
        cur = self.mode

        # OK：跟随 <-> 待机 切换开关
        if gesture == "Okay":
            if cur == InteractMode.IDLE:
                return self._set(InteractMode.FOLLOW, detail="OK->跟随模式")
            if cur in (InteractMode.FOLLOW, InteractMode.REC):
                return self._set(InteractMode.IDLE, detail="OK->待机")
            return None  # 回放中忽略

        if cur == InteractMode.PLAY:
            return None

        if gesture == "Thumb_Up":
            # 复位：回初始姿态（app 层执行，模式停在 IDLE）
            return self._set(InteractMode.IDLE, kind="reset_request",
                             detail="复位请求")

        if gesture == "ILoveYou":
            if cur in (InteractMode.IDLE, InteractMode.FOLLOW):
                self.mode = InteractMode.PLAY
                return InteractionEvent("play_request", cur, InteractMode.PLAY, "回放请求")
            return None

        return None

    def _set(self, new_mode: InteractMode, detail: str = "",
             kind: str = "mode_change") -> Optional[InteractionEvent]:
        if new_mode == self.mode and kind == "mode_change":
            return None
        ev = InteractionEvent(kind, self.mode, new_mode, detail)
        self.mode = new_mode
        return ev
