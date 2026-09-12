"""交互状态机 v5 单元测试：OK=跟随/待机切换，拇指向上=复位。"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerohandtwin.interaction.interaction_fsm import InteractMode, InteractionFSM


def test_fsm_starts_idle():
    fsm = InteractionFSM()
    assert fsm.mode == InteractMode.IDLE and not fsm.in_follow


def test_ok_toggles_follow():
    fsm = InteractionFSM()
    ev = fsm.feed_gesture("Okay")
    assert ev.detail == "OK->跟随模式" and fsm.in_follow
    ev = fsm.feed_gesture("Okay")
    assert ev.detail == "OK->待机" and fsm.mode == InteractMode.IDLE


def test_thumb_up_resets_to_idle():
    fsm = InteractionFSM()
    fsm.feed_gesture("Okay")  # -> FOLLOW
    ev = fsm.feed_gesture("Thumb_Up")
    assert ev.kind == "reset_request" and fsm.mode == InteractMode.IDLE
    # 待机中也可复位（幂等，回到就绪位）
    fsm.feed_gesture("Thumb_Up")
    assert fsm.mode == InteractMode.IDLE


def test_lost_keeps_mode():
    fsm = InteractionFSM()
    fsm.feed_gesture("Okay")  # -> FOLLOW
    fsm.feed_gesture("Lost")
    assert fsm.in_follow  # 手丢失保持模式


def test_recording_flow_keyboard_only():
    """录制由键盘 r 触发（接口不依赖手势）。"""
    from nerohandtwin.interaction.recorder import TrajectoryRecorder

    rec = TrajectoryRecorder(rate_hz=50)
    rec.start()
    assert rec.recording
    rec.record([0.1, 0, 0.3], 0.5)
    path = rec.stop_and_save(Path(tempfile.mkdtemp()), note="test")
    assert path.exists() and not rec.recording
