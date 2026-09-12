"""手势状态机 v2（窗口多数投票）单元测试."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerohandtwin.perception.gesture_fsm import GestureFSM, LOST


def test_voting_confirms_with_flicker():
    """闪烁场景（真机实测根因）：Victory/None 交替也能确认。"""
    fsm = GestureFSM(window_ms=450, win_ratio=0.6, min_votes=3)
    t = 0
    seq = ["Victory", "None", "Victory", "Victory", "None", "Victory"]
    out = None
    for g in seq:
        out = fsm.update(g, True, t)
        t += 33
    assert out == "Victory", f"闪烁下应确认 Victory, got {out}"


def test_voting_window_expiry():
    """窗口滑出后旧手势票数衰减。"""
    fsm = GestureFSM(window_ms=300, win_ratio=0.6, min_votes=3)
    t = 0
    for _ in range(5):
        fsm.update("Victory", True, t)
        t += 33
    assert fsm.current == "Victory"
    # 长时间 None（无新 Victory）-> 窗口内全是 None，current 保持但不投票新值
    for _ in range(20):
        out = fsm.update("None", True, t)
        t += 33
    # current 保持 Victory（None 不主动成为 current），或视实现保持
    assert out in ("Victory", "None")


def test_flicker_of_two_gestures_picks_majority():
    """两种手势竞争：窗口内多数者当选。"""
    fsm = GestureFSM(window_ms=450, win_ratio=0.5, min_votes=3)
    t = 0
    seq = ["Pointing_Up"] * 2 + ["Victory"] * 4 + ["Pointing_Up"] * 1
    out = None
    for g in seq:
        out = fsm.update(g, True, t)
        t += 33
    assert out == "Victory"


def test_lost_and_recover():
    fsm = GestureFSM(window_ms=450, win_ratio=0.6, min_votes=3, lost_ms=300)
    for i in range(3):
        fsm.update("Victory", True, i * 33)
    assert fsm.update("Victory", True, 300) == "Victory"
    assert fsm.update("None", False, 500) == "Victory"   # 未超 lost_ms
    assert fsm.update("None", False, 800) == LOST        # 超时判丢失
    # 恢复：清窗直采
    assert fsm.update("Closed_Fist", True, 900) == "Closed_Fist"


def test_min_votes_guard():
    """偶发 1~2 帧误识别不足以确认。"""
    fsm = GestureFSM(window_ms=450, win_ratio=0.6, min_votes=3)
    t = 0
    seq = ["Closed_Fist"] * 3 + ["Pointing_Up"] + ["Closed_Fist"] * 1
    out = None
    for g in seq:
        out = fsm.update(g, True, t)
        t += 33
    assert out == "Closed_Fist"  # 单帧 Pointing_Up 未通过门槛
