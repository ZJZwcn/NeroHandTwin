"""手势去抖状态机 v2：时间窗多数投票（抗识别闪烁、适配低帧率）。

v1 的缺陷：要求同一手势【连续】持续 stable_ms，真实识别在 13fps 下
Victory/None 交替闪烁时计时器不断重置，手势永远无法确认（真机实测）。

v2 设计：维护最近 window_ms 的帧窗口，某手势票数 ≥ min_votes 且占窗内
比例 ≥ win_ratio 即确认为当前稳定手势。单帧闪烁不影响累计票数。

- "None"（手在但无手势）参与窗口稀释但不主动成为 current；
- Lost 逻辑不变：present=False 超 lost_ms 判丢失，恢复时清窗直采。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

__all__ = ["GestureFSM"]

LOST = "Lost"
NONE_GESTURE = "None"


class GestureFSM:
    """窗口多数投票手势确认。

    用法：
        fsm = GestureFSM(window_ms=450, win_ratio=0.6, min_votes=3, lost_ms=300)
        stable = fsm.update(raw_gesture, present=True, now_ms=now)
    """

    def __init__(
        self,
        window_ms: float = 450.0,
        win_ratio: float = 0.6,
        min_votes: int = 3,
        lost_ms: float = 300.0,
        episode_none_frames: int = 6,
        **_ignored,
    ):
        self.window_ms = float(window_ms)
        self.win_ratio = float(win_ratio)
        self.min_votes = int(min_votes)
        self.lost_ms = float(lost_ms)
        self.episode_none_frames = int(episode_none_frames)
        self.current: str = NONE_GESTURE
        self.edge: bool = False          # 本次 update 手势边界事件（切换/重新出现）
        self._window: deque = deque()  # (ts_ms, gesture)
        self._pending: Optional[str] = None
        self._pending_progress: float = 0.0
        self._last_seen_ms: float = 0.0
        self._none_run: int = 0          # 连续 None 帧数（episode 结束检测）
        self._episode_ended: bool = False
        self.lost: bool = True

    @property
    def pending(self) -> Optional[str]:
        """当前投票领先但尚未确认的手势（可视化用）。"""
        return self._pending

    @property
    def pending_progress(self) -> float:
        """领先手势的确认进度 0~1（上次 update() 快照）。"""
        return self._pending_progress

    def reset(self) -> None:
        self._window.clear()
        self._pending = None
        self._pending_progress = 0.0
        self.current = NONE_GESTURE
        self.lost = True

    def update(self, raw_gesture: str, present: bool, now_ms: float) -> str:
        """输入本帧原始手势，返回投票确认后的稳定手势。"""
        now_ms = float(now_ms)
        self.edge = False  # 每帧重置：edge 只在确认瞬间为 True

        if not present:
            if self.current != LOST and now_ms - self._last_seen_ms > self.lost_ms:
                self._force(LOST)
            return self.current

        self._last_seen_ms = now_ms
        self.lost = False

        # 时间回退防御：now_ms 必须单调（时钟异常/测试时序交叠时保护窗口）
        if self._window and now_ms < self._window[-1][0]:
            self._window.clear()  # 回退即清窗，重新累计

        # 从 Lost 恢复：清窗直采，避免恢复瞬间残留旧状态/等待凑票
        if self.current == LOST:
            self._window.clear()
            self._force(raw_gesture)
            if raw_gesture != NONE_GESTURE:
                return self.current
            # 手在视野但无手势（None）：清窗后正常走投票累计

        # 滑动窗口
        self._window.append((now_ms, raw_gesture))
        while self._window and now_ms - self._window[0][0] > self.window_ms:
            self._window.popleft()

        # 多数投票：None（手在但无手势）是识别噪声，不进分母、不当选——
        # 否则手势间隙帧会稀释票仓，低帧率+闪烁下永远无法确认（真机实测）
        votes: dict[str, int] = {}
        n_votes = 0
        for _, g in self._window:
            if g == NONE_GESTURE:
                continue
            votes[g] = votes.get(g, 0) + 1
            n_votes += 1
        best_g: Optional[str] = None
        best_c = 0
        for g, c in votes.items():
            if c > best_c:
                best_g, best_c = g, c

        if best_g is not None and best_g != self.current and \
                best_c >= self.min_votes and best_c / max(n_votes, 1) >= self.win_ratio:
            # 新稳定手势确认：手势切换边沿
            self.current = best_g
            self._pending = None
            self._pending_progress = 1.0
            self._none_run = 0
            self._episode_ended = False
            self.edge = True
            return self.current

        # 同一手势经 None 间隔后重新确认（episode 结束+重现）→ 也算新边沿
        if best_g is not None and best_g == self.current and self._episode_ended and \
                best_c >= self.min_votes:
            self._episode_ended = False
            self._none_run = 0
            self.edge = True
            self._pending = None
            self._pending_progress = 1.0
            return self.current

        # None 连续帧：标记 episode 结束（下一次该手势多票时触发重现边沿）
        if raw_gesture == NONE_GESTURE:
            self._none_run += 1
            if self._none_run >= self.episode_none_frames and not self._episode_ended:
                self._episode_ended = True
                self._window.clear()  # 丢弃旧票：重现必须由新帧重新凑票
        else:
            self._none_run = 0

        # 未确认：更新 UI 进度信息
        if best_g is not None and best_g != self.current:
            self._pending = best_g
            self._pending_progress = min(1.0, best_c / max(self.min_votes, 1))
        else:
            self._pending = None
            self._pending_progress = 0.0
        return self.current

    def _force(self, gesture: str) -> str:
        self.current = gesture
        self._pending = None
        self._pending_progress = 0.0
        if gesture == LOST:
            self.lost = True
            self._window.clear()  # 丢失后清窗，恢复时重新累计
        return self.current
