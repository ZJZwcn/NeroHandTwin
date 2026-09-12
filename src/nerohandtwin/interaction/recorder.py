"""轨迹录制与回放：跟随过程中记录末端位姿流，回放时复现。

格式（JSON）：
    {
      "meta": {"fps": 50, "duration_s": 8.3, "created": "...", "note": "..."},
      "frames": [ {"t": 0.0, "xyz": [...], "gripper": 0.5}, ... ]
    }
回放以 speed 倍速推进时间轴，帧间线性插值；支持循环外中断。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

__all__ = ["TrajectoryRecorder", "TrajectoryPlayer"]


class TrajectoryRecorder:
    """按固定频率记录 (t, xyz, gripper)。"""

    def __init__(self, rate_hz: float = 50.0):
        self.rate_hz = float(rate_hz)
        self.frames: list[dict] = []
        self._t0: Optional[float] = None
        self._last_t: float = -1.0

    @property
    def recording(self) -> bool:
        return self._t0 is not None

    @property
    def duration_s(self) -> float:
        return self.frames[-1]["t"] if self.frames else 0.0

    def start(self) -> None:
        self.frames = []
        self._t0 = time.monotonic()
        self._last_t = -1.0

    def record(self, xyz, gripper: float) -> None:
        """记录一帧；内部按 rate_hz 限频（调用方可达 30Hz 也只记需要的量）。"""
        if self._t0 is None:
            return
        t = time.monotonic() - self._t0
        if t - self._last_t < 1.0 / self.rate_hz:
            return
        self._last_t = t
        self.frames.append(
            {"t": round(t, 4), "xyz": [float(v) for v in xyz], "gripper": float(gripper)}
        )

    def stop_and_save(self, out_dir, note: str = "") -> Path:
        """结束录制并保存 JSON，返回文件路径。"""
        if self._t0 is None:
            raise RuntimeError("录制未开始")
        self._t0 = None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"trajectory_{stamp}.json"
        payload = {
            "meta": {
                "rate_hz": self.rate_hz,
                "duration_s": self.duration_s,
                "created": stamp,
                "note": note,
            },
            "frames": self.frames,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def discard(self) -> None:
        self._t0 = None
        self.frames = []


class TrajectoryPlayer:
    """按时间轴迭代回放帧（速度倍速、可选循环）。"""

    def __init__(self, frames: list[dict], speed: float = 1.0, loop: bool = False):
        if not frames:
            raise ValueError("轨迹为空，无法回放")
        self.frames = sorted(frames, key=lambda f: f["t"])
        self.speed = float(speed)
        self.loop = bool(loop)
        self.finished = False
        self._idx = 0
        self._t_play: float = 0.0

    @classmethod
    def from_file(cls, path: str | Path, speed: float = 1.0, loop: bool = False) -> "TrajectoryPlayer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["frames"], speed=speed, loop=loop)

    def total_duration(self) -> float:
        return self.frames[-1]["t"]

    def step(self, dt: float) -> Optional[dict]:
        """推进 dt 秒（已被 speed 缩放），返回当前帧 {xyz, gripper}；结束返回 None。"""
        if self.finished:
            return None
        self._t_play += max(dt, 0.0) * self.speed
        t = self._t_play
        while True:
            if t >= self.frames[self._idx]["t"]:
                if self._idx + 1 >= len(self.frames):
                    if self.loop:
                        self._idx = 0
                        self._t_play = 0.0
                        return self._interp(0, 0.0)
                    self.finished = True
                    return self._interp(len(self.frames) - 1, 0.0)
                break
            if self._idx == 0:
                break
            self._idx -= 1
        while self._idx + 1 < len(self.frames) and t >= self.frames[self._idx + 1]["t"]:
            self._idx += 1
        if self._idx + 1 >= len(self.frames):
            return self._last()
        span = self.frames[self._idx + 1]["t"] - self.frames[self._idx]["t"]
        alpha = 0.0 if span <= 0 else (t - self.frames[self._idx]["t"]) / span
        return self._interp(self._idx, alpha)

    def _interp(self, i: int, alpha: float) -> dict:
        a = self.frames[i]
        if i + 1 >= len(self.frames) or alpha <= 0:
            return {"xyz": np.array(a["xyz"]), "gripper": a["gripper"]}
        b = self.frames[i + 1]
        xyz = (1 - alpha) * np.array(a["xyz"]) + alpha * np.array(b["xyz"])
        return {"xyz": xyz, "gripper": (1 - alpha) * a["gripper"] + alpha * b["gripper"]}

    def _last(self) -> dict:
        a = self.frames[-1]
        return {"xyz": np.array(a["xyz"]), "gripper": a["gripper"]}


def iter_playback(player: TrajectoryPlayer, dt: float) -> Iterator[dict]:
    """便捷生成器：按固定 dt 推进到结束。"""
    while not player.finished:
        frame = player.step(dt)
        if frame is None:
            break
        yield frame
