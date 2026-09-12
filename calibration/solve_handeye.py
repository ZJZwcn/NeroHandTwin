"""Umeyama 手眼外参求解（独立入口，可直接对已有样本文件求解）。

用法：
    python calibration/solve_handeye.py data/calibration/calib_samples_xxx.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from nerohandtwin.geometry.calib_solver import (  # noqa: E402
    calibration_residuals,
    solve_rigid_transform,
)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    samples = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    pts_cam = [np.asarray(s["pose_camera"])[:3, 3] for s in samples]
    pts_base = [
        np.asarray(s["flange_pose"])[:3, 3] + np.asarray(s.get("board_offset_flange", [0, 0, 0]))
        for s in samples
    ]
    result = solve_rigid_transform(pts_cam, pts_base)
    residuals = calibration_residuals(pts_cam, pts_base, result.transform)
    print(f"点数: {result.num_points}  RMSE: {result.rmse * 1000:.2f} mm")
    print(f"逐点残差(mm): {[round(r * 1000, 2) for r in residuals]}")
    print("R =")
    print(np.round(result.rotation, 5))
    print(f"t = {np.round(result.translation, 4)} (m)")
    print("T_base<-camera =")
    print(np.round(result.transform, 5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
