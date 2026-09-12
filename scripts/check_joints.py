"""关节方向自检（协同调试）：逐关节小幅摆动，验证真机方向与用户感知表一致。

用户感知语义表（面对机械臂、以夹爪开口方向为准）：
    1舵: 负=向右旋转 / 正=向左旋转（带动整个臂）
    2舵: 负=向后走   / 正=向前走（肩，前后）
    3舵: 负=向左旋转 / 正=向右旋转（腕自转）
    4舵: 负=向上动   / 正=向前下（肘）
    5舵: 负=向左旋转 / 正=向右旋转（腕自转）
    6舵: 负=向后仰   / 正=向前俯（腕俯仰）
    7舵: 自转

用法：python scripts/check_joints.py
机械臂会逐个关节 +0.3rad 再回到原位（速度 10%），每关节打印
系统 FK 观测到的 TCP 位移方向。请目视比对每一项是否与上表一致，
把不一致的关节号告诉我（例如 "2 反了"），即可通过配置翻转。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _describe_tcp(d: np.ndarray) -> str:
    """基座系位移向量 -> 人可读方向（基座系 x+ 前向工作区 / y+ 臂左 / z+ 上）。"""
    parts = []
    if abs(d[0]) > 0.01:
        parts.append(f"{'向前(朝工作区)' if d[0] > 0 else '向后(离工作区)'} {abs(d[0])*1000:.0f}mm")
    if abs(d[1]) > 0.01:
        parts.append(f"{'向机械臂左' if d[1] > 0 else '向机械臂右'} {abs(d[1])*1000:.0f}mm")
    if abs(d[2]) > 0.01:
        parts.append(f"{'向上' if d[2] > 0 else '向下'} {abs(d[2])*1000:.0f}mm")
    return "、".join(parts) if parts else "几乎不动（自转关节）"


def main() -> int:
    import yaml
    from nerohandtwin.control.arm_interface import create_arm
    from nerohandtwin.control.safety import SafeStartup

    cfg = yaml.safe_load((ROOT / "configs" / "arm.yaml").read_text(encoding="utf-8"))["arm"]
    print("[arm] 连接并到就绪位...")
    arm = create_arm(cfg)
    arm.connect()
    startup = SafeStartup(arm, cfg.get("ready_joints"))
    if not startup.is_ready:
        startup.move_to_ready()
    time.sleep(1.0)

    table = {
        1: "1舵 正=向左旋转（整体）【已实测确认】",
        2: "2舵 正=向后走/收臂【已实测确认，注意与直觉'正=前'相反】",
        3: "3舵 正=向右旋转（腕自转）【已实测确认】",
        4: "4舵 正=向前下【已实测确认】",
        5: "5舵 正=向右旋转（腕自转）【已实测确认】",
        6: "6舵 正=夹爪向右转（姿态关节，TCP 位置不动）【已实测确认】",
        7: "7舵 正=夹爪向下摆（姿态关节，TCP 位置不动）【已实测确认】",
    }
    amp = 0.3
    mismatch = []
    try:
        for j in range(7):
            state = arm.get_state()
            q0 = np.asarray(state.joint_angles, dtype=float).copy()
            p0 = arm.forward_kinematics(q0)[:3, 3]
            input(f"\n[{j+1}/7] {table[j+1]} —— 回车后关节{j+1}将正向摆动 {amp:.2f}rad 并返回")
            q = q0.copy()
            q[j] += amp
            arm.move_joints(q.tolist())
            time.sleep(2.0)
            p1 = arm.forward_kinematics(arm.get_state().joint_angles.tolist())[:3, 3]
            q_mid = np.asarray(arm.get_state().joint_angles, dtype=float)
            actual = q_mid[j] - q0[j]
            arm.move_joints(q0.tolist())
            time.sleep(2.0)
            d = p1 - p0
            print(f"    系统 FK 观测: 关节{j+1} 实际变化 {actual:+.2f}rad, "
                  f"TCP {_describe_tcp(d)}")
            ans = input("    与你观察到的一致吗? [y/n]> ").strip().lower()
            if ans == "n":
                mismatch.append(j + 1)
        print("\n" + "=" * 50)
        if mismatch:
            print(f"方向不一致的关节: {mismatch}")
            print("把这份清单告诉我，将按关节翻转配置修正（不影响机械臂固件）。")
        else:
            print("全部关节方向一致 ✓ 跟随方向问题应源于外参，请重跑 calibrate_touch.py")
    except KeyboardInterrupt:
        pass
    finally:
        print("[arm] 退出: 回就绪位并保持使能")
        arm.close(go_home=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
