"""夹爪独立调试工具（协同测试用）：使能/失能/开合/状态回读 全流程。

用法（机械臂 CAN 已接、夹爪已装）：
    python scripts/test_gripper.py
交互指令：
    e=使能  d=失能  o=全开(100)  c=全闭(0)  <数字>=开合度0~100
    s=读状态(使能状态/宽度/力度)  q=退出
注意：只动夹爪，不动机械臂。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from nerohandtwin.control.pyagx_arm import PyAgxArmController
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "arm.yaml").read_text(encoding="utf-8"))["arm"]
    if cfg.get("backend") != "pyagx":
        print("错误：configs/arm.yaml 的 backend 不是 pyagx")
        return 1

    arm = PyAgxArmController(
        interface=cfg.get("interface", "agx_cando"),
        channel=cfg.get("channel", "0"),
        firmware=cfg.get("firmware", "default"),
        speed_percent=int(cfg.get("speed_percent", 10)),
        stroke_m=float(cfg.get("stroke_m", 0.1)),
    )
    print("连接机械臂 CAN（不使能机械臂运动）...")
    arm.connect()  # 使能臂 + 独立使能夹爪（带回读确认）
    print(f"机械臂使能: {arm._enabled} | 夹爪使能: {arm.gripper_enabled()}")

    print("=" * 50)
    print("指令: e=使能 d=失能 o=全开 c=全闭 <0~100>=开合度 s=状态 q=退出")
    while True:
        try:
            cmd = input("gripper> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "q":
            break
        if cmd == "e":
            ok = arm.enable_gripper()
            print(f"使能结果: {ok}（回读={arm.gripper_enabled()}）")
        elif cmd == "d":
            if arm._effector is not None:
                arm._gripper_send(arm._GRIPPER_CODE_DISABLE)
                time.sleep(0.2)
                print(f"已发失能指令（回读={arm.gripper_enabled()}）")
            else:
                print("无夹爪")
        elif cmd == "o":
            arm.set_gripper(1.0)
            print("已发全开(100)")
        elif cmd == "c":
            arm.set_gripper(0.0)
            print("已发全闭(0)")
        elif cmd.isdigit():
            v = max(0, min(100, int(cmd))) / 100.0
            arm.set_gripper(v)
            print(f"已发开合度 {int(v * 100)}")
        elif cmd == "s":
            gs = arm._effector.get_gripper_status() if arm._effector is not None else None
            if gs is None:
                print("无夹爪状态反馈")
            else:
                f = gs.msg.foc_status
                print(f"使能={f.driver_enable_status} 宽度={gs.msg.value * 1000:.0f}mm "
                      f"力度={gs.msg.force:.2f}N 错误={f.driver_error_status}")
        else:
            print("未知指令")
    print("退出（机械臂与夹爪均保持使能）")
    arm.close(go_home=False)  # 调试工具不回位，保持使能
    return 0


if __name__ == "__main__":
    sys.exit(main())
