"""真机模式入口（硬件到位后使用）。

用法（Windows, USB-CAN）：
    python scripts/run_real.py                # 先就绪序列，再进入手势交互
    python scripts/run_real.py --no-camera    # 仅执行就绪序列做连通性自检
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 冻结（PyInstaller）模式下：资源从解包目录读取；源码运行时用仓库根
if getattr(sys, "frozen", False):
    ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    # 冻结控制台默认 GBK，✓ 等字符会 UnicodeEncodeError，统一转 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # mediapipe.tasks.python.vision 顶层 import matplotlib.pyplot（仅供其 3D 标注
    # 函数用，本项目叠加层走 P5，从不调用）。冻结包排除 matplotlib 以缩小体积、
    # 加速启动，这里用空桩模块顶替，避免 import 即失败
    import types

    for _stub in ("matplotlib", "matplotlib.pyplot"):
        sys.modules.setdefault(_stub, types.ModuleType(_stub))
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
else:
    ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _frozen_selftest() -> int:
    """冻结包完整性自检：只验证重依赖可导入，不碰任何硬件。"""
    import importlib

    checks = [
        ("numpy", "numpy"),
        ("cv2", "cv2"),
        ("yaml", "yaml"),
        ("aiohttp", "aiohttp"),
        ("mediapipe.tasks.python", "mediapipe"),
        ("pyrealsense2", "pyrealsense2"),
        ("pyAgxArm", "pyAgxArm"),
        ("can", "python-can"),
        ("agx_cando.bus", "agx_cando"),
        ("nerohandtwin.web.server", "nerohandtwin"),
    ]
    failed = []
    for mod, label in checks:
        try:
            importlib.import_module(mod)
            print(f"[selftest] {label:12s} OK")
        except Exception as exc:  # noqa: BLE001
            failed.append((label, exc))
            print(f"[selftest] {label:12s} FAIL: {exc}")
    if failed:
        print(f"[selftest] {len(failed)} 项失败，exe 资源不完整")
        return 1
    print("[selftest] 全部通过 ✓")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NERO 真机手势交互（危险，请先读 README）")
    parser.add_argument("--no-camera", action="store_true", help="仅连通性自检（就绪序列后退出）")
    parser.add_argument("--show-cv", action="store_true",
                        help="保留 OpenCV 窗口（默认仅网页控制台）")
    parser.add_argument("--port", type=int, default=8000, help="网页控制台端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--config", default="configs/arm.yaml")
    parser.add_argument("--frozen-selftest", action="store_true",
                        help="冻结包完整性自检（不碰硬件，打包验证用）")
    args = parser.parse_args()

    if args.frozen_selftest:
        return _frozen_selftest()

    import yaml

    cfg_path = ROOT / args.config
    arm_cfg_full = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    backend = arm_cfg_full["arm"]["backend"]
    if backend != "pyagx":
        print(f"错误：{cfg_path} 的 backend={backend}，不是 pyagx。")
        print("真机运行请把 configs/arm_real.example.yaml 复制为 configs/arm.yaml 并确认 CAN 接线。")
        return 1

    print("=" * 60)
    print("真机运行前置检查（全部确认后输入 yes 继续）:")
    print(" 1. 急停手段可用（电源插头/Ctrl+C 电子急停）")
    print(" 2. 机械臂周围 0.6m 内无人与障碍")
    print(" 3. Web 上位机碰撞保护等级已开启")
    print(" 4. 全局速度 percent 已设为低位（建议首次 10）")
    calib = yaml.safe_load((ROOT / "configs" / "calibration.yaml").read_text(encoding="utf-8"))
    cal_state = calib.get("extrinsics", {}).get("calibrated")
    if cal_state == "axes_user":
        print(" 5. 外参: 快速方向标定 ✓（相对跟随可用；绝对跟随可再跑 calibrate_touch.py）")
    elif cal_state == "touch":
        print(" 5. 外参: 对准精标 ✓（绝对跟随可用）")
    else:
        print(" 5. 外参: 未标定——跟随方向可能不对。推荐先跑: python scripts/calibrate_quick.py （30秒，无机械臂）")
    answer = input("确认以上 5 项 [yes/N]> ").strip().lower()
    if answer != "yes":
        print("已取消")
        return 1

    from nerohandtwin.app import NeroGestureApp
    from nerohandtwin.control.arm_interface import create_arm
    from nerohandtwin.sources import CameraHandSource

    arm_cfg = arm_cfg_full["arm"]
    if not Path(arm_cfg.get("model_path", "")).is_absolute() and "model_path" in arm_cfg:
        arm_cfg.pop("model_path", None)  # 真机不需要模型
    arm = create_arm(arm_cfg)

    source = None
    if not args.no_camera:
        cam_cfg = yaml.safe_load((ROOT / "configs" / "camera.yaml").read_text(encoding="utf-8"))
        model = str(ROOT / cam_cfg["tracker"]["model_path"])
        source = CameraHandSource.realsense(model, depth_mode="depth")

    app = NeroGestureApp(ROOT / "configs", obs_source=source, arm=arm,
                         show=bool(args.show_cv) and not args.no_camera)
    # 网页控制台（守护线程，127.0.0.1）
    from nerohandtwin.web.server import start_web_server
    start_web_server(app, host="127.0.0.1", port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[real] 网页控制台: {url}"
          + ("（OpenCV 窗口已保留）" if app.show else "（OpenCV 窗口已关闭）"))
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    try:
        if args.no_camera:
            app.start()
            print("[real] 就绪序列完成，连通性正常。按 Ctrl+C 退出。")
            import time

            while True:
                time.sleep(1)
        else:
            app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
