"""NeroHandTwin 项目一键打包脚本（Windows）。

用法（miniconda 环境）：
    D:/miniconda3/python.exe scripts/build_exe.py
产物：
    dist/NeroHandTwin/NeroHandTwin.exe   （onedir，双击启动，控制台保留确认提示）
可选：--onefile 打单文件（启动更慢、易被杀软误报，不推荐）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyInstaller import __main__ as pyi

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="NeroHandTwin exe 打包")
    parser.add_argument("--onefile", action="store_true", help="单文件模式（不推荐，启动慢）")
    args = parser.parse_args()

    pyi_args = [str(ROOT / "packaging" / "nerohandtwin.spec"), "--noconfirm"]
    if args.onefile:
        pyi_args.append("--onefile")
    print("执行: PyInstaller", " ".join(pyi_args))
    try:
        pyi.run(pyi_args)  # 失败时内部自行退出并打印日志
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        print(f"打包失败（退出码 {code}），请查看上方日志。")
        return code

    exe = ROOT / "dist" / "NeroHandTwin" / "NeroHandTwin.exe"
    if exe.exists():
        # 部署辅助文件随包分发（部署说明 + 诊断启动）
        import shutil

        for aux in ("部署说明.txt", "诊断启动.bat"):
            src = ROOT / "packaging" / aux
            if src.exists():
                shutil.copy2(src, exe.parent / aux)
        print(f"打包成功: {exe}")
        print("运行: 双击 exe（或命令行运行），首次运行需在确认提示输入 yes")
        return 0
    print("警告: 未找到 exe 产物，请检查 dist/ 目录。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
