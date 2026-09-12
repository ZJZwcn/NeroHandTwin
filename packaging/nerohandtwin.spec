# -*- mode: python ; coding: utf-8 -*-
"""NERO 真机控制台 PyInstaller 打包配置（onedir：启动快、更新依赖无需重打包）。

用法（在项目根目录、miniconda 环境）：
    D:/miniconda3/python.exe -m PyInstaller packaging/nerohandtwin.spec --noconfirm
产物：dist/NeroHandTwin/NeroHandTwin.exe（双击启动，同目录 _internal/ 放资源与依赖）
"""
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent

# _ctypes.pyd / _sqlite3.pyd 的运行期 DLL 依赖（conda 的 Library/bin）不在 PyInstaller
# 默认收集路径内，干净机器上会 "DLL load failed"，这里显式打包进 _internal。
# 注意：spec 内 sys.executable 不可靠，用 base_prefix 定位 conda 根目录
_LIB_BIN = Path(sys.base_prefix) / "Library" / "bin"
binaries = []
for _dll in ("ffi.dll", "ffi-8.dll", "sqlite3.dll"):
    _p = _LIB_BIN / _dll
    if _p.exists():
        binaries.append((str(_p), "."))
    else:
        print(f"[spec] 警告: 未找到 {_p}，冻结包将缺该依赖")

datas = [
    (str(ROOT / "configs"), "configs"),
    (str(ROOT / "models"), "models"),
    (str(ROOT / "src" / "nerohandtwin" / "web" / "static"), "nerohandtwin/web/static"),
]
hiddenimports = [
    "agx_cando",
    "agx_cando.bus",       # 冻结后 python-can 插件入口点丢失，BACKENDS 手动注册用
]

# 无内置 hook 的包：整体收集其非 .py 资源（.task/.binarypb/cando.dll/realsense2.dll）
for pkg in ("mediapipe", "agx_cando", "pyrealsense2"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(ROOT / "scripts" / "run_real.py")],
    pathex=[str(ROOT / "src"), str(ROOT / "third_party" / "pyAgxArm")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        # 真机模式不需要：仿真/可视化/测试/无关大件，排除以减小体积加速启动。
        # matplotlib 可安全排除：mediapipe 顶层虽 import 它，但入口 run_real.py
        # 在冻结时已用桩模块顶替（叠加层走 P5，从不用 matplotlib）
        "mujoco", "matplotlib", "pytest", "IPython", "torch", "tensorflow",
        "PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter", "unittest",
        "sphinx", "jedi", "notebook", "jupyter", "pandas",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeroHandTwin",
    console=True,          # 需要确认输入与运行日志，保留控制台
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="NeroHandTwin")
