"""官方 STEP 装配体 → 关节分组 STL（数字孪生用）。

用法（必须在项目根目录运行）：
    python scripts/convert_step_model.py

流程：
  1. cadquery(OCP) 导入官方 NERO 夹爪版 STEP 装配体并整体三角化
  2. 按三角形质心的 z 高度带分组：CAD 为零位直立姿态，
     FK 零位关节位置 z=0.138/0.448/0.718m → 底座→组0、肩段→组2、
     肘段→组4、腕+夹爪→组6（组1/3/5 为共轴关节对，无独立几何）
  3. 输出 nero_group0.stl 至 nero_group6.stl（米单位，零位姿态坐标）

运行时前端用各关节的 T_cur 与 T_zero 的合成矩阵驱动各组，
实现官方模型任意姿态同步。
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

# 项目根 = 当前工作目录（避免任何向上跳目录的路径推导）
ROOT = Path.cwd().resolve()
OUT_DIR = (ROOT / "src" / "nerohandtwin" / "web" / "static" / "model").resolve()
STEP_PATH = (ROOT / "zip_model" / "nero带夹爪以及带灵巧手模型" / "NERO夹爪版外发.STEP").resolve()
N_GROUPS = 7
TOLERANCE_MM = 5.0       # 线性偏差容差：越大网格越粗、文件越小
Z_BANDS = [0.14, 0.45, 0.72]  # 组0|组2 / 组2|组4 / 组4|组6 边界（米）


def _safe_out(name: str) -> Path:
    """输出路径白名单校验：只允许写进 OUT_DIR 下的 .stl/.json。"""
    p = (OUT_DIR / name).resolve()
    if not (p.is_relative_to(OUT_DIR) and p.suffix in (".stl", ".json")):
        raise RuntimeError(f"非法输出路径: {p}")
    return p


def write_stl(path: Path, tris: np.ndarray) -> None:
    """二进制 STL：tris (n,3,3) float32，米单位。"""
    p = _safe_out(path.name)
    with p.open("wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            v1, v2, v3 = t
            normal = np.cross(v2 - v1, v3 - v1)
            ln = float(np.linalg.norm(normal))
            normal = normal / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<9f", *t.ravel()))
            f.write(struct.pack("<H", 0))


def main() -> int:
    from cadquery import importers

    if not (ROOT / "pyproject.toml").exists():
        print("[conv] 请在项目根目录运行本脚本（需要 zip_model/ 与 configs/）", flush=True)
        return 1
    if not STEP_PATH.exists():
        print(f"[conv] 找不到 STEP 文件: {STEP_PATH}", flush=True)
        return 1

    t0 = time.time()
    comp = importers.importStep(str(STEP_PATH))
    shape = comp.val()
    print(f"[conv] 导入 {STEP_PATH.name} 耗时 {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    verts, tris_idx = shape.tessellate(TOLERANCE_MM)
    print(f"[conv] 三角化 {time.time()-t0:.0f}s：顶点 {len(verts)}，三角形 {len(tris_idx)}", flush=True)

    v = np.array([[x.x, x.y, x.z] for x in verts], dtype=np.float64)
    extent = float(v.max() - v.min())
    scale = 0.001 if extent > 5 else 1.0  # CAD 通常毫米单位
    v *= scale
    idx = np.asarray(tris_idx, dtype=np.int64)
    # OCP 顶点索引为 1 起，cadquery tessellate 可能 0 起：按越界情况自适应
    if idx.max() == len(v):
        idx -= 1
    print(f"[conv] 单位缩放 {scale}，索引范围 {idx.min()}~{idx.max()}", flush=True)

    # CAD 轴线与 FK 轴线对齐：用长杆段（0.15<z<0.70，零位直立）顶点均值
    # 估出 CAD 臂轴线在 x/y 方向的偏移，平移回 FK 原点（z 不动）
    shaft = v[(v[:, 2] > 0.15) & (v[:, 2] < 0.70)]
    axis_xy = shaft[:, :2].mean(axis=0) if len(shaft) else np.zeros(2)
    if np.linalg.norm(axis_xy) > 0.005:
        v[:, :2] -= axis_xy
        print(f"[conv] 轴线偏移校正: -{np.round(axis_xy, 4).tolist()}", flush=True)

    t0 = time.time()
    tri_xyz = v[idx]                                    # (n,3,3)
    centroid_z = tri_xyz[:, :, 2].mean(axis=1)          # 每个三角形质心 z
    g_of = np.full(len(tri_xyz), 6, dtype=np.int8)
    g_of[centroid_z < Z_BANDS[2]] = 4
    g_of[centroid_z < Z_BANDS[1]] = 2
    g_of[centroid_z < Z_BANDS[0]] = 0
    # 退化三角形（零面积）丢弃
    area2 = np.linalg.norm(np.cross(tri_xyz[:, 1] - tri_xyz[:, 0],
                                    tri_xyz[:, 2] - tri_xyz[:, 0]), axis=1)
    keep = area2 > 1e-12
    print(f"[conv] 分组完成 {time.time()-t0:.0f}s，退化三角剔除 {int((~keep).sum())}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"groups": [], "model_pose_joints": [0.0] * N_GROUPS,
                "z_bands": Z_BANDS, "tolerance_mm": TOLERANCE_MM, "source": STEP_PATH.name}
    for g in range(N_GROUPS):
        out = _safe_out(f"nero_group{g}.stl")
        sel = (g_of == g) & keep
        if sel.any():
            arr = tri_xyz[sel].astype(np.float32)
            write_stl(out, arr)
            c = arr.reshape(-1, 3).mean(axis=0)
            manifest["groups"].append({"index": g, "file": out.name,
                                       "triangles": int(len(arr)),
                                       "centroid": [round(float(x), 4) for x in c]})
            print(f"[conv] 组{g}: {len(arr)} tri, 质心 {np.round(c,3).tolist()} → {out.name}", flush=True)
        else:
            manifest["groups"].append({"index": g, "file": None, "triangles": 0})
            print(f"[conv] 组{g}: 空", flush=True)
    manifest_path = _safe_out("manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"[conv] 完成 → {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
