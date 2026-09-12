"""官方 URDF → 网页数字孪生资产（逐 link STL + 关节树 JSON）。

用法（必须在项目根目录运行）：
    python scripts/convert_urdf_model.py

来源：third_party/agx_arm_sim/agx_arm_description/urdf/nero_gripper_d435.urdf
（官方 NERO+两指夹爪+D435 支架模型，MuJoCo 仿真与真机共用此运动学）。

输出（web/static/model/）：
  link/<link名>.stl   官方逐 link 网格（米单位，link 系坐标）
  urdf.json           关节树：joints(name/type/parent/child/origin/axis/limit)
                      + links(mesh 文件与 visual origin)
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# 项目根 = 当前工作目录（避免任何向上跳目录的路径推导）
ROOT = Path.cwd().resolve()
OUT_DIR = (ROOT / "src" / "nerohandtwin" / "web" / "static" / "model").resolve()
URDF_PATH = (ROOT / "third_party" / "agx_arm_sim" / "agx_arm_description"
             / "urdf" / "nero_gripper_d435.urdf").resolve()
MESH_DIR = (ROOT / "third_party" / "agx_arm_urdf" / "nero" / "meshes").resolve()
ALLOWED_SRC = (ROOT / "third_party").resolve()
# 真机同款 100mm 行程夹爪 CAD（官方 gripper_base.stl 只有 65mm 高，与 URDF 指关节
# 挂点 135.8mm 不符 → 指悬空。用真机夹爪本体网格替换）
GRIPPER_STEP = (ROOT / "zip_model" / "100mm行程夹爪.STEP").resolve()
GRIPPER_BODY_SOLIDS = (0, 2, 3, 4)   # 本体实体（排除指板/垫片/丝杠等运动件）
GRIPPER_FLANGE_Y_MM = 65.0           # STEP 法兰安装面（y+ 端面）
GRIPPER_ROLL_180 = True              # 真机夹爪安装相对 CAD 建模姿态滚转 180°（正反颠倒修正）
GRIPPER_FINGER_SOLIDS = ((9, 10, 11, 12), (5, 6, 7, 8))
# 指板分组（link1=+y 滑动侧 / link2=−y 滑动侧）：滚转 180° 后 ±y 互换，
# 故 link1 取 STEP z<0 侧（9~12），link2 取 z>0 侧（5~8）；板+垫+螺母

# 前端 MDH 正运动学（app.js neroFK）就绪位，用于交叉验证 URDF 关节树
READY_Q = [0.0, -0.6109, 0.0, 2.0071, 0.0, 0.0, 0.2618]
MDH = [
    [0.138, 0, 0, 0],
    [0, 0, math.pi / 2, math.pi],
    [0.31, 0, math.pi / 2, math.pi],
    [0, 0, math.pi / 2, math.pi],
    [0.27001, 0, math.pi / 2, math.pi / 2],
    [0, 0, math.pi / 2, math.pi / 2],
    [0.0235, 0, math.pi / 2, 0],
]


def _safe_out(name: str) -> Path:
    """输出路径白名单校验：只允许写进 OUT_DIR 下的 .stl/.json。"""
    p = (OUT_DIR / name).resolve()
    if not (p.is_relative_to(OUT_DIR) and p.suffix in (".stl", ".json")):
        raise RuntimeError(f"非法输出路径: {p}")
    return p


def _safe_src(name: str) -> Path:
    """源网格路径校验：只允许读 third_party 下的 .stl。"""
    p = (MESH_DIR / name).resolve()
    if not (p.is_relative_to(ALLOWED_SRC) and p.suffix == ".stl" and p.exists()):
        raise RuntimeError(f"非法或缺失的源网格: {p}")
    return p


def write_stl(name: str, tris: np.ndarray) -> None:
    """二进制 STL：tris (n,3,3) float32，米单位。"""
    p = _safe_out(name)
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


def convert_gripper_body() -> bool:
    """真机 100mm 夹爪本体 → gripper_base.stl（gripper_base 系，米）。

    坐标映射（由 URDF 指板位置 0.06~0.1358m 与 STEP 指板位置交叉验证）：
      base_x = step_x        base_y = step_z        base_z = (65 − step_y)/1000
    即 STEP 法兰安装面（y=+65mm）贴合 gripper_base 原点，指沿 base ±y 滑动开合。
    """
    solids = _load_gripper_solids()
    parts = []
    for i, tri in enumerate(solids):
        if i not in GRIPPER_BODY_SOLIDS:
            continue
        parts.append(_to_base_frame(tri))
    if not parts:
        print("[urdf] 夹爪本体实体为空，保留官方 gripper_base.stl", flush=True)
        return False
    all_tris = np.concatenate(parts, axis=0).astype(np.float32)
    write_stl("link/gripper_base.stl", all_tris)
    c = all_tris.reshape(-1, 3).mean(axis=0)
    print(f"[urdf] 夹爪本体替换: {len(all_tris)} tri, 质心 {np.round(c, 4).tolist()}",
          flush=True)
    return True


def _load_gripper_solids() -> list:
    """真机夹爪 STEP 全部实体 → [tri (n,3,3) STEP 系毫米] 列表（未变换）。"""
    from cadquery import importers

    comp = importers.importStep(str(GRIPPER_STEP))
    out = []
    for s in comp.val().Solids():
        verts, tris = s.tessellate(1.5)
        v = np.array([[p.x, p.y, p.z] for p in verts])
        idx = np.asarray(tris, dtype=np.int64)
        if idx.max() == len(v):
            idx -= 1
        out.append(v[idx])
    print(f"[urdf] 100mm 夹爪实体数: {len(out)}", flush=True)
    return out


def _to_base_frame(tri: np.ndarray) -> np.ndarray:
    """STEP 实体（mm，(n,3,3) 三角形）→ base 系（米）：base=[x, z, 65−y]/1000。

    GRIPPER_ROLL_180：绕 base z 轴滚转 180°（x/y 取反），修正真机安装方向
    与 CAD 建模姿态的上下颠倒。
    """
    v = tri.astype(np.float64)
    assert v.ndim == 3 and v.shape[1:] == (3, 3), f"期望 (n,3,3) 三角形，实际 {v.shape}"
    m = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]], dtype=float)  # 列映射 [x, z, −y]
    mapped = v @ m.T * 0.001
    mapped[:, :, 2] += GRIPPER_FLANGE_Y_MM / 1000.0  # base_z = (65 − y)/1000
    if GRIPPER_ROLL_180:
        mapped[:, :, 0] *= -1.0
        mapped[:, :, 1] *= -1.0
    return mapped


def convert_gripper_fingers() -> bool:
    """真机指板 → gripper_link1/2.stl（烘焙到 URDF 棱柱关节系，闭合位）。

    官方 gripper_link*.stl 是短款夹爪的指板，与 100mm 夹爪真实指板位置不符
    （叠加对比错位 1~4cm）。这里用真机 CAD 指板实体：
      1. 平移到闭合位（+y 板内侧面贴 y=0，−y 板对称）——对应 URDF q=0
      2. 逆变换到各指关节系（origin (0,0,0.1358)，rpy(π/2,0,π)/(π/2,0,0)）
    前端棱柱动画沿关节轴 ±0.05m 滑动，即 0~100mm 行程，与 SDK 语义一致。
    """
    solids = _load_gripper_solids()
    t1 = np.array([0.0, 0.0, 0.1358])
    R1 = np.array([[-1.0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]])            # rpy(π/2,0,π) = Rz(π)·Rx(π/2)
    R2 = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])            # rpy(π/2,0,0) = Rx(π/2)
    Ri1, Ri2 = R1.T, R2.T
    ok = True
    for finger, group, R in (
            ("gripper_link1", GRIPPER_FINGER_SOLIDS[0], R1),
            ("gripper_link2", GRIPPER_FINGER_SOLIDS[1], R2)):
        parts = []
        for i, tri in enumerate(solids):
            if i not in group:
                continue
            parts.append(tri)
        if not parts:
            print(f"[urdf] {finger}: 指板实体为空", flush=True)
            ok = False
            continue
        plate = np.concatenate(parts, axis=0)
        plate_base = _to_base_frame(plate)    # STEP 毫米 → base 系（米）
        # 平移到闭合位：+y 板 min_y→0；−y 板 max_y→0（对应 URDF q=0、开口 0mm）
        if finger == "gripper_link1":
            plate_base[:, :, 1] -= plate_base[:, :, 1].min()
        else:
            plate_base[:, :, 1] -= plate_base[:, :, 1].max()
        # base → 关节系（行向量约定：p_local_row = (p−t)_row @ R）
        local = (plate_base - t1) @ R
        write_stl(f"link/{finger}.stl", local.astype(np.float32))
        c = local.reshape(-1, 3).mean(axis=0)
        print(f"[urdf] {finger} 烘焙: {len(local)} tri, 质心 {np.round(c, 4).tolist()}",
              flush=True)
    return ok


def rpy_to_quat(r, p, y):
    """URDF rpy(ZYX 顺序: R=Rz(y)Ry(p)Rx(r)) → 四元数 [x,y,z,w]。"""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy]


def mdh_fk_flange(q):
    T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def mul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
                for i in range(4)]

    for i in range(7):
        d, a, al, off = MDH[i]
        ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(q[i] + off), math.sin(q[i] + off)
        T = mul(T, [[ct, -st, 0, a],
                    [ca * st, ca * ct, -sa, -sa * d],
                    [sa * st, sa * ct, ca, ca * d],
                    [0, 0, 0, 1]])
    return [round(T[i][3], 5) for i in range(3)]


def urdf_fk_flange(joints, q):
    """沿 joint1→joint7 链求 link7 原点位置（URDF 语义）。"""
    T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def mul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
                for i in range(4)]

    def rpy_mat(r, p, y):
        cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                  math.sin(p), math.cos(y), math.sin(y))
        return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr]]

    qi = 0
    for j in joints:
        x, y, z = j["origin"]["xyz"]
        R = rpy_mat(*j["origin"]["rpy"])
        Tj = [[R[0][0], R[0][1], R[0][2], x],
              [R[1][0], R[1][1], R[1][2], y],
              [R[2][0], R[2][1], R[2][2], z],
              [0, 0, 0, 1]]
        if j["type"] == "revolute":
            ax = j["axis"]
            th = q[qi] if qi < len(q) else 0.0
            qi += 1
            n = math.sqrt(sum(a * a for a in ax)) or 1.0
            ux, uy, uz = [a / n for a in ax]
            c, s = math.cos(th), math.sin(th)
            C = 1 - c
            Rz = [[c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
                  [uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s],
                  [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C]]
            Tj = mul(Tj, [[Rz[0][0], Rz[0][1], Rz[0][2], 0],
                          [Rz[1][0], Rz[1][1], Rz[1][2], 0],
                          [Rz[2][0], Rz[2][1], Rz[2][2], 0],
                          [0, 0, 0, 1]])
        T = mul(T, Tj)
    return [round(T[i][3], 5) for i in range(3)]


def main() -> int:
    if not (ROOT / "pyproject.toml").exists():
        print("[urdf] 请在项目根目录运行本脚本", flush=True)
        return 1
    if not URDF_PATH.exists():
        print(f"[urdf] 找不到 URDF: {URDF_PATH}", flush=True)
        return 1

    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    joints = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        axis = j.find("axis")
        limit = j.find("limit")
        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "origin": {
                "xyz": [float(v) for v in (origin.get("xyz", "0 0 0")).split()],
                "rpy": [float(v) for v in (origin.get("rpy", "0 0 0")).split()],
            },
            "axis": [float(v) for v in (axis.get("xyz", "0 0 1")).split()] if axis is not None else [0, 0, 1],
            "limit": ([float(limit.get("lower")), float(limit.get("upper"))]
                      if limit is not None else None),
        })

    links = {}
    for lk in root.findall("link"):
        vis = lk.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        pkg_uri = mesh.get("filename", "")
        fname = pkg_uri.rsplit("/", 1)[-1]          # 如 link2.dae → link2.stl
        stl_name = Path(fname).stem + ".stl"
        if stl_name in ("gripper_link1.stl", "gripper_link2.stl"):
            # 官方短款指板弃用：网格由 convert_gripper_fingers 用真机 CAD 烘焙，
            # 但 manifest 仍需登记 link 条目（前端按关节树建节点）
            vorigin = vis.find("origin")
            links[lk.get("name")] = {
                "mesh": f"link/{stl_name}",
                "mesh_origin": {
                    "xyz": [float(v) for v in (vorigin.get("xyz", "0 0 0")).split()] if vorigin is not None else [0, 0, 0],
                    "rpy": [float(v) for v in (vorigin.get("rpy", "0 0 0")).split()] if vorigin is not None else [0, 0, 0],
                },
            }
            continue
        try:
            src = _safe_src(stl_name)
        except RuntimeError as exc:
            print(f"[urdf] 跳过 {lk.get('name')}: {exc}", flush=True)
            continue
        vorigin = vis.find("origin")
        links[lk.get("name")] = {
            "mesh": f"link/{stl_name}",
            "mesh_origin": {
                "xyz": [float(v) for v in (vorigin.get("xyz", "0 0 0")).split()] if vorigin is not None else [0, 0, 0],
                "rpy": [float(v) for v in (vorigin.get("rpy", "0 0 0")).split()] if vorigin is not None else [0, 0, 0],
            },
        }
        dst = _safe_out(f"link/{stl_name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[urdf] link {lk.get('name'):16s} ← {stl_name}", flush=True)

    # 官方 gripper_base.stl 只有 65mm 高，与 URDF 指关节挂点 0.1358m 不符
    # （真机为 100mm 行程夹爪）——用 zip_model 的真机夹爪 CAD 替换本体网格；
    # 指板网格同样与真机不符（短款），用 CAD 指板烘焙到 URDF 关节系
    try:
        convert_gripper_body()
        convert_gripper_fingers()
    except Exception as exc:  # noqa: BLE001 - 替换失败保留官方网格
        print(f"[urdf] 夹爪网格替换失败（保留官方网格）: {exc}", flush=True)

    # 交叉验证：URDF 关节链 FK vs 前端 MDH FK（就绪位法兰位置应一致）
    p_urdf = urdf_fk_flange([j for j in joints if j["name"].startswith("joint")
                             and j["type"] == "revolute" and j["parent"].startswith("link")
                             or j["name"] == "joint1"][:0] or
                            [j for j in joints if j["name"] in
                             {f"joint{i}" for i in range(1, 8)}], READY_Q)
    p_mdh = mdh_fk_flange(READY_Q)
    print(f"[urdf] URDF FK  法兰: {p_urdf}", flush=True)
    print(f"[urdf] MDH  FK  法兰: {p_mdh}", flush=True)
    dev = max(abs(a - b) for a, b in zip(p_urdf, p_mdh))
    print(f"[urdf] 最大偏差 {dev*1000:.2f} mm" + ("  ✓ 一致" if dev < 0.01 else "  ⚠ 不一致!"),
          flush=True)

    manifest = {
        "root": "base_link",
        "joints": joints,
        "links": links,
        "source": URDF_PATH.name,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _safe_out("urdf.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[urdf] 完成 → {OUT_DIR}（joints={len(joints)}, links={len(links)}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
