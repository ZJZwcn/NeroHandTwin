"""NERO 本体运动学：MDH 正解 + 臂平面 2R 逆解 + 腕部水平协同。

参数与 pyAgxArm.api.constants.ROBOT_MDH_PRESET["nero"] 完全一致（已用
SDK fk() 与官方 MuJoCo 模型双向核对）。本模块离线可用，不依赖 CAN/SDK。

经数值验证的结构（q3 = q5 = 0 时成立，为协同跟随的设计基础）：

- J1 底座旋转整个「臂平面」，平面内水平前向 f = (cos q1, sin q1, 0)；
- J2/J4 构成平面 2R 链（肘前支），法兰位于 (f, z) 竖直平面内：
    肩   S  = (0, 0, d1)
    肘   E  = S + L1·dir(a1),  a1 = π/2 + q2        （L1 = 0.31）
    法兰 W  = E + L2·dir(a2),  a2 = π/2 + q2 + q4   （L2 = 0.27001）
- 法兰 x 轴 = 夹爪进给方向（手指所指），面内角 φ = a2 + q7：
    φ = 0 ⇔ 夹爪水平朝前（开口正对手）⇔ q7 = -a2；
- 法兰 z 轴 = 手指开合方向，q6 = 0 时恒保持水平（与 q2/q4 无关的结构
  不变量）——即夹爪开口永远水平，配合 J1 对方位即「开口与手对准」；
- 法兰相对腕心沿开合方向偏移 -d7 = -0.0235 m（面外 2.35cm，忽略级）；
  夹爪手指滑块原点在法兰前方约 GRIPPER_TIP_M（仅用于安全说明/诊断）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "D1", "L1", "L2", "D7", "GRIPPER_TIP_M",
    "fk_flange", "flange_inplane", "approach_angle", "opening_tilt",
    "solve_level_pose", "LevelIKResult", "fk_chain",
]

# ---------- NERO MDH：(d, a, alpha, theta_offset)，米/弧度 ----------
_MDH = (
    (0.138, 0.0, 0.0, 0.0),
    (0.0, 0.0, np.pi / 2, np.pi),
    (0.31, 0.0, np.pi / 2, np.pi),
    (0.0, 0.0, np.pi / 2, np.pi),
    (0.27001, 0.0, np.pi / 2, np.pi / 2),
    (0.0, 0.0, np.pi / 2, np.pi / 2),
    (0.0235, 0.0, np.pi / 2, 0.0),
)

D1 = 0.138        # 底座 -> 肩轴高度 (m)
L1 = 0.31         # 肩 -> 肘 (m)
L2 = 0.27001      # 肘 -> 法兰 (m)
D7 = 0.0235       # 腕 -> 法兰面外偏移 (m)
GRIPPER_TIP_M = 0.175  # 法兰 -> 手指滑块原点近似 (m，MuJoCo 实测)

_R_MIN = 0.10            # 法兰-肩最小半径（奇异保护，正常安全盒不会触及）
_R_MAX = L1 + L2 - 0.005  # 最大伸展半径


def _link(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    """单连杆 MDH 齐次矩阵（与 SDK mdh_kinematics 逐元素一致）。"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [ca * st, ca * ct, -sa, -sa * d],
        [sa * st, sa * ct, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_flange(q) -> tuple[np.ndarray, np.ndarray]:
    """关节角 (7,) -> 法兰位姿 (pos(3,), R(3,3))，基座系，米/弧度。"""
    qv = np.asarray(q, dtype=float).reshape(7)
    T = np.eye(4)
    for i, (d, a, al, off) in enumerate(_MDH):
        T = T @ _link(d, a, al, qv[i] + off)
    return T[:3, 3].copy(), T[:3, :3].copy()


def flange_inplane(q) -> tuple[float, float]:
    """法兰在臂平面内的坐标 (s, z)：s 沿 f=(cos q1,sin q1,0) 前向，z 高度。"""
    p, _ = fk_flange(q)
    q1 = float(np.asarray(q, dtype=float).reshape(7)[0])
    s = float(p[0] * np.cos(q1) + p[1] * np.sin(q1))
    return s, float(p[2])


def approach_angle(q) -> float:
    """夹爪进给方向（法兰 x 轴）的面内角 φ（弧度）。

    φ=0 水平朝前，φ>0 抬头，φ<0 低头。q3=q5=0 时 φ = π/2 + q2 + q4 + q7。
    """
    _, R = fk_flange(q)
    q1 = float(np.asarray(q, dtype=float).reshape(7)[0])
    f = np.array([np.cos(q1), np.sin(q1), 0.0])
    ax = R[:, 0]
    return float(np.arctan2(ax[2], ax @ f))


def opening_tilt(q) -> float:
    """手指开合方向（法兰 z 轴）偏离水平面的仰角（弧度）。0=开口水平。"""
    _, R = fk_flange(q)
    return float(np.arcsin(np.clip(R[2, 2], -1.0, 1.0)))


@dataclass
class LevelIKResult:
    """solve_level_pose 的结果。"""

    q2: float
    q4: float
    q7: float
    a2: float = 0.0                 # 前臂面内角（诊断用）
    s: float = 0.0                  # 实际解出的法兰面内坐标（半径裁剪后）
    z: float = 0.0
    ok: bool = True                 # 位置与水平朝向均达成
    notes: list[str] = field(default_factory=list)


def solve_level_pose(s_t: float, z_t: float,
                     q2_lim: tuple[float, float] = (-1.70, 0.40),
                     q4_lim: tuple[float, float] = (-1.00, 2.10),
                     q7_lim: tuple[float, float] = (-1.50, 1.50)) -> LevelIKResult:
    """面内 (s, z) -> {q2, q4, q7}：肘前支 2R 逆解 + q7 保持夹爪水平朝前。

    进给方向面内角 φ = a2 + q7，取 φ = 0（水平朝前、开口正对手）得
    q7 = -a2。半径超限时目标沿射线方向收拢到可达域；关节越限时裁剪
    并在 notes 标记（位置/水平度进入降级模式，仍保持安全）。
    """
    notes: list[str] = []
    dx, dz = float(s_t), float(z_t) - D1
    r = float(np.hypot(dx, dz))
    if r < _R_MIN or r > _R_MAX:
        r_c = float(np.clip(r, _R_MIN, _R_MAX))
        dx, dz = dx * r_c / r, dz * r_c / r
        notes.append(f"半径裁剪 {r:.3f}->{r_c:.3f}")
        r = r_c
    gamma = float(np.arctan2(dz, dx))
    cos_beta = float(np.clip((r * r + L1 * L1 - L2 * L2) / (2.0 * r * L1), -1.0, 1.0))
    beta = float(np.arccos(cos_beta))
    a1 = gamma - beta                       # 肘前支（肘在法兰前方/上方）
    e_s, e_z = L1 * np.cos(a1), D1 + L1 * np.sin(a1)
    a2 = float(np.arctan2(dz + D1 - e_z, dx - e_s))

    q2 = a1 - np.pi / 2
    q4 = a2 - a1
    q7 = -a2

    ok = True
    solved = {"q2": q2, "q4": q4, "q7": q7}
    for name, val, lim in (("q2", q2, q2_lim), ("q4", q4, q4_lim), ("q7", q7, q7_lim)):
        clipped = float(np.clip(val, *lim))
        if not np.isclose(clipped, val, atol=1e-9):
            notes.append(f"{name} 越限裁剪")
            ok = False
        solved[name] = clipped

    return LevelIKResult(q2=solved["q2"], q4=solved["q4"], q7=solved["q7"], a2=a2,
                         s=float(dx), z=float(dz + D1), ok=ok, notes=notes)


def fk_chain(q) -> list[np.ndarray]:
    """关节角 -> 逐连杆原点位置链（3D 渲染骨架用）。

    返回 9 个点：[基座原点, J1..J7 各帧原点 (8 个), 夹爪指尖近似]。
    与 fk_flange 同一套 MDH 链，逐元素一致。
    """
    qv = np.asarray(q, dtype=float).reshape(7)
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for i, (d, a, al, off) in enumerate(_MDH):
        T = T @ _link(d, a, al, qv[i] + off)
        pts.append(T[:3, 3].copy())
    p_flange, R = T[:3, 3].copy(), T[:3, :3].copy()
    pts.append(p_flange + R[:, 0] * GRIPPER_TIP_M)  # 指尖近似
    return pts
