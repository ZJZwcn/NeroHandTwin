# NeroHandTwin

> **NERO 机械臂手势交互与手部跟随数字孪生系统**
>
> 基于 **AgileX Robotics 松灵 NERO 七自由度机械臂**、**Intel RealSense D435i** 与 **MediaPipe** 构建的视觉手势交互与协同跟随系统，支持 **MuJoCo 仿真、网页控制台、轨迹录制/回放、3D 数字孪生、实时遥测与安全控制抽象**。

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-000000?style=flat-square)](https://mujoco.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/guide)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square)](LICENSE)

---

## 目录

- [项目概览](#项目概览)
- [核心能力](#核心能力)
- [系统工作流](#系统工作流)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [网页控制台](#网页控制台)
- [手势与交互](#手势与交互)
- [运行参数](#运行参数)
- [项目结构](#项目结构)
- [测试与验收](#测试与验收)
- [真机部署](#真机部署)
- [离线打包](#离线打包)
- [依赖与模型](#依赖与模型)
- [安全设计](#安全设计)
- [许可证与第三方资产](#许可证与第三方资产)

---

## 项目概览

NeroHandTwin 是一个面向 **视觉手势交互 + 机械臂协同跟随 + 数字孪生** 的工程化原型系统。

系统以固定外置视角（**eye-to-hand**）的 RealSense D435i 采集 RGB 与深度信息，通过 MediaPipe 获取手部关键点与手势，再完成 3D 位置估计、坐标变换、工作空间映射和安全裁剪，最终生成机械臂跟随目标与夹爪控制指令。

当前阶段机械臂尚未到位，因此项目首先在 **MuJoCo 官方 NERO 模型**中完成完整链路验证；控制层使用统一的 `ArmInterface` 抽象，后续可通过替换后端切换到 `pyAgxArm` 真机控制，并结合手眼标定完成实际部署。  

### 当前状态

| 能力 | 状态 |
|---|---|
| 手势识别 | ✅ 已实现 |
| MuJoCo 仿真闭环 | ✅ 已验证 |
| Web 控制台 | ✅ 已实现 |
| 3D 数字孪生 | ✅ 已实现 |
| 轨迹录制 / 回放 | ✅ 已实现 |
| 安全限速 / 手丢失保持 | ✅ 已实现 |
| D435i 真机输入 | ✅ 已提供测试入口 |
| NERO 真机控制 | ⏳ 等待硬件到位与联调 |
| pyAgxArm 真机后端 | ✅ 已完成控制层抽象，待硬件联调 |

---

## 核心能力

### 1. 手势驱动的模式切换

- **OK**：切换“跟随模式 / 待机模式”，系统上电默认待机。
- **拇指向上**：回到初始姿态，并保持使能后停在待机状态。
- **ILoveYou**：回放最近一次录制轨迹。
- **手离开视野 / 深度无效**：保持当前位姿，避免无效观测驱动机械臂继续运动。

### 2. 三轴协同跟随

跟随模式下，根据手部运动生成机械臂协同运动：

- 手向左 / 右移动 → J1 转动，对准手部方向；
- 手向上 / 下移动 → 调整机械臂姿态，使夹爪保持在相近水平线；
- 手向远 / 近移动 → 前伸 / 后缩，保持安全参考距离。

同时，系统通过逆解与限速模块生成连续目标流，当前设计上限为 **≤ 80 mm/s**，并进行加速度约束。

### 3. 夹爪 1:1 指距镜像

跟随模式下，夹爪开度直接映射手部 **拇指尖—中指尖** 的实测距离：

> 手指张开多少毫米 → 夹爪对应张开多少毫米

夹爪满行程为 **100 mm**，输入超过范围后进行饱和限制。

### 4. 网页数字孪生控制台

浏览器端同时提供：

- MJPEG 实时视频；
- P5.js 手部骨架与交互叠加；
- Three.js NERO 3D 数字孪生；
- D3 实时遥测仪表；
- 参数设置与运行模式控制；
- 录制、回放、复位、锚定等控制操作。

### 5. 仿真优先、硬件可替换

核心控制接口采用后端抽象：

```text
ArmInterface
├── MujocoArmController
└── PyAgxArmController
```

这样可以在 **不改变上层感知与交互逻辑** 的情况下切换仿真和真机。

---

## 系统工作流

```text
┌─────────────────────────────── 感知 ───────────────────────────────┐
│ RealSense D435i / Webcam                                          │
│   └─ RGB + Depth                                                  │
│       └─ MediaPipe Hand / Gesture                                 │
│           └─ 21 Keypoints + Gesture State                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────── 几何 ────────────────────────────────┐
│ 深度反投影 → 3D 手部位置 → 相机坐标系 → 基座坐标系                │
│                         ↓                                         │
│                 工作空间映射 + 安全盒裁剪                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────── 交互 ────────────────────────────────┐
│ 450 ms 手势窗口多数投票 → Interaction FSM                         │
│      ├─ 跟随 / 待机                                                 │
│      ├─ 复位                                                         │
│      ├─ 锚定                                                         │
│      └─ 录制 / 回放                                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────── 控制 ────────────────────────────────┐
│ Follow Controller → IK / Gripper Mapping → Rate Limit              │
│                        ↓                                            │
│                 ArmInterface 后端                                  │
│                  ├─ MuJoCo                                          │
│                  └─ pyAgxArm                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────── 安全 ────────────────────────────────┐
│ 看门狗 500 ms / 手丢失保持 / 工作空间硬限幅 / 关节限速            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 系统架构

```text
                    ┌────────────────────────────┐
                    │       Camera / Sensor      │
                    │ D435i RGB+Depth / Webcam   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │       Perception Layer     │
                    │ MediaPipe / 21 Keypoints   │
                    │ Gesture / Depth / Filter   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │        Geometry Layer      │
                    │ 3D Reprojection / TF       │
                    │ Workspace / Calibration    │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │      Interaction Layer     │
                    │ Gesture FSM / Follow /     │
                    │ Record / Replay / Gripper  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │       Control Layer        │
                    │   ArmInterface Abstraction │
                    └──────────┬──────────┬──────┘
                               │          │
                  ┌────────────▼───┐  ┌──▼────────────────┐
                  │ MuJoCo Backend  │  │ pyAgxArm Backend  │
                  │  DLS-IK         │  │ CAN / Real Robot  │
                  │  Gravity Comp.  │  │ Hardware Pending  │
                  └────────────┬────┘  └──────────┬────────┘
                               │                  │
                               └────────┬─────────┘
                                        ▼
                            ┌────────────────────────┐
                            │      Safety Layer       │
                            │ Watchdog / Rate Limit   │
                            │ Workspace Hard Limits   │
                            └────────────────────────┘

             ┌────────────────────────────────────────────┐
             │               Web Console                  │
             │ Video / P5 / Three.js / D3 / Settings     │
             └────────────────────────────────────────────┘
```

### 分层职责

| 层 | 主要职责 |
|---|---|
| 感知层 | RGB / 深度采集、手部关键点、手势识别、深度反投影、滤波 |
| 几何层 | 相机—基座坐标变换、标定、工作空间映射与裁剪 |
| 交互层 | 手势 FSM、跟随策略、夹爪映射、录制与回放 |
| 控制层 | IK、关节控制、速度限制与后端抽象 |
| 安全层 | 看门狗、手丢失保持、工作空间硬限幅、运动边界约束 |
| Web 层 | 实时视频、数字孪生、遥测、控制与参数配置 |

---

## 快速开始

### 1. 创建 Python 环境

项目当前已验证 **Python 3.12**：

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/Scripts/python.exe -r requirements.txt
```

### 2. 下载手势模型

模型约 **16 MB**，首次使用下载一次即可：

```bash
python scripts/download_models.py
```

### 3. 无相机仿真冒烟测试

使用合成手部观测运行主循环，约 12 秒自动退出：

```bash
python scripts/run_sim.py --headless
```

### 4. 交互式 MuJoCo 仿真

```bash
python scripts/run_sim.py
```

OpenCV 键盘控制：

| 按键 | 功能 |
|---|---|
| `o` | 跟随 / 待机 |
| `h` | 复位 |
| `c` | 锚定参考 |
| `r` | 开始 / 停止录制 |
| `y` | 回放最近轨迹 |
| `q` | 退出 |

### 5. 摄像头输入

使用笔记本摄像头验证 MediaPipe 链路：

```bash
python scripts/run_sim.py --camera webcam
```

---

## 网页控制台

推荐从网页控制台启动系统：

```bash
python scripts/run_real.py
```

启动后自动打开：

```text
http://127.0.0.1:8000
```

如需保留旧版 OpenCV 窗口：

```bash
python scripts/run_real.py --show-cv
```

### 控制台功能

#### 相机视图

MJPEG 实时画面 + P5.js 叠加，可按设置开关：

- 手部骨架；
- 中心十字；
- 掌心位置；
- 手—机械臂跟随连线；
- 指尖距离线。

#### 3D 数字孪生

Three.js 加载官方 NERO CAD 转换后的关节模型，根据实时关节角同步姿态，支持拖拽旋转与缩放。

模型转换：

```bash
python scripts/convert_step_model.py
```

如果官方模型加载失败，系统自动回退到程序化模型。

#### 遥测仪表

D3 仪表盘提供：

- 夹爪开度；
- J1 / J2 / J4；
- 伺服误差趋势；
- 红 / 蓝 / 橙三类状态数据曲线。

#### 参数设置

网页中可实时调整：

- 明暗主题；
- 全局速度；
- 跟随关节限速；
- 三轴跟随增益；
- 死区；
- 手—相机参考距离。

当前参数运行时即时生效，不落盘；部分默认配置同时位于 `configs/mapping.yaml`。

---

## 手势与交互

### 手势速查

| 手势 / 状态 | 行为 |
|---|---|
| **OK** | 跟随模式 ↔ 待机模式 |
| **拇指向上** | 复位到初始姿态并进入待机 |
| **ILoveYou** | 回放最近一次录制 |
| 跟随模式 | 手部三轴位移驱动机械臂协同跟随 |
| 跟随模式中的手势距离 | 1:1 映射夹爪开度，最大 100 mm |
| 手离开视野 / 深度无效 | 保持当前位姿 |

### 手势稳定机制

手势确认采用 **450 ms 窗口多数投票**：

- 至少获得 3 票；
- 目标手势占比 ≥ 60%；
- 通过窗口确认减少识别闪烁导致的误触发。

### 跟随锚定

按 `c` 可以重新建立参考锚点。当前跟随逻辑中：

- J1：底座旋转对准手部方向；
- J2 / J4：通过逆解保持高度与安全距离；
- J7：保持夹爪水平朝前。

---

## 运行参数

项目提供三层速度调节机制，建议从低到高逐级联调：

| 参数 | 默认 / 建议 | 作用 |
|---|---|---|
| 全局速度 `%` | 默认 `10`，建议逐步 `30 → 50` | SDK 总倍率 |
| `max_joint_rate` | 默认 `0.5 rad/s`，约 29°/s | 跟随关节限速 |
| `k_yaw` / `k_pitch` / `k_depth` | 按响应情况调节 | 三轴跟随增益 |

经验上，增益越大响应越快，但也更容易产生抖动；出现抖动时应优先降低对应增益或限速。

---

## 项目结构

```text
NeroHandTwin/
├── configs/
│   ├── arm.yaml                 # 机械臂后端配置
│   ├── mapping.yaml             # 跟随映射、增益、限速
│   ├── calibration.yaml         # 外参
│   ├── gestures.yaml            # 手势配置
│   └── camera.yaml              # 相机配置
│
├── calibration/
│   ├── calibrate.py             # 标定数据采集
│   └── solve_handeye.py         # 手眼 / 外参求解
│
├── src/nerohandtwin/
│   ├── perception/
│   │   ├── camera_realsense.py
│   │   ├── hand_tracker.py
│   │   ├── gesture_fsm.py
│   │   └── filters.py
│   │
│   ├── geometry/
│   │   ├── transforms.py
│   │   ├── calib_solver.py
│   │   └── workspace.py
│   │
│   ├── control/
│   │   ├── arm_interface.py
│   │   ├── mujoco_arm.py
│   │   ├── pyagx_arm.py
│   │   ├── safety.py
│   │   └── trajectory.py
│   │
│   ├── interaction/
│   │   ├── interaction_fsm.py
│   │   ├── follow_controller.py
│   │   └── recorder.py
│   │
│   ├── endeffector/
│   │   └── gripper.py
│   │
│   ├── web/
│   │   └── server.py            # Web 服务与前端资源
│   │
│   ├── ui/
│   │   └── visualizer.py        # OpenCV 可视化
│   │
│   ├── app.py                   # 主循环编排
│   └── sources.py               # 感知数据源
│
├── scripts/
│   ├── run_sim.py
│   ├── run_real.py
│   ├── test_camera.py
│   ├── download_models.py
│   ├── build_exe.py
│   ├── convert_step_model.py
│   └── 其他标定 / 模型转换脚本
│
├── tests/                       # 几何 / 滤波 / FSM / 控制 / E2E / Web
├── packaging/
│   └── nerohandtwin.spec
├── models/                      # MediaPipe 模型（运行时下载）
├── third_party/                 # 官方 NERO 模型 / pyAgxArm SDK
├── data/                        # 轨迹与标定运行数据
├── doc_readme/                  # NERO 官方 3D 模型获取说明
├── requirements.txt
└── LICENSE
```

---

## 测试与验收

运行完整测试集：

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

当前测试规模为 **52 个用例**，覆盖：

- 几何变换；
- 滤波；
- FSM；
- 控制逻辑；
- MuJoCo 闭环跟随；
- Web 服务。

其中 **9 个测试依赖官方模型 / SDK**；当 `third_party/` 缺失时会自动跳过。

### 当前仿真验收结果

| 指标 | 目标 | 实测 / 状态 |
|---|---:|---:|
| 闭环跟随稳态误差 | < 5 mm | **1.74 mm** |
| IK 定位精度 | — | **亚毫米级（两阶段 DLS）** |
| OK 跟随 / 待机切换 | 稳定防误触 | **通过** |
| 夹爪 1:1 指距镜像 | 无跳变 | **EMA + 饱和限制** |
| 录制 → 保存 → 回放 | 可复现轨迹 | **通过，50 Hz JSON** |
| 无真机全流程演示 | 可独立运行 | **通过** |

---

## 真机部署

> ⚠️ 真机联调前必须先确认 NERO 官方安全约束、机械臂固件状态、网络连接、电源与碰撞保护配置。以下步骤对应当前项目的既定部署流程。

### 1. 网络与机械臂状态

电脑使用静态 IP：

```text
10.90.0.153/24
```

NERO Web 上位机：

```text
http://10.90.0.150
```

启动前确认机械臂处于可控状态，并将全局速度设置为较低值。

### 2. 安装 pyAgxArm

```bash
uv pip install -e third_party/pyAgxArm
```

项目当前真机链路依赖 `python-can`；Windows 环境使用 `agx_cando` 接口配合 USB-CAN 模块。

### 3. 切换真机后端

复制示例配置：

```text
configs/arm_real.example.yaml
→ configs/arm.yaml
```

将：

```yaml
backend: pyagx
```

并按照实际固件版本填写 `firmware`。

### 4. 外参标定

推荐使用对准法：

```bash
python scripts/calibrate_touch.py
```

机械臂自动走到 8 个位姿，操作者让手掌中心靠近夹爪尖端约 2～3 cm，并按空格完成采样。系统据此求解完整外参。

备用方法：

```bash
calibrate_axes.py
```

用于轴向快速标定。

### 5. 连通性自检

```bash
python scripts/run_real.py --no-camera
```

确认机械臂就绪序列能够正常执行，并确保关节空间安全离开零位区域。

### 6. 首次联调

建议：

```text
speed_percent = 10
```

同时将 D435i 固定在可以同时观察手部与机械臂工作空间的位置，参考工作距离约 **0.4～1.5 m**。

停止演练时可使用：

- OK → 退出跟随进入待机；
- `Ctrl+C`；
- 断电。

### 7. 正式运行

```bash
python scripts/run_real.py
```

---

## 相机自检

D435i 到位后：

```bash
# RealSense：验证真实深度反投影 / 3D
python scripts/test_camera.py --source realsense

# Webcam：验证手宽估距伪 3D
python scripts/test_camera.py --source webcam
```

---

## 离线打包

项目支持通过 PyInstaller 生成 Windows onedir 发行目录。

```bash
D:/miniconda3/python.exe scripts/build_exe.py
```

产物：

```text
dist/NeroHandTwin/
├── NeroHandTwin.exe
└── _internal/
```

完整性自检：

```bash
dist/NeroHandTwin/NeroHandTwin.exe --frozen-selftest
```

离线部署时请将整个 `NeroHandTwin/` 文件夹一起复制到目标电脑。详细注意事项见：

[`docs/DEPLOY_EXE.md`](docs/DEPLOY_EXE.md)

---

## 依赖与模型

### 核心依赖

完整依赖见 `requirements.txt`，主要包括：

- `mediapipe 1.0.x`（Tasks API）
- `pyrealsense2`
- `mujoco 3.x`
- `numpy`
- `opencv-python`
- `scipy`
- `pyyaml`
- `python-can`

### 模型与第三方组件

MediaPipe 模型由：

```bash
python scripts/download_models.py
```

下载到本地，不直接纳入仓库。

官方 NERO MuJoCo 模型与 `pyAgxArm` SDK 请根据项目约定自行放置到：

```text
third_party/
```

缺失官方资源时，与之绑定的相关测试将自动跳过。

---

## 安全设计

NeroHandTwin 将安全控制放在感知与执行链路之间，当前包含：

### Watchdog

指令流看门狗周期为 **500 ms**。当控制链路失去有效更新时，不继续盲目推进目标指令。

### 手丢失保持

当手部离开视野或深度无效时：

```text
无有效观测 → 保持当前位姿
```

而不是生成不可靠的运动目标。

### 工作空间硬限幅

所有跟随目标在进入机械臂控制后经过工作空间边界裁剪，避免目标点超出允许范围。

### 速度与加速度约束

跟随目标流加入关节限速和加速度限制，设计目标为平滑、保守的联调方式。

> **重要：** 真机运行前仍应依据 NERO 官方手册完成机械与电气安全确认。项目软件层面的限制不能替代机械臂原生安全机制。

---

## 上手路径建议

推荐按照下面的顺序验证，而不是一开始直接连接真机：

```text
① 环境与依赖
      ↓
② 无相机 MuJoCo 冒烟测试
      ↓
③ 交互式 MuJoCo 仿真
      ↓
④ Webcam 验证 MediaPipe 手势链路
      ↓
⑤ D435i 验证真实深度与 3D 反投影
      ↓
⑥ Web 控制台 / 数字孪生
      ↓
⑦ 真机连通性自检
      ↓
⑧ 低速首次联调
      ↓
⑨ 正式运行
```

这样可以把 **感知问题、几何问题、控制问题和硬件问题** 分开定位，降低首次联调复杂度。

---

## 许可证与第三方资产

本项目代码采用：

**MIT License**

详见 [`LICENSE`](LICENSE)。

以下第三方资产不在本项目 MIT 授权范围内：

- **NERO 3D 网格**：`src/nerohandtwin/web/static/model/` 中的网格由官方 NERO CAD（STEP）通过 `scripts/convert_step_model.py` 转换得到，相关权利归 © 松灵机器人（AgileX Robotics）所有；
- **MediaPipe 手势模型**：由 Google MediaPipe 官方渠道分发，并通过脚本下载；
- **NERO 官方文档、MuJoCo 模型、pyAgxArm SDK**：归 AgileX 所有，请从官方渠道获取并遵循其许可条款。

---

## 项目说明

NeroHandTwin 当前定位为 **仿真优先、真机可切换、网页可视化、控制安全可约束** 的机械臂视觉交互系统工程原型。

系统最重要的工程设计原则是：

> **感知与控制解耦、仿真与真机解耦、交互与执行解耦，安全约束贯穿控制链路。**

通过统一接口和模块化目录，可以在不重写上层业务逻辑的情况下替换相机、手势模型、机械臂后端和可视化方式，为后续真机联调与功能扩展保留清晰边界。
