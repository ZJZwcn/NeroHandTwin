# NeroHandTwin — NERO 机械臂手势交互与手部跟随数字孪生系统

基于 **松灵 AgileX NERO 七自由度机械臂** + **Intel RealSense D435i**（eye-to-hand 外置固定视角）的 Python 手势交互系统：

- **OK**（拇指食指捏圈+三指伸直）→ 跟随模式 ↔ 待机 切换（上电默认待机）
- **跟随模式** → 臂协同追手：手左/右移→J1 转动对准，手远/近→前伸/后缩保持安全距离，手高/低→夹爪保持同一水平线
- **跟随模式中夹爪** → 1:1 镜像拇指尖-中指尖实测距离（满行程 100mm，捏拢=闭合/张开=全开）
- **拇指向上** → 复位回初始姿态（保持使能，停在待机）
- **ILoveYou** → 回放最近一次录制（倍速可调）；轨迹录制用键盘 `r` 或网页「录制」按钮
- **手离开视野 / 深度无效** → 保持当前位姿（安全默认）

当前阶段机械臂未到位，**全部功能在 MuJoCo 仿真中验证**（官方 NERO 模型），
控制层做了硬件抽象，真机到位后切换 `pyAgxArm` 后端 + 手眼标定即可上真机。

## 快速开始

```bash
# 1. 创建环境（本机已验证 Python 3.12）
uv venv .venv --python 3.12
uv pip install -p .venv/Scripts/python.exe -r requirements.txt

# 2. 下载手势识别模型（~16MB，一次即可）
python scripts/download_models.py

# 3. 纯仿真冒烟自检（无相机，合成手部观测跑通主循环，12s 自动退出）
python scripts/run_sim.py --headless

# 4. 交互式仿真（OpenCV 窗口键盘兜底：o跟随/待机 h复位 c锚定 r录制 y回放 q退出）
python scripts/run_sim.py

# 5. 接笔记本摄像头（伪 3D 模式，验证 MediaPipe 链路）
python scripts/run_sim.py --camera webcam
```

### 运行测试

```bash
.venv/Scripts/python.exe -m pytest tests/ -q      # 52 个用例，含 MuJoCo 闭环跟随精度验收与 web 服务测试
                                                  # 其中 9 个依赖官方模型/SDK（third_party/ 缺失时自动跳过）
```

## 网页控制台（推荐运行方式）

```bash
python scripts/run_real.py        # 启动后自动打开浏览器 http://127.0.0.1:8000
python scripts/run_real.py --show-cv   # 需要旧 OpenCV 窗口时加此开关
```

- **相机视图**：MJPEG 实时视频 + P5.js 叠加（手部骨架/中心十字/掌心/跟随连线/指距线，设置面板可逐项开关）
- **3D 数字孪生**：Three.js 加载官方 CAD 模型（`scripts/convert_step_model.py` 从 STEP 转换为关节分组 STL），按实时关节角同步姿态，可拖拽旋转/缩放；加载失败自动回退程序化模型
- **遥测仪表**：D3 圆弧仪表（夹爪开度/J1/J2/J4）+ 伺服误差滚动趋势线（红=eu 蓝=ev 橙=深度）
- **设置面板**：明暗主题切换 + 机械臂运行参数（全局速度%、跟随关节限速、三轴增益、死区、手-相机参考距离），运行时即时生效不落盘
- 网页按钮（跟随/复位/锚定/录制/回放/退出）与键盘手势完全同源，命令经主循环串行执行，控制安全边界不变

### 跟随速度调整

三层由慢到快（都在网页「设置 → 机械臂参数」里即时可调，也写进了 `configs/mapping.yaml`）：
1. **全局速度 %**（SDK 总倍率，默认 10 是联调保守值）→ 建议逐级 30 → 50
2. **跟随关节限速** `max_joint_rate`（rad/s，默认 0.5≈29°/s）→ 建议先 0.8，稳定后 1.0~1.2
3. **三轴增益** `k_yaw/k_pitch/k_depth`（越大响应越快、越易抖动，抖了就回调）

## 打包为 exe（离线部署）

```bash
D:/miniconda3/python.exe scripts/build_exe.py     # 产物 dist/NeroHandTwin/
dist/NeroHandTwin/NeroHandTwin.exe --frozen-selftest    # 完整性自检（不碰硬件）
```

onedir 结构（`NeroHandTwin.exe` + `_internal/`），首次双击需在控制台确认安全提示后输入 yes。
**分发到其他电脑**：见 [docs/DEPLOY_EXE.md](docs/DEPLOY_EXE.md)（整个文件夹一起拷贝、简单路径、解除锁定）。


### 相机自检（D435i 到货后）

```bash
python scripts/test_camera.py --source realsense   # 深度反投影真 3D
python scripts/test_camera.py --source webcam      # 手宽估距伪 3D
```

## 系统架构

```
感知层  D435i(RGB+对齐深度) → MediaPipe GestureRecognizer(21关键点+手势)
        → 像素反投影(掌心/食指尖 3D) → One Euro 滤波
几何层  相机系 → 基座系 (T_base←camera, Umeyama 手眼标定)
        → 工作空间映射(绝对/相对) → 安全盒裁剪
交互层  手势窗口多数投票(450ms) → 交互模式 FSM(OK 开关：跟随↔待机)
        → 协同跟随控制器 / 夹爪镜像指距 / 轨迹录制回放
        → 限速目标流 (≤80mm/s + 加速度限幅)
控制层  ArmInterface 抽象 ─┬─ MujocoArmController (DLS-IK + 重力补偿 + 零空间避奇异)
                           └─ PyAgxArmController  (真机 CAN, 待硬件)
安全层  指令流看门狗(500ms) / 手丢失保持 / 安全盒硬限幅
```

## 目录结构

```
configs/                  arm.yaml(后端) mapping.yaml(映射/限速) calibration.yaml(外参) gestures.yaml camera.yaml
calibration/              calibrate.py(ArUco采集) solve_handeye.py(Umeyama求解)
src/nerohandtwin/
  perception/             camera_realsense.py / hand_tracker.py / gesture_fsm.py / filters.py
  geometry/               transforms.py / calib_solver.py / workspace.py
  control/                arm_interface.py / mujoco_arm.py / pyagx_arm.py / safety.py / trajectory.py
  interaction/            interaction_fsm.py / follow_controller.py / recorder.py
  endeffector/            gripper.py(捏合→开合度)
  web/                    网页控制台(server.py + Three.js 数字孪生 / P5 叠加 / D3 仪表)
  ui/visualizer.py        OpenCV 叠加(关键点/手势/状态/FPS)
  app.py sources.py       主循环编排与感知数据源
scripts/                  run_sim.py / run_real.py / test_camera.py / download_models.py / build_exe.py / 标定与模型转换脚本
tests/                    52 用例：几何/滤波/FSM/控制/端到端/web 服务
packaging/                nerohandtwin.spec（PyInstaller 打包配置）
doc_readme/               NERO 官方 3D 模型下载链接
models/                   MediaPipe 任务模型（download_models.py 下载，不入库）
third_party/（自行准备）   官方 MuJoCo 模型与 pyAgxArm SDK：从松灵支持页获取后放这里（缺失时相关测试自动跳过）
data/                     录制轨迹 / 标定样本（运行时生成，不入库）
```

## 手势速查（v5 最终方案）

| 手势 | 动作 |
|---|---|
| **OK**（拇指食指捏圈+三指伸直） | **跟随模式 ↔ 待机 切换**（上电默认待机） |
| 跟随模式中 | 臂协同追手：手左/右移→臂转动跟随；手上/下移→夹爪保持同一水平线；手远/近→臂前伸/后缩保持安全距离 |
| 跟随模式中夹爪 | **1:1 镜像拇指-中指指尖距离**：指尖张开多少毫米夹爪就开多少毫米（满行程 100mm 饱和） |
| **拇指向上** | **复位**：回初始姿态（保持使能），停在待机 |
| ILoveYou | 回放最近一次录制（data/trajectory_*.json） |
| 手离开视野 / 深度无效 | 保持当前位姿（安全默认） |

跟随为**协同多关节伺服**（无需任何标定）：J1 底座旋转对准手、J2/J4 平面
逆解保持高度与安全距离、J7 保持夹爪水平朝前。按 **C 键**重新锚定参考。

手势确认采用**窗口多数投票**（450ms 窗口，≥3 票且占比 ≥60%，抗识别闪烁）。
键盘兜底：`o`=跟随/待机切换 `h`=复位 `c`=锚定 `r`=录制 `y`=回放 `q`=退出。

## 真机部署清单（硬件到位后）

> ⚠️ NERO 手册安全约束：**零位是奇异点，不能从原点直接下点位指令**；
> 运动学控制避免 J2~J5 处于 0 位；供电 DC24V 严禁超 25V。

1. **网络**：网口直连，电脑配静态 IP `10.90.0.153/24`，Web 上位机 `http://10.90.0.150` 确认臂状态、开启碰撞保护、全局速度置低。
2. **SDK**：`uv pip install -e third_party/pyAgxArm`（依赖 python-can，Windows 用 `agx_cando` 接口 + USB-CAN 模块）。
3. **切换后端**：复制 `configs/arm_real.example.yaml` → `configs/arm.yaml`（`backend: pyagx`），按固件版本填 `firmware`。
4. **外参标定（对准法，推荐）**：
   - **python scripts/calibrate_touch.py**：机械臂自动走到 8 个位姿，你把手掌中心贴近夹爪尖（2~3cm）按空格采样。解出**完整外参**（方向+位置），无需理解任何坐标系、无需标定板。约 2 分钟。
   - 跟随方向仍不对时重跑一次即可（每次采样保持手稳定）。
   - （备用）`calibrate_axes.py` 为轴向快速标定，对手势移动方向理解要求高。
5. **连通自检**：`python scripts/run_real.py --no-camera`（就绪序列：关节空间离开零位）。
6. **首次联调**：`speed_percent: 10` 起步；D435i 固定在能看到手与机械臂工作空间的支架上（手距相机 0.4~1.5m 深度有效）；停止演练（OK 退出跟随回待机 / Ctrl+C / 断电）。
7. **正式运行**：`python scripts/run_real.py`。

### 上手顺序建议（第 1 天）

```
手势识别（不需要机械臂）：python scripts/test_camera.py --source realsense
       ↑ 只用摄像头就能验证：21 关键点、手势名、捏合度都在画面上
手部跟随（要接机械臂）  ：python scripts/run_real.py
       ↑ OK→跟随/待机；拇指向上→复位；ILoveYou→回放；跟随中夹爪镜像指距
```

## 验收状态（仿真）

| 指标 | 目标 | 实测 |
|---|---|---|
| 闭环跟随稳态误差 | <5 mm | **1.74 mm** |
| IK 定位精度（可达点） | — | 亚毫米（两阶段 DLS） |
| OK 切换跟随/待机 | 边沿触发 + 复位宽限防误触 | 通过（test_pipeline_e2e.py） |
| 夹爪 1:1 镜像指距 | 无跳变 | EMA 平滑 + 满行程饱和 |
| 录制→保存→回放 | 复现轨迹 | 通过（50Hz JSON，倍速可调） |
| 无真机依赖全流程演示 | — | 通过（tests/test_pipeline_e2e.py） |

## 依赖

见 `requirements.txt`。核心：mediapipe 1.0.x（Tasks API）、pyrealsense2、mujoco 3.x、numpy、opencv-python、scipy、pyyaml、python-can（真机）。

## 许可证

本项目代码以 [MIT License](LICENSE) 发布。

第三方资产不受 MIT 覆盖：
- **3D 网格**（`src/nerohandtwin/web/static/model/`）：由官方 NERO CAD（STEP）经 `scripts/convert_step_model.py` 转换，版权归 © 松灵机器人（AgileX Robotics），仅供本项目配套的仿真与数字孪生展示使用。
- **MediaPipe 手势模型**（`models/`）：Google MediaPipe 官方分发模型，由 `scripts/download_models.py` 下载，不入库。
- NERO 机械臂官方文档、MuJoCo 模型与 pyAgxArm SDK 归 AgileX 所有，请从官方渠道获取。
