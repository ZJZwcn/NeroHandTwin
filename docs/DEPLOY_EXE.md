# NeroHandTwin.exe 免安装部署指南（目标电脑无需 Python/conda）

## 分发物

`dist/NeroHandTwin.zip`（约 170MB，由 `dist/NeroHandTwin/` 整个目录压缩而来）。
每次 `scripts/build_exe.py` 重新打包后，需要重新压缩分发。

## 目标电脑部署步骤

1. 解压 `NeroHandTwin.zip` → 得到 `NeroHandTwin` 文件夹。
2. **整个文件夹一起拷贝**到目标电脑（不能只拷 exe，`NeroHandTwin.exe` 必须与 `_internal/` 同级）。
3. 放到**简单路径**（如 `D:\NeroHandTwin`），避免中文/空格/& 等特殊字符；不要放 OneDrive 或网盘同步目录。
4. zip 若经网络/U盘传输：右键 zip → 属性 → 勾选「解除锁定」→ 再解压（否则 Windows 会拦截 DLL）。
5. 接好硬件：RealSense D435（USB 3.0）+ AgileX USB-CAN + 机械臂通电。
6. 双击 `NeroHandTwin.exe`：SmartScreen 点「更多信息→仍要运行」；控制台 5 项安全确认输入 `yes` 回车；浏览器自动打开 <http://127.0.0.1:8000>。

## 运行环境要求

- Windows 10/11 64 位（exe 自带 Python 3.13 与全部依赖，零安装）
- 硬件三件套缺一不可：D435 相机 / USB-CAN / NERO 机械臂
- 机械臂连不上：安装 AgileX CANdo 驱动（随适配器附赠）

## 故障诊断

```bat
cd /d 部署目录
NeroHandTwin.exe --frozen-selftest
```

- 全部通过 → 包完整，问题在硬件/驱动/接线。
- 有 FAIL → 包不完整或被拦截 → 解除锁定、加入杀软白名单后重试。
- 双击一闪而过：运行包内 `诊断启动.bat` 捕获完整报错。

## 外机跑不起来的常见原因（按概率排序）

| 现象 | 原因 | 处理 |
|---|---|---|
| 双击无反应/一闪而过 | 只拷了 exe 没带 `_internal/` | 整个文件夹一起拷 |
| DLL load failed | zip 未解除锁定 / 被杀软拦截 | 属性→解除锁定；加白名单 |
| 找不到模块/路径错误 | 部署在含中文/空格/& 的路径或 OneDrive | 换 `D:\NeroHandTwin` 这类简单路径 |
| 提示找不到相机 | D435 未接或非 USB3 口 | 换 USB3 口重插 |
| CAN 初始化失败 | USB-CAN 未接或 CANdo 驱动未装 | 装 AgileX 驱动 |
| SmartScreen 蓝屏警告 | exe 未签名 | 「更多信息」→「仍要运行」 |
| 控制台问 yes 后没动静 | 没输入 yes | 输入 `yes` 回车 |
