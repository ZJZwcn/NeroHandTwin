@echo off
chcp 65001 >nul
cd /d %~dp0
echo ==== NERO 程序包自检测试 ====
NeroHandTwin.exe --frozen-selftest
echo.
echo ==== 正式启动（输入 yes 后回车） ====
NeroHandTwin.exe
echo.
echo 程序已退出，以上为全部日志。按任意键关闭窗口。
pause >nul
