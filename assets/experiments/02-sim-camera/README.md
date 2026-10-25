# 相机图像实验 · 公开文件包

本包提供 capture_frame.py、frame.png、requirements.txt、公开复现配置 environment.json 和验收摘要 validation.json。
图像为实际采集的 256×256 RGB，原图未修改。完整本地日志和含本机路径的原始记录保留在作者本机。

已验证：Python 3.11.9、ManiSkill 3.0.1、SAPIEN 3.0.3、PyTorch 2.14.1；PushCube-v1、panda、单环境、seed=42、base_camera。
使用 CPU 物理仿真与 CPU 可访问图像路径，脚本不调用 env.step()，不训练模型。

在新建目录中解压后，可使用 PowerShell：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --only-binary=:all: -r requirements.txt
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) '.cache'
& '.\.venv\Scripts\python.exe' -B capture_frame.py --output-dir result
```

第一次末端位姿控制器初始化因 PinocchioModel 不可用而失败，改用 pd_joint_delta_pos 后采图成功。
逆运动学、动作执行、推物成功率和训练均不属于本次验证范围。其他平台与驱动组合需要实际检查。
