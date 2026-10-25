---
title: "第一次机器人仿真实验：保存一张相机图像"
description: "用 ManiSkill 初始化一个推方块任务，保存真实相机 RGB，检查图像、数据维度与运行配置，并记录一次控制器初始化问题。"
date: 2026-10-25
series: "没有机器人，也能开始的具身智能实践"
order: 2
tags:
  - 仿真实践
  - ManiSkill
  - 相机观测
  - 具身智能
---

# 第一次机器人仿真实验：保存一张相机图像

上一篇，我们把机器人任务拆成了观测、动作和闭环。这一次，先让“观测”变成一份能打开、能检查的实际文件：启动一个推方块环境，获取相机 RGB，保存成 PNG。

本篇实验已于 2026-10-25 在本机完成。最终只保存一张图，没有调用 `env.step()`，也没有训练模型。下面的画面和数值都来自这次运行。

## 先看我们要得到的结果

<figure class="experiment-figure"><img src="../assets/experiments/02-sim-camera/frame.png" alt="PushCube-v1 的真实相机观测：机械臂前臂与夹爪、木质桌面、蓝色方块及红白目标区域" width="256" height="256" loading="lazy" decoding="async"><figcaption>seed=42，base_camera，256×256。保留原始传感器视角与像素，未重绘或添加标注。</figcaption></figure>

图中可以辨认机械臂夹爪、方块和目标区域。机械臂上部没有完整进入视野，这是相机位置决定的；第一次检查的重点，是确认任务相关物体确实出现在观测里。

完成本篇，你应当能回答三个问题：图片来自哪个相机？代码中图像的维度和类型是什么？换一台机器时，需要记录哪些条件才能尝试复现？

[下载公开实验文件包](../assets/experiments/02-sim-camera/sim-camera-files.zip)，其中包含脚本、实际图片、公开复现配置和依赖版本。也可以单独查看 [完整脚本](../assets/experiments/02-sim-camera/capture_frame.py)、[环境记录](../assets/experiments/02-sim-camera/environment.json) 和 [验收记录](../assets/experiments/02-sim-camera/validation.json)。

## 第一步：确认运行条件

本次使用的主要配置如下。它是一组已经跑通过的配置，不代表所有 Windows 或其他平台组合都适用。

| 项目 | 本次实际配置 |
| --- | --- |
| 系统 | Windows，版本 10.0.26200，25H2 |
| GPU / 驱动 | RTX 4060 Ti，约 8 GB；610.88 |
| Vulkan 探测 | 设备可见，API 1.4.341 |
| Python | 3.11.9 |
| ManiSkill / SAPIEN | 3.0.1 / 3.0.3 |
| PyTorch | 2.14.1，CPU 构建 |
| Gymnasium / NumPy / Pillow | 1.4.0 / 2.4.6 / 13.0.0 |

检查时，默认 Python 没有安装仿真依赖，可复用的旧虚拟环境也不在预期位置。因此，这次在工作区建立了独立虚拟环境，没有升级已有研究环境。安装后运行 `python -m pip check`，没有发现依赖冲突。

ManiSkill 官方系统支持表列出了 Windows 上的 CPU 物理仿真与渲染支持，但 Windows 的整体支持仍有限。本篇明确选择单环境、CPU 物理仿真和 CPU 可访问的图像路径。渲染仍依赖 SAPIEN/Vulkan；仅仅检测到显卡或 CUDA，不能代替实际渲染检查。[安装与系统支持](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)

## 第二步：准备脚本与独立环境

将实验文件包解压到自己的实验目录。下面使用 PowerShell，安装到该目录的 `.venv`，通过完整解释器路径运行，避免混用不同 Python：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) '.cache'
& '.\.venv\Scripts\python.exe' -B '.\capture_frame.py' --output-dir '.\result'
```

`requirements.txt` 记录了本次实际安装的版本。安装需要联网；首次加载任务时，也可能下载模型资源。脚本在导入 ManiSkill 前设置资源与缓存目录，避免把新增资源放到难以找到的位置。

本机实际使用的是工作区里的 `work/sim-camera-env/`，成功运行的核心命令是：

```powershell
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) 'work\sim-camera-cache'
& '.\work\sim-camera-env\Scripts\python.exe' -B '.\outputs\02-sim-camera\capture_frame.py'
```

如果结果目录已经有 `frame.png` 或 `environment.json`，脚本会拒绝覆盖。复现时通过 `--output-dir` 指定一个新目录即可。

## 第三步：初始化任务并读取相机

完整脚本还处理了配置记录、失败信息和文件检查。采图的核心部分如下：

```python
import gymnasium as gym
import mani_skill.envs
from PIL import Image

env = gym.make(
    "PushCube-v1",
    robot_uids="panda",
    num_envs=1,
    obs_mode="rgb",
    sim_backend="cpu",
    render_backend="cpu",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
    sensor_configs={"width": 256, "height": 256, "shader_pack": "minimal"},
)
try:
    obs, _ = env.reset(seed=42)
    batch = obs["sensor_data"]["base_camera"]["rgb"]
    rgb = batch[0].detach().cpu().numpy()
    Image.fromarray(rgb).save("frame.png")
finally:
    env.close()
```

`num_envs=1` 表示这次只创建一个环境。`reset(seed=42)` 设置此次保存场景的初始化种子。ManiSkill 3.0.1 在构造环境时还会进行一次内部 reset；脚本随后显式 reset，并只保存后者的观测。初始化可能包含库内部的物理稳定步骤，不能把“没有调用 step”理解为物理引擎完全没有工作。[快速开始](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)

`sim_backend="cpu"` 在运行中解析为 `physx_cpu`，`render_backend="cpu"` 解析为 `sapien_cpu`。本次返回的图像张量在 CPU 上，没有走 CUDA 图像张量路径。不要只凭参数中的 “cpu” 就判断没有发生渲染。

`sensor_configs` 将观测相机分辨率设为 256×256；其他相机参数沿用任务配置，实际相机内外参已保存到 `environment.json`。固定种子有助于控制初始化随机性，仍需同时记录版本和配置；本篇没有验证跨机器逐像素复现。

## 第四步：看懂图像的数据结构

本次读取到的 `batch` 是：

```text
shape:  [1, 256, 256, 3]
dtype:  torch.uint8
device: cpu
```

四个维度依次表示：环境批次、图像高度、图像宽度、RGB 通道。前面的 `1` 不是多出来的一张图片，而是单环境的批次维度；用 `batch[0]` 取出后，才得到保存为 PNG 的三维数组。

本次数组是 `uint8`，保存前不需要再次乘以 255。若后续改了着色器、观测模式或封装方式，应先打印实际维度和类型，再处理数组，不能直接套用本篇的假设。[观测数据说明](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)

还要分清相机的用途。这里保存的是 `obs` 中 `base_camera` 的传感器数据，可作为策略输入；实验没有运行策略。用于制作展示视频的相机，以及 GUI 查看场景的相机，是另外的类别。画面更适合展示，不代表它就是机器人策略读取的观测。[相机分类与配置](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html)

## 第五步：保存之后，重新打开检查

成功退出还不够。完整脚本会用 Pillow 重新打开 PNG，并与保存前的数组逐像素比较；随后又独立检查了文件、配置和哈希。

| 检查项 | 本次实际结果 |
| --- | --- |
| 保存尺寸与通道 | 256×256，RGB 三通道 |
| 数据类型 | uint8 |
| 实际像素范围 | 34～255 |
| 不同 RGB 颜色数量 | 10649 |
| PNG 读回与原始数组 | 逐像素一致 |
| 实际查看 | 可辨认夹爪、桌面、方块与目标区域 |
| 最终保存帧数 / 动作调用数 | 1 / 0 |

这些数值用于检查这次文件的完整性，不是图像质量评分，也不是机器人任务成功率。只检查“文件存在”无法发现纯黑画面、错误相机或数组维度处理问题；只检查像素变化，也不能确认任务物体是否真的可见，所以实际打开图片仍然必要。

公开的 `environment.json` 提供复现配置，`validation.json` 提供本次本地核验的摘要。原始记录保留在本机。原图 SHA-256 为：

```text
89d665a348ce18c86542c12795cc6d63f673f7ed7922c946cb51cf6508a4f4e6
```

## 这次实际遇到的控制器问题

第一次尝试用了 `pd_ee_delta_pose`，程序在控制器初始化时失败。日志先提示缺少 pinocchio，随后在逆运动学模块调用 `PinocchioModel` 时出现：

```text
TypeError: 'NoneType' object is not callable
```

当前安装中的 `PinocchioModel` 不可用。由于本篇只需要初始化与采图，将控制器改为 `pd_joint_delta_pos` 后，第二次尝试成功。本机保留了完整失败记录，公开实验包提供问题说明与可复现脚本。

成功运行仍有缺少 pinocchio 的 warning。它没有阻止此次配置下的初始化和图像获取，却也不意味着逆运动学功能可用了。下一次真正执行动作时，需要针对选用的控制器重新验证依赖、动作含义和单位。

## 完成之后，下一步是什么

现在我们已经得到一份来源明确、可以检查的相机观测：它来自哪个任务、哪个相机、什么版本、什么随机种子，都有记录。

本篇验证到这里：单环境初始化、真实相机 RGB 获取、PNG 保存和读回。动作执行、推物成功率、策略训练与实机迁移尚未验证。

接下来会把这张图片放回完整的数据结构里：除了 RGB，机器人还返回哪些观测？一条动作记录应该保存什么？怎样把当前观测、动作和下一时刻观测对应起来？这就是下一篇《机器人数据长什么样：读懂一条观测与动作记录》的任务。
