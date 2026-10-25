---
title: "从一条记录到一段轨迹：保存机器人动作序列"
description: "采集20个真实控制步，保存21份观测和原始相机帧，读懂轨迹的时间轴、回合边界及GIF回放，并检查数据是否一一对应。"
date: 2026-10-26
series: "没有机器人，也能开始的具身智能实践"
order: 4
tags:
  - 仿真实践
  - 机器人轨迹
  - ManiSkill
  - 数据采集
---

# 从一条记录到一段轨迹：保存机器人动作序列

第三篇，我们读懂了一条 `观测 → 动作 → 下一时刻观测`。这次把它接成一段序列：让同一个仿真环境连续执行动作，保存每一步，再从文件生成回放。

本篇实验已于2026-10-26实际完成。只启动一个 PushCube-v1，固定 seed=42，共执行20次 `env.step()`，保存21份观测与21张原始PNG。没有训练模型，没有为了让画面变化更大而追加动作。

## 先看真实回放

<figure class="experiment-figure" data-trajectory-replay><img id="trajectory-preview" src="../assets/experiments/04-first-trajectory/frames/0000.png" data-replay-src="../assets/experiments/04-first-trajectory/replay.gif" data-duration-ms="1050" alt="真实机器人轨迹：机械臂夹爪在木桌上方小幅往返，方块没有可见移动；默认显示原始第0帧" width="256" height="256" loading="lazy" decoding="async"><figcaption>base_camera，256×256。默认显示原始第0帧，点击后播放真实帧编码的GIF。<div class="replay-controls"><button class="button" type="button" data-replay-toggle aria-controls="trajectory-preview" aria-pressed="false" hidden>播放回放</button><span class="replay-status" role="status" data-replay-status>当前显示原始第0帧。</span></div><a href="../assets/experiments/04-first-trajectory/replay.gif" download>下载GIF回放</a></figcaption></figure>

动作幅度较小，主要观察夹爪与前臂的轻微往返。方块没有被成功推到目标。GIF只用于展示：颜色经过量化，实际数值检查使用原始PNG和NPZ中的RGB。

[下载公开轨迹文件包](../assets/experiments/04-first-trajectory/trajectory-files.zip)，包含采集脚本、21张原图、数值样本、字段说明、公开配置、验收摘要和GIF。也可以单独查看 [脚本](../assets/experiments/04-first-trajectory/capture_trajectory.py)、[trajectory.npz](../assets/experiments/04-first-trajectory/trajectory.npz)、[逐步统计](../assets/experiments/04-first-trajectory/trajectory.json)、[字段与维度](../assets/experiments/04-first-trajectory/schema.json) 和 [验收结果](../assets/experiments/04-first-trajectory/validation.json)。原始运行日志与本机路径留在作者本机。

## 第一步：复用第三篇的环境与控制方式

本次复用原有环境：Python3.11.9、ManiSkill3.0.1、SAPIEN3.0.3、PyTorch2.14.1 CPU、Gymnasium1.4.0、NumPy2.4.6、Pillow13.0.0。82个已安装包的版本均未改变，控制器与任务源码SHA256也与第三篇一致。

使用单个 panda 机器人、`obs_mode="rgb"`、`pd_joint_delta_pos`，相机为策略观测相机 base_camera，256×256、minimal shader。沿用CPU物理和CPU可访问的SAPIEN/Vulkan图像路径，解析后的后端为 physx_cpu/sapien_cpu。

先核对控制器，再发动作。[ManiSkill控制器文档](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)说明动作空间由控制器决定；本次还将运行时配置与第三篇记录直接比较，确认以下语义相同：

- 前7维输入范围为[-1,1]，转换为机械臂关节角目标增量：`delta_q=0.1*a` rad。
- 目标相对本步测量位置计算，`use_delta=True, use_target=False`。
- 最后一维控制夹指绝对目标，`target_m=0.015+0.025*a[7]`；输入1对应每指0.04 m。

这也意味着，其他机械臂维度填0只是目标增量为0；夹爪维度填0会改变目标位置。具体解释可回看[第三篇](observations-and-actions.html)。

## 第二步：固定动作计划，先定好停止条件

这次不使用随机动作，也不设计求解策略。第一关节的归一化输入按五步一组切换，其他机械臂输入为0，夹爪输入始终为1：

| 控制步 | 第一关节输入 | 每步请求的目标增量 |
| --- | --- | --- |
| 1—5 | +0.1 | +0.01 rad |
| 6—10 | -0.1 | -0.01 rad |
| 11—15 | +0.1 | +0.01 rad |
| 16—20 | -0.1 | -0.01 rad |

脚本有两个停止条件：达到最多20步，或遇到 `terminated` / `truncated`。后者出现时，先保留这一步返回的最后观测，然后立即停止；不 reset 后把新回合接到原轨迹后面。

解压文件包后，使用已经配置好相同依赖的 Python：

```powershell
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) '.cache'
python -B capture_trajectory.py --seed 42 --steps 20 --output-dir result
```

`result` 必须是新的结果目录。`--steps` 的允许范围为1到20。若需要逐项比对第三篇记录，可额外传入 `--reference-env`，指向第三篇的环境JSON；不提供时，脚本仍会检查动作映射、缩放和边界，但不会声称完成参考文件比对。

## 第三步：为什么20个动作需要21份观测

初始化就有一份观测。每执行一个动作，又得到一份下一时刻观测：

```text
obs[0] -- actions[0] --> obs[1]
obs[1] -- actions[1] --> obs[2]
...
obs[19] -- actions[19] --> obs[20]
```

因此，N个动作连接N+1份观测。只有20张图时，可能漏掉了初始帧或最后一帧；不能仅凭“文件看起来不少”就判断轨迹完整。

核心采集顺序如下，完整脚本还保存全部字段、控制器目标、时间和原图：

```python
observations = [copy_observation(obs)]
for action in planned_actions[:20]:
    next_obs, reward, terminated, truncated, info = env.step(action)
    actions.append(action.copy())
    observations.append(copy_observation(next_obs))
    # 此处同时保存本次 reward、flags 和 info。
    if terminated.any() or truncated.any():
        break
```

`copy_observation` 递归复制每个数值叶子。对于Torch张量，采用 `.detach().cpu().numpy().copy()`，避免旧记录随仿真缓存更新。采集结束时还重新计算每个历史快照的指纹，确认没有变化。

## 第四步：把时间轴和环境轴分清楚

按[ManiSkill观测接口](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)，单环境数据仍可能带环境批次维度。本篇在第三篇形状的最前面增加时间轴，不随意 squeeze 掉环境轴：

| 数组 | 实际形状 | 维度顺序 |
| --- | --- | --- |
| `obs/agent/qpos`、`qvel` | `(21,1,9)` | 观测时间、环境、关节 |
| `obs/extra/tcp_pose` | `(21,1,7)` | 观测时间、环境、位姿分量 |
| `obs/sensor_data/base_camera/rgb` | `(21,1,256,256,3)` | 观测时间、环境、高、宽、RGB |
| `actions` | `(20,8)` | 动作时间、动作维度，无额外环境轴 |
| `rewards`、`terminated`、`truncated` | `(20,1)` | 动作时间、环境 |
| `info/elapsed_steps`、`info/success` | `(20,1)` | 动作返回时间、环境 |
| `step_index`、`time_s` | `(21,1)` | 观测时间、环境 |

qpos的前7维单位为rad，后2维为m；qvel对应rad/s和m/s。TCP前三项是世界坐标xyz，单位m，后四项是qw,qx,qy,qz。RGB为uint8、0..255。相机矩阵也沿时间堆叠，但本次固定相机的参数没有改变。

`reset_info/*` 是初始化的一次性信息，不沿时间堆叠。本次 elapsed_steps 和 success 为 `(1,)`，环境轴在0；reconfigure为标量。公开schema和验收记录给出这些特殊字段的轴说明，不能套用所有字段都具有 `(时间,环境,...)` 的规则。

方块真值和控制器目标仍在独立的 `diagnostic/` 字段中。它们用于验收，不加入策略观测。

## 第五步：仿真1秒，为什么GIF有1.05秒

本次控制频率20 Hz，物理频率100 Hz。一个控制步推进0.05秒仿真，内部包含5个物理子步。观测时间为：

```text
0.00, 0.05, 0.10, ... , 0.95, 1.00 秒
```

相邻21份观测之间只有20个时间间隔，所以仿真长度是1.00秒。GIF则让包括最终状态在内的21张图各显示50毫秒，每轮为 `21×50=1050` 毫秒。多出的50毫秒是最终状态的展示时间。

按照[Pillow13.0.0的GIF接口](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif)，duration以毫秒表示，loop=0表示播放器循环。本篇重新解码得到21帧，每帧50毫秒，没有帧合并。播放循环不会调用仿真器，也没有新采集回合。

GIF使用256色量化，解码图像与预期量化结果一致，但不能要求它与原PNG逐像素相同。本次原始RGB与GIF解码图的平均绝对像素误差约0.752658，使用0..255数值单位计算；数值研究与后续训练数据应读取PNG或NPZ中的原始RGB。

## 第六步：重新读取文件，检查每个对应关系

NPZ中共21个纯数值数组，使用[NumPy的禁用pickle读取方式](https://numpy.org/doc/2.4/reference/generated/numpy.load.html)：

```python
import numpy as np

with np.load('trajectory.npz', allow_pickle=False) as d:
    n = len(d['actions'])
    q = d['obs/agent/qpos']
    assert len(q) == n + 1
    assert np.allclose(np.diff(d['time_s'][:, 0]), 0.05)
    assert not d['terminated'][:-1].any()
    assert not d['truncated'][:-1].any()
    assert np.allclose(
        d['diagnostic/arm_target_qpos'],
        q[:-1, :, :7] + 0.1*d['actions'][:, None, :7]
    )
    print('第0/5/10/15/20步：', q[[0, 5, 10, 15, 20], 0, 0])
```

最后一条公式检查每个控制器目标都使用对应的上一份观测计算。若错把 `q[1:]` 与当前动作配对，记录的目标与输入含义就错位了。

本次35项独立检查还覆盖：所有数值有限、动作边界、步号连续、每张PNG与对应NPZ逐像素一致、GIF帧数与时长、快照独立性，以及第二、三篇原始文件和依赖版本未改变。验收只读已保存文件，没有重新运行成功的采集。

第一关节的实测值如下：

| 观测步号 | 仿真时间 | 第一关节位置 |
| --- | --- | --- |
| 0 | 0.00 s | 0.009934283 rad |
| 5 | 0.25 s | 0.027843354 rad |
| 10 | 0.50 s | 0.011843494 rad |
| 15 | 0.75 s | 0.027844150 rad |
| 20 | 1.00 s | 0.011843315 rad |

虽然正负动作数量相同，最后仍比起点多约0.001909 rad。指令不是瞬时位移，目标也不等于实测状态；不能据动作计划对称就认定机器人精确回到起点。

方块净位移范数约 `7.66369e-7 m`，没有可见推送；所有success均为false。奖励范围约0.06982到0.07188，正奖励同样不意味着任务成功。

## 回合边界与常见问题

**20步结束，为什么truncated仍然是false？**

这是采集脚本达到预算后停止，并未触发环境返回的截断。[Gymnasium的step接口](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)区分任务终止与外部截断；本次保留真实返回值，并单独记录 `stop_reason=requested_step_limit`。这是一段回合片段，不声称保存了完整任务回合。

**如果中途成功或超时，最后一帧还要保存吗？**

要。先保存导致结束的那次动作及返回的最后观测，再停止。N/N+1关系仍成立。不要在同一段序列中继续接上reset后的初始状态。

**为什么不直接把观测字典塞进NPZ？**

本篇把字段路径作为键，只保存数值数组，避免对象数组和pickle依赖。JSON负责解释字段、单位和对应关系，NPZ负责保留数值。

**GIF看起来变化很小，需要追加动作吗？**

本次20步就是采集上限。小幅运动同样能检验时间对应关系；先看关节数据和逐步像素差，不为了画面效果改写实验范围。

**PNG数量与GIF帧数不同，一定是采集错了吗？**

需要分别检查。编码器可能合并相同画面并延长显示时间，所以要读取真实帧数和每帧时长。本次没有合并，但不能把这个结果推广到所有轨迹或编码设置。

## 官方参考资料

- [ManiSkill3.0.1观测说明](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
- [ManiSkill3.0.1控制器说明](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
- [Gymnasium Env.step](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)
- [Pillow13.0.0 GIF读写参数](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif)
- [NumPy2.4 numpy.load](https://numpy.org/doc/2.4/reference/generated/numpy.load.html)

本篇记录的是上述版本、驱动和本机实际运行结果。文档的latest链接可能更新；精确配置和本地实现SHA256保存在公开复现记录中。
