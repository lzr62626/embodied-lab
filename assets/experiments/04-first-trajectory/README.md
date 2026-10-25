# 第一段机器人动作序列

2026-10-26 实际采集：单个 PushCube-v1、panda、seed=42，20 个动作、21 份独立观测和 21 张原始 PNG。没有训练模型、追加动作或在终止后 reset 拼接。35 项保存文件验收通过。

## 文件与运行

`capture_trajectory.py` 是本次实际执行的脚本。`trajectory.npz` 保存纯数值数组，`trajectory.json` 保存计划、步数、时间、逐步统计和回放参数，`schema.json` 提供字段形状、单位与经审阅的维度说明。`validation.json` 补充独立验收和 reset-only 字段轴说明。`frames/0000.png` 到 `frames/0020.png` 是实际策略相机 RGB；`replay.gif` 为展示用的量化副本。

在现有项目根目录，复用原有解释器：

```powershell
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) 'work\sim-camera-cache'
& '.\work\sim-camera-env\Scripts\python.exe' -B '.\outputs\04-first-trajectory\capture_trajectory.py' --seed 42 --steps 20 --reference-env '.\outputs\03-observation-action\environment.json' --output-dir '.\work\04-first-trajectory\my-check'
```

输出目录必须是新的；不要覆盖当前交付结果。`--reference-env` 会在动作前比较第三篇的版本、源码 SHA256、控制器、动作空间、后端与时间配置，发现差异立即停止。公开第三篇的环境配置也可用作参考。

解压独立下载包后，使用已经配置好相同依赖的 Python：

```powershell
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) '.cache'
python -B capture_trajectory.py --seed 42 --steps 20 --output-dir result
```

没有提供第三篇参考文件时，脚本仍会在动作前检查当前动作映射、增量/绝对控制、缩放范围和动作边界；报告不会声称完成了参考文件比对。无需升级已有环境来读取 NPZ 或查看 GIF。

## 本次实际配置与动作

Python 3.11.9，ManiSkill 3.0.1，SAPIEN 3.0.3，PyTorch 2.14.1 CPU，Gymnasium 1.4.0，NumPy 2.4.6，Pillow 13.0.0。沿用 `rgb`、`base_camera`、256×256、minimal shader；CPU 物理与 CPU 可访问 SAPIEN/Vulkan 图像路径，实际后端为 physx_cpu/sapien_cpu。

控制模式 pd_joint_delta_pos：前七维是关节角增量输入，每维 `delta_rad=0.1*a`，目标相对本步测量位置计算；最后一维是夹指绝对位置输入，`target_m=0.015+0.025*a`，两根夹指 mimic 联动。

20 个动作中，第一维按五步一组依次为 +0.1、-0.1、+0.1、-0.1；其他机械臂维度为0，夹爪为1。每次第一关节请求的目标增量为 ±0.01 rad，夹指目标保持0.04 m。固定动作正负相抵，不保证实际位置精确回到起点。

## N 与 N+1

一次 step 消耗当前观测对应的动作，返回下一份观测。因此 N 个动作连接 N+1 个状态：`obs[k] → actions[k] → obs[k+1]`。本次 N=20。

| 数组 | 实际形状 | 含义 |
| --- | --- | --- |
| `obs/agent/qpos`、`qvel` | `(21,1,9)` | 时间、环境、关节；前7维 rad/rad·s⁻¹，后2维 m/m·s⁻¹ |
| `obs/extra/tcp_pose` | `(21,1,7)` | 世界 xyz(m)、qw,qx,qy,qz |
| `obs/sensor_data/base_camera/rgb` | `(21,1,256,256,3)` | 时间、环境、高、宽、RGB；uint8、0..255 |
| `actions` | `(20,8)` | 时间、动作维；这里没有额外环境轴 |
| `rewards`、终止/截断标志、step info | `(20,1)` | 动作时间、环境 |
| `step_index`、`time_s` | `(21,1)` | 观测时刻、环境；步号0..20、秒数0..1 |
| `diagnostic/cube_pose` | `(21,1,7)` | 仅供验收的仿真器物体真值，不属于策略观测 |
| `diagnostic/arm_target_qpos` | `(20,1,7)` | 每个动作设置的控制器目标，单位rad |
| `diagnostic/gripper_target_qpos` | `(20,1,2)` | 两根夹指目标，单位m |

相机外参、内参分别为 `(21,1,3,4)`、`(21,1,3,3)`，cam2world_gl 为 `(21,1,4,4)`。它们在本次固定相机轨迹中没有变化。

`reset_info/*` 只在初始化时保存，不沿时间堆叠：elapsed_steps 和 success 为 `(1,)`，环境轴为0；reconfigure 为标量，没有环境轴。公开 schema 已补充这些特殊字段的轴说明，validation.json 的 reset_info_axes 也提供对应解释；原始采集文件保留在本机。

## 时间、回合边界和实际结果

控制频率20 Hz，物理频率100 Hz；一个控制步含5个物理子步，推进0.05秒。20步共1.00秒仿真时间。结束原因是达到采集预算，所有 terminated、truncated、success 均为 false。它是一段回合片段；没有把预算停止伪装成环境的 truncated=true。

第一关节在步0、5、10、15、20的实测值约为0.009934283、0.027843354、0.011843494、0.027844150、0.011843315 rad。结束相对起点仍有 +0.001909032 rad 偏差。方块净位移范数约7.66369e-7 m，未发生可见推送，任务没有成功。

## GIF 回放

仅从保存的21张PNG编码，重新解码仍是21帧，没有帧合并；每帧50 ms，名义20 fps，每轮1050 ms。相比1.00秒仿真，多出的50 ms是最终状态展示时间。loop=0表示播放器循环；不会新增仿真步，也不表示采集了多个回合。

GIF每帧使用256色量化，原图与解码GIF的平均RGB绝对误差约0.752658（0..255数值单位）。解码图像与预期量化结果一致；颜色有损，PNG及NPZ的RGB才是原始数据。GIF帧数与时长必须重新读取，不能从保存参数猜测。

## 验收读取

```python
import numpy as np
from PIL import Image

with np.load('trajectory.npz', allow_pickle=False) as d:
    n = len(d['actions'])
    assert n <= 20 and len(d['obs/agent/qpos']) == n + 1
    assert all(not d[k].dtype.hasobject and np.isfinite(d[k]).all() for k in d.files)
    assert np.allclose(np.diff(d['time_s'][:, 0]), 0.05)
    assert not d['terminated'][:-1].any() and not d['truncated'][:-1].any()
    print('N=', n, '关节轨迹=', d['obs/agent/qpos'][:, 0, 0])

with Image.open('replay.gif') as gif:
    durations = []
    for i in range(gif.n_frames):
        gif.seek(i)
        gif.load()
        durations.append(gif.info['duration'])
    print('GIF帧数=', gif.n_frames, '每轮毫秒数=', sum(durations))
```

35项验收还检查了21张PNG与对应NPZ逐像素相等、所有历史快照指纹未变、动作与目标对应、相机固定，以及第二/三篇原始文件和82个依赖版本未改变。验收过程没有重新创建仿真。原始运行日志和退出码保存在本机工作目录，不在网站公开包内。

本次 SAPIEN 仍有 Pinocchio 未安装提示，关节控制正常完成，未安装新包。其他平台、驱动、控制模式的兼容性仍须实际验证。

## 官方参考

- [ManiSkill Observation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
- [ManiSkill Controllers](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
- [Gymnasium Env.step](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)
- [Pillow 13.0.0 GIF](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif)
- [NumPy 2.4 load](https://numpy.org/doc/2.4/reference/generated/numpy.load.html)
