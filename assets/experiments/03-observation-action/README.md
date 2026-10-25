# 一条真实的观测与动作记录

本实验于 2026-10-26 完成：PushCube-v1、panda、单环境、seed=42，执行一次 `env.step()`，不训练模型。图像由 SAPIEN/Vulkan 实际渲染，取自策略观测相机 `base_camera`，未重绘、调色或标注。

## 交付文件

- `capture_transition.py`：本次实际执行的脚本，SHA256 与原始环境报告一致；仅一个 `env.step` 调用点。
- `before.png`、`after.png`：同一相机、相邻时刻的原始 256×256 RGB 图像。
- `transition.npz`：31 个数值数组，可用 `allow_pickle=False` 读取。
- `transition.json`：采集时生成的字段路径、形状、类型、动作、控制器目标和变化统计。
- `schema.json`：独立验收时补充的完整字段含义、单位与动作语义，不改写采集记录。
- `environment.json`：公开复现配置，包含版本、控制器、后端与时间参数，已移除本机路径。
- `validation.json`：之后完成的 33 项独立检查与两张图像的视觉检查摘要。
- `requirements.txt`：本次使用的主要依赖版本；迁移环境中的其他包未升级。

## 运行命令

在现有项目根目录运行，复用第二篇已经跑通的环境：

```powershell
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) 'work\sim-camera-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\work\sim-camera-env\Scripts\python.exe' -B '.\outputs\03-observation-action\capture_transition.py' --seed 42 --output-dir '.\work\03-observation-action\my-check'
```

`my-check` 必须尚无结果。不要使用已有交付目录作为输出位置。脚本检测到已有 PNG、NPZ 或环境报告时会拒绝覆盖。初始化过程中 ManiSkill 自己也会 reset；记录中的 `explicit_reset_calls=1` 指脚本显式的种子初始化，不声称构造阶段只 reset 一次。

如果解压网站下载包到独立目录，使用已经配置好相同依赖的 Python：

```powershell
$env:SIM_CAMERA_CACHE = Join-Path (Get-Location) '.cache'
python -B capture_transition.py --seed 42 --output-dir result
```

不需要通过升级已有环境来读数据。纯数据检查只依赖 NumPy；查看 PNG 可使用普通图片查看器或 Pillow。

## 实际配置

Python 3.11.9，ManiSkill 3.0.1，SAPIEN 3.0.3，PyTorch 2.14.1（CPU 构建），Gymnasium 1.4.0，NumPy 2.4.6，Pillow 13.0.0。Windows 10.0.26200，RTX 4060 Ti，驱动 610.88。

`obs_mode=rgb`，`control_mode=pd_joint_delta_pos`，相机 256×256，`shader_pack=minimal`。请求 `sim_backend=cpu, render_backend=cpu`，解析为 `physx_cpu, sapien_cpu`；图像是 CPU 可访问的，渲染仍使用 GPU 的 Vulkan 路径。

控制频率 20 Hz，物理仿真频率 100 Hz；本次一次 step 为 0.05 秒，内部有 5 个物理子步。这个时间是仿真时间，不是程序执行耗时。

## 读懂观测

`obs_t/` 和 `obs_tp1/` 分别表示动作前、后的观测；以下路径省略这两个前缀。所有叶子原先都是 CPU Torch 张量，保存后是相同形状与数值的 NumPy 数组。

| 字段 | 形状 / 类型 | 含义与单位 |
| --- | --- | --- |
| `agent/qpos` | `(1, 9)` float32 | 前 7 维为 `panda_joint1..7` 关节角，rad；后 2 维为两个夹指的位置，m |
| `agent/qvel` | `(1, 9)` float32 | 同样顺序的速度；前 7 维 rad/s，后 2 维 m/s |
| `extra/tcp_pose` | `(1, 7)` float32 | 世界坐标中的工具中心点位姿，`x,y,z,qw,qx,qy,qz`；位置 m，四元数无量纲 |
| `sensor_data/base_camera/rgb` | `(1, 256, 256, 3)` uint8 | batch、高、宽、RGB；0..255，未归一化 |
| `sensor_param/base_camera/extrinsic_cv` | `(1, 3, 4)` float32 | 世界到 OpenCV 相机坐标的 `[R\|t]`，旋转无量纲、平移 m |
| `sensor_param/base_camera/cam2world_gl` | `(1, 4, 4)` float32 | OpenGL/Blender 相机坐标到世界坐标的齐次变换 |
| `sensor_param/base_camera/intrinsic_cv` | `(1, 3, 3)` float32 | OpenCV 相机内参；焦距、主点用像素表达 |

第一维 `1` 表示只有一个环境。实际返回的 `agent` 没有 `controller` 字段：当前控制器没有需要暴露的额外观测状态。也没有 `extra/obj_pose` 或 `extra/goal_pos`；`rgb` 模式下不能把仿真器额外读取的方块真值当作策略输入。

官方概念页列出的矩阵形状不能替代运行检查：本次 `extrinsic_cv` 实际为 `(1,3,4)`。`schema.json` 和 NPZ 是本次实际字段的依据。

## 读懂动作

传入 `env.step` 的动作形状是 `(8,)`、float32，八维范围均为 `[-1,1]`。它与 `(1,9)` 的观测关节数组不同：两根夹指由一个 mimic 控制输入驱动。

- `a[0:7]`：归一化关节角增量；`delta_q = 0.1*a` rad。`use_delta=True, use_target=False`，目标从动作前测得的关节位置计算：`q_target = q_t + delta_q`。
- `a[7]`：夹指绝对目标位置；`target_m = 0.015 + 0.025*a[7]`。第二根夹指跟随第一根。`-1` 对应 -0.01 m 的闭合驱动目标，`0` 对应 0.015 m，`1` 对应 0.04 m；负目标不是实际可达到的负开口。
- 因此，全零动作会把夹指目标设到 0.015 m，不能把它解释成整个机器人保持原状态。

本次动作：`[0.2,0,0,0,0,0,0,1]`。它要求第一个关节目标增加 0.02 rad，其他机械臂关节目标保持动作前测量值，夹爪保持 0.04 m 的打开目标。动作不是速度、力矩，也不是末端 xyz 位移。

## 实际结果

第一个关节：`q_t=0.0099342829` rad，控制器目标 `0.0299342833` rad；0.05 秒后实测 `q_tp1=0.0148822637` rad，实际增量 `0.0049479809` rad。目标与下一帧测量值并不相等。其他关节也有小幅动态变化，不能假定零增量会让实际位置严格不变。

两张 PNG 都是 uint8 RGB，像素范围 34..255。2,986/65,536 个像素变化，平均绝对像素差约 0.694585（原始 0..255 灰度级单位，按全部 RGB 通道计算）。方块真值位移范数约 `7.6654e-7` m，不到 1 微米，没有可见方块位移。

`reward=[0.0700626820]`，`terminated=[false]`，`truncated=[false]`，`info_tp1/success=[false]`。奖励为正不代表成功。本次只验证相邻观测与动作正确对应；不能据此认定推物成功，也没有通过额外零动作对照分离物理沉降与动作的因果效果。

## 验收方法

```python
import numpy as np

with np.load('transition.npz', allow_pickle=False) as d:
    assert len(d.files) == 31
    assert all(not d[k].dtype.hasobject and np.isfinite(d[k]).all() for k in d.files)
    a = d['action']
    q0, q1 = d['obs_t/agent/qpos'], d['obs_tp1/agent/qpos']
    assert a.shape == (8,) and a.dtype == np.float32
    assert ((a >= -1) & (a <= 1)).all()
    assert d['step_index_t'].tolist() == [0]
    assert d['step_index_tp1'].tolist() == [1]
    assert np.allclose(d['time_tp1_s'] - d['time_t_s'], 0.05)
    assert np.allclose(d['diagnostic/arm_target_qpos'], q0[:, :7] + 0.1*a[None, :7])
    print('实际关节变化：', q1 - q0)
    print('奖励和终止标志：', d['reward'], d['terminated'], d['truncated'])
```

`diagnostic/*`、`info_*`、时间与步计数是单独记录的验收信息，不增加到策略观测。两张 PNG 与对应 NPZ 的 `rgb[0]` 逐像素一致；动作前快照使用 `.detach().cpu().numpy().copy()`，采集脚本还检查 step 后快照指纹未变。验收读取保存文件，不重新启动仿真。

原始运行日志与退出码留在作者本机，不包含在此公开文件包中。本次首轮运行成功，没有失败重跑；SAPIEN 提示 Pinocchio 未安装，但这里的关节控制无需使用末端逆运动学，未新增依赖。

## 参考资料

- [ManiSkill Observation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
- [ManiSkill Controllers](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
- [Panda 控制器配置](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/agents/robots/panda/panda.py)
- [PD 关节位置控制器](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/agents/controllers/pd_joint_pos.py)
- [动作裁剪和缩放](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/utils/gym_utils.py)
- [PushCube 任务实现](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/tasks/tabletop/push_cube.html)
- [NumPy 2.4 数据读取](https://numpy.org/doc/2.4/reference/generated/numpy.load.html)

参考链接的 latest/main 内容会随维护更新。本次精确行为以已安装版本、保存的控制器配置、本地源码 SHA256 和实际数值为准。跨平台或驱动变化后的像素级一致性不作保证。
