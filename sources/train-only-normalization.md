---
title: "只用训练集计算统计量：标准化机器人状态"
description: "从两条训练轨迹的42份唯一状态拟合统计，冻结后应用到train与validation窗口，核对近似常量、逆变换和末批维度。"
date: 2026-10-26
series: "没有机器人，也能开始的具身智能实践"
order: 8
tags:
  - 数据实践
  - 标准化
  - 数据泄漏
  - PyTorch
---

# 只用训练集计算统计量：标准化机器人状态

[第七篇](multi-trajectory-splits.html)把三份真实来源固定分成train与validation。接下来处理关节状态的数值尺度：统计从哪里来，哪些记录计数，又怎样把同一份变换应用到两个集合？

本篇于2026-10-26完成验证。只用seed42、43的42份原始状态拟合qpos/qvel统计，保存并冻结，再处理34个训练窗口与17个验证窗口。58项真实数据检查、12项单独标注的私有数值测试通过；35文件公开包独立解压运行，97项包检查通过。没有启动仿真、reset、step、新采集、GPU计算或模型训练。

[下载完整离线示例](../assets/experiments/08-train-only-normalization/normalization-files.zip)，包含3份完整真实数值轨迹、训练统计、批次和便携代码。也可查看[统计说明](../assets/experiments/08-train-only-normalization/train_stats.json)、[20数组统计文件](../assets/experiments/08-train-only-normalization/train_stats.npz)、[18维详细报告](../assets/experiments/08-train-only-normalization/normalization_report.json)、[独立验收](../assets/experiments/08-train-only-normalization/validation.json)和[运行说明](../assets/experiments/08-train-only-normalization/README.md)。

## 第一步：确定哪些状态参与拟合

划分保持不变，先确定来源再计算统计：

| 集合 | 来源 | 原始状态记录 | 窗口数 | 窗口内状态取用次数 |
|---|---|---:|---:|---:|
| train | seed42、seed43 | 21+21=42 | 34 | 34×5=170 |
| validation | seed44 | 21 | 17 | 17×5=85 |

一个K=4窗口含5份状态。相邻窗口共享状态；从窗口累计均值和标准差，会重复加权位于轨迹中间的记录。本例直接读取两条训练来源的obs[0:N+1]，包括初始与最后观测。

“唯一”指trajectory_id与状态索引组成的记录身份。同一来源的状态3只计一次；两个时刻数值碰巧相同，也仍是两条记录，不按值去重。两个集合的均值、标准差不分别拟合，validation也不参与阈值选择。

```python
# raw_qpos的形状为(T+1,E,J)，此处E=1。
states = raw_qpos[:, 0, :].astype(np.float64, copy=True)
# 只拼接训练来源的状态记录，不能先拼接来源后切时间窗口。
train_states = np.concatenate(train_state_chunks, axis=0)  # (42,9)
mean = train_states.mean(axis=0, dtype=np.float64)
std = train_states.std(axis=0, dtype=np.float64, ddof=0)
```

这里拼接的是用于离线统计的训练记录，没有改变任何轨迹边界。使用训练轨迹全部状态也不代表窗口起点能看到未来观测；模型的上下文与预测目标仍需另行定义。

## 第二步：按关节分别计算，不混合单位

实际环境为Python3.11.9、NumPy2.4.6、PyTorch2.14.1+cpu（包元数据2.14.1）、Pillow13.0.0，复用原环境。484份既有文件、82个包版本保持不变。读取前核对了第七篇清单、划分、字段、窗口、验收、重复报告与3份完整NPZ。

本地ManiSkill3.0.1的panda模块与panda_v2.urdf确认前7个为转动关节，最后2个为手指移动关节。qpos单位分别为rad、m，qvel分别为rad/s、m/s。每个字段各计算9个均值与标准差，不能把混合单位合成一个总均值。

统计使用float64和总体标准差。[NumPy的std接口](https://numpy.org/doc/2.4/reference/generated/numpy.std.html)在axis=0沿记录轴计算，显式ddof=0。[PyTorch的std接口](https://docs.pytorch.org/docs/2.14/generated/torch.std.html)默认correction=1，本篇核对时显式改为0，避免统计口径不同。

以下为真实训练统计，表格做有效位数舍入，精确值保存在JSON和NPZ。每行样本数均为42。

### qpos：位置统计

| 字段 | 原始单位 | train均值 | 总体std | 实际scale | 保护 |
|---|---|---:|---:|---:|---|
| 关节1 | rad | 0.0169537 | 0.00615026 | 0.00615026 | 否 |
| 关节2 | rad | 0.382238 | 0.00770221 | 0.00770221 | 否 |
| 关节3 | rad | 0.00272402 | 0.0102961 | 0.0102961 | 否 |
| 关节4 | rad | -1.9536 | 0.0205822 | 0.0205822 | 否 |
| 关节5 | rad | 0.00623744 | 0.0109223 | 0.0109223 | 否 |
| 关节6 | rad | 2.34972 | 0.00178891 | 0.00178891 | 否 |
| 关节7 | rad | 0.806172 | 0.0108102 | 0.0108102 | 否 |
| 手指1 | m | 0.0399998 | 3.40683e-07 | 1e-06 | 近似常量 |
| 手指2 | m | 0.0399998 | 2.70433e-07 | 1e-06 | 近似常量 |

### qvel：速度统计

| 字段 | 原始单位 | train均值 | 总体std | 实际scale | 保护 |
|---|---|---:|---:|---:|---|
| 关节1 | rad/s | 0.000700452 | 0.0652216 | 0.0652216 | 否 |
| 关节2 | rad/s | -1.59595e-05 | 0.000144927 | 0.000144927 | 否 |
| 关节3 | rad/s | 0.000359153 | 0.00479784 | 0.00479784 | 否 |
| 关节4 | rad/s | -2.50102e-06 | 8.03566e-05 | 8.03566e-05 | 否 |
| 关节5 | rad/s | -1.06763e-05 | 0.00015533 | 0.00015533 | 否 |
| 关节6 | rad/s | 2.31128e-06 | 1.16791e-05 | 1.16791e-05 | 否 |
| 关节7 | rad/s | -2.23951e-06 | 2.77377e-05 | 2.77377e-05 | 否 |
| 手指1 | m/s | 3.12221e-05 | 0.0002386 | 0.0002386 | 否 |
| 手指2 | m/s | -5.48428e-06 | 0.000238674 | 0.000238674 | 否 |

## 第三步：保护近似常量维度，再冻结统计

本篇在查看验证结果之前固定每维epsilon数值为1e-6，其单位跟随原始字段。它是本教程的数值保护配置，并非所有机器人的通用选择。

```python
scale = np.maximum(std, epsilon)
z64 = (raw.astype(np.float64) - mean) / scale
z32 = z64.astype(np.float32)
```

两根手指qpos的标准差约3.406835e-7m和2.704333e-7m，均小于epsilon，实际scale使用1e-6m。它们是近似常量，真实来源没有精确常量维度。qvel各维均未触发保护。

| qpos维度 | 原std，m | 实际scale，m | 训练状态变换后std |
|---|---:|---:|---:|
| 手指1 | 3.406835e-7 | 1e-6 | 0.340683 |
| 手指2 | 2.704333e-7 | 1e-6 | 0.270433 |

受保护维度的标准差小于1是公式的实际结果。精确常量按原始min==max识别；protected按std<=epsilon识别；near_constant是受保护且并非精确常量。不会因为std很小就把真实值写成全0。

[normalization.py](../assets/experiments/08-train-only-normalization/normalization.py)保存均值、原std、最小/最大值、epsilon、实际scale、三类掩码和计数，共20个纯数值数组。JSON还记录训练来源、状态索引、来源SHA与数值指纹、字段顺序、单位及算法版本。

统计保存后不再随getitem或验证加载更新。FrozenNormalizer用不可写bytes支撑内部数组，公开属性返回副本；加载时核对NPZ的SHA、JSON数值以及算法/顺序/单位。接入Dataset时再核对训练组和来源指纹，不能靠改文件名套用另一份统计。

## 第四步：用同一份统计读取两个集合

公开包解压后，在包含input目录与脚本的位置使用已有兼容Python执行：

```powershell
python -B inspect_normalization.py --input-dir input --output-dir result
```

此命令先从train拟合一次，保存并冻结，然后读取validation与窗口并导出报告。结果目录已有同名文件会被保护，再次离线复现请改用result-2。也可用[fit_train_statistics.py](../assets/experiments/08-train-only-normalization/fit_train_statistics.py)单独生成统计，再用FrozenNormalizer.load读取；[标准化窗口包装器](../assets/experiments/08-train-only-normalization/normalized_window_dataset.py)只应用变换。

```python
sample = base[index]  # 第七篇已验证的独立窗口
for field in ("qpos", "qvel"):
    sample["observations"][field] = normalizer.transform(
        field, sample["observations"][field]
    )
```

原有K4/stride1、来源映射、全局/局部窗口索引和时间保持不变。[DataLoader](https://docs.pytorch.org/docs/2.14/data.html)采用batch_size4、shuffleFalse、drop_lastFalse、num_workers0、pin_memoryFalse与CPU generator seed42，实际遍历为：

```text
train:      34 windows -> [4,4,4,4,4,4,4,4,2]
validation: 17 windows -> [4,4,4,4,1]
```

| 字段 | 首批B4 | train末批B2 | validation末批B1 |
|---|---|---|---|
| 标准化qpos/qvel | (4,5,9) | (2,5,9) | (1,5,9) |
| RGB | (4,5,3,256,256) | (2,5,3,256,256) | (1,5,3,256,256) |
| actions | (4,4,8) | (2,4,8) | (1,4,8) |

B是批次轴，5是K+1状态时间轴。没有用无条件squeeze删除末批维度。qpos/qvel先以float64计算，再返回float32 CPU张量；RGB继续CHW并除以255，不拟合图像通道统计。

actions保持原始归一化控制指令，reward、terminated、truncated及所有索引未改。控制器把归一化动作映射成目标，与这里把观测减均值、除尺度是不同操作。物体真值、控制器目标、success及来源/集合标签仍不进入模型观测。

## 第五步：验收原值、变换与逆变换

[train首批](../assets/experiments/08-train-only-normalization/train_batch.npz)和[validation首批](../assets/experiments/08-train-only-normalization/validation_batch.npz)各21个纯数值数组，均可allow_pickle=False读取。新增raw_observations/qpos与qvel仅供导出核对，Dataset的模型观测仍只有qpos、qvel、RGB，不自动把原始核对副本加入输入。

逆变换为raw_hat=z*scale+mean。验收之前已固定rtol=1e-6、每原始单位atol=1e-7，实际float32舍入误差逐维记录。两集合合计最大误差按物理单位分别为：

| 原始字段范围 | 最大绝对还原误差 | 单位 |
|---|---:|---|
| qpos机械臂7维 | 2.39754e-9 | rad |
| qpos手指2维 | 5.10911e-14 | m |
| qvel机械臂7维 | 3.87472e-9 | rad/s |
| qvel手指2维 | 2.58062e-11 | m/s |

58项真实数据验收覆盖42份记录、全体窗口公式、原值与来源对应、未修改字段、B2/B1、冻结统计、首批重读及独立副本。修改返回窗口不会影响来源、再次读取或相邻窗口；修改公开返回的统计副本也不会改变冻结值。

12项私有测试另外覆盖常量、近似常量、非有限值、错误维度/字段顺序/训练组/来源指纹。在私有内存测试中改变validation数值，训练均值、std和scale完全不变；记录到的拟合读取只有两条train NPZ。人工测试不作为真实轨迹公开，也没有覆盖原始数据。

主导出首次通过。验收工具前两次分别遇到单窗口字符串与批次列表的适配问题、对NumPy数组误用Tensor的fill_；仅修正私有验收代码后通过，失败日志仍在本机，没有重写成功导出或重新采集。

## 常见问题与本次限制

**为什么不能分别标准化train和validation？** 那会让验证变换依赖验证资料的分布。本篇固定训练统计后，validation第一关节qpos的变换均值约-3.66646、标准差约0.927608。这是这份资料在固定尺度下的实际结果，不据此重拟合。

**为什么不要求每个训练批次均值0、标准差1？** 统计对象是42份唯一训练状态，批次包含的窗口重复权重与状态范围不同。只对未受保护维度的完整唯一训练记录检查接近0/1，不能把这个要求强加给单个窗口或批次。

**相同数值要去重吗？** 不按值去重。记录身份是来源加状态索引；窗口重复引用同一记录才是本篇避免的重复加权。

**标准化后能否直接解释为弧度或米？** z是逐维无量纲结果。需要物理量时用相同统计逆变换；不要混淆观测标准分数与动作的[-1,1]控制输入。

**是否证明模型更好或数据已经独立？** 没有。本次不训练、不设测试集、不报告泛化。3条同任务短片段仅演示预处理；只用train拟合也不能证明所有泄漏已经消除。

## 官方参考资料

- [NumPy2.4：std的axis、dtype与ddof](https://numpy.org/doc/2.4/reference/generated/numpy.std.html)
- [PyTorch2.14：std与correction](https://docs.pytorch.org/docs/2.14/generated/torch.std.html)
- [PyTorch2.14：Dataset与DataLoader](https://docs.pytorch.org/docs/2.14/data.html)
- [NumPy2.4：禁用pickle读取数值文件](https://numpy.org/doc/2.4/reference/generated/numpy.load.html)
- [Pillow13.0.0版本资料](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)
