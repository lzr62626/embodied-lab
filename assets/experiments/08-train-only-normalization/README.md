# 第八篇：只用训练集计算统计量，标准化机器人状态

本篇完成离线CPU数据处理，复用第七篇3份真实轨迹，没有初始化仿真、reset、step、采集、GPU计算或模型训练。原始数据及日志保留，484份既有文件和82个依赖版本未改变。

## 运行方式

公开包解压后，在含本文件、脚本和input目录的位置，使用已有兼容Python执行：

```powershell
python -B inspect_normalization.py --input-dir input --output-dir result
```

命令先从train来源拟合一次，保存并冻结统计，再读取各集合窗口并导出示例。不会导入采集模块。同名结果会被保护；再次离线运行请改用result-2。

只有拟合与加载需求时，可分别使用：

```powershell
python -B fit_train_statistics.py --input-dir input --output-dir fitted-statistics
```

随后用FrozenNormalizer.load读取已有统计，配合NormalizedWindowDataset；Dataset构造与getitem不进行拟合。

实际环境：Python3.11.9、NumPy2.4.6、PyTorch运行时2.14.1+cpu（包元数据2.14.1）、Pillow13.0.0。requirements.txt记录实际包元数据版本。CPU版torch的安装来源需与所用环境匹配；本篇没有安装或升级依赖，新平台安装不在本次验证范围。

## 统计规则

- 固定train：seed-0042、seed-0043；validation：seed-0044。来源与第七篇一致。
- 每条20动作21状态，显式选择环境0，使用obs[0:N+1,0,:]。42个训练状态记录按“trajectory_id、状态索引”计数；相同数值仍是不同记录。
- 不从重叠窗口累计统计。34训练窗口共有170次状态取用，不等于170份独立来源状态；validation有21份状态、85次窗口取用。
- qpos、qvel各9维，逐维float64总体统计（NumPy ddof=0，PyTorch correction=0核对）。初始与最后观测均参与训练来源统计。这是离线预处理，不代表未来状态在窗口起点可见。
- 前7个qpos为rad，后2个为m；qvel分别为rad/s与m/s。安装的panda模块和panda_v2.urdf核对了7个转动、2个移动关节及顺序。
- scale=max(std,epsilon)，z=(x-mean)/scale。每维epsilon固定为1e-6，单位与原始字段相同。这是教程配置，不是所有机器人的通用阈值。
- exact_constant以原始min==max识别；protected以std<=epsilon识别；near_constant是受保护但并非精确常量的维度。

实际两根手指qpos标准差约为3.406835e-7m、2.704333e-7m（精确值以train_stats.json为准），scale均为1e-6m。它们是近似常量；真实来源没有精确常量维度。qvel各维均未触发保护。受保护训练维度的变换后标准差不必为1，validation也不必均值0/标准差1，不据此重拟合或调阈值。

## Dataset与导出

沿用K4/stride1；train34窗，9批4/4/4/4/4/4/4/4/2；validation17窗，5批4/4/4/4/1。DataLoader采用shuffle=False、drop_last=False、num_workers=0、pin_memory=False和CPU generator seed42。

标准化后的qpos/qvel为float32，首批(4,5,9)，末批保留B2/B1。RGB仍为float32/CHW/255，首批(4,5,3,256,256)。actions仍为(4,4,8)的原始归一化控制指令，其控制器语义与状态统计标准化不同。奖励、回合标志和来源/步数/时间元数据未改。

模型观测只有qpos、qvel、RGB。物体真值、控制器目标、success和标签不进入观测。首批另有raw_observations/qpos与qvel，仅是导出核对副本，不自动加入Dataset模型输入。两个首批各21个纯数值数组，统计文件20个数组，均可allow_pickle=False读取；字符串trajectory_id通过JSON映射和DataLoader元数据保留。

统计保存后以不可写bytes支撑内部数组，公开属性返回副本。每次返回的样本独立，修改不会影响原始缓存、冻结统计、重读或重叠窗口。

## 验收及遇到的问题

58项真实数据/来源/统计/窗口/批次检查和12项单独标注的私有数值测试通过。涵盖42份唯一记录、冻结与来源指纹、NumPy/PyTorch统计口径、全部窗口公式及未修改字段、B2/B1、逆变换、独立存储和首批保存重读。具体单位相关误差见normalization_report.json及validation.json。

逆变换预先固定rtol=1e-6、各原单位atol=1e-7，float32舍入后的实际误差逐维报告，没有为了通过验收放宽容差。

主导出首次通过，之后只读取已保存文件。验收工具首轮把单窗口trajectory_id字符串当作批次列表处理，第二轮将Tensor的fill_用于NumPy数组；仅修正私有验收适配，第三轮通过。两次失败日志保留在工作目录；没有覆盖成功导出或改动来源数据。人工常量/近似常量、NaN/Inf、字段顺序、错误来源和validation内存扰动测试与真实数据明确分开，不作为仿真数据公开。

validation.json是最终独立验收；batch_summary.json记录实际加载形状，normalization_report.json提供18维真实统计及变换/还原结果。统计SHA与训练来源指纹可追踪，不能仅靠改文件名切换统计来源。

## 限制与资料

3条1秒短片段只是预处理示例，不证明大规模数据分布、来源统计独立、推物成功或模型效果。没有测试集或训练。仅用train拟合并不能消除所有数据泄漏；未来需要结合任务和来源相关性继续评估。

- NumPy2.4：https://numpy.org/doc/2.4/reference/generated/numpy.std.html
- PyTorch2.14：https://docs.pytorch.org/docs/2.14/generated/torch.std.html
- Dataset/DataLoader：https://docs.pytorch.org/docs/2.14/data.html
- Pillow13.0.0：https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html

公开包只保留便携代码、真实数值来源、过滤配置及整理报告。原始日志、本机路径、设备原始枚举、人工夹具和维护/发布工具留在工作区。
