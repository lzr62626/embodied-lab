# 从单步到序列 · 公开时间窗口示例

本包使用第四篇完整真实轨迹，在CPU上离线构造窗口。没有仿真初始化、reset、step、新采集、训练或GPU使用。

## 运行

已验证Windows、Python3.11.9、NumPy2.4.6、PyTorch2.14.1+cpu、Pillow13.0.0。requirements.txt记录实际版本与CPU版PyTorch索引；作者复用现有环境，未安装或升级。其他系统与新安装组合未在本次验证。

使用兼容Python，在解压目录运行：

```powershell
python -B inspect_windows.py --trajectory source/trajectory.npz --output-dir result
```

可省略--trajectory，默认读取脚本旁的source/trajectory.npz。脚本默认K=4、stride=1，保护已有输出；再次执行请使用result-2等新目录。

trajectory_dataset.py是第五篇原样代码副本，哈希记录在environment.json；它与本篇两个脚本放在同一目录，没有作者工作区路径依赖。新环境可按需要使用python -m pip install -r requirements.txt，这不是本次作者环境的执行步骤。

## 真实结果

来源PushCube-v1/panda、seed42、base_camera256×256、20个动作、21份观测，仿真时间仍为1秒。source/trajectory.npz与第四篇数值文件逐字节一致。

实际K4、stride1得到17个完整窗口，起点0～16。首窗口状态0～4/动作0～3/时间0～0.20秒；末窗口状态16～20/动作16～19/时间0.80～1.00秒。

DataLoader配置batch_size4、shuffleFalse、drop_lastFalse、num_workers0、pin_memoryFalse、CPU generator seed42。5批大小4/4/4/4/1。首批17个纯数值数组；qpos/qvel为(4,5,9)，RGB为(4,5,3,256,256)，actions为(4,4,8)，rewards/terminated/truncated为(4,4)。末批RGB保持(1,5,3,256,256)，没有squeeze。

39项真实数据检查PASS；16项单独标注的私有人工数值边界测试PASS。真实来源两个终止标志全为false；人工末动作终止测试不是新的仿真结果，夹具和维护验证器不在本包中。

window_preview.png为560×1592，首窗口5个256×256原始RGB图块未裁剪、缩放或改色。对照图逐像素检查和重新打开目视检查通过，标注在图外，不属于模型输入。

## 语义与限制

每个窗口内observations[j]→actions[j]→observations[j+1]。只使用qpos/qvel/RGB观测子集，显式选择环境0。B是批次轴，K/K+1分别是动作/状态时间轴，C/H/W为通道/高/宽。

RGB转float32/CHW并除以255。qpos/qvel/action保持原值；关节位置前7维rad、后2维m，速度分别rad/s和m/s。动作前7维对应0.1*a rad目标增量，末维夹爪目标为0.015+0.025*a m。动作不是实测状态。

source_id=0通过window_index.json映射到来源SHA256，表示单个文件片段。end_index_exclusive=s+K是动作右开端点，也是最后一份被包括状态的索引；状态切片右开端点是s+K+1。

不拼接不同文件/reset，不允许终止后继续包含动作。末动作终止可以保留最后观测；K>N返回空Dataset，导出明确拒绝且不写空示例；无效参数报错，尾窗不完整舍弃。ends_fragment独立于原始terminated/truncated，不改写回合标志。

返回窗口存储独立；相邻窗口共享内容，不共享可修改缓存。17个重叠窗口不是17条独立轨迹，没有新增仿真时间。当前整体读入小轨迹再stack，未验证大规模性能。

本篇是离线序列记录，不默认未来观测在窗口起点可见。上下文、目标和可见时刻须为具体模型另行定义。本次不拆训练/验证、不训练或报告泛化；未来应考虑先按轨迹/回合划分再切窗。

## 文件与验收

window_index.json保存17个起点和完整索引表；sample_batch.npz/json提供真实首批与实际批次统计。sample_batch.json保留导出阶段的pending状态，最终独立验收结果见validation.json。

validation.json的original_artifacts_sha256指本地原始导出；public_artifacts_sha256指本包整理后的文件。过滤后的环境说明与便携README不宣称和本地原文件逐字节相同。原始日志、本机路径、人工夹具和维护工具均不公开。

官方资料：
- https://docs.pytorch.org/docs/2.14/data.html
- https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html
- https://numpy.org/doc/2.4/user/basics.indexing.html
- https://pillow.readthedocs.io/en/stable/reference/Image.html
