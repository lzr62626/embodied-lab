# 按轨迹划分训练与验证 · 公开示例

默认命令只做CPU离线数据组织与加载，不启动仿真，不训练。包含3份完整数值轨迹及63张原图，可解压独立运行。

## 复现

已验证Windows、Python3.11.9、NumPy2.4.6、PyTorch2.14.1+cpu、Pillow13.0.0。requirements.txt记录离线依赖与CPU版索引。作者复用现有环境，没有安装/升级；其他系统和新安装组合未在本次验证。

在解压目录，用兼容Python执行：

```powershell
python -B inspect_splits.py --input-dir trajectories --output-dir result
```

已有同名导出时拒绝覆盖，重复离线复现请换result-2。三个依赖模块与数据都在包内，不依赖作者的绝对路径。

capture-requirements.txt与三个capture脚本只是可选采集源码；本次作者40次预算已经完成。默认复现不导入它们。新仿真实验须另行确定环境、预算和新目录，不能把重采当作验收。

## 真实来源与固定划分

- source_id0 / seed-0042：复用第四篇历史采集，train，20动作21观测17窗口；本次新调用0。
- source_id1 / seed-0043：本次新增，train，20动作21观测17窗口；20次尝试全部返回。
- source_id2 / seed-0044：本次新增，validation，20动作21观测17窗口；20次尝试全部返回。

固定PushCube-v1/panda、单环境、pd_joint_delta_pos、base_camera256×256、minimal shader、控制20Hz/物理100Hz。顺序采集43→44，动作计划五步+.1、五步-.1重复一次，其他臂0、夹爪1。40次新尝试在调用前登记，没有重采seed42，没有训练。

每份来源是0～1秒片段，不拼成一条3秒轨迹。总计60来源动作/63状态，其中新增40动作/42状态。success均false，方块净位移约7.66e-7～8.22e-7m，没有推物成功结论。

物理解析physx_cpu、渲染后端标签sapien_cpu，CPU可访问uint8图像。运行时Device接口返回cpu，原始设备枚举另存在本机；没有测量物理GPU选用或利用率，不能把接口标签解释成全程无GPU渲染。离线DataLoader无GPU张量。

## 窗口与批次

先固定train42+43/validation44，再分别在每条来源内部以K4/stride1切窗。总计51窗，train34窗9批(4×8批+2)，validation17窗5批(4×4批+1)。末批B2/B1保留。batch_size4、shuffleFalse、drop_lastFalse、num_workers0、pin_memoryFalse、CPU generator seed42。

train_batch.npz和validation_batch.npz各19个纯数值数组，可使用np.load(...,allow_pickle=False)读取。RGB为(4,5,3,256,256)，actions为(4,4,8)。trajectory_id字符串留在Dataset/DataLoader元数据及batch_summary.json，NPZ通过数值source_id和manifest映射还原，不使用object或字符串数组。

一个批次可以包含不同来源的完整窗口，单条窗口不会跨来源。全局16是seed42状态16～20，全局17是seed43状态0～4。B不是拼接后的时间轴。

## 验收与限制

66项真实数据/索引/批次检查PASS；11项单独标注的私有索引和重打包测试PASS。219份旧文件、82个包版本、81个采集文件/脚本哈希保持不变。原始PNG与对应NPZ RGB逐像素一致，新seed首末图和三首帧预览已重新打开；560×984预览标签位于图块外，不是模型输入。

来源集合无交集，数值/模型内容指纹不重复，跨集合精确RGB重复0。文件SHA包含压缩细节；内容指纹按键排序、形状、规范字节序及数值字节计算。私有重打包测试文件SHA不同，数值/模型指纹相同，防止改名或重新压缩伪装来源。

这不是统计独立、完全无泄漏或泛化能力证明。三个片段共享任务、资产和固定动作计划；没有测试集、训练、拟合归一化或模型效果。未来需要拟合统计时只用训练来源。

leakage_check.json中qpos最大数值差不是单一物理单位；validation.json另列前7个机械臂关节的rad差和两指m差。sample导出摘要保留pending，最终独立验收结果见validation.json。

## 文件与公开边界

dataset_manifest.json记录来源/指纹，splits.json记录预先固定的划分，window_index.json记录全局与轨迹内窗口；multi_trajectory_dataset.py纠正单轨迹固定source_id0的适用范围。source_utils.py提供指纹与划分检查。前两篇读取模块原样附带，并记录哈希。

原始日志、实际机器路径、GPU原始枚举、采集账本、人工夹具和维护验证器不在包内。每条来源的公开environment/schema/trajectory JSON经过整理，原始数值NPZ与PNG字节保持不变。本机派生GIF未包含。validation.json分别记录original_artifacts_sha256、original_source_artifacts_sha256和public_artifacts_sha256，不宣称过滤JSON和原始本机JSON逐字节相同。

官方资料：
- https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html
- https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html
- https://docs.pytorch.org/docs/2.14/data.html
- https://numpy.org/doc/2.4/reference/generated/numpy.load.html
- https://pillow.readthedocs.io/en/stable/reference/Image.html
