# 把轨迹变成数据集 · 公开文件包

本包只执行离线CPU数据读取，不启动仿真、不训练模型、不使用GPU。包含真实来源轨迹，解压后无需另行采集或依赖作者的目录。

## 已验证环境

Windows、Python3.11.9、NumPy2.4.6、PyTorch2.14.1+cpu、Pillow13.0.0。requirements.txt列出实际版本和CPU版PyTorch索引。本次复用已有环境，未安装或升级依赖；其他系统和新环境安装组合未在本次验证。

若已有兼容环境，进入解压目录后执行：

```powershell
python -B inspect_dataset.py --trajectory source/trajectory.npz --output-dir result
```

也可以省略 --trajectory，默认读取脚本旁的 source/trajectory.npz。result必须没有同名导出文件；再次运行请使用新的目录，例如 result-2。此脚本不会修改来源轨迹，也不会覆盖既有导出。

在独立的新环境中可按需要使用 python -m pip install -r requirements.txt；这不是本次作者环境的执行步骤。

## 已验证结果

来源为第四篇真实PushCube-v1/panda、seed42、base_camera256×256轨迹：20个动作、21份观测、仿真1秒。源文件SHA256记录在environment.json和sample_batch.json。

Dataset长度20，DataLoader配置batch_size4、shuffleFalse、drop_lastFalse、num_workers0、pin_memoryFalse，CPU generator seed42。实际得到5个批次，索引0～19无遗漏重复。

sample_batch.npz包含首批次17个纯数值数组，可用np.load(...,allow_pickle=False)读取。qpos/qvel为(4,9)，RGB为(4,3,256,256)，action为(4,8)，reward/terminated/truncated为(4,)。原始RGB实际范围34～255，首批次当前RGB为约0.13333334～1.0；图像只改变通道顺序和数值尺度。

36项独立数值/存储/对应关系检查通过；sample_preview.png已重新打开目视检查。8个图块与原始RGB逐像素一致，标签位于图外，PNG为560×1288。相邻样本共享时刻内容，但返回张量存储独立。

## 字段与限制

每条样本为obs[i]→action[i]→obs[i+1]。显式选择环境0；观测只包含qpos/qvel/RGB。源文件仍保留其他原始观测和独立诊断字段，Dataset不把它们加入输入。

qpos前7维为rad、后2维为m；qvel前7维为rad/s、后2维为m/s。动作保留归一化原值；前7维对应0.1*a rad的关节目标增量，第8维对应0.015+0.025*a m的夹爪绝对目标，不是速度。

最后一条连接step19→20、0.95→1.0秒。is_last_sample=true只表示本文件片段末尾；terminated和truncated都保持false。

20条相邻样本只演示数据管线，没有训练/验证划分或模型效果。方块未明显移动，不代表推物成功。后续多轨迹实验应考虑先按轨迹/回合划分再采样，减少共享观测泄漏。当前实现整体读入一个小NPZ，未验证大数据集性能。

## 文件与验收

- trajectory_dataset.py、inspect_dataset.py：可运行的离线示例。
- source/trajectory.npz、source/schema.json、source/environment.json：真实来源数据及已过滤的公开说明。
- sample_batch.npz、sample_batch.json：本次真实首批次及实际统计。
- sample_preview.png、schema.json：真实样本对照图与输出字段说明。
- environment.json、validation.json：公开版本配置、独立验收摘要和哈希。

validation.json中的original_artifacts_sha256指本地原始导出；public_artifacts_sha256指本包中的已整理文件。过滤后的配置与说明不宣称和本地文件逐字节相同。原始运行日志、机器路径和维护用验收工具不在包内。

## 官方资料

- https://docs.pytorch.org/docs/2.14/data.html
- https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html
- https://numpy.org/doc/2.4/reference/generated/numpy.load.html
- https://pillow.readthedocs.io/en/stable/reference/Image.html
