# 第20篇公开复核包：从估计误差到动作指令

解压后，在本README所在目录，用已有兼容环境执行：

```text
python -B inspect_diagnostics.py --output-dir result-r001
```

输出目录必须不存在。默认只读复算：不启动仿真、不reset/step、不采集、不训练、不拟合统计、不改参数。需要NumPy2.4.6和Pillow13.0.0，作者实际验证环境Python3.11.9；不要升级已有研究环境。Matplotlib3.11.2只记录原图表生成版本，默认验收不导入它。

若要另外重建诊断文件到新目录：

```text
python -B diagnose_estimation_action.py --output-dir recomputed-r001
```

六条完整数值轨迹和126张原始PNG均为历史复用：seed45/46各U来自第13篇，CE/PE来自第19篇；四份feedback_audit.npz是原始真实反馈审计。本篇新增仿真运行0，实际诊断44条，原动作9～19，状态20不额外生成决策。相对路径由source_manifest提供，无需作者本机路径。

数据和脚本一览：diagnostic_records.npz保存全部44行数值；sample_index和source_manifest保存来源字符串映射；两个CSV覆盖完整字段；case_summary保留所有预定时刻与峰值并列；metrics提供逐组及总计统计；historical_motion是第19篇实际状态9～20；schema标注单位和轴。全部NPZ均可np.load(...,allow_pickle=False)读取。

保持gain1、0.1rad物理尺度、原float32边界0.10000000149011612；float64 atol=1e-12、rtol=0，控制器目标容差1e-6rad。sent_current只是在本条当前状态上离线重算的即时测量指令，没有实际执行，不是已知正确标签，不具有新reward/下一状态或成功记录。

真实结果：A0/B28/C0/D16。A为raw相同；B为raw不同但f64限幅相同；C为f64限幅不同但f32发送相同；D为发送不同。采用严格数值相等，+0与-0相等，位模式另列；触边不等于严格截断。位置误差rad，指令差为归一化控制量；不能直接把指令差当作真实运动差。

独立验收从原始qpos逐条重算，不调用生成器的核心函数。检查原始动作/审计按位一致、N/N+1、时间、边界、PNG逐像素一致、外推/指令差恒等式、四类互斥、CSV和指标，重开30个未缩放图块。files.json封存包内文件SHA。人工数值测试仅保留作者工作目录，未作为机器人数据公开。独立目录复核只运行离线脚本。

这些数据和起点均已使用过，参考轨迹与参考时钟已知。分类仅解释保存数据中的数值关系；不能推出假想即时控制器或去掉限幅后的运动，也不能证明普遍稳定性、恢复能力或推物能力。
