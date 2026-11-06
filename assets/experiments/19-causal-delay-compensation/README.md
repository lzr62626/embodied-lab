# 第19篇公开复核包：因果线性外推补偿

解压后，在README所在目录，用已有兼容Python环境执行：

```text
python -B evaluate.py --output-dir result-r001
```

输出目录必须不存在。默认只读CPU离线验收：不启动仿真、不reset/step、不采集、训练、拟合或搜索参数。已用Python3.11.9、NumPy2.4.6、Pillow13.0.0实际验证，无需升级研究依赖。默认评估不要求安装机器人仿真器或训练库。

14份完整真实数值来源、294张原始PNG：两个seed各U、C0、P0、C1、P1、CE、PE。U为第13篇参考，C0为第16篇即时无干预反馈，P0为第15篇即时有干预反馈，C1/P1为第18篇延迟反馈；只新增本篇CE/PE四组。旧审计保持原字段，没有补造外推查询。

本篇最新可用测量仍为t-1，另读t-2，以float64先算delta=latest-previous，再算estimated=latest+delta。gain1、原动作限幅、当前参考时钟和底层PD当前状态锚点不变。动作9使用状态7/8，动作10使用8/9，动作19使用17/18。当前真值在指令算完后仅作诊断。没有使用当前qvel、RGB、方块、success或历史动作。

estimate_error=estimated-current与hold_error=latest-current在同一实际新运行的决策9～19上比较。hold不是另一条执行过的控制器。跟踪偏差和配对r0/r1/rE在各组共同状态9～20计算，不能混淆两种时间范围和误差。四组估计RMSE更低但最大估计误差更大；CE更接近参考，PE仍有末端偏差，全部任务success=false。

evaluate.py先验证files.json SHA，再以allow_pickle=False读取纯数值。独立核对N/N+1、PNG像素、全部新旧反馈审计、两个过去索引/时间、限幅标志、当前PD锚点、指标和20份CSV；不调用实际缓存、决策或标量反馈函数来验证自己。输出metrics.json、estimation_metrics.json及全部comparison/estimation NPZ与本包根目录对应。

schema.json说明字段/单位/维度；module_sources.json和provenance.json记录代码来源SHA。作者已核对私人80次调用账本及5083个旧文件，公开入口只重算数值证据，不访问私人库存。validation.json中人工测试只报告结论，人工夹具不包含在包内。

手写外推不是训练出的动力学模型。两个seed已经使用，参考/时钟已知，各组1秒，历史对照复用；不作普遍稳定、恢复、泛化、推物或真实机器人时延结论。原始日志、本机绝对路径、设备原始枚举、人工夹具和维护工具均不公开。包内没有自动采集入口。
