# 第16篇公开复现包：参考反馈的配对评估

解压后在本文件所在目录，使用已有兼容Python环境运行：

```text
python -B evaluate.py --output-dir result-r001
```

只读CPU验收，不创建仿真环境、不reset、不step、不训练或拟合参数。输出目录必须不存在。需要Python3.11、NumPy2.4.6、Pillow13.0.0；这是已实际验证的兼容环境说明，不要求升级现有研究环境。

包内有8条完整数值轨迹及168张原始PNG：两个seed各U/A/D/E四条件。U/A/D为历史复用，只有E来自第16篇新增运行。dataset_manifest.json提供稳定来源ID、相对路径、条件和原始NPZ的SHA-256。

输出metrics.json、逐步CSV/NPZ和validation.json。指标应与本包根目录metrics.json完全一致；保存的comparison NPZ数组也应逐项相同。程序复算D/E反馈、PNG像素、N/N+1、动作来源、时间和回合标志，重新计算配对响应。默认不包含任何自动采集行为。

文件SHA-256清单为files.json。provenance.json记录读取模块及反馈控制器的来源哈希，公开适配只移除了私人采集账本与本机文件库存的检查；数值计算与真实数据保持一致。私有账本和2061项旧文件保护已在作者项目内核对，不在这个公开包中重新检查。

所有NPZ均可使用allow_pickle=False读取；字符串映射在JSON。模型没有在本篇训练；两个seed已使用过，不是独立测试集。全部条件success=false，数据不证明推物能力。
