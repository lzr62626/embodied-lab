# 第11篇 · 用短历史预测机器人动作

本包包含真实数值轨迹、原尺寸相机 PNG、冻结训练统计、已保存权重和 CPU 离线评估脚本。使用 Python 3.11.9 及 numpy 2.4.6、torch 2.14.1、Pillow 13.0.0 验证。

## 解压后运行

在解压目录执行 `python -B evaluate.py --output-dir result`。已有 result 时选择一个新目录。

脚本核对 package-manifest.json 中的 SHA-256，重新读取全部 NPZ 和 PNG，重新加载权重并复算指令与报告。结果写入 result/report.json。图像、物体真值、时间和来源标签保留用于验收，模型只读取关节状态。

## 本次结果

- S 完成 20 步，第一关节相对脚本最大偏差 0.03071988 rad，success 步数 0。
- H 完成 20 步，第一关节相对脚本最大偏差 0.02588769 rad，success 步数 0。

每个动作 t 对应 obs[t] 到 obs[t+1]。各组末步为预算结束，原 terminated/truncated 均为 false。

第十篇比较同一起点的脚本、常数和单状态 MLP。第十一篇比较同结构 S、H 模型，S 把当前状态重复四次，H 拼接最近四份状态。三条源轨迹共用同一换向脚本，seed44 已被用于先前的离线及闭环检查，报告的范围限于这些固定资料。

requirements.txt 列出实际验证版本。评估依赖均从现有环境复用，数据处理和模型推理使用 CPU。
