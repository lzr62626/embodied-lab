# 第12篇 · 公开离线复核包

解压后在当前目录，使用已安装兼容NumPy、PyTorch CPU和Pillow依赖的Python执行：

```bash
python -B evaluate.py --output-dir result
```

result须为尚不存在的新目录。脚本验证package-manifest.json的SHA-256，读取全部数值数组，重建输入、复算指令或诊断结果，并输出result/report.json。默认不训练、不启动仿真、不reset或step。

requirements.txt记录实际复核版本。所有NPZ均为纯数值或布尔数组，用allow_pickle=False读取。原始轨迹、权重和统计的数值内容保持不变。配置、来源说明和验收摘要经过公开整理。

公开目录仅包含离线复核代码与必要资料，不包含原始日志、本机路径、设备枚举或人工测试夹具。

结果只适用于教程中的短轨迹与固定控制规则。任务success为false的记录不表示推物成功。第15篇控制器额外使用参考轨迹和时钟，不能当作与历史模型输入条件相同的比较。
