# 第九篇 · 用 PyTorch 训练第一个行为克隆基线

本实验用当前关节位置与速度预测同一时刻演示动作的第 1 维。输入来自三条固定来源轨迹，标准化使用第八篇冻结的训练集统计。训练为 seed42、43 的 40 条 transition，验证为 seed44 的 20 条 transition。

## 解压后复现离线评估

实测环境为 Python 3.11.9、PyTorch 2.14.1+cpu、NumPy 2.4.6。用已有兼容环境，在解压目录执行以下命令。

```powershell
python -B evaluate_bc.py --input-dir input --run-dir run --output-dir reproduced-evaluation
```

`reproduced-evaluation` 应为新目录。脚本检查冻结统计和轨迹指纹，加载 `run/model_weights.npz`，用 CPU 重算 `predictions.csv`、`predictions.npz` 和 `metrics.json`。与随包的 `evaluation/` 对照即可。

若要另行复现训练，以下命令会创建一个新实验。固定参数记录在 `run/config.json`。

```powershell
python -B train_bc.py --input-dir input --output-dir reproduced-training
```

## 文件与结果

- `input/` 保存三个完整数值轨迹、固定划分和冻结统计。`input-provenance.json` 保存每份源数据与参考代码的 SHA-256。
- `run/` 保存最终数值权重、训练配置、每轮 MSE、更新次数、首末批索引，以及重载对照用的训练集预测。
- `evaluation/` 保存 seed44 的 20 步预测和按样本汇总的指标，`t` 从 0 开始，5、10、15 标记为换向。
- `validation.json` 保存独立验收结果。各 NPZ 均可用 `np.load(..., allow_pickle=False)` 读取。

正式训练一次，共 100 轮和 500 次优化器更新。训练均值基线预测 0。验证集上，基线与 MLP 的 MSE 分别约为 0.01000 和 0.01600，MAE 分别约为 0.10000 和 0.08001。三个换向点中，MLP 在 t=5 和 t=15 预测了与演示相反的方向。三条轨迹使用同一换向脚本，指标的范围是离线动作拟合。
