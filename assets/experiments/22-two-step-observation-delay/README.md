# 第22篇 · 观测晚到两步

用 PushCube-v1 的两条已知起点，比较直接用旧位置和用旧速度外推。每个 seed 复用无干预脚本 U、一步 Hold 和一步 Velocity，并各新增一条两步 Hold 和两步 Velocity。四个反馈条件使用同一覆盖动作。实际新增四组，各 20 步，共 80 次 `env.step`。运行时钟每步 0.05 s，两个周期为 0.10 s。

## 从保存文件复算

用兼容的 Python 3.11 和 NumPy、Pillow，在解压后的目录或本地交付目录执行。已验证版本见 `environment.json`。把输出写到一个新目录。

```powershell
python -B inspect_two_step_delay.py --output replay-check --compare-saved
```

脚本从 `source_manifest.json` 找到十条真实轨迹。公开包内原始轨迹和 PNG 都随包提供。它核对源文件 SHA-256、数值指纹、步序、动作、限幅、控制器目标、原始图片，再复算输出指标和逐步文件。解压复核使用保存文件，所需数组都可用 `np.load(..., allow_pickle=False)` 读取。

## 结果和文件

- `seed-0045`、`seed-0046` 保存新轨迹、实际反馈审计及原图。历史文件在公开包 `history` 内。每份 `rollout.npz` 都保留动作轴、状态轴和环境轴。
- `actual-feedback.csv` 和 `actual-feedback.npz` 包含 88 条真实反馈决策，其中新增 44 条。`same-state-candidates.csv` 和同名 NPZ 包含 176 条同状态复算，`is_actual_rule` 标明已执行规则。
- 每个 seed 的 `*-comparison.npz`、`*-actions.csv` 和 `*-states.csv` 提供共同时间轴上的四组对照及 U。`timing-cases.json` 保留两步反馈在 t=9、10、11 的实测索引。
- `metrics.json` 按 seed 给出实际闭环指标及各候选的同状态误差。`validation.json` 记录逐项数值和像素验收。`figure-validation.json` 记录图表输入检查及 16 个拼图块的像素校验。
- 两步 Velocity 在四条新轨迹上的同状态估计 RMSE 均大于两步 Hold。真实闭环里，同 seed 的两步 Velocity 跟踪 RMSE 略小而峰值偏差较大。两个 seed 的所有任务成功标志均为 false。

本地交付附 `run_two_step_delay.py` 和 `preflight.py` 供审计。正式采集已封存。原始调用账本及私有预检留在项目工作目录。观察到的现象来自一次固定动作覆盖和约 1 秒的仿真反馈。`sapien_cpu` 是逻辑接口记录，渲染的物理 GPU 使用率未测量。
