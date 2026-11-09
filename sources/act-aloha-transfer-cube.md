---
title: "ACT 双臂传方块复现 · ALOHA 的 50 次仿真评测"
description: "使用 LeRobot 官方 ACT 权重完成 50 次双臂交接评测，37 次成功。公开逐回合结果、成功与未达标录像，并与官方相同 seed 对照。"
slug: act-aloha-transfer-cube
date: 2026-11-09
section: reproduction
series: "开源模型评测复现"
order: 2
tags:
  - ACT
  - ALOHA
  - 双臂操作
---

# ACT 双臂传方块复现 · ALOHA 的 50 次仿真评测

[下载公开评测资料](../assets/reproductions/act-aloha-transfer-cube/act-aloha-results.zip) · [逐回合 CSV](../assets/reproductions/act-aloha-transfer-cube/results.csv) · [验收摘要](../assets/reproductions/act-aloha-transfer-cube/validation.json)

两只机械臂隔着一张桌子。右臂要拾起红色方块，再递到左臂的夹爪里。这个双臂交接任务要求策略协调两只手的运动，还要在方块离开桌面后把握递交时机。本次用 LeRobot 官方 ACT 权重做预训练策略的仿真推理评测。作者在 Windows 上完成了 2 个试跑回合和 50 个正式回合，正式结果为 **37/50 成功（74%）**。

## 一次看一帧，一次生成一段动作

ACT 是 Action Chunking with Transformers。这个 checkpoint 每次读取一张 640×480 的俯视 RGB 图像和 14 维关节与夹爪状态，输出 100 个连续动作。每个动作包含左右两臂合计 14 个目标值。控制频率为 50 Hz，所以一个动作块对应 2 秒仿真时间。环境每步接收一个动作。策略把动作块放入队列，队列用完后才根据新观测预测下一块。若环境已报告成功，回合立即结束。

论文介绍了 ACT 的方法与双臂系统，本次使用公开的模拟器任务和预训练权重。 [ACT 原论文](https://arxiv.org/abs/2304.13705) · [LeRobot 官方模型卡](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/ba73b2766f1371cdc133ca4efb97eb090d744625)

## 评测条件和成功判定

| 项目 | 本次设置 |
| --- | --- |
| 任务 | `AlohaTransferCube-v0`，gym-aloha 0.1.3，MuJoCo 3.1.6，dm-control 1.0.20 |
| 代码 | LeRobot v0.3.3，commit `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` |
| 权重 | `lerobot/act_aloha_sim_transfer_cube_human`，revision `ba73b2766f1371cdc133ca4efb97eb090d744625` |
| 推理 | 单环境，batch=1，FP32，关闭 AMP 与 TF32，每次生成并依次执行 100 步 |
| 正式回合 | seed 1000–1049，每回合最多 400 步 |
| 试跑回合 | seed 9000、9001，两次均成功，单独记录 |

环境通过接触关系给奖励。奖励 1 对应右夹爪接触方块，2 对应右夹爪接触且方块离开桌面。奖励 3 对应左夹爪接触方块。当左夹爪接触方块、方块又没有接触桌面时，环境给奖励 4，设置 `is_success=True` 并结束回合。这里的成功标准由接触信号决定，评测在首次达标时结束，因此没有测量左臂继续持稳多久。 [gym-aloha 0.1.3 任务说明](https://pypi.org/project/gym-aloha/0.1.3/)

在 Windows 上先固定代码版本、校验权重哈希，再用俯视画面和一次动作预测做预检。构建模型时跳过单独下载 ImageNet 初始化，随后严格加载官方 checkpoint 的参数与缓冲区。正式运行前，检查试跑与正式评测使用的代码、权重和运行配置一致。 [公开配置与哈希](../assets/reproductions/act-aloha-transfer-cube/protocol.json)

## 37 次成功，13 次超时

| 指标 | 正式结果 |
| --- | --- |
| 正式回合 | 50 |
| 成功 | 37（74%） |
| 400 步内未达标 | 13（26%） |
| 程序异常 | 0 |
| 平均最大奖励 | 3.34 / 4 |
| 正式循环总步数 | 15,932 |
| 正式循环总耗时 | 130.129 秒，含渲染和视频写入 |

在 13 个未达标回合里，7 个回合的最大奖励为 1，6 个为 2。前者达到右夹爪接触阶段，后者曾达到抬起阶段。各回合的最大奖励表明曾达到的阶段，无法仅凭该值确定随后失手的原因。

## 看两段完整录像

按 seed 升序选取首个成功和首个未达标回合。两段均为 640×480 的俯视相机录像，包含初始帧和每步执行后的画面，按仿真的 50 FPS 播放。

### 成功 · seed 1001

右臂把方块从桌上举起并送向左夹爪。第 229 步，环境给出奖励 4，回合结束。

<figure class="experiment-figure"><video controls playsinline preload="metadata" width="640" height="480" style="aspect-ratio:4/3" poster="../assets/reproductions/act-aloha-transfer-cube/images/seed-1001-poster.png" aria-label="seed 1001 成功交接回合"><source src="../assets/reproductions/act-aloha-transfer-cube/videos/seed-1001.mp4" type="video/mp4"><a href="../assets/reproductions/act-aloha-transfer-cube/videos/seed-1001.mp4">下载成功回合录像</a></video><figcaption>seed 1001 · 229 步 · 最大奖励 4。</figcaption></figure>

### 超时 · seed 1000

右夹爪曾接触方块，最大奖励为 1。运行到 400 步时，环境仍未报告交接成功。

<figure class="experiment-figure"><video controls playsinline preload="metadata" width="640" height="480" style="aspect-ratio:4/3" poster="../assets/reproductions/act-aloha-transfer-cube/images/seed-1000-poster.png" aria-label="seed 1000 超时未达标回合"><source src="../assets/reproductions/act-aloha-transfer-cube/videos/seed-1000.mp4" type="video/mp4"><a href="../assets/reproductions/act-aloha-transfer-cube/videos/seed-1000.mp4">下载超时回合录像</a></video><figcaption>seed 1000 · 400 步 · 最大奖励 1。</figcaption></figure>

## 与官方记录逐个 seed 对照

| 评测范围 | 成功数 | 成功率 |
| --- | --- | --- |
| 本次 seed 1000–1049 | 37 / 50 | 74% |
| 官方记录筛选出相同的 50 个 seed | 38 / 50 | 76% |
| 官方记录全部 500 个回合 | 415 / 500 | 83% |

50 个相同环境 seed 中，37 个两边都成功，12 个两边都未成功。唯一成败不同的是 seed 1020，本次跑到 400 步，最大奖励为 2，官方记录为成功。官方模型卡的评测设置使用 batch=50，本次使用单环境。两份结果只能在所列条件下对照，逐回合数据无法单独确定 seed 1020 差异的原因。 [官方逐回合记录](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/blob/ba73b2766f1371cdc133ca4efb97eb090d744625/eval_info.json) · [对照数据](../assets/reproductions/act-aloha-transfer-cube/reference-comparison.json)

## 结果如何核对

评测结束后，从逐步记录复算了全部回合的最大与累计奖励、成功标记和结束步数，并完整解码 52 段录像，核对尺寸、帧率和帧数。两段代表录像另外抽帧目视检查。公开包包含全部 50 个正式回合的汇总行、公开配置和这两段代表录像。复现范围为官方预训练权重在这一仿真任务的闭环评测。 [资料字段说明](../assets/reproductions/act-aloha-transfer-cube/README.md)
