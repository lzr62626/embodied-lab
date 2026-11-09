# ACT 双臂传方块评测 · 公开资料

在 gym-aloha 的 `AlohaTransferCube-v0` 上运行 LeRobot 官方预训练 ACT 权重。正式评测固定 seed 1000–1049，共 50 回合，其中 37 次成功、13 次超时、0 程序异常。seed 9000 与 9001 是试跑回合，单独统计。

- `results.csv` 包含每个正式 seed 的成功标记、步数、最大与累计奖励、耗时和同 seed 的官方成功标记。成功列的 1 表示环境报告交接成功。
- `summary.json` 为本次正式结果及单列的试跑统计。耗时包含渲染和视频写入。
- `protocol.json` 为公开的模型、代码、依赖版本、哈希和固定评测条件。
- `reference-comparison.json` 为固定 revision 的官方记录与本次结果的按 seed 对照。
- `validation.json` 为只读验收摘要，核对了 52 回合的逐步记录和完整视频。
- `videos/` 与 `images/` 包含首个成功和首个超时回合的原始录像及对应末帧封面。

官方模型记录有 500 回合，415 次成功。筛选 seed 1000–1049 后为 38/50，本次为 37/50。seed 1020 是唯一成败不同的回合。本次使用单环境，官方模型卡记录的评测命令使用 batch=50。

来源 [ACT 论文](https://arxiv.org/abs/2304.13705) · [固定的官方权重](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/ba73b2766f1371cdc133ca4efb97eb090d744625)
