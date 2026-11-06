# Diffusion Policy / PushT 公开评测资料

LeRobot 实现的预训练策略评测：固定环境 seed 1000—1049，单环境、batch=1，35/50 成功（70%）。两个测试回合不计入正式分母。没有训练或微调。

## 配置与来源

- LeRobot v0.3.3，commit b883328e6c95681ca90a18b102e4ae5e1f91e2bf。
- 权重 lerobot/diffusion_pusht，revision 84a7c23178445c6bbf7e1a884ff497017910f653。
- 权重 SHA256：995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8。
- Python 3.11.9；torch 2.7.1+cu128；torchvision 0.22.1+cu128；diffusers 0.33.1；gym-pusht 0.1.5；gymnasium 0.29.1；pymunk 6.11.1；safetensors 0.5.3。
- 图像观察96×96，录像384×384；观察2帧，预测长度16，执行8步，DDPM100次去噪。FP32参数，AMP关闭，TF32开启。
- 成功：原始覆盖率严格大于0.95；奖励：clip(coverage/0.95,0,1)。最多300步。
- 模型采样种子为20261106+环境seed，每回合重置Python、NumPy、torch及CUDA随机数。

原方法：https://github.com/real-stanford/diffusion_policy
LeRobot：https://github.com/huggingface/lerobot/tree/b883328e6c95681ca90a18b102e4ae5e1f91e2bf
权重：https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653

## 资料范围

results.csv包含全部50个正式回合的整理结果；summary.json是汇总；protocol.json是公开配置节选，参考比较口径见protocol-clarifications.md和reference-comparison.json。
images包含全50回合覆盖率图（PNG和SVG）、两份抽帧图和录像封面。videos只提供按seed顺序选出的首个成功（1000）及首个失败（1001）录像，不代表全部原始录像。
validation.json给出本次只读数值和视频验收摘要。

本公开资料不包含模型权重、训练数据集、本机原始日志、完整环境审计或网站维护工具。模型、上游代码及论文属于各自作者，许可按其固定版本的说明执行。
