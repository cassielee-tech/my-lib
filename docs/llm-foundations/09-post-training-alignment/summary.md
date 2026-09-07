# 第 9 章总结｜后训练与对齐

> 返回：[第 9 章首页](../09-post-training-alignment.md)

::: tip 本章核心结论
1. 预训练学习续写能力，SFT 用示范塑造指令行为，偏好对齐进一步调整回答倾向。
2. LoRA 冻结基础权重，只训练低秩增量；QLoRA 再把冻结的基础权重量化。
3. 对齐改善行为偏好，但不自动保证事实正确。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

1. **预训练模型是不是已经能对话？** 它可能模仿对话格式，但不等于稳定遵循用户意图。
2. **继续预训练和 SFT 有什么区别？** 前者通常在领域原始语料上继续预测全部 Token；后者使用明确的指令—回答示范塑造行为。
3. **LoRA 会把 Base Model 参数变少吗？** 不会，它减少的是可训练参数；未量化时 Base 权重仍然完整存在。
4. **Rank 越大一定越好吗？** 不一定。更大 Rank 提高容量，也增加资源和过拟合风险。
5. **QLoRA 会训练 4-bit Base 权重吗？** 通常不会，量化 Base 冻结，只训练 Adapter。
6. **DPO 不需要 Reference Model 吗？** 标准 DPO 目标包含参考策略；实现中可通过预计算等方式减少常驻开销。
7. **偏好对齐能让知识自动变正确吗？** 不能，它主要改变回答偏好和行为分布。

## 14. 动手练习

### 练习 1：计算 LoRA 参数量

对一个 `[d,k]=[4096,11008]` 的 MLP 投影，分别计算 $r=8$ 和 $r=64$ 时的 LoRA 参数量，并与原矩阵参数量比较。

### 练习 2：制作一条 SFT 样本

为“解释 AllReduce”编写 System、User、Assistant 三段数据，标出哪些 Token 参与 Loss。再思考：如果 Assistant 答案中有错误，模型会学到什么？

### 练习 3：制作一条偏好样本

为同一 Prompt 编写 Chosen 和 Rejected。不要只让 Rejected 明显更短，而要让两者在长度相近的情况下体现事实性或解释质量差异。

## 15. 自测题

1. Base Model 与 Chat/Instruction Model 的主要差别是什么？
2. 预训练和 SFT 的 Loss 都是交叉熵，它们为什么会产生不同效果？
3. 继续预训练与 SFT 的数据形态有什么区别？
4. LoRA 为什么能减少可训练参数？
5. $W\in\mathbb{R}^{d\times k}$ 的 LoRA 参数量怎样计算？
6. LoRA、QLoRA 和量化之间是什么关系？
7. 偏好数据的三元组包含什么？
8. 经典 RLHF 的 Reward Model 做什么？
9. DPO 相比经典 PPO-RLHF 省去了哪些环节？
10. LoRA 为什么降低梯度通信，却不一定消除模型计算和激活显存？

::: details 自测答案

1. Base Model 主要学习续写数据分布；Chat/Instruction Model 又经过示范和偏好训练，以更稳定地遵循指令和对话角色。
2. 预训练在大规模原始序列上预测广泛位置；SFT 在指令—回答示范上重点优化期望回答，数据分布和监督位置不同。
3. 继续预训练通常使用领域原始文本或代码；SFT 使用明确的 Prompt 与目标回答。
4. 它冻结完整 $W_0$，只训练低秩矩阵 $A$、$B$ 表示权重增量。
5. $r\times k+d\times r=r(k+d)$，不含 Bias 等其他可训练项。
6. LoRA 是参数高效微调；量化降低数值存储位宽；QLoRA 通常用量化的冻结 Base Model 配合可训练 LoRA。
7. Prompt $x$、Chosen Response $y_w$ 和 Rejected Response $y_l$。
8. 它学习为 Prompt—Response 输出标量奖励，使人偏好的回答得分更高。
9. 它不显式训练 Reward Model，也不需要在训练循环中用 PPO 在线 Rollout 和策略优化，而是直接优化成对偏好目标。
10. Base 权重冻结后只需同步 Adapter 梯度，但前向/反向仍要经过基础模型，基础权重和中间激活仍占资源。

:::

## 16. 本章小结

- 预训练建立通用 Token 预测能力，SFT 用示范塑造指令和对话行为；
- 继续预训练适应领域分布，不能与 SFT 混为一谈；
- LoRA 冻结 Base 权重，用低秩矩阵表示任务相关的权重增量；
- QLoRA 将量化 Base 与 LoRA 结合，进一步降低基础权重显存；
- 经典 RLHF 训练 Reward Model，再用强化学习优化 Policy；
- DPO 直接使用 Chosen/Rejected Pair 优化相对偏好；
- 对齐改变行为倾向，不保证事实正确，也不能替代系统评测；
- LoRA 显著减少优化器状态和梯度通信，但基础模型计算和激活依然存在。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT：LoRA](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

下一课将进入 MoE，学习 Router 怎样只激活少数 Expert，以及 Expert Parallel 为什么会把 Token 送入 AlltoAll 通信。

---

<!-- chapter-navigation -->
[进入第 10 章 →](../10-moe-expert-parallel.md)
