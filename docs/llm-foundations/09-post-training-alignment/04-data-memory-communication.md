# 单元 09-4｜数据质量、显存与通信

> 所属章节：[第 9 章｜后训练与对齐](../09-post-training-alignment.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **数据质量、显存与通信** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

不同阶段都遵循“数据决定模型学到什么”：

| 阶段 | 典型数据 | 主要风险 |
| --- | --- | --- |
| 预训练 | 大规模原始文本、代码 | 重复、污染、版权、低质量和有害内容 |
| 继续预训练 | 领域原始语料 | 领域过窄、遗忘通用能力 |
| SFT | Prompt 与目标回答 | 模板单一、答案错误、风格过拟合 |
| 偏好对齐 | Chosen / Rejected Pair | 标准不一致、偏好偏差、长度偏好 |

少量高质量、覆盖真实任务分布的数据，往往比大量机械生成但未经验证的数据更有价值。训练集、验证集、测试集还要避免同源泄漏，否则评测会虚高。

## 12. 与训练显存和集合通信的关系

### 12.1 Full Fine-Tuning

所有参数都需要梯度和优化器状态。数据并行时，全部参数梯度都要进行 AllReduce 或 ReduceScatter；参数规模直接决定通信量级。

### 12.2 LoRA

Base 权重冻结，不需要保存其梯度和 AdamW 状态。数据并行训练时，通常只同步 LoRA 等可训练参数的梯度，因此梯度通信量可以显著降低。

但 LoRA 并没有让 Base Model 消失：

- 前向和反向仍要经过基础模型计算图，以求出 Adapter 的梯度；
- Base 权重仍需加载到设备，除非再结合量化或分片；
- 激活显存仍然存在；
- 模型太大时仍可能需要 Tensor Parallel 或 Pipeline Parallel。

### 12.3 RLHF / DPO

经典 RLHF 可能同时涉及 Policy、Reference Model、Reward Model 和 Value Model，还要生成样本，系统资源和调度比普通 SFT 更复杂。

DPO 不需要显式 Reward Model 和在线 PPO Rollout，但通常仍要计算当前 Policy 与 Reference Model 对 Chosen/Rejected 的 Log Probability。工程实现可以预计算 Reference Log Probability，或共享/卸载参考权重来降低成本。

从 AI Infra 视角看，不能只问“训练多少参数”，还要问：同时驻留几个模型、保存哪些激活、每步同步哪些梯度、数据怎样送入设备。

---

[进入本章总结 →](./summary.md)
