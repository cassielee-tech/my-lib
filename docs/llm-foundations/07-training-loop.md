# 第 7 章｜大模型训练循环

> 本章目标：把数据、前向传播、Loss、反向传播和参数更新连成完整闭环；理解 AdamW、学习率、梯度累积、混合精度与梯度裁剪各自解决的问题；知道分布式训练中的梯度通信发生在哪里。

## 本章导学

::: tip 本章核心结论
1. 一个训练 Step 是数据、前向、Loss、反向、梯度同步和参数更新的闭环。
2. 梯度累积扩大有效 Batch，混合精度减少资源，AdamW 根据梯度和历史统计更新参数。
3. 数据并行中，同一参数的梯度必须在更新前聚合。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，沿一次参数更新的时间顺序推进。

**先看这几张核心图：** Training Step Lifecycle、Gradient Accumulation 和 AdamW State Update 三张图。

- [ ] 我能按顺序说出一次参数更新的步骤
- [ ] 我能区分 Micro-batch 与有效 Batch
- [ ] 我能指出梯度通信发生的位置

## 本章单元

- **07-1（约 15 分钟）**：[一次训练 Step 与梯度](./07-training-loop/01-training-step-gradient.md)
- **07-2（约 15 分钟）**：[梯度累积与 AdamW 状态](./07-training-loop/02-gradient-accumulation-adamw.md)
- **07-3（约 15 分钟）**：[学习率、混合精度与梯度裁剪](./07-training-loop/03-lr-mixed-precision-clipping.md)
- **07-4（约 15 分钟）**：[训练显存与梯度通信](./07-training-loop/04-training-memory-communication.md)

- **本章总结**：[串联知识、练习与检查](./07-training-loop/summary.md)
