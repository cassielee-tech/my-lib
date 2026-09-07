# 单元 07-4｜训练显存与梯度通信

> 所属章节：[第 7 章｜训练循环](../07-training-loop.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **训练显存与梯度通信** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

训练显存不只是模型参数，大致还包括：

$$
M_{train}\approx M_{parameter}+M_{gradient}+M_{optimizer}+M_{activation}+M_{temporary}
$$

- **参数**：当前模型权重；
- **梯度**：与可训练参数对应；
- **优化器状态**：AdamW 的 $m$、$v$ 等；
- **激活值**：前向保存、反向使用，随 Batch、序列长度和层数增长；
- **临时缓冲区**：算子 Workspace、通信 Buffer、内存碎片等。

参数、梯度和优化器状态主要随参数量增长；激活值主要随 Micro-batch 和序列长度增长。所以减小 Micro-batch 或使用 Activation Checkpointing 能降低激活显存，但不能消除 AdamW 状态。

## 10. 与集合通信的关系

单设备训练的反向传播只产生本地梯度。进入数据并行后，每个 Rank 读取不同 Micro-batch，得到不同的本地梯度，必须在 `optimizer.step()` 前聚合为一致的平均梯度：

$$
g=\frac{1}{N}\sum_{r=0}^{N-1}g^{(r)}
$$

这通常通过 **AllReduce** 完成。DDP 会在反向传播过程中按 Bucket 触发梯度通信，并尝试与尚未完成的反向计算重叠。

梯度累积带来一个重要优化：中间 Micro-step 只在本地累加，最后一个 Micro-step 再同步，否则每个 Micro-step 都 AllReduce 会产生多余通信。PyTorch DDP 中通常借助 `no_sync()` 控制，但必须保证最后一次反向确实触发同步。

从时间线看，正确性约束是：

> 本地反向 → 梯度聚合 → 反缩放/裁剪 → Optimizer Step → 所有 Rank 获得一致的新参数

混合精度还会影响通信数据类型；Bucket 大小影响通信启动次数与重叠机会；梯度裁剪的全局范数在分片训练中可能需要额外归约。这些会在分布式训练和集合通信章节展开。

---

[进入本章总结 →](./summary.md)
