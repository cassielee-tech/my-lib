# 单元 04-4｜计算量、显存与 Flash Attention

> 所属章节：[第 4 章｜Self-Attention](../04-self-attention.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **计算量、显存与 Flash Attention** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

忽略常数与多头 reshape，核心有两次矩阵乘法：

```text
QKᵀ: [S,D] × [D,S] → [S,S]，约 2S²D FLOPs
AV:  [S,S] × [S,D] → [S,D]，约 2S²D FLOPs
```

合计约：

$$
4S^2D
$$

标准实现还可能保存 `[B,N,S,S]` 的分数或概率矩阵用于反向传播。这就是序列长度翻倍时，Attention 相关计算与中间数据可能接近四倍的原因。

### 不要只看 S²

一个 Transformer 层还包含 QKV/输出投影和 MLP，它们通常与 `S×H²` 相关。在不同 H、S 与硬件上，Attention 是否占主导需要具体分析，不能只看到平方复杂度就断言它一定最慢。

## 10. Flash Attention 为什么重要

朴素实现可能把完整 `S×S` 分数矩阵写入 HBM，再读回来做 Softmax 和 `A×V`。Flash Attention 类算法通过分块与在线 Softmax，把更多中间计算留在更快的片上存储中，避免完整物化大矩阵。

关键点是：

- 数学结果仍是同一个 Attention；
- 优化重点是减少 HBM 读写，而不是改变 QKV 语义；
- 需要处理分块归一化、数值稳定、Mask 和反向重计算；
- PyTorch 的 SDPA 会根据条件选择可用的优化后端。

这类优化体现了 AI Infra 的核心思维：不仅统计 FLOPs，还要分析数据在存储层级间如何移动。

## 11. 与分布式和集合通信的关系

Attention 本身描述单个逻辑张量上的运算。当隐藏维、Head 或序列被切到多张设备后，局部结果需要通信才能恢复下一步所需的数据分布。

常见思路包括：

- **Tensor Parallel**：切分 QKV 或输出投影的特征维，可能需要 AllReduce/ReduceScatter；
- **Sequence Parallel**：让不同 Rank 保存不同序列片段，降低重复激活；
- **Context Parallel**：长上下文跨 Rank 切分，Attention 需要交换 K/V 或中间结果；
- **Head Parallel**：不同 Rank 计算不同 Head，后续输出投影前后需要匹配数据布局。

后续学习张量并行时，会从本章 Shape 出发推导通信发生在哪里。

---

[进入本章总结 →](./summary.md)
