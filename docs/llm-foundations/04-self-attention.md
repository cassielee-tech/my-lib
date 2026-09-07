# 第 4 章｜Self-Attention、QKV 与多头注意力

## 本章导学

::: tip 本章核心结论
1. Q 表示“我要找什么”，K 表示“我有什么特征”，V 表示“真正取回什么内容”。
2. $QK^T$ 产生位置之间的匹配分数，Softmax 把它变成权重，再加权汇总 V。
3. Causal Mask 阻止当前 Token 偷看未来，多头注意力让模型并行学习不同关系。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。先建立 Q/K/V 直觉，再进入公式和性能。

**先看这几张核心图：** QKV Retrieval、Causal Attention Matrix、Multi-Head Attention 三张图；手算部分留到第二次。

- [ ] 我能用自己的话解释 Q、K、V
- [ ] 我能说出 Softmax 前后的数据含义
- [ ] 我能解释为什么需要 Causal Mask

## 本章要解决的问题

上一章把文本变成了 `[B,S,H]` 的 Token 向量，但每个向量主要表示当前位置自身。模型还需要让每个 Token 读取上下文，例如判断“它”指代“书”还是“小明”。Self-Attention 就是完成上下文信息交换的核心机制。

本章的主线是：

```text
输入 X
→ 投影得到 Q、K、V
→ Q 与 K 计算位置间匹配分数
→ Mask 屏蔽不可见位置
→ Softmax 得到注意力权重
→ 权重与 V 加权求和
→ 得到融合上下文的新表示
```

## 本章单元

- **04-1（约 15 分钟）**：[Attention 直觉与 Q/K/V](./04-self-attention/01-qkv-intuition.md)
- **04-2（约 15 分钟）**：[缩放点积与 Causal Mask](./04-self-attention/02-scaled-dot-product-mask.md)
- **04-3（约 15 分钟）**：[多头注意力与 Shape](./04-self-attention/03-multi-head-shape.md)
- **04-4（约 15 分钟）**：[计算量、显存与 Flash Attention](./04-self-attention/04-attention-performance.md)

- **本章总结**：[串联知识、练习与检查](./04-self-attention/summary.md)
