# 单元 04-2｜缩放点积与 Causal Mask

> 所属章节：[第 4 章｜Self-Attention](../04-self-attention.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **缩放点积与 Causal Mask** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{D}}+M\right)V
$$

其中 `M` 是 Mask。把公式拆开看：

![Scaled Dot-Product Attention 的 Shape 变化](/images/llm/attention-shapes.svg)

### 第一步：QKᵀ 生成位置关系矩阵

```text
Q:  [B,S,D]
Kᵀ: [B,D,S]
QKᵀ: [B,S,S]
```

`QKᵀ[b,i,j]` 表示第 `b` 条序列中，Query 位置 `i` 与 Key 位置 `j` 的匹配分数。

### 第二步：为什么除以 √D

当 `D` 增大时，点积数值的方差也倾向增大。过大的 logits 会让 Softmax 过早接近 0 或 1，梯度可能变小。除以 `√D` 可以让分数尺度更稳定。

若 Q、K 各维独立、均值为 0、方差约为 1，则点积是 D 项之和，方差约为 D；除以 `√D` 后，方差回到约 1 的量级。

### 第三步：逐行 Softmax

每一个 Query 位置对所有可见 Key 的分数做 Softmax，因此注意力矩阵每一行的权重和为 1。

### 第四步：权重乘 V

```text
A: [B,S,S]
V: [B,S,Dv]
O = A×V: [B,S,Dv]
```

输出仍有 `S` 个位置，但每个位置的向量已经混合了上下文信息。

## 5. Causal Mask 怎样防止看到未来

Decoder-only 模型训练时，完整序列已经放进张量。如果不遮罩，位置“我”可能直接读取未来的“喜欢”，下一个 Token 预测就等于偷看答案。

![因果遮罩如何改变注意力矩阵](/images/llm/causal-attention-matrix.svg)

Mask 在 Softmax 之前把未来位置的分数变成 `-∞` 或一个足够小的数：

$$
e^{-\infty}=0
$$

因此 Softmax 后未来位置的权重为 0。第 `i` 行只会在位置 `0..i` 之间重新分配权重。

### 为什么遮罩是上三角

矩阵的行是当前 Query 位置，列是被读取的 Key 位置。当列号 `j > i` 时，Key 位于 Query 的未来，所以右上半部分需要遮罩。

---

[继续单元 04-3 →](./03-multi-head-shape.md)
