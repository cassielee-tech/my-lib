# 单元 03-2｜特殊 Token、Mask 与 Embedding

> 所属章节：[第 3 章｜文本怎样成为模型输入](../03-tokenization-embedding.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **特殊 Token、Mask 与 Embedding** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

常见特殊 Token 包括：

- `[BOS]`：序列开始；
- `[EOS]`：序列结束；
- `[PAD]`：为了组成 Batch 而补齐的空位置；
- `[UNK]`：无法被词表表示的内容，字节级方案通常很少需要；
- 模型还可能定义对话角色、工具调用等专用 Token。

不同文本的 Token 数量不同，但一个普通张量要求每行长度相同，所以短序列需要 Padding，过长序列需要 Truncation。

### Attention Mask 的作用

假设：

```text
input_ids:
[[1, 42, 815, 9, 2],
 [1, 73,   2, 0, 0]]
```

对应的 Padding Attention Mask 可以是：

```text
attention_mask:
[[1, 1, 1, 1, 1],
 [1, 1, 1, 0, 0]]
```

`1` 表示真实 Token，`0` 表示 Padding。模型应避免让 Padding 位置参与有效注意力或 Loss。

注意不要混淆两种 Mask：

- **Padding Mask**：屏蔽为了对齐长度而补出的 `[PAD]`；
- **Causal Mask**：屏蔽未来 Token，保证当前位置只能看见自己和左侧上下文。

## 4. Embedding：从离散 ID 到连续向量

Embedding 层是一张可训练的参数表：

$$
E \in \mathbb{R}^{V \times H}
$$

其中：

- `V` 是 vocabulary size；
- `H` 是 hidden size；
- 第 `i` 行就是 Token ID `i` 的向量。

![Token ID 通过 Embedding 查表变成向量](/images/llm/embedding-lookup.svg)

输入输出 shape 为：

```text
input_ids: [B, S]       整数
E:         [V, H]       可训练浮点参数
输出:      [B, S, H]    浮点激活
```

### Embedding 是查表，不是把 ID 当数字计算

对于 Token ID `815`，Embedding 前向操作等价于选择 `E[815]`。它在数学上也可以写成 one-hot 向量与矩阵相乘，但实际实现不会构造巨大的 one-hot 矩阵，而是直接 Gather 对应行。

### Embedding 如何学到语义

Embedding 初始通常是随机的。训练时，如果某个 Token 的向量导致后续预测产生 Loss，反向传播会计算该行参数的梯度，优化器再更新它。

出现在相似上下文中的 Token 会收到相似的训练约束，因此它们的向量可能逐渐形成有意义的几何关系。但模型中的“语义”不只存在于 Embedding 表，也分布在后续 Transformer 层中。

### 位置信息从哪里来

Token Embedding 只表示“这是什么 Token”，无法区分同一个 Token 出现在第 1 位还是第 100 位。模型还需要位置信息。

位置表示的具体方法包括：

- 可学习位置 Embedding；
- 正弦/余弦位置编码；
- RoPE（Rotary Position Embedding）等相对位置方法。

这一课只需记住：进入 Transformer 的表示必须同时包含 Token 内容与位置信息。RoPE 会在 Attention 课详细解释。

---

[继续单元 03-3 →](./03-causal-lm-target.md)
