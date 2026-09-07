# 第 4 章总结｜Self-Attention

> 返回：[第 4 章首页](../04-self-attention.md)

::: tip 本章核心结论
1. Q 表示“我要找什么”，K 表示“我有什么特征”，V 表示“真正取回什么内容”。
2. $QK^T$ 产生位置之间的匹配分数，Softmax 把它变成权重，再加权汇总 V。
3. Causal Mask 阻止当前 Token 偷看未来，多头注意力让模型并行学习不同关系。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

### Attention 权重不等于最终答案概率

Attention 权重描述一个位置如何混合上下文 Value；语言模型最终 Token 概率还要经过多层 Transformer 和 LM Head。

### Q、K、V 不是三份相同数据

它们来自同一个 X，但经过不同可训练投影，因此数值和职责不同。

### 每个位置都有自己的 Query

不是整句话只有一个 Query。Self-Attention 对每个 Token 位置都生成 Q，并形成注意力矩阵的一行。

### Softmax 沿最后一维执行

对 `[B,N,S,S]` 分数张量，通常沿最后一个 Key 位置维做 Softmax，使每个 Query 的一行权重和为 1。

### Attention 图不一定是可靠解释

较高权重表示当前层当前 Head 中更强的 Value 聚合，不应直接当成人类可验证的推理过程或因果解释。

## 13. 动手任务

### 必做 1：Shape 推导

给定：

```text
B=4, S=256, H=1024, heads=16
```

回答：

1. 每头维度 D 是多少？
2. 拆头后的 Q/K/V shape 是什么？
3. Attention scores shape 是什么？共有多少个元素？
4. scores 若使用 BF16，理论占多少 Byte？

::: details 必做 1 参考答案

每头维度：

```text
D = H / heads = 1024 / 16 = 64
```

拆头后的 Q/K/V：

```text
[B,heads,S,D] = [4,16,256,64]
```

Attention scores：

```text
[B,heads,S,S] = [4,16,256,256]
```

元素数量：

```text
4 × 16 × 256 × 256 = 4,194,304
```

BF16 每元素 2 Byte：

```text
4,194,304 × 2 = 8,388,608 Byte = 8 MiB
```

这只是一个 scores 张量的数据量，不包含 Q/K/V、Softmax 输出、反向保存值、Allocator 和其他层。

:::

### 必做 2：运行并观察 Mask

运行手写 Attention 代码，确认：

- `weights[0]` 的上三角全部为 0；
- 每行权重和为 1；
- 第一行只有第一个位置权重为 1；
- 修改输入后，Q/K/V、权重和输出都会改变。

### 选做：与框架实现对比

使用 `torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)` 计算相同输入，注意该 API 常用 shape 为 `[B,heads,S,D]`，对比结果与朴素实现是否接近。

## ✅ 理解检查

1. Q、K、V 分别解决什么问题？
2. 为什么 Self-Attention 中 Q、K、V 都来自 X，但仍要使用三组投影？
3. `QKᵀ` 为什么得到 `[B,S,S]`？矩阵元素 `[i,j]` 表示什么？
4. 为什么要除以 `√D`？
5. Causal Mask 为什么在 Softmax 前加入？
6. 为什么 Attention 权重矩阵每行和为 1？
7. Multi-Head Attention 怎样保持总隐藏维 H 不变？
8. 序列长度从 S 变成 2S，标准 Attention 的关系矩阵元素数如何变化？
9. Flash Attention 主要减少什么开销？它是否改变 Attention 数学语义？
10. Attention 与 Tensor/Context Parallel 可能怎样产生集合通信？

## 15. 参考答案

::: details 1. Q、K、V 分别解决什么问题？

Q 表达当前位置的检索需求，K 提供各位置用于匹配的特征，V 提供匹配后实际被聚合的内容。Q 与 K 决定权重，权重作用于 V。

:::

::: details 2. 为什么使用三组投影？

虽然三者输入都是 X，但“查询什么”“怎样被匹配”“传递什么内容”是不同角色。独立参数让模型为三个角色学习不同的特征空间，而不是被迫使用同一种表示。

:::

::: details 3. QKᵀ 为什么得到 [B,S,S]？

Q 是 `[B,S,D]`，K 转置最后两维后是 `[B,D,S]`，中间维 D 相消，得到 `[B,S,S]`。元素 `[b,i,j]` 是第 b 条序列中 Query 位置 i 与 Key 位置 j 的匹配分数。

:::

::: details 4. 为什么除以 √D？

D 增大时点积通常变大，使 Softmax 容易饱和。若各维方差约为 1，D 项之和的方差约为 D；除以 `√D` 可把分数方差缩回较稳定的量级，有利于梯度和训练稳定性。

:::

::: details 5. Causal Mask 为什么在 Softmax 前加入？

把未来位置的 logit 设为 `-∞` 后，其指数为 0，Softmax 概率精确变成 0，剩余可见位置重新归一化。如果在 Softmax 后简单修改，还必须再次归一化，也更容易产生错误。

:::

::: details 6. 为什么每行和为 1？

Softmax 对每个 Query 的所有 Key 分数进行归一化。每行因此成为一个离散权重分布，随后用于对所有 Value 做加权和。

:::

::: details 7. 多头怎样保持 H 不变？

通常设置每头维度 `D=H/N`。N 个 Head 各输出 D 维，拼接后是 `N×D=H`，再经过一个 `[H,H]` 输出投影，所以输入输出隐藏维都为 H。

:::

::: details 8. S 变成 2S 时关系矩阵怎样变化？

元素数从 `S²` 变成 `(2S)²=4S²`，即四倍。相应的两次核心矩阵乘法计算量和朴素保存的注意力中间数据也近似按四倍增长。

:::

::: details 9. Flash Attention 主要减少什么？

它通过分块和在线 Softmax 减少完整 S×S 中间矩阵在 HBM 的写入与读取，提高 IO 效率。它计算的数学 Attention 语义不变，但执行顺序、存储方式以及反向策略经过重组。

:::

::: details 10. Attention 并行为什么产生通信？

当 Q/K/V、Head、隐藏维或序列维被分散到不同 Rank 时，单个 Rank 只有局部数据。下一步若需要完整投影结果、跨片段 K/V 或聚合后的输出，就需要 AllGather、AllReduce、ReduceScatter 或点对点交换。具体原语取决于切分维度和后续期望的数据布局。

:::

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
- [PyTorch：MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch：Scaled Dot Product Attention 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)

下一课会把 Attention、MLP、残差连接和 RMSNorm 组合成一个完整 Decoder Block，计算一层 Transformer 的参数量、FLOPs 与激活内存。

---

<!-- chapter-navigation -->
[进入第 5 章 →](../05-decoder-block.md)
