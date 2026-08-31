# 04｜Self-Attention、QKV 与多头注意力

## 这一课要解决的问题

上一课把文本变成了 `[B,S,H]` 的 Token 向量，但每个向量主要表示当前位置自身。模型还需要让每个 Token 读取上下文，例如判断“它”指代“书”还是“小明”。Self-Attention 就是完成上下文信息交换的核心机制。

这一课的主线是：

```text
输入 X
→ 投影得到 Q、K、V
→ Q 与 K 计算位置间匹配分数
→ Mask 屏蔽不可见位置
→ Softmax 得到注意力权重
→ 权重与 V 加权求和
→ 得到融合上下文的新表示
```

## 1. Attention 到底在做什么

对序列中的每个当前位置，Attention 都执行两步：

1. 找出上下文中哪些位置与当前任务更相关；
2. 按相关程度汇总这些位置携带的信息。

![QKV 如何完成上下文检索](/images/llm/qkv-retrieval.svg)

### Q、K、V 的直观职责

- **Query**：当前位置想寻找什么信息；
- **Key**：每个位置用什么特征供别人匹配；
- **Value**：如果某位置被关注，实际取走什么信息。

这个解释只用于建立直觉。Q、K、V 并不是字符串或字段，而是由训练学习出来的浮点向量。

### 为什么 Key 和 Value 要分开

用于“判断是否相关”的特征，不一定等于最终要传递的内容。Key 服务于寻址，Value 服务于内容聚合。把两者分开让模型可以学习不同的表示空间。

## 2. 从向量相似度开始

Attention 使用点积衡量 Query 与 Key 的匹配程度：

$$
score(q,k)=q\cdot k=\sum_{i=1}^{D}q_i k_i
$$

例如：

```text
q  = [1, 2]
k₁ = [2, 1]  → q·k₁ = 1×2 + 2×1 = 4
k₂ = [-1, 0] → q·k₂ = 1×(-1) + 2×0 = -1
```

在这个例子中，`q` 与 `k₁` 的匹配分数更高。

### 点积不是严格的余弦相似度

余弦相似度会除以两个向量的长度，只比较方向；Attention 的普通点积同时受方向和模长影响。模型会通过投影参数和归一化机制学习合适的表示。

## 3. Q、K、V 从哪里来

Self-Attention 的 Q、K、V 都来自同一个输入 `X`，但使用三组不同参数投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

若暂时忽略多头：

```text
X:  [B,S,H]
WQ: [H,D]  → Q: [B,S,D]
WK: [H,D]  → K: [B,S,D]
WV: [H,Dv] → V: [B,S,Dv]
```

“Self” 表示查询方和被查询方来自同一序列。Cross-Attention 则可以让 Q 来自一组表示，K/V 来自另一组表示。

## 4. Scaled Dot-Product Attention 完整公式

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

## 6. Multi-Head Attention

单头 Attention 只在一组投影空间中计算关系。多头注意力把隐藏维度切成多个 Head，让每个 Head 使用独立投影参数计算注意力，然后拼接结果。

![多头注意力的切分与合并](/images/llm/multi-head-attention.svg)

设隐藏维 `H=512`、头数 `N=4`，通常每头维度：

$$
D=H/N=128
$$

多头 Shape 常写为：

```text
投影前: [B,S,H]
拆头后: [B,N,S,D]
每头分数: [B,N,S,S]
每头输出: [B,N,S,D]
合并后: [B,S,H]
```

最终还要乘输出投影 `WO [H,H]`，让不同 Head 的信息重新混合。

### 多头不是把完整 H 维计算复制 N 次

经典实现中，总隐藏维 H 被切成 N 个大小为 D 的 Head，且 `N×D=H`。增加 Head 数量通常会减小单头维度，而不是按头数成倍增加总表示宽度。

## 7. 一个完整的 Shape 例子

给定：

```text
B = 2
S = 128
H = 512
N = 8
D = H/N = 64
```

| 张量 | Shape | 含义 |
| --- | --- | --- |
| X | `[2,128,512]` | 输入隐藏状态 |
| Q/K/V 投影后 | `[2,128,512]` | 尚未显式拆头 |
| Q/K/V 拆头后 | `[2,8,128,64]` | 8 个独立 Head |
| Attention scores | `[2,8,128,128]` | 每头的所有位置关系 |
| Attention weights | `[2,8,128,128]` | Mask、Softmax 后权重 |
| 每头输出 | `[2,8,128,64]` | 聚合后的 Value |
| 合并输出 | `[2,128,512]` | 恢复隐藏维 |

## 8. 用 PyTorch 手写单头 Attention

```python
import math
import torch
from torch import nn

torch.manual_seed(0)

B, S, H, D = 2, 4, 8, 8
x = torch.randn(B, S, H)

wq = nn.Linear(H, D, bias=False)
wk = nn.Linear(H, D, bias=False)
wv = nn.Linear(H, D, bias=False)

q = wq(x)                              # [B,S,D]
k = wk(x)                              # [B,S,D]
v = wv(x)                              # [B,S,D]

scores = q @ k.transpose(-2, -1)       # [B,S,S]
scores = scores / math.sqrt(D)

causal_mask = torch.triu(
    torch.ones(S, S, dtype=torch.bool),
    diagonal=1,
)
scores = scores.masked_fill(causal_mask, float('-inf'))

weights = torch.softmax(scores, dim=-1) # [B,S,S]
output = weights @ v                    # [B,S,D]

print(q.shape)
print(scores.shape)
print(weights[0])
print(weights[0].sum(dim=-1))           # 每行都是 1
print(output.shape)
```

生产代码应优先使用框架提供的 `scaled_dot_product_attention` 等优化实现，而不是物化所有中间张量的朴素版本。

## 9. Attention 的计算量与显存

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

后续学习张量并行时，会从本课 Shape 出发推导通信发生在哪里。

## 12. 容易混淆的点

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

## 14. 本课自测

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

## 15. 本课自测答案

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
