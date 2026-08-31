# 06｜RoPE、位置编码与长上下文

> 本课目标：理解注意力为什么必须获得位置信息；看懂 RoPE 如何旋转 Query 和 Key；解释它为什么让注意力分数感知相对位置，并建立长上下文的基本认识。

## 1. 注意力为什么不知道顺序

上一课已经得到 Decoder Block 的主干，但还缺一个关键问题：`Attention(Q, K, V)` 只比较向量内容，本身并不知道某个 Token 位于第几个位置。

如果把输入 Token 的顺序整体打乱，且不加入任何位置线索，自注意力也会以相同方式打乱输出。换句话说，它能看到“有哪些 Token”，却无法仅凭注意力公式区分“我爱于金凤”和“于金凤爱我”。自然语言的顺序会改变含义，因此必须把位置注入模型。

![没有位置时，注意力只能随输入一起置换](/images/llm/attention-without-position.svg)

常见方法包括：

- **可学习绝对位置向量**：第 0、1、2……个位置各有一个可训练向量，与 Token Embedding 相加；
- **正弦位置编码**：用不同频率的正弦、余弦函数生成固定位置向量；
- **相对位置偏置**：根据两个 Token 的距离，直接修改注意力分数；
- **RoPE（Rotary Position Embedding）**：按照位置旋转 Query 和 Key，让它们的点积自然包含相对位置信息。

现代 Decoder-only 模型广泛使用 RoPE 或它的变体，本课重点学习它。

## 2. 先理解二维旋转

二维向量旋转角度 $\phi$，可以写成：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}
$$

$$
R(\phi)
\begin{bmatrix}x_1\\x_2\end{bmatrix}
=
\begin{bmatrix}
x_1\cos\phi-x_2\sin\phi\\
x_1\sin\phi+x_2\cos\phi
\end{bmatrix}
$$

旋转只改变方向，不改变向量长度。RoPE 将每个注意力头的维度两两配对，例如 $D=8$ 时形成 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)` 四个二维平面；位置越靠后，旋转角度越大。

![RoPE 在二维平面旋转向量，相对角度只由位置差决定](/images/llm/rope-relative-rotation.svg)

## 3. RoPE 怎样进入 Attention

输入仍然先通过线性投影得到 $Q$、$K$、$V$。RoPE 位于投影之后、计算 $QK^T$ 之前，通常只旋转 $Q$ 和 $K$，不旋转 $V$：

$$
Q'=\operatorname{RoPE}(Q, position),\qquad
K'=\operatorname{RoPE}(K, position)
$$

$$
Attention(Q,K,V)=\operatorname{softmax}\left(\frac{Q'K'^T}{\sqrt{D}}+Mask\right)V
$$

![RoPE 在注意力计算链路中的准确位置](/images/llm/rope-attention-flow.svg)

这一区分很重要：

- $Q$ 和 $K$ 决定“应该关注谁”，因此把位置写进二者的匹配分数；
- $V$ 携带被取回的内容，通常不需要同样旋转；
- RoPE 没有改变 Attention 的输出形状，也通常不引入可训练参数。

## 4. 为什么点积能得到相对位置

对同一组二维维度，设位置 $m$ 的 Query 旋转 $m\theta$，位置 $n$ 的 Key 旋转 $n\theta$：

$$
q'_m=R(m\theta)q_m,\qquad k'_n=R(n\theta)k_n
$$

二者点积为：

$$
\begin{aligned}
(q'_m)^T k'_n
&=q_m^T R(m\theta)^T R(n\theta)k_n\\
&=q_m^T R((n-m)\theta)k_n
\end{aligned}
$$

结果只通过 $n-m$ 感知位置。若把两个 Token 同时向后移动 $c$ 个位置，距离仍为：

$$
(n+c)-(m+c)=n-m
$$

所以对应的相对旋转不变。这就是 RoPE 的核心：**用绝对位置决定各自旋转多少，再通过点积把它转化为相对位置关系。**

注意，这并不表示整个模型对位置平移完全不变；因果 Mask、上下文边界和 Token 内容仍然会影响结果。这里说的是单个 RoPE 点积中的相对位置性质。

## 5. 为什么需要多种旋转频率

如果所有维度都使用同一个 $\theta$，模型只能用一种尺度观察距离。RoPE 为不同维度对设置不同频率。常见写法之一是：

$$
\theta_i=base^{-2i/D},\qquad i=0,1,\ldots,D/2-1
$$

高频维度对旋转得快，擅长区分邻近位置；低频维度对旋转得慢，可以表达更长距离。模型把多个尺度组合起来判断两个 Token 的位置关系。

![不同频率共同编码近距离和远距离](/images/llm/rope-multi-frequency.svg)

具体实现中的维度排列、频率公式和 `base` 可能不同，阅读源码时要以对应模型配置为准。

## 6. 从 Shape 看一次 RoPE

假设：

- Batch Size：$B=2$
- 注意力头数：$N=4$
- 序列长度：$S=128$
- 每头维度：$D=64$

则 $Q$、$K$ 的 Shape 都是：

$$
[B,N,S,D]=[2,4,128,64]
$$

每头的 64 维被组成 32 个二维对。预计算的角度表可以看成 `[S, D/2] = [128, 32]`，再沿 Batch 和 Head 维广播。旋转前后 Shape 完全不变。

最小实现如下：

```python
import torch

def apply_rope(x, cos, sin):
    # x: [B, N, S, D]；cos/sin: [1, 1, S, D/2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

B, N, S, D = 2, 4, 128, 64
q = torch.randn(B, N, S, D)

base = 10_000
inv_freq = 1.0 / (base ** (torch.arange(0, D, 2) / D))
angles = torch.outer(torch.arange(S), inv_freq)
cos = angles.cos()[None, None, :, :]
sin = angles.sin()[None, None, :, :]
q_rotated = apply_rope(q, cos, sin)

assert q_rotated.shape == q.shape
```

这里的代码用于理解数学过程。生产实现通常会缓存角度表，并把 RoPE 与相邻算子融合，以减少中间张量和内存访问。

## 7. 长上下文为什么仍然困难

假设模型只在最大长度 $L_{train}$ 内训练。直接输入远超该长度的位置时，会出现几个问题：

1. 模型遇到训练阶段没有充分学习过的旋转相位组合；
2. 高频维度已经旋转许多圈，远距离位置可能出现难以区分的周期现象；
3. 即使位置编码能够外推，标准 Attention 的计算量仍随 $S^2$ 增长；
4. 推理时 KV Cache 通常随序列长度 $S$ 线性增长。

因此“支持更大的 position id”不等于“模型已经可靠、高效地支持长上下文”。

常见扩展方法包括线性位置缩放、Dynamic NTK、YaRN、LongRoPE 和分段频率缩放等。它们大体是在压缩位置或调整各频率的变化速度，但效果与模型训练方式、原始上下文长度和缩放参数强相关。使用已有模型时，应采用该模型配置声明的 RoPE 类型，不要随意替换。

## 8. 与 AI Infra 和集合通信的关系

### 8.1 算子与访存

RoPE 主要是逐元素乘加，计算量不大，但需要读取 $Q/K$ 和正余弦表、再写回结果。它可能更受内存带宽和 Kernel 启动开销影响，因此常被融合进 Q/K 投影后的处理或 Attention Kernel。

### 8.2 KV Cache

自回归推理中，常见实现会在写入 Cache 前把 $K$ 按它的绝对位置旋转。之后 Decode 只需旋转当前 Token 的 $Q$，就能与缓存中的 $K'$ 计算注意力。具体缓存布局仍要以框架实现为准。

### 8.3 并行执行

RoPE 本身通常不触发集合通信，但分布式实现仍有正确性约束：

- Tensor Parallel 切分 Head 时，一个二维旋转对的两个元素必须留在一起；
- Sequence/Context Parallel 下，各 Rank 必须使用全局位置编号，而不是都从 0 重新开始；
- 不同 Rank 若使用了不同的 `base`、缩放规则或 position offset，结果会静默出错。

这类问题很适合用集合通信算子开发中的思路检查：不仅看张量 Shape，还要追踪每个分片对应的全局语义。

## 9. 容易混淆的点

1. **RoPE 是给 Token Embedding 加一个位置向量吗？** 不是。它通常在得到 Q/K 后旋转其维度对。
2. **RoPE 会旋转 V 吗？** 主流用法通常不旋转 V；以具体模型实现为准。
3. **RoPE 有可训练参数吗？** 标准 RoPE 的频率按公式生成，通常没有可训练参数。
4. **相对位置是否意味着不需要绝对 position id？** 不是。每个 Q/K 仍先根据绝对 position id 旋转，点积后呈现相对关系。
5. **扩大 RoPE 缓存就能获得长上下文吗？** 不能。还涉及模型外推能力、训练/微调、Attention 计算量和 KV Cache 显存。

## 10. 动手练习

### 练习 1：验证长度保持不变

运行上面的最小实现，再比较旋转前后的向量二范数：

```python
print(torch.max(torch.abs(q.norm(dim=-1) - q_rotated.norm(dim=-1))))
```

结果应接近 0，因为二维旋转不改变向量长度。

### 练习 2：验证共同平移不改变相对点积

任选二维向量 $q,k$ 和位置 $m,n$，分别计算：

$$
[R(m\theta)q]^T[R(n\theta)k]
$$

以及同时平移 10 个位置后的：

$$
[R((m+10)\theta)q]^T[R((n+10)\theta)k]
$$

两者应近似相等。

## 11. 自测题

1. 没有位置编码的 Self-Attention 缺少什么信息？
2. RoPE 在 Attention 链路中的位置在哪里？
3. 为什么标准 RoPE 通常只作用于 Q 和 K？
4. $D=128$ 的一个注意力头包含多少个二维旋转对？
5. RoPE 旋转前后为什么不改变向量长度？
6. 如何从公式说明 RoPE 点积依赖相对位置？
7. 为什么不同维度对要使用不同频率？
8. 直接增大 position id 为什么不等于获得可靠长上下文？
9. RoPE 本身通常需要集合通信吗？
10. Context Parallel 中 position id 使用错误会发生什么？

::: details 自测答案

1. 缺少 Token 的顺序和距离信息；它只能根据内容进行匹配。
2. 在线性投影得到 Q/K 后、计算 $QK^T$ 前。
3. Q/K 决定匹配分数，旋转二者就能把位置关系写进分数；V 负责携带被取回的内容。
4. $128/2=64$ 个。
5. 旋转矩阵是正交矩阵，满足 $R^TR=I$，因此保持二范数不变。
6. $(R(m\theta)q)^T(R(n\theta)k)=q^TR((n-m)\theta)k$，位置通过差值 $n-m$ 出现。
7. 高频擅长区分近距离，低频能覆盖更长距离；多种尺度共同表达位置。
8. 超出训练长度后模型会遇到陌生相位；同时 Attention 的平方计算量和 KV Cache 显存问题仍然存在。
9. 通常不需要，它是本地逐元素计算。
10. 各 Rank 会把不同的全局 Token 当成相同位置，导致 Q/K 旋转错误，最终注意力结果错误。

:::

## 12. 本课小结

- Self-Attention 本身不能从内容计算中恢复 Token 顺序；
- RoPE 把注意力头维度两两配对，并按位置和频率旋转 Q/K；
- 两次旋转后的点积包含 $(n-m)$，因而能够表达相对位置；
- 多频率让模型同时观察近距离与远距离；
- 长上下文不仅是位置编码问题，还受训练分布、Attention 计算量和 KV Cache 限制；
- RoPE 通常没有通信，但并行切分必须保持旋转对和全局 position id 正确。

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Hugging Face Transformers：RoPE utilities](https://huggingface.co/docs/transformers/internal/rope_utils)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

下一课将进入训练循环，把 Loss、反向传播、AdamW、学习率、混合精度和梯度累积连成一次完整的参数更新。
