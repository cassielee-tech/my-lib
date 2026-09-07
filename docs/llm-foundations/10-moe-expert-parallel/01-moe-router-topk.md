# 单元 10-1｜MoE、Router 与 Top-k

> 所属章节：[第 10 章｜MoE 与 Expert Parallel](../10-moe-expert-parallel.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **MoE、Router 与 Top-k** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

普通 Decoder Block 的每个 Token 都经过同一套 Attention 和 FFN。若把 FFN 扩大一倍，每个 Token 的计算量也大致随之增加。

MoE（Mixture of Experts）提供了另一种扩展思路：准备多个不同参数的 FFN Expert，但每个 Token 只选择少数几个 Expert 计算。

```text
Dense FFN：每个 Token → 同一个 FFN
Sparse MoE：每个 Token → Router → Top-k Experts
```

这样可以增加**总参数量**，同时让每个 Token 的**激活参数量和计算量**只与 Top-k Expert 有关。

![Dense FFN 与 Sparse MoE 的计算路径差异](/images/llm/dense-vs-moe.svg)

::: warning 总参数不等于激活参数
MoE 的所有 Expert 权重仍需要存储在设备集群中，只是单个 Token 不会经过全部 Expert。因此“计算稀疏”不等于“权重不用占显存”。
:::

## 2. MoE 通常替换 Decoder Block 的哪一部分

常见 Sparse MoE Transformer 保留 Attention、RMSNorm 和残差连接，只把部分或全部 Dense FFN 替换为 MoE Layer：

$$
h'=h+Attention(RMSNorm(h))
$$

$$
y=h'+MoE(RMSNorm(h'))
$$

每个 Expert 本身通常仍是一个独立 FFN，例如 SwiGLU：

$$
Expert_i(x)=W_{down}^{(i)}left(SiLU(W_{gate}^{(i)}x)\odot W_{up}^{(i)}x\right)
$$

所以“8 个 Expert”通常意味着这一层拥有 8 套不同的 FFN 参数，而不是 8 个完整 Transformer 模型。Attention 和其他公共层仍然共享。

有些架构还会同时使用：

- **Routed Experts**：由 Router 为每个 Token 动态选择；
- **Shared Experts**：所有 Token 都执行，用于承载公共能力。

## 3. Router 怎样为 Token 选择 Expert

设一层有 $E$ 个 Expert，Token Hidden State 为 $x\in\mathbb{R}^{H}$。Router 通常先做一个小型线性投影：

$$
z=W_rx,\qquad W_r\in\mathbb{R}^{E\times H}
$$

经过 Softmax 得到路由概率：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{E}e^{z_j}}
$$

再选择概率最高的 $k$ 个 Expert：

$$
\mathcal{T}(x)=TopK(p,k)
$$

输出是选中 Expert 结果的加权和：

$$
y=\sum_{i\in\mathcal{T}(x)}\tilde p_i Expert_i(x)
$$

$\tilde p_i$ 通常是对 Top-k 概率重新归一化后的权重，具体实现可能不同。

![Router 为每个 Token 产生不同的 Top-k 路径](/images/llm/moe-router-topk.svg)

需要注意：

- 不同 Token 可以选择不同 Expert；
- 同一个 Token 在不同 MoE 层也可能选择不同 Expert；
- Expert 是训练中自动形成的参数分支，不应简单假设它们分别固定代表“数学”“代码”“英语”；
- Top-1 只执行一个 Expert，Top-2 会执行两个并加权合并。

## 4. 从 Shape 看一次路由

把 Batch 和 Sequence 展平，设：

- Token 数 $T=B\times S$
- Hidden Size $H$
- Expert 数 $E$
- 每 Token 选择 $k$ 个 Expert

则主要张量为：

| 张量 | Shape | 含义 |
| --- | --- | --- |
| 输入 | `[T, H]` | 展平后的 Token Hidden State |
| Router Logits | `[T, E]` | 每个 Token 对所有 Expert 的分数 |
| Top-k Index | `[T, k]` | 被选中的 Expert ID |
| Top-k Weight | `[T, k]` | 合并 Expert 输出的权重 |
| Expert 输入 | 逻辑上 `[T × k, H]` | 每个 Token 复制到所选 Expert |
| MoE 输出 | `[T, H]` | 按原 Token 顺序恢复后的结果 |

当 $k=2$ 时，一个 Token 会产生两条 Expert Assignment，因此逻辑 Expert 输入数是 $2T$。实际实现还会根据 Expert 分组、容量 Padding 或 Dropless Routing 改变物理布局。

---

[继续单元 10-2 →](./02-load-balance-capacity.md)
