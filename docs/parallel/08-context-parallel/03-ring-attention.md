# 单元 08-3｜Ring Attention 与负载均衡

> 所属章节：[第 8 章｜Context Parallel（序列并行）](../08-context-parallel.md)

::: info 本单元目标
围绕 **"causal 三角与 zigzag 均衡"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 5. Ring Attention 的运转细节

把 08-2 的轮转展开成时间表（N=4，块 $K_0..K_3$ 沿环逆时针流动）：

```text
轮次：      0         1         2         3（回到初始）
rank 0: 算[Q0×K0]  算[Q0×K3]  算[Q0×K2]  算[Q0×K1]
rank 1: 算[Q1×K1]  算[Q1×K0]  算[Q1×K3]  算[Q1×K2]
rank 2: 算[Q2×K2]  算[Q2×K1]  算[Q2×K0]  算[Q2×K3]
rank 3: 算[Q3×K3]  算[Q3×K2]  算[Q3×K1]  算[Q3×K0]
        ↑ 本轮收到的块同时传给左邻——传与算并行
```

每轮的 partial attention 用**在线 softmax** 累加（max/sum 修正后合并），N 轮后各 rank 的 $O_{local}$ 即最终结果——这正是 FlashAttention 分块计算的序列间版本。

## 6. causal 三角：Ring 的隐藏地雷

causal mask（[《模型全景》04-2](../../model/04-self-attention/02-scaled-dot-product-mask.md)）：第 $i$ 个 token 只能看 $j \le i$——投影到块上：

```text
块计算量矩阵（行=Q块i，列=K块j）：
        K0   K1   K2   K3
Q0      △    0    0    0      △ = 对角块（半量）
Q1      ■    △    0    0      ■ = 满量
Q2      ■    ■    △    0
Q3      ■    ■    ■    △      → 总工作量 = 三角形
```

**问题**：朴素轮转下，持 Q3 的 rank 每轮都算满量，持 Q0 的 rank 三轮在算空气——**同一批人里有人忙死有人闲死**，而集合式的轮转节拍由最慢者决定（[《集合通信》04-4 的木桶效应](../../collective/04-gather-scatter-alltoall/04-moe-imbalance.md)再次现形）。

**解法：均衡重排（zigzag / 斜切）**——把三角形区域按"每 rank 的总块权相等"重新分派：例如把 $(i,j)$ 的计算责任偏移交错，使每 rank 分到约 $\frac{N+1}{2}$ 个块当量，而不是 $0..N$ 个不等。核心思想一句话：**矩阵的分块所有权不必沿对角线，沿"等权斜线"切**。

::: tip 一图记住
AllGather 的账在**显存峰值**，Ring 的账在**显存 + 调度**：它把通信藏进了计算，又把"谁算哪块"从自由题变成了排列组合题——均衡做不好，省下的通信时间全赔给空转。
:::

---

[继续单元 08-4 →](./04-cp-engineering.md)
