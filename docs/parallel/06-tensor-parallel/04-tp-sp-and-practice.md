# 单元 06-4｜TP+SP 与工程要点

> 所属章节：[第 6 章｜Tensor Parallel](../06-tensor-parallel.md)

::: info 本单元目标
围绕 **"AllReduce 拆半与 TP 的工程清单"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 7. TP+SP：把 AllReduce 拆成两半

TP 每层的 4 次 AllReduce 中，有两处"出口"其实不必拿**完整**输出——**Norm 层沿序列维切开后（Sequence Parallel，SP）**，出口只需要序列分片：

$$
\underbrace{\text{AllReduce}(S)}_{\text{每卡拿全长}} \;=\; \underbrace{\text{ReduceScatter}(S)}_{\text{规约后各留一段}} + \underbrace{\text{AllGather}(S)}_{\text{需要时再拼}}
$$

- 前向出口：只要下一段 → **RS 代替 AllReduce**（通信减半：$S(N-1)/N$ 代替 $2S(N-1)/N$，回扣 [《集合通信》04-2](../../collective/04-gather-scatter-alltoall/02-reduce-scatter.md)）；
- 反向入口：需要拼回完整序列 → **AG 补上**；
- Norm/Dropout 等**沿序列独立**的层顺势切到序列维——三大件的分片之外，激活也开始省。

Megatron 的 TP+SP 正是"恒等式落地"的经典（[《集合通信》01-2 的黄金恒等式](../../collective/01-collective-semantics-cost/02-primitive-cards.md)）：**语义上够用的最小原语**再次立功。SP 的完整推导是[第 8 章](../index.md)的主角，这里先记住"TP+SP = 每层通信减半"。

## 8. 工程清单与收束

把 TP 投入生产要过的检查单：

1. **组划分**：`tp_group` = 机内 8 卡（第 4 章 node-major），跨机不设 TP；
2. **显存账**：三大件 ÷tp；权重加载时各 rank 只载自己的分片；
3. **通信原语**：出口 AllReduce（或 TP+SP 的 RS+AG）落在 tp_group 上——HCCL 按拓扑自动选机内高速通路（[H01-6](../../ascend/hccl-source/06-coll-algorithms.md)）；
4. **重叠期望管理**：TP 通信在关键路径上——优化方向是**压通信量**（TP+SP、量化激活）而不是找重叠；
5. **组合原则**：单机内 TP，跨机交给 DP/PP——`tp=8` 之外的世界由后两章接管。

::: tip 本章收束
三种并行的性格已经显现：**DP 通信大而低频、可重叠**（宽容，用慢网也行）；**TP 通信小而高频、不可重叠**（挑剔，必须住机内）；下一章的 **PP 通信极小但引入空泡**（省通信、费调度）——三者互补，拼出混合并行的完整版图。
:::

---

[进入本章总结 →](./summary.md)
