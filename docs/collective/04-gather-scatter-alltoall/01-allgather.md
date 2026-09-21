# 单元 04-1｜AllGather：从分片到完整

> 所属章节：[第 4 章｜AllGather、ReduceScatter 与 AlltoAll](../04-gather-scatter-alltoall.md)

::: info 本单元目标
围绕 **AllGather 的布局变化与使用场景** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 1. 语义：每人贡献一片，人人拿到全量

```text
开始：rank 0 有 [A]   rank 1 有 [B]   rank 2 有 [C]   rank 3 有 [D]
结束：每个 rank 都有 [A|B|C|D]（按 rank 顺序拼接）
```

- 输入每人 **S/N**，输出每人 **S**——**数据被复制了 N 份**（对比 AlltoAll：总量守恒，下一单元）；
- 没有计算，纯搬运拼装。

### 1.1 算法：Ring AllReduce 的下半场

第 2 章 AG 阶段**原封不动**就是一个完整的 AllGather 算法：N−1 轮完全块传阅。账本也自然是 AllReduce 的一半：

$$
T_{AG} \approx (N-1)\alpha + \frac{S(N-1)}{N}\cdot\frac{1}{BW}
$$

（另有 RD 风格的 log 轮次变体，选型逻辑同第 3 章。）

## 2. 谁在用 AllGather

| 场景 | 用在哪 | 为什么 |
| --- | --- | --- |
| **TP 前向**（列并行后的激活拼装） | 每层线性层输出后 | 各 rank 算出激活的不同切片，拼全才能进下一个非并行算子 |
| **SP 前向** | Norm/Attention 前的序列维拼装 | 序列并行把序列切段，算子要全序列时先 AG |
| **FSDP** | forward/backward 取参数前 | 参数平时分片存着，用时 AG 展开成完整副本 |

::: tip TP+SP 的"一半魔法"
Megatron 的 TP+SP 组合里，前向用 **AG**、反向用 **RS**（下一单元）——恰好把 AllReduce 拆成两半分别挂在前后向，通信量减半。这正是第 1 章恒等式在工程里的变现（详见 [大模型基础：并行通信总览](../../model/01-landscape/02-parallelism-communication.md)）。
:::

---

[继续单元 04-2 →](02-reduce-scatter.md)
