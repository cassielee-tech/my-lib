# 第 2 章｜Ring AllReduce 完整推导

> 本课目标：理解环形拓扑为什么带宽最优；亲手完成 4 rank 的 ReduceScatter 与 AllGather 两阶段模拟，说清每轮谁发哪一块；掌握 Ring 的时间公式与适用边界，了解 2N 分块的改进。

## 本章导学

::: tip 本课只记住 3 件事
1. Ring 的魔法：**每人只跟左右邻居说话，同时收发**——无论多少人，每条链路负载恒定，带宽最优。
2. AllReduce 的 Ring 实现 = **ReduceScatter（N−1 轮累积轮转）+ AllGather（N−1 轮完整块轮转）**——上一章恒等式的落地。
3. 时间公式 $T \approx 2(N-1)\alpha + \frac{2S(N-1)}{N \cdot BW}$：**带宽项随 N 递增收敛，延迟项线性膨胀**——大消息的天选算法、小消息的噩梦。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。推荐拿一张纸，跟着正文手工模拟——Ring 是"手推一遍、终身受益"的算法。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](02-ring-allreduce/quick.md)：一个"教室传卷子"的类比讲完整章，再回来按单元深入。

- [ ] 我能不看表说出 RS 阶段每一轮谁发哪块
- [ ] 我能手推 4 rank 的完整两阶段并验证最终状态
- [ ] 我能解释"带宽项与 N 无关"的含义和代价

## 本章单元

- **02-1（约 15 分钟）**：[为什么是环：拓扑的自然性](02-ring-allreduce/01-why-ring.md)
- **02-2（约 15 分钟）**：[第一阶段：ReduceScatter 轮转](02-ring-allreduce/02-reduce-scatter-phase.md)
- **02-3（约 15 分钟）**：[第二阶段：AllGather 轮转](02-ring-allreduce/03-allgather-phase.md)
- **02-4（约 15 分钟）**：[Ring 的账本与变体](02-ring-allreduce/04-ledger-and-variants.md)

- **本章总结**：[练习、自测与串联](02-ring-allreduce/summary.md)

## 这一课在整条路线中的位置

上一章立了语义与账本，本章推导第一个完整算法。Ring 是 HCCL/NCCL 等通信库大消息路径的支柱，也是 [HCCL 源码专题 H01-6](../ascend/hccl-source/06-coll-algorithms.md) 中 Ring/RHD/NHR 一族的原点——手推过 Ring，那一章的算法对比就能秒懂。
