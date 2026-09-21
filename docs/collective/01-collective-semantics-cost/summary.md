# 第 1 章总结｜Collective 语义与代价模型

> 返回：[第 1 章首页](../01-collective-semantics-cost.md)

## 把本章串成一条主线

请先合上各单元正文，沿"坐标系 → 语义卡片 → 代价账本 → 选型"复述整章；遇到断点时，再回到对应单元查阅。

```text
坐标系（01-1）：Rank × 通信域 × 消息
        ↓
六大原语（01-2）：开始谁有什么 → 结束谁有什么
   黄金恒等式：AllReduce = Reduce+Broadcast = ReduceScatter+AllGather
        ↓
代价模型（01-3）：T ≈ α×轮次 + β×字节/BW + γ×计算
   小消息 α 主导 · 大消息 β 主导 · AllReduce 的账 ≈ 2 倍
        ↓
选型两层（01-4）：选原语（语义，不能错）→ 选算法（大小×N×拓扑）
        ↓
第 2 章：Ring——恒等式 RS+AG 的经典实现
```

## 9. 动手练习

### 练习 1：画语义快照

对 AllGather、ReduceScatter、AlltoAll 各画"开始/结束"两张快照图（4 个 rank，每人 4 块数据），标出每个 rank 手里的块。

### 练习 2：算一次账

8 个 rank 做 1 GB 的 AllReduce，有效带宽 25 GB/s，每轮 α = 20 µs。某算法需要 2×(N−1)=14 轮、每 rank 经手 2·S·(N−1)/N 字节。分别计算 β 项与 α 项，判断谁主导。

### 练习 3：抓误用

某代码用 AllGather 收集所有 rank 的梯度后各自本地求和。对照账本计算它比 AllReduce 多传多少字节（N=8），并说出正确原语。

## 10. 自测题

1. Rank、World Size、通信域分别是什么？
2. 为什么说"rank 是逻辑编号，谈拓扑必须映射回物理位置"？
3. 六个原语各自的"结束时谁拿到什么"？
4. 写出 AllReduce 的两种分解，哪种是优化重点、为什么？
5. AllGather 和 ReduceScatter 的对偶关系是什么？
6. AlltoAll 与其他五个原语的本质区别是什么？
7. 写出 α-β-γ 公式并说明各项何时主导。
8. 为什么 AllReduce 的理想账约是 AllGather 的两倍？
9. 列举两个"选错原语"的例子及其代价。
10. 算法选型看哪三个维度？

::: details 自测答案

1. Rank：参与通信的逻辑单位（通常一进程一卡）；World Size：组内 rank 总数；通信域：一组成员的集合，一个任务可建多个域。
2. 物理卡编号（NPU ID/Chip ID/逻辑 ID）与 rank 编号可能错位；算法性能取决于物理位置（谁和谁近），不取决于逻辑编号。
3. Broadcast 全员有 X；Reduce 仅 root 有结果；AllReduce 全员有归约结果；AllGather 全员有完整拼接；ReduceScatter 各自持有归约结果的一块；AlltoAll 各自收到定向投递给它的块。
4. Reduce+Broadcast 与 ReduceScatter+AllGather。后者是重点：中间态每人只经手 1/N，是 Ring 等高效算法的结构。
5. AllGather 是"拼接"（[S/N]→[S]），ReduceScatter 是"归约后切分"（[S]→[S/N]）——方向相反、互为逆操作的一半。
6. AlltoAll 不是"拼/切"而是**定向重分发**：出发前每块就指定了收件人，总量不变但布局按目标重排。
7. T ≈ α×轮次 + β×字节/带宽 + γ×本地计算。小消息/多轮次算法 α 主导；大消息 β 主导；含规约或压缩的操作 γ 不可忽略。
8. 因为 AllReduce 语义上等于 RS+AG 两段，每 rank 经手约 2·S·(N−1)/N，而单段原语只有 S·(N−1)/N。
9. 如：AllGather+本地求和替代 AllReduce（多传约 N 倍字节）；AllReduce 替代 Reduce（所有人陪跑并多收数据）。
10. 消息大小、rank 数、拓扑。

:::

## 11. 本课小结

- 坐标系：Rank × 通信域 × 消息，语义=两张快照的差别；
- 黄金恒等式：AllReduce = RS + AG，一切高效算法的出发点；
- AlltoAll 是定向重分发，与其他原语本质不同；
- 三本账：轮次 × α、字节 × β、计算 × γ；小消息与大消息是两个世界；
- 选型两层：原语由数据需求决定（错了无法补救），算法由大小×N×拓扑决定。

## 参考资料

- [NCCL 官方文档：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/ops.html)
- [HCCL 源码专题 H01-6：集合通信算法与代价模型](../../ascend/hccl-source/06-coll-algorithms.md)
- [《模型全景》第 1 章：并行为什么产生通信](../../model/01-landscape/02-parallelism-communication.md)
- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)

下一课完整推导第一个算法：**Ring AllReduce**——恒等式 RS+AG 的经典实现，每轮谁发哪一块，账本为什么漂亮。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
