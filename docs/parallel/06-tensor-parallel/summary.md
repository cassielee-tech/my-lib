# 第 6 章总结｜Tensor Parallel

> 返回：[第 6 章首页](../06-tensor-parallel.md)

## 把本章串成一条主线

请先合上各单元正文，沿"动机 → 切法 → 账本 → 工程"复述整章；遇到断点时，再回到对应单元查阅。

```text
动机（06-1）：DP 不省显存、不加速单层 → 切模型本身
   W 按列切：权重 ÷N · 输入需完整 · 输出天然分片
        ↓
切法（06-2）：ColP（输出分片/输入复制）⇄ RowP（输入分片/输出部分和）
   经典组合 ColP→GeLU→RowP：中间零通信，出口 AllReduce
   Attention：多头即天然切分，同样的 ColP…RowP 框架
        ↓
账本（06-3）：每层 4 次 AllReduce（激活级、MB 级、128 次/步）
   不可重叠（下一层输入=本层 AllReduce 输出）
   机内 0.14s/步 vs 机间 1.1s/步 → TP 组必须住机内，tp=每机卡数
        ↓
工程（06-4）：TP+SP 把 AllReduce 拆成 RS+AG（通信减半）
   显存三大件 ÷tp · 优化方向是压量不是找重叠 · 跨机交给 DP/PP
```

## 9. 动手练习

### 练习 1：Shape 推导

MLP：$W_1 \in \mathbb{R}^{4096 \times 16384}$、$W_2 \in \mathbb{R}^{16384 \times 4096}$、输入 $X \in \mathbb{R}^{8 \times 2048 \times 4096}$（BF16），TP=8。写出每卡持有的 $W_1, W_2$ 形状、中间激活形状，标出两次通信的位置、方向与数据量（MB）。

### 练习 2：通信账对比

同一模型（7.5B、32 层、h=4096）分别用纯 DP（8 卡）与纯 TP（8 卡）训练：估算两者每步的通信总量，并从"对象/频率/可重叠性"三个维度列表对比。

### 练习 3：位置判断

某团队把 `tp=16` 跨两台机器（每机 8 卡）部署，实测吞吐反而低于 `tp=8`。用本章账本解释原因，并给出不改硬件的两种改法。

## 10. 自测题

1. DP 的两个做不到是什么？TP 分别怎么解决？
2. 写出按列切 $W$ 后 $Y$ 的分块表达式。三个观察分别是什么？
3. Column Parallel 的输入/输出各是什么状态？Row Parallel 呢？
4. 为什么 ColP→GeLU→RowP 中间零通信？两处"恰好"分别是什么？
5. Attention 的 TP 为什么多头就是天然切分？哪一步产生通信？
6. 每层 4 次 AllReduce 分别在哪些位置（前向/反向各哪里）？
7. TP 通信为什么不可重叠？与 DDP 可重叠的差别根源是什么？
8. 用数字说明 TP 组为什么必须住机内。
9. TP+SP 怎样把 AllReduce 通信减半？用到的恒等式是什么？
10. TP=8 时显存三大件怎么变？还有什么没省？

::: details 自测答案

1. 不省显存（每卡完整模型）→ TP 三大件 ÷N；不加速单层 → TP 把单层矩阵乘拆到多卡。
2. $W = [W_1 | \cdots | W_N]$，$Y = [XW_1 | \cdots | XW_N]$。权重 ÷N；每卡需完整输入 X；输出天然列分片。
3. ColP：输入完整（复制/AG），输出列分片。RowP：输入列分片（天然），输出部分和（需求和）。
4. ColP 的输出列分片恰好是 RowP 需要的输入分片；GeLU 逐元素在分片上各自做照样正确。
5. 头与头独立，按 head 维 ColP 后每卡的注意力计算在自己的头上完整进行；输出投影 RowP 后的部分和求和（AllReduce）产生通信。
6. 前向：Attention 出口、MLP 出口各一次；反向：两处的入口梯度各一次。
7. 下一层输入 = 本层 AllReduce 输出——通信在关键路径上；DDP 通信的是梯度、消费者是 step，中间整段反向可并行，而 TP 没有这个空隙。
8. 128 MB×128 次：机内 200 GB/s ≈ 0.14 s/步可接受；机间 25 GB/s ≈ 1.1 s/步——每步白烧约一秒纯通信。
9. 前向用 RS（各留一段，S(N-1)/N）代替 AllReduce（2S(N-1)/N），反向用 AG 拼回——AllReduce = RS + AG 恒等式；Norm 等沿序列独立的层顺势切到序列维。
10. 参数/梯度/优化器状态全部 ÷8；输入激活仍复制（SP 之后连激活也省一部分）。

:::

## 11. 本课小结

- TP 切权重不切数据，与 DP 正交——两把刀切不同的轴；
- ColP 与 RowP 互补，经典组合让逐元素层零通信，出口 AllReduce 收口；
- 每层 4 次、激活级、不可重叠——TP 通信小而贵，只能住机内（tp = 每机卡数）；
- 三大件 ÷tp 是 TP 的显存红利；TP+SP 用恒等式再砍一半通信；
- DP/TP/PP 三种性格初现：宽容 / 挑剔 / 下一章揭晓。

## 参考资料

- [Megatron-LM 论文：Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- [《模型全景》05-4：Block 张量并行](../../model/05-decoder-block/04-tensor-parallel-block.md)
- [《集合通信》04-1/04-2：AllGather 与 ReduceScatter](../../collective/04-gather-scatter-alltoall/01-allgather.md)
- [《集合通信》05-4：藏不住的关键路径通信](../../collective/05-topology-hierarchical-overlap/04-overlap-practice.md)
- [HCCL 源码专题 H01-6：算法与拓扑感知](../../ascend/hccl-source/06-coll-algorithms.md)

下一章 **Pipeline Parallel** 换一把刀：不再切层内部，而是把层**分段接力**——Stage、Micro-batch、1F1B 与 Bubble，通信最少、调度最讲究的一种并行。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
