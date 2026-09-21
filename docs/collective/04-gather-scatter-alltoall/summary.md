# 第 4 章总结｜AllGather、ReduceScatter 与 AlltoAll

> 返回：[第 4 章首页](../04-gather-scatter-alltoall.md)

## 把本章串成一条主线

请先合上各单元正文，沿"AG → RS → A2A → MoE 失衡"复述整章；遇到断点时，再回到对应单元查阅。

```text
AG（04-1）：[S/N]→[S] 复制拼全 · Ring 下半场 · TP/SP 前向、FSDP 取参数
        ↓
RS（04-2）：[S]→[S/N] 规约分持 · Ring 上半场 · TP/SP 反向、FSDP 梯度
   （AllReduce = AG + RS：恒等式在场景与账本两层都成立）
        ↓
A2A（04-3）：定向搬运 · 总量守恒 · 逐对交换 · MoE dispatch/combine
        ↓
MoE 失衡（04-4）：路由动态 → 流量不均 → 木桶效应
   缓解：模型侧（均衡约束/限流）+ 通信侧（等块 padding/流水）
```

## 9. 动手练习

### 练习 1：画 shape 图

4 rank、S=256 MB。分别画出 AllGather、ReduceScatter、（均衡）AlltoAll 三者在"开始/结束"时每个 rank 持有的数据形状与大小，并标注每个原语每 rank 的实际传输量。

### 练习 2：TP+SP 的前后向通信

Megatron TP+SP 组合中，前向用 AG、反向用 RS。设每层激活 512 MB、8 卡：写出一层前向+反向的总通信量，并与"前向反向各做一次 AllReduce"的朴素方案对比，算出节省比例。

### 练习 3：诊断 MoE 瓶颈

一个 8 rank MoE 训练任务，Profiling 显示 A2A 阶段耗时波动巨大（有时 2 ms、有时 15 ms），且各 rank 进入 A2A 的时间几乎相同。给出两条最可能的原因与对应的验证手段。

## 10. 自测题

1. AllGather 的输入/输出形状分别是什么？数据被复制了几份？
2. Ring AG 的轮次与每 rank 流量是多少？
3. TP 前向为什么需要 AllGather？接在什么算子之后？
4. ReduceScatter 结束时每个 rank 手里有什么？
5. FSDP 反向用 RS 而不是 AllReduce 规约梯度，省在哪里？
6. AlltoAll 与 AllGather 的本质区别（从"总量"和"谁拿什么"两点答）？
7. 均衡 A2A 的逐对交换每轮在做什么？结构上像哪个算法的舞步？
8. 为什么说"同步性放大了 MoE 失衡的伤害"？
9. 容量因子限流的收益与代价分别是什么？
10. "语义上够用的最小原语"是什么意思？举一个本章的例子。

::: details 自测答案

1. 输入每人 S/N（一片），输出每人 S（按 rank 顺序拼接的全量）；数据被复制了 N 份。
2. N−1 轮完全块传阅；每 rank 流量 S(N−1)/N。
3. 列并行的线性层输出后：各 rank 持有激活的不同切片，拼全才能进入下一个需要全量输入的算子（如 Norm/非线性）。
4. 恰好一段"该段的全体之和"（S/N 的完全规约块）。
5. AllReduce 让每人拿全长梯度（2S(N−1)/N），RS 只拿自己参数段（S(N−1)/N）——参数本就分片存，别人的梯度段无用，省一半通信。
6. 总量：AG 变多（复制）、A2A 守恒（搬运）；谁拿什么：AG 人人拿一样的全量、A2A 人人拿不一样的专属子集。
7. rank r 与 (r XOR k) 互换双方互发的那一块；舞步同 RD（配对地址异或），节奏同 Ring（N−1 轮带宽友好）。
8. 集合通信全员到齐才散场：7 个方向发完也得陪最慢的方向等——最慢方向决定整体时长（木桶效应）。
9. 收益：通信变均匀、峰值可控；代价：超限 token 被丢，牺牲少量计算/精度。
10. 选语义够用且通信量最小的原语，而不是无脑 AllReduce；例：FSDP 梯度用 RS 而非 AllReduce。

:::

## 11. 本课小结

- AG 复制拼全、RS 规约分持——Ring AllReduce 的两个半场各自独立成军，支撑 TP/SP/FSDP；
- A2A 是定向搬运：总量守恒、按目的地重排，MoE 的 dispatch/combine 骨架；
- MoE 之难在动态失衡：模型侧调流量分布、通信侧在失衡下求效率；
- 选原语第一直觉：语义上够用的最小原语。

## 参考资料

- [《模型全景》第 1 章：并行与通信总览](../../model/01-landscape/02-parallelism-communication.md)
- [《训练与推理系统》第 1 章：训练中的显存与通信](../../systems/01-training-loop/04-training-memory-communication.md)
- [《并行策略》第 1 章：Token 的 AlltoAll 分发](../../parallel/01-moe-expert-parallel/03-token-alltoall.md)
- [第 1 章：六大原语卡片](../01-collective-semantics-cost/02-primitive-cards.md)
- [第 2 章：Ring 两阶段手推](../02-ring-allreduce/02-reduce-scatter-phase.md)

下一课收官《集合通信》：**拓扑、分层与重叠**——真实集群"机内高速 + 机间低速"的两层世界，以及算法、chunk/channel、重叠如何托举前面所有原语。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
