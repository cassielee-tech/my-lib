# 第 4 章｜AllGather、ReduceScatter 与 AlltoAll

> 本课目标：掌握三大单原语的语义（数据布局怎么变）、典型算法与账本；把它们挂到 TP/SP/FSDP/MoE 的真实用法上；理解 MoE AlltoAll 失衡的根源与缓解路线。

## 本章导学

::: tip 本课只记住 3 件事
1. **AllGather：`[S/N] → [S]`**（每人一碎片 → 人人有全量，复制拼装）；**ReduceScatter：`[S] → [S/N]`**（规约后各持一段）——它们是 Ring AllReduce 拆开的两个半场，也是 TP/SP/FSDP 的日常动词。
2. **AlltoAll 是"定向搬运"不是"复制"**：每人发给每人不同的一块，**总量守恒、按目的地重排**——MoE dispatch/combine 的骨架。
3. **MoE 的 A2A 之难在动态失衡**：路由在运行时决定流量，同步集合通信被最慢的方向拖死——**木桶效应**。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。建议用一张卡片记"shape 变化图"，比记文字快。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](04-gather-scatter-alltoall/quick.md)：一个"办公室分发动作"的类比讲完整章，再回来按单元深入。

- [ ] 我能不看书画出三个原语的 shape 变化图
- [ ] 我能说出 TP/SP/FSDP/MoE 各在哪个阶段用哪个原语
- [ ] 我能解释 MoE A2A 为什么失衡，以及模型侧/通信侧两条缓解路线

## 本章单元

- **04-1（约 15 分钟）**：[AllGather：从分片到完整](04-gather-scatter-alltoall/01-allgather.md)
- **04-2（约 15 分钟）**：[ReduceScatter：规约后分着拿](04-gather-scatter-alltoall/02-reduce-scatter.md)
- **04-3（约 15 分钟）**：[AlltoAll：定向重分发](04-gather-scatter-alltoall/03-alltoall.md)
- **04-4（约 15 分钟）**：[MoE 的 AlltoAll 为什么难](04-gather-scatter-alltoall/04-moe-imbalance.md)

- **本章总结**：[练习、自测与串联](04-gather-scatter-alltoall/summary.md)

## 这一课在整条路线中的位置

第 1 章的原语卡片在这里展开成完整推导，第 2 章的 Ring 两半场在这里各自独立成军。模型侧的对应叙事在 [《模型全景》第 1 章（并行通信总览）](../model/01-landscape/02-parallelism-communication.md) 与 [《训练与推理系统》第 5 章 MoE](../parallel/01-moe-expert-parallel/03-token-alltoall.md)——本章是它们的系统视角补全。
