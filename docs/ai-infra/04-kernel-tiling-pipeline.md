# 第 4 章｜Kernel、Tiling 与流水线

> 本课目标：理解 Kernel 的 Host/Device 两半结构与启动开销；掌握 Tiling 的两个维度（多核切分与核内分块）及其约束；理解流水线与 Double Buffer 怎样让搬运和计算重叠，为阅读真实算子代码建立骨架认知。

## 本章导学

::: tip 本课只记住 3 件事
1. Kernel 有两半：Host 侧负责"切分与下发"，Device 侧每个执行单元"按索引认领自己的份"。
2. Tiling 是第 3 课账本的工程化：多核切分决定利用率（第 2 课），核内分块决定复用次数（第 3 课）。
3. 流水线的最小可行方案是"队列 + 双缓冲"：正在算 A 块时，B 块的数据已经在路上。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。先解剖 Kernel 结构，再学切分，然后让流水转起来，最后走读一个最小算子把三章知识合龙。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./04-kernel-tiling-pipeline/quick.md)：一个工程怎么分包的类比讲完整章，再回来按单元深入。

- [ ] 我能说出 Host 侧与 Device 侧各自的职责和 Kernel Launch 开销的影响
- [ ] 我能为 GEMM 设计一个两级切分方案并核对容量、对齐、尾块
- [ ] 我能画出一个 Tile 的串行/流水时间线并判断流水收益

## 本章单元

- **I04-1（约 15 分钟）**：[Kernel 的解剖：入口、索引与启动开销](./04-kernel-tiling-pipeline/01-kernel-anatomy.md)
- **I04-2（约 15 分钟）**：[Tiling：把大算子切成块](./04-kernel-tiling-pipeline/02-tiling-strategy.md)
- **I04-3（约 15 分钟）**：[流水线与 Double Buffer](./04-kernel-tiling-pipeline/03-pipeline-double-buffer.md)
- **I04-4（约 15 分钟）**：[最小算子走读与调优检查单](./04-kernel-tiling-pipeline/04-minimal-kernel-walkthrough.md)

- **本章总结**：[练习、自测与串联](./04-kernel-tiling-pipeline/summary.md)

## 这一课在整条路线中的位置

第 2 章认识了计算单元，第 3 章认识了存储层次，本章回答"软件怎样把一个算子铺到这些资源上"。它是理解真实算子代码（CANN 篇第 3、4 章 Ascend C）的通用前置，也是第 5 章 Roofline 分析之后"动手改"的第一站。
