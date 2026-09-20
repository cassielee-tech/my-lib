# 第 2 章｜GPU/NPU 执行模型与计算单元

> 本课目标：理解 CPU、GPU、NPU 为什么采用不同执行方式；说清 GPU 的 SM/Warp/SIMT 与 NPU 的 Cube/Vector/Scalar 各自承担什么；解释一块芯片的理论算力为什么经常跑不满。

## 本章导学

::: tip 本课只记住 3 件事
1. CPU 用复杂控制换**低延迟**，GPU 用海量线程换**吞吐**，NPU 用专用数据通路换**矩阵计算效率**。
2. 执行模型决定利用率：Warp 分支分歧、数据搬运跟不上、尾块浪费，都是"峰值跑不满"的结构性原因。
3. Core、Warp、Tile 这些术语跨平台通用，但具体含义必须回到具体硬件确认。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。先建立三种执行哲学的对比，再分别走进 GPU 与 NPU 内部，最后用执行模型解释利用率损失。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./02-execution-model/quick.md)：三支施工队的类比讲完整章，再回来按单元深入。

- [ ] 我能解释 CPU、GPU、NPU 执行模型的本质差异
- [ ] 我能说出 SM/Warp/SIMT 与 Cube/Vector/Scalar 各自的角色
- [ ] 我能列举至少 4 种"理论算力跑不满"的原因

## 本章单元

- **I02-1（约 15 分钟）**：[三种执行哲学：CPU、GPU 与 NPU](./02-execution-model/01-execution-philosophy.md)
- **I02-2（约 15 分钟）**：[GPU 内部：SM、Warp 与 SIMT](./02-execution-model/02-gpu-sm-warp-simt.md)
- **I02-3（约 15 分钟）**：[NPU 内部：达芬奇架构与计算单元](./02-execution-model/03-npu-davinci-core.md)
- **I02-4（约 15 分钟）**：[从执行模型到利用率](./02-execution-model/04-utilization-tail-effects.md)

- **本章总结**：[练习、自测与串联](./02-execution-model/summary.md)

## 这一课在整条路线中的位置

第 1 章把一次模型执行拆成了框架、编译、Runtime、Kernel 与硬件阶段。本章下探到硬件层，回答"最后一层里发生了什么"；下一课再进入存储层次与数据搬运。理解执行模型，才能看懂后面的 Kernel/Tiling（第 4 课）和利用率指标（贯穿性能分析与 CANN 调优）。
