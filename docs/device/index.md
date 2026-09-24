# 单卡执行系统

> 返回：[课程总览](../roadmap.md)

**专栏目标**：性能三件套视角——算子怎样铺到硬件、瓶颈怎样判断、任务怎样异步并发。这是理解 HCCL 执行模型（引擎、任务下发、Stream/Notify）的硬件与系统基础。

## 章节列表

上半程：**性能分析方法**（第 1-5 章）

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [全景与性能分析](01-landscape-performance.md) | 一个请求怎样穿过框架、编译器、Runtime、算子和硬件？ | ✅ |
| 2 | [GPU/NPU 执行模型与计算单元](02-execution-model.md) | CPU、GPU、NPU 为什么采用不同执行方式？昇腾 Cube/Vector 做什么？ | ✅ |
| 3 | [存储层次与数据搬运](03-memory-hierarchy.md) | HBM、Cache、片上 Buffer 有什么区别？数据为什么比计算贵？ | ✅ |
| 4 | [Kernel、Tiling 与流水线](04-kernel-tiling-pipeline.md) | 大算子怎样切成 Tile？搬运和计算怎样形成流水？ | ✅ |
| 5 | [FLOPs、带宽与 Roofline](05-flops-bandwidth-roofline.md) | 怎样判断算子是 Compute-bound 还是 Memory-bound？ | ✅ |

下半程：**执行系统**（第 6-8 章）

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 6 | [Eager、计算图与图编译](06-eager-graph-compilation.md) | 一行 Python 怎样变成可执行计算图？图模式为什么快？ | ✅ |
| 7 | [Stream、Event 与异步执行](07-stream-event-async.md) | 计算和通信怎样并发？什么时候必须同步？ | ✅ |
| 8 | [数据类型、布局与算子融合](08-dtype-layout-fusion.md) | BF16/FP8、Layout 和 Fusion 怎样影响精度与性能？ | ✅ |

**专题**：[2026 主流 AI 加速卡全景](accelerator-cards-2026.md)——NVIDIA、昇腾、AMD 与 Intel 的算力/显存/互联规格对照（选修，《集合通信》拓扑章的前置材料）。

## 阅读提示

- 主线章：1 → 3 → 5 → 7（分析框架、搬运代价、瓶颈标尺、异步执行）；
- 二梯队：2（执行模型细节）、4（Kernel/Tiling）、6（图编译）、8（精度/布局/融合）——写 HCCL 引擎与读 Profiling 时回补；
- **第 7 章是 HCCL 执行模型的直接前置**：Stream/Event/Notify 与通信计算重叠全在这一章。

## 验收清单

- [ ] 为矩阵乘或逐元素算子计算 FLOPs、访存量和算术强度，用 Roofline 判断瓶颈；
- [ ] 画出一次算子从框架下发到设备完成的生命周期；
- [ ] 解释同步、数据布局转换和 Kernel Launch 开销的来源。

---

[返回课程总览 →](../roadmap.md) · [进入《并行策略》 →](../parallel/index.md)
