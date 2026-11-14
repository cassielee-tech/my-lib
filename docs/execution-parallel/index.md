# 执行与并行

> 返回：[首页](../index.md)

**专栏目标**：上篇（单卡执行系统）回答"算子怎样在单卡上跑得快"——性能三件套视角：算子怎样铺到硬件、瓶颈怎样判断、任务怎样异步并发；下篇（并行策略）回答"模型为什么要通信"——每一种并行策略都从张量如何切分出发推导通信原语，而不是背结论。

**时间紧？** 每篇各有速通：[⚡ 上篇速通｜单卡执行](00-速通-上篇.md)（8 章）、[⚡ 下篇速通｜并行策略](00-速通-下篇.md)（9 章）。

## 上篇｜单卡执行系统（第 1-8 章 + 专题）

性能分析方法（第 1-5 章）：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [全景与性能分析](01-landscape-performance.md) | 一个请求怎样穿过框架、编译器、Runtime、算子和硬件？ | ✅ |
| 2 | [GPU/NPU 执行模型与计算单元](02-execution-model.md) | CPU、GPU、NPU 为什么采用不同执行方式？昇腾 Cube/Vector 做什么？ | ✅ |
| 3 | [存储层次与数据搬运](03-memory-hierarchy.md) | HBM、Cache、片上 Buffer 有什么区别？数据为什么比计算贵？ | ✅ |
| 4 | [Kernel、Tiling 与流水线](04-kernel-tiling-pipeline.md) | 大算子怎样切成 Tile？搬运和计算怎样形成流水？ | ✅ |
| 5 | [FLOPs、带宽与 Roofline](05-flops-bandwidth-roofline.md) | 怎样判断算子是 Compute-bound 还是 Memory-bound？ | ✅ |

执行系统（第 6-8 章）：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 6 | [Eager、计算图与图编译](06-eager-graph-compilation.md) | 一行 Python 怎样变成可执行计算图？图模式为什么快？ | ✅ |
| 7 | [Stream、Event 与异步执行](07-stream-event-async.md) | 计算和通信怎样并发？什么时候必须同步？ | ✅ |
| 8 | [数据类型、布局与算子融合](08-dtype-layout-fusion.md) | BF16/FP8、Layout 和 Fusion 怎样影响精度与性能？ | ✅ |

**专题**：[2026 主流 AI 加速卡全景](accelerator-cards-2026.md)——NVIDIA、昇腾、AMD 与 Intel 的算力/显存/互联规格对照（选修，上篇拓扑章的前置材料）。

- 主线章：1 → 3 → 5 → 7（分析框架、搬运代价、瓶颈标尺、异步执行）；
- 二梯队：2（执行模型细节）、4（Kernel/Tiling）、6（图编译）、8（精度/布局/融合）——写 HCCL 引擎与读 Profiling 时回补；
- **第 7 章是 HCCL 执行模型的直接前置**：Stream/Event/Notify 与通信计算重叠全在这一章。

**验收**：为矩阵乘或逐元素算子计算 FLOPs、访存量和算术强度，用 Roofline 判断瓶颈；画出一次算子从框架下发到设备完成的生命周期；解释同步、数据布局转换和 Kernel Launch 开销的来源。

## 下篇｜并行策略（第 1-9 章）

模型视角速览（第 1-3 章，已备）：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [MoE 与 Expert Parallel](01-moe-expert-parallel.md) | Router/Top-k 怎样工作？Token 的 AlltoAll 分发为什么失衡？ | ✅ |
| 2 | [效率设计](02-efficient-llm-design.md)【选修】 | MQA/GQA、量化、长上下文各自省什么？ | ✅ |
| 3 | [系统分析](03-llm-systems-analysis.md) | 读配置估参数量/显存/FLOPs，模型切分与性能模型怎样建立？ | ✅ |

系统视角深化（第 4-9 章）：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 4 | [分布式基础：Rank、通信域与拓扑](04-distributed-basics.md) | Process、Device、Rank、World Size、Process Group 分别是什么？ | ✅ |
| 5 | [数据并行与 DDP](05-ddp.md) | 梯度为什么需要 AllReduce？Bucket 和通信计算重叠怎样工作？ | ✅ |
| 6 | [Tensor Parallel](06-tensor-parallel.md) | Column/Row Parallel 怎样切矩阵？为什么产生 AllGather、AllReduce 或 ReduceScatter？ | ✅ |
| 7 | [Pipeline Parallel](07-pipeline-parallel.md) | Stage、Micro-batch、1F1B 和 Bubble 分别是什么？ | ✅ |
| 8 | [Sequence/Context Parallel](08-context-parallel.md) | 长序列怎样切到多卡？Attention 为什么需要交换 K/V？ | ✅ |
| 9 | [ZeRO、FSDP 与混合并行](09-zero-fsdp.md) | 参数、梯度、优化器状态怎样分片？DP/TP/PP/CP/EP 怎样组合？ | ✅ |

- 第 1-3 章是模型侧的并行速览；第 4-9 章是系统侧深化；
- 与《集合通信》的衔接：第 6 章 TP 推导出 AG/RS，第 9 章 FSDP 推导出 RS——它们正是集合通信的"需求方"。

**验收**：为 DP、TP、PP、CP、EP 各画一张张量分布图；计算每种并行的单次通信量，说明通信发生在前向还是反向；为给定模型/集群选一套并行组合并说明理由。

---

[返回首页 →](../index.md) · [进入《通信与昇腾》 →](/comm-ascend/)
