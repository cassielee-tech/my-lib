# AI Infra

::: tip 学习导航
不要一次消化 24 课。当前只需要看“学习主线”，然后进入侧边栏中第一篇未完成的课程；每课完成“3 个核心结论＋1 张图＋3 道自测”即可继续。
:::

AI Infra 研究的是：怎样让大模型在真实硬件和集群上**放得下、算得快、扩得开、跑得稳**。

这部分承接“大模型基础”12 课，从单个算子怎样执行开始，逐步进入训练与推理系统、分布式并行、集合通信，最终落到昇腾与 HCCL 源码。

## 学习主线

```text
模型计算
  → 加速器与存储层次
  → Kernel、编译和 Runtime
  → 训练与推理系统
  → 分布式并行
  → 集合通信语义与算法
  → PyTorch / HCCL 调用链
  → HCCL 源码与性能分析
```

整个 AI Infra 阶段规划为 **6 个模块、24 个主题章**。每个主题章继续拆成若干个约 **15 分钟学习单元**；概念、推导、实验和博客输出分开完成。

## 第一阶段：硬件与性能基础

这一阶段先建立单设备视角。分析通信之前，必须知道计算、HBM 搬运和片上存储分别花了多少时间。

| 课次 | 主题 | 学完能够回答的问题 |
| ---: | --- | --- |
| 01 | [AI Infra 全景与性能分析方法](./01-landscape-performance.md) | 一个大模型请求怎样穿过框架、编译器、Runtime、算子和硬件？ |
| 02 | GPU/NPU 执行模型与计算单元 | CPU、GPU、NPU 为什么采用不同执行方式，昇腾 Cube/Vector 单元分别做什么？ |
| 03 | 存储层次与数据搬运 | HBM、Cache、片上 Buffer 和寄存器有什么区别，数据为什么经常比计算更贵？ |
| 04 | Kernel、Tiling 与流水线 | 一个大算子怎样被切成 Tile，搬运和计算怎样形成流水？ |
| 05 | FLOPs、带宽、算术强度与 Roofline | 怎样判断一个算子是 Compute-bound 还是 Memory-bound？ |

阶段产出：能够为矩阵乘法或逐元素算子计算 FLOPs、访存量和算术强度，并用 Roofline 判断理论瓶颈。

第 1 章已经拆成以下 15 分钟单元：

- **I01-1**：[AI Infra 分层与执行对象](./01-landscape-performance/01-stack-execution.md)
- **I01-2**：[性能指标与时间线](./01-landscape-performance/02-metrics-timeline.md)
- **I01-3**：[计算、访存与通信瓶颈](./01-landscape-performance/03-performance-bottlenecks.md)
- **I01-4**：[数量级、算术强度与峰值](./01-landscape-performance/04-arithmetic-intensity.md)
- **I01-5**：[性能分析流程与常见误区](./01-landscape-performance/05-analysis-workflow.md)

## 第二阶段：框架、编译与 Runtime

这一阶段解释“模型代码怎样真正变成设备任务”，并为阅读 PyTorch 到 HCCL 的调用链补齐基础。

| 课次 | 主题 | 学完能够回答的问题 |
| ---: | --- | --- |
| 06 | Eager、计算图与图编译 | Python 中的一行模型代码怎样变成可执行计算图？ |
| 07 | Stream、Event 与异步执行 | Host、Device、计算和通信怎样并发，什么时候必须同步？ |
| 08 | 数据类型、布局与算子融合 | BF16/FP16/FP8、Layout 和 Fusion 怎样影响精度与性能？ |

阶段产出：画出一次算子从框架下发到设备完成的生命周期，并解释同步、数据布局转换和 Kernel Launch 开销。

## 第三阶段：训练与推理系统

这一阶段从单个算子上升到完整任务，分别分析训练和在线推理怎样使用显存与算力。

| 课次 | 主题 | 学完能够回答的问题 |
| ---: | --- | --- |
| 09 | 训练显存与计算优化 | Activation Checkpointing、混合精度、梯度累积和 Offload 分别节省什么？ |
| 10 | 推理引擎与 KV Cache 管理 | Continuous Batching、Paged Attention 和 Prefix Cache 怎样提高并发？ |
| 11 | Prefill/Decode 调度与推理指标 | TTFT、TPOT、吞吐为什么互相制约，Chunked Prefill 和 PD 分离解决什么？ |

阶段产出：为同一模型分别画出训练和推理的资源账本，并分析一次推理服务的延迟与吞吐瓶颈。

## 第四阶段：分布式并行

这一阶段回答“模型为什么要通信”。每一种并行策略都从张量如何切分出发推导通信原语，而不是背结论。

| 课次 | 主题 | 学完能够回答的问题 |
| ---: | --- | --- |
| 12 | 分布式基础：Rank、通信域与拓扑 | Process、Device、Rank、World Size、Process Group 分别是什么？ |
| 13 | 数据并行与 DDP | 梯度为什么需要 AllReduce，Bucket 和通信计算重叠怎样工作？ |
| 14 | Tensor Parallel | Column/Row Parallel 怎样切矩阵，为什么产生 AllGather、AllReduce 或 ReduceScatter？ |
| 15 | Pipeline Parallel | Stage、Micro-batch、1F1B 和 Bubble 分别是什么？ |
| 16 | Sequence/Context Parallel | 长序列怎样切到多卡，Attention 为什么需要交换 K/V？ |
| 17 | ZeRO、FSDP 与混合并行 | 参数、梯度、优化器状态怎样分片，DP/TP/PP/CP/EP 怎样组合？ |

阶段产出：为 DP、TP、PP、CP、EP 和 ZeRO/FSDP 画出张量分布，计算单次通信量，并说明通信发生在前向还是反向。

## 第五阶段：集合通信原理与工程

这一阶段从 Collective 的最终语义进入通信算法、代价模型和工程优化，是学习 HCCL 源码前最重要的准备。

| 课次 | 主题 | 学完能够回答的问题 |
| ---: | --- | --- |
| 18 | Collective 语义与通信代价模型 | Broadcast、Reduce、AllReduce、AllGather、ReduceScatter、Alltoall 的输入输出是什么？ |
| 19 | Ring AllReduce 完整推导 | Ring 为什么等于 ReduceScatter 加 AllGather，每轮发送哪一块数据？ |
| 20 | Tree、Recursive Doubling 与算法选择 | 小消息和大消息为什么选择不同算法，Latency 与 Bandwidth 如何权衡？ |
| 21 | AllGather、ReduceScatter 与 Alltoall | 数据布局如何变化，MoE Alltoall 为什么更容易失衡？ |
| 22 | 拓扑、分层通信与通信计算重叠 | 机内和机间链路怎样组合，Channel、Chunk、Pipeline 分别解决什么？ |

阶段产出：手工模拟 4～8 个 Rank 的 Ring AllReduce，推导通信轮次、单 Rank 数据量和理论耗时，并能按消息大小与拓扑比较算法。

## 第六阶段：昇腾与 HCCL 源码

最后从上层模型语义一路追到 HCCL。源码中的具体类名和目录会以工作环境的 CANN/HCCL 版本为准，避免混用不同版本。

| 课次 | 主题 | 学完能够回答的问题 |
| ---: | --- | --- |
| 23 | 从 PyTorch 到 HCCL 的调用链 | `loss.backward()` 怎样经过 DDP、ProcessGroup 和适配层到达 HCCL API？ |
| 24 | HCCL 架构、AllReduce 源码与性能诊断 | 通信域怎样初始化，拓扑、算法、Executor、Transport 怎样协作完成一次 AllReduce？ |

第 24 课是源码学习入口，而不是终点。完成后再按工作内容继续拆分专题：

- 通信域初始化与 RankTable；
- 拓扑发现、Plane 构建与建链；
- AllReduce / AllGather / ReduceScatter Executor；
- Alltoall 与变长数据交换；
- Stream、Notify、Task 与并发调度；
- 超时、错误传播、Profiling 与性能定位；
- 不同拓扑和消息规模下的算法选择。

## 课程依赖关系

```text
01～05 单卡性能基础
        ↓
06～08 框架和 Runtime
        ↓
09～11 完整训练/推理系统
        ↓
12～17 并行策略：解释为什么通信
        ↓
18～22 集合通信：解释怎样通信
        ↓
23～24 HCCL：解释代码怎样实现通信
```

三条问题贯穿全部课程：

1. **数据在哪里？** 当前张量位于 Host、HBM、片上存储还是其他 Rank？
2. **数据有多少？** Shape、数据类型和切分方式决定多少字节需要计算或搬运。
3. **谁在等待？** 计算、访存和通信之间是否存在严格依赖，能否流水或重叠？

## 建议学习节奏

一个主题章通常拆为 **3～6 个 15 分钟单元**：概念、核心图、公式推导、工程关联、自测各自独立。小实验或 Profiling 另安排 30～60 分钟，不和概念阅读混在一起。

按每周完成 8～12 个单元推进，主线约需 4～6 个月。工作中正在接触的 HCCL 问题可以提前插入，但遇到不熟悉的概念时，应回到对应前置单元补齐。

## 完成后的能力目标

完成 24 课后，应当能够：

- 从 FLOPs、HBM 字节数和算术强度判断算子瓶颈；
- 理解计算图、Stream、Event、Kernel 与 Runtime 的关系；
- 分析训练显存、推理 KV Cache、TTFT、TPOT 和吞吐；
- 从张量切分推导 DP、TP、PP、CP、EP 所需通信；
- 推导 Ring、Tree 等集合通信算法的步骤与成本；
- 追踪 PyTorch 分布式操作到 HCCL 的概念调用链；
- 带着 Shape、拓扑、数据量和关键路径阅读 HCCL 源码；
- 区分计算、访存、通信和负载不均衡瓶颈。

从这里开始：[第 1 课｜AI Infra 全景与性能分析方法](./01-landscape-performance.md)。

## 专题文章

- [2026 主流 AI 加速卡全景：NVIDIA、昇腾、AMD 与 Intel](./accelerator-cards-2026.md)
