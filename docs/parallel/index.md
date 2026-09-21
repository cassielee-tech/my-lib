# 并行策略

> 返回：[课程总览](../roadmap.md)

**专栏目标**：回答"模型为什么要通信"。每一种并行策略都从张量如何切分出发推导通信原语，而不是背结论——这是理解集合通信需求的系统视角。

## 章节列表

模型视角速览（第 1-3 章，已备）：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [MoE 与 Expert Parallel](./01-moe-expert-parallel.md) | Router/Top-k 怎样工作？Token 的 AlltoAll 分发为什么失衡？ | ✅ |
| 2 | [效率设计](./02-efficient-llm-design.md)【选修】 | MQA/GQA、量化、长上下文各自省什么？ | ✅ |
| 3 | [系统分析](./03-llm-systems-analysis.md) | 读配置估参数量/显存/FLOPs，模型切分与性能模型怎样建立？ | ✅ |

系统视角深化（第 4-9 章）：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 4 | [分布式基础：Rank、通信域与拓扑](./04-distributed-basics.md) | Process、Device、Rank、World Size、Process Group 分别是什么？ | ✅ |
| 5 | [数据并行与 DDP](./05-ddp.md) | 梯度为什么需要 AllReduce？Bucket 和通信计算重叠怎样工作？ | ✅ |
| 6 | [Tensor Parallel](./06-tensor-parallel.md) | Column/Row Parallel 怎样切矩阵？为什么产生 AllGather、AllReduce 或 ReduceScatter？ | ✅ |
| 7 | [Pipeline Parallel](./07-pipeline-parallel.md) | Stage、Micro-batch、1F1B 和 Bubble 分别是什么？ | ✅ |
| 8 | [Sequence/Context Parallel](./08-context-parallel.md) | 长序列怎样切到多卡？Attention 为什么需要交换 K/V？ | ✅ |
| 9 | [ZeRO、FSDP 与混合并行](./09-zero-fsdp.md) | 参数、梯度、优化器状态怎样分片？DP/TP/PP/CP/EP 怎样组合？ | ✅ |

## 阅读提示

- 第 1-3 章是模型侧的并行速览；第 4-9 章是系统侧深化（第 4-9 章已全部备齐——本专栏九讲收官）；
- 本专栏与《集合通信》的衔接：第 6 章 TP 推导出 AG/RS，第 9 章 FSDP 推导出 RS——它们正是《集合通信》的"需求方"；
- 《集合通信》已备，不必等本专栏写完：可先读其第 1 章（语义与代价模型）再回来补并行细节。

## 验收清单

- [ ] 为 DP、TP、PP、CP、EP 各画一张张量分布图；
- [ ] 计算每种并行的单次通信量，说明通信发生在前向还是反向；
- [ ] 为给定模型/集群选一套并行组合并说明理由。

---

[返回课程总览 →](../roadmap.md) · [进入《集合通信》 →](../collective/index.md)
