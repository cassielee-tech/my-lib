# 专题｜HCCL 源码学习

> 专题目标：从"会用 HCCL 跑分布式训练"进入"能读懂 HCCL 源码"。围绕 HCCL 开源仓库建立 **架构 → 概念 → 算法 → 调用链** 的完整认知，最终能独立走读一个通信算子的实现。

## 为什么要做这个专题

前面的路线已经回答了两个问题：大模型为什么需要集合通信（大模型基础第 1 章），以及 Collective 在通用系统里怎样被组织（AI Infra）。但只要还在调用层面，就永远绕不开三个问号：

1. **一次 `HcclAllReduce` 从 API 到硬件，到底经过了哪些模块？**
2. **HCCL 为什么把代码拆成 selector、template、executor 这些层？**
3. **换个芯片、换个拓扑、换个数据量，代码里是哪一段在替我做选择？**

回答这三个问题，必须回到源码。HCCL 已于 2025 年 11 月开源（[gitcode.com/cann/hccl](https://gitcode.com/cann/hccl)），仓库自带一套完整的中文文档（`docs/zh`），本专题就是这份官方文档的**导读与消化笔记**：先建立地图，再按图读码。

## 专题信息

- **参考资料**：[HCCL 开源仓库](https://gitcode.com/cann/hccl) 及其 `docs/zh` 目录（架构简介、用户指南、API 参考、算法介绍）；配套的 [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)；
- **前置知识**：[第 1 章｜认识昇腾与 CANN](./01-ascend-cann.md)、[从 PyTorch 调用到 HCCL](../llm-foundations/01-landscape/03-pytorch-to-hccl.md)；
- **学习节奏**：**7 个学习单元 + 1 篇总结**，每单元约 15 分钟；建议边读边在本地 clone 仓库对照目录与代码。

## 单元列表

| 单元 | 主题 | 你将回答的问题 |
| --- | --- | --- |
| H01-1 | [HCCL 全景与仓库地图](./hccl-source/01-hccl-repo-map.md) | HCCL 是什么，源码放在哪？ |
| H01-2 | [软件架构：HCCL 与 HCOMM 分层](./hccl-source/02-architecture-layering.md) | 为什么要拆成两个库、五层 API？ |
| H01-3 | [通信域、Rank 与 RankGraph](./hccl-source/03-comm-domain-rank-graph.md) | "谁和谁、怎么连"如何被建模？ |
| H01-4 | [通信原语与同步机制](./hccl-source/04-primitives-and-sync.md) | 数据面到底提供了哪几个动词？ |
| H01-5 | [通信引擎与任务执行](./hccl-source/05-comm-engines.md) | 谁在真正搬运数据？ |
| H01-6 | [集合通信算法与代价模型](./hccl-source/06-coll-algorithms.md) | Ring、RHD、NHR 各自何时占优？ |
| H01-7 | [源码走读：AllReduce 调用链](./hccl-source/07-allreduce-call-chain.md) | 一个算子的源码怎么读？ |
| 总结 | [HCCL 源码阅读地图](./hccl-source/summary.md) | 读完后往哪里继续深入？ |

## 本专题的读码方法

读通信库源码最容易迷路，因为入口多、抽象多。本专题始终用三个问题约束注意力：

```text
1. 入口在哪：这次调用的第一行代码在哪个文件？
2. 数据在哪：buffer 从用户内存到链路，经过了哪些拷贝和映射？
3. 谁在等待：同步点在哪个层级，等的是本核线程还是远端 rank？
```

 ::: warning 版本提醒
HCCL 处于快速演进期：开源仓库的目录结构、模块命名（如 HCOMM 拆仓、legacy 目录）与 CANN 商用版本存在差异。本专题基于 2026 年开源 master 分支整理；读码时务必以本地 `git log` 与实际目录为准，不要把博客里的路径当作所有版本的真理。
 :::

---

[开始单元 H01-1 →](./hccl-source/01-hccl-repo-map.md)
