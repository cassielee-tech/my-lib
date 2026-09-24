# 课程｜HCOMM 源码学习

> 课程目标：从"读过 HCCL 源码"进入"能基于 HCOMM 开发自定义通信算子"。围绕 HCOMM 通信基础库建立 **仓库地图 → 编程模型 → 控制面/数据面 → 三引擎算子开发 → 实战起步** 的完整认知，最终能独立写出第一个扩展通信算子。

## 为什么要做这个课程

[HCCL 源码学习](hccl-source.md)回答了"内置算子怎么跑"，但岗位技能版图还有另一半：当内置算法在特定拓扑/数据量下不够快，或通算融合需要新的通信语义时，怎么办？答案在 HCOMM——华为把 HCCL 的**底座**（通信域、拓扑、资源、原语）整体开放，提供轻量级通信算子开发接口，实现**通信算子全栈可编程**。本课程回答三个问题：

1. **一次建域，HCOMM 内部到底做了什么**（控制面源码）？
2. **Write/Read 这些"动词"在源码里落在哪**（数据面源码）？
3. **自定义通信算子从零到跑通，完整路径是什么**（三引擎开发流程）？

## 课程信息

- **参考资料**：[HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)（2025/11/30 开源）及其 `docs/zh` 目录（架构文档、通信算子开发指南、API 参考）；配套样例分布在 HCOMM 仓 `examples/` 与 [HCCL 仓](https://gitcode.com/cann/hccl) `examples/`；
- **前置知识**：[HCCL 源码学习](hccl-source.md)（分层架构与调用链）、[第 1 章｜Runtime 与任务执行](01-runtime-task-execution.md)、[第 0 章（HCCL）｜从 PyTorch 走向 HCCL](00-pytorch-to-hccl.md)；
- **学习节奏**：**7 个学习单元 + 1 篇总结**，每单元约 15 分钟；强烈建议本地 `git clone --depth 1 https://gitcode.com/cann/hcomm` 对照目录与头文件阅读。

## 单元列表

| 单元 | 主题 | 你将回答的问题 |
| ---: | --- | --- |
| 0 | [HCOMM 全景与仓库地图](hcomm-source/00-hcomm-repo-map.md) | 仓库怎么切、对外接口分几族？ |
| 1 | [编程模型：通信、并发与拓扑](hcomm-source/01-prog-models.md) | Channel/Thread/拓扑模型怎么抽象？ |
| 2 | [控制面走读：通信域的一生](hcomm-source/02-control-plane-communicator.md) | 建域时源码里发生了什么？ |
| 3 | [数据面走读：原语与资源](hcomm-source/03-data-plane-primitives.md) | 30 个数据面动词怎么用、落在哪？ |
| 4 | [AICPU 算子开发：七步流程](hcomm-source/04-aicpu-op-dev.md) | "下发在前、编排在后"怎么实现？ |
| 5 | [三引擎对比：AICPU / AIV / CCU](hcomm-source/05-engine-comparison.md) | 编排时机与编程模型的差异？ |
| 6 | [实战：examples 与第一个算子](hcomm-source/06-examples-first-op.md) | 从样例到自定义算子的路线图？ |
| 总结 | [HCOMM 源码阅读地图](hcomm-source/summary.md) | 两课程联合地图与岗位能力对照 |

## 本课程的读码方法

HCCL 课程用三个问题读算子层；读 HCOMM 源码的三个问题换成了：

```text
1. 在哪个平面：这段代码属于控制面（建域/建图/配资源）还是数据面（搬运/同步）？
2. 资源归谁：这个 Thread/Channel/Mem 由谁创建、生命周期多长、谁能复用？
3. 同步在等谁：这个 Notify 等的是本实体的另一个 Thread，还是远端 rank？
```

::: warning 版本提醒
HCOMM 与 HCCL 同处快速演进期（开源仓库与 CANN 商用版本目录存在差异，`legacy/` 目录仅做兼容）。本课程基于 2026 年开源 master 分支 + 官方文档整理；读码时以本地 `git log` 与实际目录为准。
:::

---

[开始单元 0 →](hcomm-source/00-hcomm-repo-map.md)
