# 课程｜HCCL 源码学习

> 课程目标：从"会用 HCCL 跑分布式训练"进入"能读懂并扩展 HCCL 源码"。概念地图（官方文档导读）与逐段精读（源码行级走读）**双轨融合**，建立 **架构 → 概念 → 算法 → 调用链 → 支撑系统 → 扩展** 的完整认知，最终能独立走读一个通信算子的实现、写出一个自定义通信算子的骨架。

## 为什么要做这个课程

前面的路线已经回答了两个问题：大模型为什么需要集合通信（《模型全景》第 1 章），以及 Collective 在通用系统里怎样被组织（AI Infra）。但只要还在调用层面，就永远绕不开三个问号：

1. **一次 `HcclAllReduce` 从 API 到硬件，到底经过了哪些模块？**
2. **HCCL 为什么把代码拆成 selector、template、executor 这些层？**
3. **换个芯片、换个拓扑、换个数据量，代码里是哪一段在替我做选择？**

回答这三个问题，必须回到源码。HCCL 已于 2025 年 11 月开源（[gitcode.com/cann/hccl](https://gitcode.com/cann/hccl)），仓库自带一套完整的中文文档（`docs/zh`）。本课程把两条学习线融合为一个序列：

- **概念地图**：基于官方文档，回答"是什么、为什么"——建立分层、拓扑、原语、引擎、算法的世界观；
- **逐段精读**：基于源码快照，回答"怎么写的"——`文件:行号` 级走读，结论卡片 + 逐段拆解 + 思考题。

每个单元都是"先结论 → 概念地图 → 源码精读 → 小结 + 自检"的完整闭环。

## 课程信息

- **参考资料**：[HCCL 开源仓库](https://gitcode.com/cann/hccl) 及其 `docs/zh` 目录（架构简介、用户指南、API 参考、算法介绍）；配套的 [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)；
- **前置知识**：[第 0 章｜认识昇腾与 CANN](00-ascend-cann.md)、[第 1 章｜Runtime 与任务执行](01-runtime-task-execution.md)、[第 0 章（HCCL）｜从 PyTorch 走向 HCCL](00-pytorch-to-hccl.md)；
- **学习节奏**：**10 个学习单元 + 1 篇总结**，每单元约 20～30 分钟；强烈建议本地 `git clone --depth 1 https://gitcode.com/cann/hccl` 对照目录与代码阅读；
- **行号约定**：精读段的行号引用格式 `文件:行号`，均以课程整理时的仓快照为准。

## 单元列表

| 单元 | 主题 | 概念地图 | 源码精读对象 |
| ---: | --- | --- | --- |
| 0 | [HCCL 全景与仓库地图](hccl-source/00-hccl-repo-map.md) | ✅ | 仓库目录、调用链总览 |
| 1 | [软件架构：分层与对外 API](hccl-source/01-architecture-layering.md) | ✅ | `include/hccl.h`（262 行） |
| 2 | [通信域、Rank 与 RankGraph](hccl-source/02-comm-domain-rank-graph.md) | ✅ | —（hcomm 仓，由 HCOMM 课程覆盖） |
| 3 | [通信原语与同步机制](hccl-source/03-primitives-and-sync.md) | ✅ | —（hcomm 仓，由 HCOMM 课程覆盖） |
| 4 | [通信引擎与任务执行](hccl-source/04-comm-engines.md) | ✅ | AICPU/AIV/CCU 三引擎对照 |
| 5 | [集合通信算法与 selector 选择器](hccl-source/05-coll-algorithms.md) | ✅ | `op_common/selector/` + `all_reduce/selector/` |
| 6 | [AllReduce 调用链走读](hccl-source/06-allreduce-call-chain.md) | ✅ | `all_reduce_op.cc`（281 行） |
| 7 | [executor 与 template：执行机制](hccl-source/07-executor-template.md) | — | executor 注册表 + 13 个执行器 + Mesh RS 模板 |
| 8 | [资源地基与 dlsym 解耦](hccl-source/08-resources-dlsym.md) | — | `op_common` 资源链 + `topo_host.cc` + `hcomm_dlsym/` |
| 9 | [MC2 自定义算子框架](hccl-source/09-mc2-custom-ops.md) | — | `hccl_mc2.h` + `examples/04、05` |
| 总结 | [HCCL 源码阅读地图](hccl-source/summary.md) | 十单元知识地图 + 继续深入五条路线 | |

**阅读主线**：`API（1）→ 算子入口（6）→ selector 选名（5）→ executor 组装（7）→ template 落原语（7）→ 资源地基（8）→ dlsym 解耦（8）→ MC2 扩展（9）`；单元 0/2/3/4 提供全景与概念地基，可按需穿插。

## 本课程的读码方法

读通信库源码最容易迷路，因为入口多、抽象多。本课程始终用三个问题约束注意力：

```text
1. 入口在哪：这次调用的第一行代码在哪个文件？
2. 数据在哪：buffer 从用户内存到链路，经过了哪些拷贝和映射？
3. 谁在等待：同步点在哪个层级，等的是本核线程还是远端 rank？
```

::: warning 版本提醒
HCCL 处于快速演进期：开源仓库的目录结构、模块命名（如 HCOMM 拆仓、legacy 目录）与 CANN 商用版本存在差异。本课程基于 2026 年开源 master 分支整理；读码时务必以本地 `git log` 与实际目录为准，不要把博客里的路径当作所有版本的真理。
:::

---

[开始单元 0 →](hccl-source/00-hccl-repo-map.md)
