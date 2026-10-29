# 课程｜HCCL 与 HCOMM 源码学习

> 课程目标：从"会用 HCCL 跑分布式训练"进入"能读懂内置算子、并能基于 HCOMM 写出扩展通信算子"。围绕 [hccl](https://gitcode.com/cann/hccl) 与 [hcomm](https://gitcode.com/cann/hcomm) 两个开源仓，概念地图（官方文档导读）与逐段精读（源码行级走读）**双轨融合**，建立 **全景 → 架构 → 底座 → 算法与调用链 → 支撑系统 → 扩展** 的完整认知，最终能独立走读一个通信算子的实现、写出第一个自定义通信算子。

## 为什么要做这个课程

前面的路线已经回答了两个问题：大模型为什么需要集合通信（《模型全景》第 1 章），以及 Collective 在通用系统里怎样被组织（《集合通信》专栏）。但只要还在调用层面，就永远绕不开四个问号：

1. **一次 `HcclAllReduce` 从 API 到硬件，到底经过了哪些模块？**
2. **HCCL 为什么把代码拆成 selector、template、executor 这些层，又为什么把底座拆进另一个仓？**
3. **换个芯片、换个拓扑、换个数据量，代码里是哪一段在替我做选择？**
4. **当内置算法在特定拓扑/数据量下不够快，或通算融合需要新的通信语义时，怎么办？**

前三问的答案在 **HCCL 仓**（内置算子怎么跑），第四问的答案在 **HCOMM 仓**（底座开放，自己写）。两仓均于 2025 年 11 月开源（hccl 在前、hcomm 在后），仓库自带完整的中文文档（`docs/zh`）。本课程把两个仓当作一个整体，两条学习线融合为一个序列：

- **概念地图**：基于官方文档，回答"是什么、为什么"——建立分层、拓扑、原语、引擎、算法的世界观；
- **逐段精读**：基于源码快照，回答"怎么写的"——`文件:行号` 级走读，结论卡片 + 逐段拆解 + 思考题。

每个单元都是"先结论 → 概念地图 → 源码精读 → 小结 + 自检"的完整闭环。

## 课程信息

- **参考资料**：[HCCL 开源仓库](https://gitcode.com/cann/hccl) 与 [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm) 及各自的 `docs/zh` 目录（架构简介、用户指南、通信算子开发指南、API 参考）；
- **前置知识**：[第 0 章｜认识昇腾与 CANN](00-ascend-cann.md)、[第 1 章｜Runtime 与任务执行](01-runtime-task-execution.md)、[第 2 章｜从 PyTorch 走向 HCCL](02-pytorch-to-hccl.md)；
- **学习节奏**：**12 个学习单元 + 1 篇总结**，分四幕推进，每单元约 20～30 分钟；强烈建议本地 `git clone --depth 1` 两个仓库对照目录与代码阅读；
- **行号约定**：精读段的行号引用格式 `文件:行号`，均以课程整理时的仓快照为准。

## 单元列表

### 第一幕·全景与架构

| 单元 | 主题 | 精读对象 |
| ---: | --- | --- |
| 0 | [双仓全景与一次 AllReduce 的旅程](hccl-hcomm/00-repo-map.md) | 两仓目录地图、调用链总览 |
| 1 | [软件架构：分层、五层 API 与两仓边界](hccl-hcomm/01-architecture-layering.md) | `include/hccl.h`（262 行） |

### 第二幕·底座：HCOMM 的名词与动词

概念与源码走读配对，一个单元内"概念 → 源码"零跳转：

| 单元 | 主题 | 精读对象 |
| ---: | --- | --- |
| 2 | [控制面：通信域的一生](hccl-hcomm/02-control-plane.md) | RankGraph 七概念 + 建域流水线（hcomm 仓） |
| 3 | [数据面：原语与资源](hccl-hcomm/03-data-plane.md) | 四原语概念 + `hcomm_primitives.h` 30 个动词 |
| 4 | [通信引擎：模型、瀑布与三引擎编排](hccl-hcomm/04-comm-engines.md) | 统一模型 + 编排时机 + CCU 深潜 |

### 第三幕·算子层：一次 AllReduce 的下半生

| 单元 | 主题 | 精读对象 |
| ---: | --- | --- |
| 5 | [集合通信算法与 selector 选择器](hccl-hcomm/05-coll-algorithms.md) | `op_common/selector/` + `all_reduce/selector/`（728 行） |
| 6 | [AllReduce 调用链走读](hccl-hcomm/06-allreduce-call-chain.md) | `all_reduce_op.cc`（281 行） |
| 7 | [executor 与 template：执行机制](hccl-hcomm/07-executor-template.md) | executor 注册表 + 13 个执行器 + Mesh RS 模板 |
| 8 | [资源地基与 dlsym 解耦](hccl-hcomm/08-resources-dlsym.md) | `op_common` 资源链 + `topo_host.cc` + `hcomm_dlsym/` |

### 第四幕·扩展：写出你自己的通信算子

| 单元 | 主题 | 精读对象 |
| ---: | --- | --- |
| 9 | [自定义算子开发：AI CPU 七步流程](hccl-hcomm/09-custom-op-dev.md) | 官方七步时序 + EngineCtx/TaskCache |
| 10 | [MC2 自定义算子框架与官方样例](hccl-hcomm/10-mc2-custom-ops.md) | `hccl_mc2.h` + `examples/04、05`（五步法 ↔ 七步对照） |
| 11 | [实战：examples 与第一个自定义算子](hccl-hcomm/11-examples-first-op.md) | 建域三式 + ACL Graph + 五步起步路线图 |
| 总结 | [HCCL 与 HCOMM 源码阅读地图](hccl-hcomm/summary.md) | 概念清单 + 岗位能力对照 + 六条深入路线 |

**阅读主线**：`双仓地图（0）→ 六族 API（1）→ 底座三单元（2/3/4）→ 算法选名（5）→ 算子入口（6）→ executor 组装（7）→ 资源与 dlsym（8）→ 自定义算子（9 → 10 → 11）`；赶时间可走最小闭环 `0 → 1 → 6 → 5 → 7`，再按需补底座与扩展。

## 本课程的读码方法

读通信库源码最容易迷路，因为入口多、抽象多。本课程始终用两组问题约束注意力——

读**底座**（单元 2～4）的三个问题：

```text
1. 在哪个平面：这段代码属于控制面（建域/建图/配资源）还是数据面（搬运/同步）？
2. 资源归谁：这个 Thread/Channel/Mem 由谁创建、生命周期多长、谁能复用？
3. 同步在等谁：这个 Notify 等的是本实体的另一个 Thread，还是远端 rank？
```

读**算子层**（单元 5～8）的三个问题：

```text
1. 入口在哪：这次调用的第一行代码在哪个文件？
2. 数据在哪：buffer 从用户内存到链路，经过了哪些拷贝和映射？
3. 谁在等待：同步点在哪个层级，等的是本核线程还是远端 rank？
```

::: warning 版本提醒
HCCL 与 HCOMM 同处快速演进期：开源仓库的目录结构、模块命名与 CANN 商用版本存在差异，`legacy/` 目录仅做兼容。本课程基于 2026 年开源 master 分支 + 官方文档整理；读码时务必以本地 `git log` 与实际目录为准，不要把博客里的路径当作所有版本的真理。
:::

---

[开始单元 0 →](hccl-hcomm/00-repo-map.md)
