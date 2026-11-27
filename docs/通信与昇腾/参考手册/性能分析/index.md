# profiling：性能分析

> **深度参考** · 前置阅读：[CANN 常用命令速查（Profiling 节）](/通信与昇腾/CANN常用命令.md)。本篇为参考手册体例，检索导向；事实基线见文首引用块。

> 集群性能受 AI 处理器类型、网络、通信算法、通信配置等多方面因素影响。HCCL 性能问题通过 **Profiling 工具**分析：采集**全量 rank** 的 Profiling 数据 → 判断整网瓶颈点 → 按通信算子**下发/执行**不同阶段做进一步分析优化。
>
> 集合通信是通信域内全局协同行为，**只通过一个 rank 的 Profiling 数据往往难以分析**，需采集全量 rank 数据才能精确定位瓶颈。
>
> 事实来源（可信度：代码 > 仓内文档 > 社区资料）：
> `hccl/docs/zh/user_guide/perf_analysis/`（`analysis_flow.md`、`perf_data_collect.md`、`perf_data_analysis.md`、`profiling_op_behavior.md`、`typical_op_behavior.md`、`slow_fast_card_analysis.md`、`switch_congestion_loss.md`）。

---

## 1. 分析流程

```mermaid
flowchart LR
    A["① 采集全量 Profiling 数据<br/>（所有 rank）"] --> B["② 判断整网性能瓶颈点"]
    B --> C["③ 按下发/执行阶段<br/>进一步分析与优化"]
    C --> D["MindStudio Insight<br/>工具分析"]
```

| 步骤 | 说明 | 详见 |
| --- | --- | --- |
| ① 采集 | 采集全量 rank 的 Profiling 数据 | [collect_analyze.md](/通信与昇腾/参考手册/性能分析/采集与分析.md) |
| ② 判断瓶颈 | 根据通信算子下发、执行的不同阶段判断整网瓶颈 | [collect_analyze.md](/通信与昇腾/参考手册/性能分析/采集与分析.md) |
| ③ 分析优化 | 识别 HCCL Profiling 信息、常见案例分析 | 本目录各篇 |

---

## 2. 通信算子的两层：下发 + 执行

通信算子在 Profiling 中分**下发**与**执行**两层——这是理解 HCCL 性能的关键：

| 层 | Profiling 位置 | 含义 |
| --- | --- | --- |
| **下发** | CANN 层，`AscendCL@OpType::ALLGATHER` | host 侧编排下发，一次对应一个算子下发 |
| **执行** | Communication(HCCL) 层 | device 侧异步执行 |

![通信算子下发](/images/cann/hccl/wiki/hierarchical/comm_op_dispatch.png)

> 图源：`hccl/docs/zh/user_guide/perf_analysis/figures/comm_op_dispatch.png`

![通信算子执行](/images/cann/hccl/wiki/hierarchical/comm_op_execution.png)

> 图源：`hccl/docs/zh/user_guide/perf_analysis/figures/comm_op_execution.png`

> **下发与执行互相掩盖**：通信算子 host 侧编排下发、device 侧异步执行，一般下发时间与执行时间互相掩盖以充分利用 device 资源。**当下发成为瓶颈**时，device 侧需等待下发，出现空泡、利用率下降。

Communication(HCCL) 层的结构：

| 元素 | 含义 |
| --- | --- |
| **Group** | 一个通信域 |
| **Plane0-X** | 不同通信流，每个 Plane 对应一个通信流——HCCL 通过**多流并发**充分利用 HCCS 物理链路资源 |
| **hcom_allReduce_xx** | 通信算子执行流程，详细信息含耗时、数据量、数据类型 |

> 通信算子由多个 **notify 同步任务** + **memcpy 内存拷贝任务**编排而成。要在 Profiling 中显示具体通信任务编排信息，需至少采集 **level 1** 级别的 Profiling 数据。

---

## 3. 两类 Profiling 任务

### 3.1 同步任务

![同步任务](/images/cann/hccl/wiki/profiling/syn_task.png)

> 图源：`hccl/docs/zh/user_guide/perf_analysis/figures/syn_task.png`

| 任务 | 含义 |
| --- | --- |
| **Notify Record** | 置 notify 寄存器为 1 |
| **Notify Wait** | 等待 notify 寄存器为 1，然后清 0 |
| **RDMASend**（同步） | 机间 RoCE 同步任务，置对端 notify 寄存器为 1 |

> 同步任务详细信息含：耗时、notify_id、本端（src rank）、对端（dst rank）。

### 3.2 数据通信任务

![数据通信任务](/images/cann/hccl/wiki/profiling/data_comm_task.png)

> 图源：`hccl/docs/zh/user_guide/perf_analysis/figures/data_comm_task.png`

| 任务 | 含义 |
| --- | --- |
| **Memcpy** | 内存拷贝任务（机内/片内） |
| **Reduce_Inline** | 内存拷贝同时完成随路归约计算 |
| **RDMASend**（数据） | 机间 RoCE 通信任务（机间内存拷贝） |

> 数据通信任务详细信息含：耗时、本端/对端 rank、数据量（size）、带宽（bandwidth）。

### 3.3 RDMASend 的区分与耗时口径

| 要点 | 说明 |
| --- | --- |
| **区分同步/数据** | RDMASend 按数据量区分：**同步任务固定 4 字节**，数据通信任务以实际通信量为准 |
| **耗时口径** | RDMASend 数据通信任务的耗时**不等于实际通信耗时**——只是把 WQE 下发到 QP 队列的耗时；实际通信耗时可参考数据量/带宽计算，或参考其后紧跟的下一个 notify wait 任务耗时 |

---

## 4. 典型算子行为（AllReduce 端到端）

AllReduce 的完整 Profiling 任务对应（usermem→hcclbuffer、L0 RS、L1 AR、L0 AG、hcclbuffer→usermem）已在 [../03_L0L1L2分级通信实现原理.md §5](/通信与昇腾/参考手册/总览/L0-L1-L2分级原理.md) 详述，此处不重复。相关图见 ../assets/层次化混合组网.md/（图库）。

---

## 5. 文档索引

| 文档 | 内容 |
| --- | --- |
| [collect_analyze.md](/通信与昇腾/参考手册/性能分析/采集与分析.md) | **数据采集与分析**：两种采集方式（Profiling 工具 / HCCL Test）、`HCCL_TEST_PROFILING`、msprof 解析、分析流程 |
| [slowfast_congestion.md](/通信与昇腾/参考手册/性能分析/慢快与拥塞.md) | **快慢卡分析**（notify wait 被统计算子耗时、前置计算波动、下发瓶颈）+ **交换机拥塞丢包**（4s notify wait、`roce_new_pkt_rty_num`、hccn_tool） |

---

## 6. 参考资料

- `hccl/docs/zh/user_guide/perf_analysis/`：`analysis_flow.md`、`perf_data_collect.md`、`perf_data_analysis.md`、`profiling_op_behavior.md`（本文主要来源）、`typical_op_behavior.md`、`slow_fast_card_analysis.md`、`switch_congestion_loss.md`
- 《性能调优工具用户指南》（msprof）、《MindStudio Insight 工具用户指南》、《性能问题通用定位指南》（TopN 性能问题 > 通信问题优化方案）
- 《HCCL 性能测试工具用户指南》（cann/oam-tools 仓 hccl_test）
- 关联：[README.md](/通信与昇腾/参考手册/性能分析/)、[../03_L0L1L2分级通信实现原理.md §5](/通信与昇腾/参考手册/总览/L0-L1-L2分级原理.md)（AllReduce 端到端 Profiling）、[../debug/](/通信与昇腾/参考手册/排障手册/)（故障诊断）、[../engine/aicpu_ts.md](/通信与昇腾/参考手册/通信引擎/aicpu_ts.md)（下发段静态开销）

