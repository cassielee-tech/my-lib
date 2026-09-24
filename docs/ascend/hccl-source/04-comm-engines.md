# 单元 4｜通信引擎与任务执行

> 所属课程：[HCCL 源码学习](../hccl-source.md) · 第 4 单元（共 10 单元）

::: info 本单元目标
读完后，你能够说清 **通信引擎的统一模型（Thread + 线程执行调度器 + 通信硬件）**，对比 **AICPU_TS / CPU_TS / AIV / CCU** 四种引擎的取舍（含引擎瀑布降级链），并把引擎模型与源码里的 `template/{aicpu,aiv,ccu}` 目录对应起来。
:::

## 先记住 4 个结论

1. **通信引擎 = Thread（执行上下文）+ 线程执行调度器（调度执行）+ 通信硬件**：Thread 承载一串数据面算子（LocalReduce、ChannelRead/Write、Notify 等），调度器把它们派给硬件。引擎差异就是"谁来执行、谁来调度"。
2. **四种引擎是同一模型在不同硬件上的投影**：AICPU_TS（AICPU 控制核跑通信 Kernel，不占计算核）、CPU_TS（Host CPU 跑通信逻辑）、AIV（Vector 核直接执行，占计算核换低延迟）、CCU（IO Die 专用硬化单元）。
3. **selector 的引擎瀑布是从"专用"到"通用"的降级**：CCU_MS → CCU_SCHED → AIV → AICPU——专用硬件越快，能力面越窄，`NOT_MATCH` 就是"这块硬件干不了这件事"。
4. **同一通信域默认只用一种引擎**，由算法选择器自动选择（单元 5）——这就是算子源码里 selector 与 template 分离的原因。

## 1. 引擎解决什么问题

单元 3 留下一个问题：数据面动词（Write/Read/Notify）总得有"人"来执行。谁执行、在哪执行、怎么调度，就是通信引擎的问题。统一模型（官方示意图）：

![通信引擎模型：向上接收资源与任务编排，向下经调度器驱动通信硬件（图源：HCCL 官方文档）](/images/cann/hccl/official/comm-engine-model.svg)

拆开看三部分：

```text
        ┌─────────────────────────────────────┐
        │            通信引擎                  │
        │  Thread 0: [Read → Reduce → Write → Notify …]   │
        │  Thread 1: [Read → Notify …]        │   ← Thread 间用
        │  Thread 2: […]                      │     ThreadNotify 协同
        │───────────────┬─────────────────────│
        │      线程执行调度器（TS / STARS / OS）│
        │───────────────┴─────────────────────│
        │  通信硬件：RoCE 网卡 / SDMA / UB …   │
        └─────────────────────────────────────┘
```

- **Thread**：通信任务的执行上下文，承载一串数据面算子；一个引擎可含多个 Thread 并发执行；
- **线程执行调度器**：把 Thread 上的算子调度到硬件，如 TS（Task Scheduler）、STARS、操作系统；
- **通信硬件**：真正搬移数据的 RoCE 网卡、SDMA、UB 网卡。

关键认知：**"Thread"不是 OS 线程的简单等价物**，而是引擎模型里的执行抽象——在 AICPU_TS 引擎上，Thread 抽象对应 NPU Stream。

## 2. 四种引擎对比

概念总表（官方文档视角）：

| 通信引擎 | Thread 抽象 | 介绍 | 特点 | 适用场景 |
| --- | --- | --- | --- | --- |
| **AICPU_TS** | NPU Stream | AICPU 运行通信 Kernel 并下发通信 Task 描述符，TS 调度到硬件执行 | 不占计算核，Task 描述符下发 | 大数据量通信 |
| **CPU_TS** | NPU Stream | Host CPU 运行通信逻辑，TS 调度下发 | 不占计算核，下发开销大 | Atlas A2 专用 |
| **AIV** | AICore Block | Vector Core 直接执行通信算子 | 低延迟，但占 Vector 核 | 小数据低延迟 |
| **CCU** | Mission | IO Die 上的专用集合通信协处理器，微码执行 | 硬化调度，高带宽低时延，少占计算核与访存带宽 | 专用硬件通信（950PR/950DT） |

再补一张源码视角的对照表（读 selector/template 代码时用，权威出处：`docs/zh/architecture/architecture-brief.md` 2.4 节 + `src/common/comm_engine_utils.h`）：

| | AICPU（_TS） | AIV | CCU |
| --- | --- | --- | --- |
| **全称** | AI CPU | AI Vector Core | Collective Communication Unit |
| **物理位置** | NPU SoC 内的**控制核**（不是 AI 计算核） | AI Core 里的**向量计算核**（和跑 AI 算子的是同一批核） | IO Die 上的**专用集合通信协处理器**（950 系新增） |
| **执行方式** | AICPU 跑通信 Kernel，把搬运翻译成 **Task 描述符**，交 TS 调度到 SDMA 等硬件执行 | 通信算子直接编译成 Vector Kernel，在向量核上**直接执行**搬运与归约 | Host 下发 **CCU 指令流**，CCU 微码执行，经 **URMA**（统一远端内存访问）搬数据 |
| **优点** | 不占计算核；通用性最强；数据量上限最高 | **时延最低**（无需描述符层层转发） | **高带宽 + 低时延 + 不占任何计算核** |
| **代价/约束** | 描述符下发有开销，小数据时延不占优 | **占 Vector 计算核**（与 AI 计算抢核）；数据走 cclBuffer 中转，有缓冲区上限（单元 5 的 `AIV_MAX_PER_RANK_DATA_SIZE`/`cclBufferSize × loop` 两条资源闸） | 片上资源有限：**支持的通信域数量有限**；不支持 inplace、int8、PROD、64 位类型、保序模式（单元 5 CCU 分支五道排除闸的来源）；仅 Ascend 950PR/950DT |
| **算法名前缀** | `Aicpu*`（如 `AicpuAllReduceSoleMeshTwoShot`） | `Aiv*`（如 `AivAllReduceSoleMeshOneShot`） | `CcuMS*` / `CcuSched*`（如 `CcuMSAllReduceSoleMeshOneShot`） |
| **代码位置** | `template/aicpu/` | `template/aiv/` | `template/ccu/` |

读表的方法不是背参数，而是看 **trade-off 轴**：

- **延迟轴**：AIV 直接在计算核上跑，路最短 → 小数据占优；CCU 硬化微码，同样极低延迟；
- **成本轴**：AIV 占用 Vector 核（挤占计算）；AICPU_TS 不占计算核但要经 Task 描述符下发（多一跳）；CCU 什么都不占，但受片上资源限制（支持的通信域数量有限）；
- **带宽轴**：AICPU_TS 与 CCU 服务大数据高带宽场景。

## 3. AICPU_TS：任务描述符下发模式

AICPU_TS 是最典型的引擎，理解它就理解了"通信任务如何被下发"（官方调度流程图）：

![AICPU + TS 调度流程：AICPU Kernel 下发通信 Task 描述符，TS 调度到执行器（图源：HCCL 官方文档）](/images/cann/hccl/official/aicpu-ts-schedule.png)

```text
1. Host 提交 AICPU Kernel 至任务队列
2. TS 调度器将 AICPU Kernel 分发至 AICPU 执行
3. AICPU 提交通信 Task 描述符至 TS 队列
4. TS 调度器将通信 Task 分发至执行器
```

注意第 3 步的深意：AICPU 上的通信 Kernel **不直接搬数据**，而是生成"Task 描述符"交给 TS，由 TS 派给真正的硬件执行器。这样 AICPU 承担的是"编排"角色（生成一串 Task），重活由专用硬件干——**不占计算核**的代价是描述符下发这一跳，换来大数据高带宽场景的吞吐。

## 4. CCU：硬化通信单元

CCU（Collective Communication Unit）位于 IO Die，Thread 抽象为 Mission（官方执行模式图）：

![CCU 通信流程：Host 下发 CCU 指令序列，CCU 微码执行并经 URMA 搬运数据（图源：HCCL 官方文档）](/images/cann/hccl/official/ccu-communication.png)

```text
1. Host 将 CCU 指令序列下发至 CCU 指令空间，同时提交 CCU Kernel 任务至任务队列
2. CCU Kernel 被调度器调度后发送至 CCU 执行
3. CCU 执行指令流，利用 URMA（统一远端内存访问）完成数据搬运
```

与 AICPU_TS 对比：AICPU_TS 下发的是"Task 描述符"（通用描述，TS 解释执行）；CCU 下发的是"预置指令流"（CCU 直接微码执行）。这是典型的**软硬件分工升级**——把高频通信模式硬化，软件只负责拼指令。代价写在官方文档里：受片上资源限制，支持的通信域数量有限（Ascend 950PR/950DT）。

CCU 有两种指令流组织模式：`CCU_MS`（Mesh Step 模式）与 `CCU_SCHED`（Schedule 调度模式），对应 selector 里 `SelectCcuMsAlgo` / `SelectCcuScheduleAlgo` 两个分支（单元 5）。

## 5. AIV：拿计算核换延迟

![AIV 通信流程：AIV Kernel 被调度到 Vector Core 直接执行数据搬运（图源：HCCL 官方文档）](/images/cann/hccl/official/aiv-communication.png)

```text
1. Host 提交 AIV Kernel 至任务队列
2. TS 调度器将 AIV Kernel 分发至 Vector Core
3. Vector Core 利用不同协议完成数据搬运
```

AIV 的价值在**小数据低延迟**：少了 AICPU 编排与描述符下发的开销，直接在 Vector Core 上执行通信算子。代价是**占用 Vector 计算核**——通信与计算争抢同一资源。这解释了一个性能现象：小消息场景用 AIV 快，但如果通信与计算本想重叠（单元 0 的训练视角），AIV 反而可能互相挤占。

之所以还有 `AIV_ONLY` 模式（selector 里禁止回退的显式开关），是给"愿意用算力换时延"的场景（如推理、微基准）留的——**降级静默、显式要求响亮**（单元 5 的 `HCCL_AIV_NOT_MATCH_LOG` 宏）。

## 6. 引擎瀑布：从"专用"到"通用"的降级

单元 5 将精读 selector 的降级链 `CCU_MS → CCU_SCHED → AIV → AICPU`，这里先读出它的物理含义：

```text
专用协处理器（最快、约束最多）
    ↓ 不满足约束（inplace? int8? PROD? 64bit? 通信域超额?）
向量计算核（次快、占计算核、缓冲区受限）
    ↓ 不满足约束（数据 > 8M/卡? 缓冲区循环上限?）
控制核 + TS 调度（最慢、约束最少、什么都能跑）← 兜底
```

**性能与通用性的取舍**：专用硬件越快，能力面越窄。`NOT_MATCH` 不是错误，而是"这块硬件干不了这件事"的正规表达。

"占不占计算核"是核心权衡的另一面：

- AICPU/CCU 不占 AI 计算核 → 训练的计算吞吐不受影响；
- AIV 占 Vector 核 → 通信与计算**抢核**——这是选择引擎时必须算的账。

## 7. 完整引擎清单（不止四个）

`src/common/comm_engine_utils.h:30` 的映射表给出全部引擎枚举：

```c
{COMM_ENGINE_RESERVED, "RESERVED"}, {COMM_ENGINE_CPU, "CPU"},
{COMM_ENGINE_CPU_TS, "CPU_TS"},     {COMM_ENGINE_AICPU, "AICPU"},
{COMM_ENGINE_AICPU_TS, "AICPU_TS"}, {COMM_ENGINE_AIV, "AIV"},
{COMM_ENGINE_CCU, "CCU"}
```

- **CPU / CPU_TS**：Host CPU 跑通信逻辑（Atlas A2 专用，下发开销大）——selector 的 DPU 分支对应 `HOSTCPU` 引擎；
- **AICPU 与 AICPU_TS**：是否经 TS 调度 Task 的两种形态；
- 架构文档的一句话总结：**通信引擎 = Thread（执行上下文）+ 线程调度器（调度执行）**，引擎差异就是"谁来执行、谁来调度"。

## 8. 引擎模型在源码里的落点

回看单元 0 的仓库地图，引擎不是独立进程，而是**算法模板的维度**：

```text
src/ops/all_reduce/
├── selector/    all_reduce_auto_selector.cc   ← 决定用哪个算法、哪个引擎
├── template/
│   ├── aicpu/   ← AICPU_TS 引擎的算法模板
│   ├── aiv/     ← AIV 引擎的算法模板
│   └── ccu/     ← CCU 引擎的算法模板
└── executor/    sequence / parallel / omnipipe / concurrent …
                 ← 选定模板后的执行器实现
```

同一个 AllReduce，selector 依据数据量、拓扑、芯片能力选择引擎，再进入对应 template 分支——**"同一通信域默认只用一种引擎"** 的约束就落在 selector 的选择逻辑里。单元 6 走读调用链、单元 7 精读 executor/template 时会再次遇到这三个目录。

## 9. 自测题

1. 通信引擎的统一模型由哪三部分组成？
2. AICPU_TS 引擎中，AICPU 上的通信 Kernel 实际做什么？
3. AIV 引擎为什么低延迟？代价是什么？
4. CCU 与 AICPU_TS 在"下发什么给硬件"上有何不同？
5. 引擎瀑布的降级方向是什么？为什么 `NOT_MATCH` 不是错误？
6. 源码里引擎维度体现在哪个目录结构上？

::: details 自测答案

1. Thread（执行上下文，承载一串数据面算子）+ 线程执行调度器（TS/STARS/OS，调度到硬件）+ 通信硬件（RoCE 网卡、SDMA、UB 等）。
2. 它不直接搬运数据，而是编排——提交通信 Task 描述符至 TS 队列，由 TS 调度器派发给硬件执行器执行。
3. 少了 AICPU 编排与描述符下发的中间跳数，Vector Core 直接执行通信算子；代价是占用 Vector 计算核，通信与计算争抢资源。
4. AICPU_TS 下发通用的 Task 描述符（由 TS 解释执行）；CCU 下发 CCU 可直接识别的预置指令序列（微码执行，配合 URMA 搬数据）。
5. 从专用引擎（CCU/AIV）降级到通用引擎（AICPU 兜底）。因为责任链上每个选择器只知道自己"能不能干"，`NOT_MATCH` 是把机会让给下一级的正规信号，全部 NOT_MATCH 才是错误。
6. 每个算子的 `template/` 目录按引擎分子目录（aicpu / aiv / ccu）；selector 决定进入哪个分支，同一通信域默认只用一种引擎。

:::

## 本单元小结

- **统一模型**：Thread + 调度器 + 通信硬件，Thread 间靠 ThreadNotify 协同（接单元 3）；
- **四引擎**：AICPU_TS（大数据、不占核、通用兜底）、CPU_TS（A2 专用）、AIV（小数据低延迟、占核）、CCU（硬化微码、950 系）；
- **两条 trade-off 轴**：延迟 vs 占用成本；编排深度 vs 带宽；
- **引擎瀑布**：专用 → 通用的降级链，性能与通用性互相交换；
- **源码落点**：`template/{aicpu,aiv,ccu}` + selector 自动选择（单元 5/7）；
- **知识连接**：AIV 引擎与第 1 章"AI Core 中 Vector 承担通信算子"的伏笔在此闭环。

## 参考资料

- [HCCL & HCOMM 软件架构简介：通信引擎（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [通信算子执行行为（性能分析）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/perf_analysis/profiling_op_behavior.md)

---

下一单元进入 **[5｜集合通信算法与 selector 选择器](05-coll-algorithms.md)**。

[返回课程导学 →](../hccl-source.md)
