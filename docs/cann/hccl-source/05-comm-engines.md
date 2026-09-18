# 单元 H01-5｜通信引擎与任务执行

> 所属专题：[HCCL 源码学习](../hccl-source.md)

::: info 本单元目标
读完后，你能够说清 **通信引擎的统一模型（Thread + 线程执行调度器 + 通信硬件）**，对比 **AICPU_TS / CPU_TS / AIV / CCU** 四种引擎的取舍，并把引擎模型与源码里的 `template/{aicpu,aiv,ccu}` 目录对应起来。
:::

## 先记住 3 个结论

1. **通信引擎 = Thread（执行上下文）+ 线程执行调度器（调度执行）+ 通信硬件**：Thread 承载一串数据面算子（LocalReduce、ChannelRead/Write、Notify 等），调度器把它们派给硬件。
2. **四种引擎是同一模型在不同硬件上的投影**：AICPU_TS（AICPU 跑通信 Kernel，不占计算核）、CPU_TS（Host CPU 跑通信逻辑）、AIV（占 Vector 核换低延迟）、CCU（专用硬化单元）。
3. **同一通信域默认只用一种引擎**，由算法选择器自动选择——这就是算子源码里 selector 与 template 分离的原因。

## 1. 引擎解决什么问题

H01-4 留下一个问题：数据面动词（Write/Read/Notify）总得有"人"来执行。谁执行、在哪执行、怎么调度，就是通信引擎的问题。统一模型（官方示意图）：

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

| 通信引擎 | Thread 抽象 | 介绍 | 特点 | 适用场景 |
| --- | --- | --- | --- | --- |
| **AICPU_TS** | NPU Stream | AICPU 运行通信 Kernel 并下发通信 Task 描述符，TS 调度到硬件执行 | 不占计算核，Task 描述符下发 | 大数据量通信 |
| **CPU_TS** | NPU Stream | Host CPU 运行通信逻辑，TS 调度下发 | 不占计算核，下发开销大 | Atlas A2 专用 |
| **AIV** | AICore Block | Vector Core 直接执行通信算子 | 低延迟，但占 Vector 核 | 小数据低延迟 |
| **CCU** | Mission | IO Die 上的专用集合通信协处理器，微码执行 | 硬化调度，高带宽低时延，少占计算核与访存带宽 | 专用硬件通信（950PR/950DT） |

读表的方法不是背参数，而是看** trade-off 轴**：

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

## 5. AIV：拿计算核换延迟

![AIV 通信流程：AIV Kernel 被调度到 Vector Core 直接执行数据搬运（图源：HCCL 官方文档）](/images/cann/hccl/official/aiv-communication.png)

```text
1. Host 提交 AIV Kernel 至任务队列
2. TS 调度器将 AIV Kernel 分发至 Vector Core
3. Vector Core 利用不同协议完成数据搬运
```

AIV 的价值在**小数据低延迟**：少了 AICPU 编排与描述符下发的开销，直接在 Vector Core 上执行通信算子。代价是**占用 Vector 计算核**——通信与计算争抢同一资源。这解释了一个性能现象：小消息场景用 AIV 快，但如果通信与计算本想重叠（H01-1 的训练视角），AIV 反而可能互相挤占。

## 6. 引擎模型在源码里的落点

现在回看 H01-1 的仓库地图，引擎不是独立进程，而是**算法模板的维度**：

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

同一个 AllReduce，selector 依据数据量、拓扑、芯片能力选择引擎，再进入对应 template 分支——**"同一通信域默认只用一种引擎"** 的约束就落在 selector 的选择逻辑里。H01-7 走读调用链时会再次遇到这三个目录。

## 7. 自测题

1. 通信引擎的统一模型由哪三部分组成？
2. AICPU_TS 引擎中，AICPU 上的通信 Kernel 实际做什么？
3. AIV 引擎为什么低延迟？代价是什么？
4. CCU 与 AICPU_TS 在"下发什么给硬件"上有何不同？
5. 源码里引擎维度体现在哪个目录结构上？

::: details 自测答案

1. Thread（执行上下文，承载一串数据面算子）+ 线程执行调度器（TS/STARS/OS，调度到硬件）+ 通信硬件（RoCE 网卡、SDMA、UB 等）。
2. 它不直接搬运数据，而是编排——提交通信 Task 描述符至 TS 队列，由 TS 调度器派发给硬件执行器执行。
3. 少了 AICPU 编排与描述符下发的中间跳数，Vector Core 直接执行通信算子；代价是占用 Vector 计算核，通信与计算争抢资源。
4. AICPU_TS 下发通用的 Task 描述符（由 TS 解释执行）；CCU 下发 CCU 可直接识别的预置指令序列（微码执行，配合 URMA 搬数据）。
5. 每个算子的 `template/` 目录按引擎分子目录（aicpu / aiv / ccu）；selector 决定进入哪个分支，同一通信域默认只用一种引擎。

:::

## 本单元小结

- **统一模型**：Thread + 调度器 + 通信硬件，Thread 间靠 ThreadNotify 协同（接 H01-4）；
- **四引擎**：AICPU_TS（大数据、不占核）、CPU_TS（A2 专用）、AIV（小数据低延迟、占核）、CCU（硬化、950 系）；
- **两条 trade-off 轴**：延迟 vs 占用成本；编排深度 vs 带宽；
- **源码落点**：`template/{aicpu,aiv,ccu}` + selector 自动选择；
- **知识连接**：AIV 引擎与第 1 章"AI Core 中 Vector 承担通信算子"的伏笔在此闭环。

## 参考资料

- [HCCL & HCOMM 软件架构简介：通信引擎（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [通信算子执行行为（性能分析）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/perf_analysis/profiling_op_behavior.md)

---

下一单元进入 **H01-6：集合通信算法与代价模型**。

[返回专题导学 →](../hccl-source.md)
