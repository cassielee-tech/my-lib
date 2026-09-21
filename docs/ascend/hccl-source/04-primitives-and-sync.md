# 单元 H01-4｜通信原语与同步机制

> 所属专题：[HCCL 源码学习](../hccl-source.md)

::: info 本单元目标
读完后，你能够说出数据面的 **四个原语概念（Endpoint / Channel / CommMem / CommEngine）**、**两组动词（网络语义与内存语义）** 和 **两种同步（ThreadNotify / ChannelNotify）**，并理解"数据面只剩几个动词"背后的设计逻辑。
:::

## 先记住 3 个结论

1. **基础通信由四个原语概念构成**：通信设备（Endpoint）、通信通道（Channel）、通信内存（CommMem）、通信引擎（CommEngine）；Channel = 两端 Endpoint + 协议 + N 个 Notify。
2. **数据面动词分两组**：网络语义（Write/Read/Notify，走 Channel）与内存语义（像操作本地内存一样远端读写，走映射内存）；选哪组取决于底层协议。
3. **同步只有两种**：实体内 ThreadNotify、跨实体 ChannelNotify——所有看似复杂的集合通信同步，最终都由这两原语组合而成。

## 1. 数据面为什么"瘦"

H01-2 说控制面/数据面分离，控制面"厚"（建域、建图、分配资源），数据面必须"瘦"：数据面在**通信热路径**上，每多一层抽象就多一份开销。所以它的接口设计目标可以概括为一句话：

> 资源由控制面备好，数据面只提供少量动词反复使用。

这个设计直接体现在 L3-prim（`hcomm_primitives.h`）：Write / Read / Reduce + Notify，几乎没有别的。

## 2. 四个原语概念

四个原语概念的组织关系如下图，它是本单元的"总纲"：

![基础通信模型：Endpoint、Channel、CommMem 与 CommEngine（图源：HCCL 官方文档）](/images/cann/hccl/official/base-comm-model.svg)

| 概念 | 一句话解释 | 主要对应硬件 |
| --- | --- | --- |
| **通信设备（Endpoint）** | 网络通信的逻辑接口，包含协议与地址 | NPU 网口 / Host NIC |
| **通信通道（Channel）** | 两端通信设备间的数据通道（含同步 Notify） | RoCE QP / UB Jetty 连接 |
| **通信内存（CommMem）** | 注册到通信域、可被 Endpoint 访问的内存段 | NPU HBM / Host 内存 |
| **通信引擎（CommEngine）** | 执行通信任务的模块，含 Thread 与调度器，驱动通信硬件搬数据 | AICPU_TS、CCU、AIV |

组合关系值得背下来：

```text
Channel = 两端通信设备（Endpoint）+ 通信协议 + N 个 Notify
```

与 H01-3 的递进链对上：Channel 由 Link 实例化，Notify 内嵌在 Channel 上，是同步的载体。

## 3. 两组动词：网络语义 vs 内存语义

| | **网络语义原语** | **内存语义原语** |
| --- | --- | --- |
| 核心对象 | Channel（通信通道） | 通信设备 + 映射内存 |
| 操作方式 | Write / Read / Notify | 本地拷贝（像操作本地内存） |
| 通信模型 | 单边操作、双边操作（需两端配合） | 单边操作（只需一端发起） |
| 适用协议 | RoCE、UB | UB_MEM、HCCS |

怎么理解两者的差别？用"寄快递"与"共享白板"类比：

- **网络语义 = 寄快递**：你把数据打包交给通道（Write），对方收完回执（Notify）。两端都要参与协议配合，但通道承载的协议很通用（跨以太网也行）。
- **内存语义 = 共享白板**：先把对方的内存映射进来，然后直接写那块地址——对你来说是本地拷贝，物理上却发生在远端。只有一端发起，延迟更低，但要求底层协议支持（HCCS、UB_MEM 这类"够得着对方内存"的链路）。

两种语义的官方模型图：

![网络语义模型：以 Channel 为核心的 Write / Read / Notify（图源：HCCL 官方文档）](/images/cann/hccl/official/semantic-communication.png)

![内存语义模型：通信设备 + 映射内存的本地化操作（图源：HCCL 官方文档）](/images/cann/hccl/official/memory-semantic-model.png)

::: tip 与 AI Infra 的知识连接
这就是通用课程里"单边/双边通信"概念在昇腾上的具体化：RoCE 的 Write 类似 RDMA one-sided；需要远端配合的握手流程则是双边模型。选择哪种语义**取决于底层协议与场景需求**，由底层组件对上层屏蔽。
:::

## 4. 同步机制：两种 Notify

通信任务天然是异步的——Write 发出去了，怎么知道对方收完？同步分两种场景，官方示意图如下：

![同步机制：ThreadNotify（实体内）与 ChannelNotify（跨实体）（图源：HCCL 官方文档）](/images/cann/hccl/official/sync-mechanism.svg)

| 同步方式 | 场景 | 说明 | 接口原型 |
| --- | --- | --- | --- |
| **ThreadNotify** | 同一通信实体内 | Thread 向同实体内另一个 Thread 发送/等待同步信号 | `ThreadNotifyRecord` / `ThreadNotifyWait` |
| **ChannelNotify** | 不同通信实体间 | Thread 通过 Channel 上的 Notify 与数据通道，向远端实体的 Thread 发送/等待同步信号 | `ChannelNotifyRecord` / `ChannelNotifyWait` |

用一次跨机 AllReduce 的片段感受两者的分工：

```text
本机 Thread A（搬运）──ThreadNotify──► 本机 Thread B（规约）
本机 Thread B ──ChannelNotify + 数据──► 远端 Thread C
远端 Thread C ◄──等待 ChannelNotify──  确认数据已到
```

- **ThreadNotify 管"本核内部秩序"**：先搬完这块数据，才能开始规约；
- **ChannelNotify 管"跨 rank 秩序"**：数据落盘远端后，对方才能读。

Record/Wait 成对出现，本质上就是一对轻量的"信号量记账"。H01-5 讲引擎时会看到：一个引擎可含多个 Thread，**Thread 间靠 ThreadNotify/ChannelNotify 协调执行顺序**——同步不是外挂能力，而是内嵌在引擎模型里。

## 5. CommMem：被通信的内存

通信内存容易被忽视，但有两个读码要点：

1. **必须注册**：只有注册到通信域、可被 Endpoint 访问的内存段才能被通信操作使用——这就是为什么用户 buffer 进通信算子前，HCCL 需要确认/建立内存映射关系（回顾性能分析文档里"usermem → hcclbuffer → usermem"的数据通路）；
2. **两种位置**：NPU HBM 或 Host 内存。数据放在哪里、由哪个 Endpoint 访问，直接影响可用协议与性能路径。

结合 Channel 与 CommMem，一次网络语义 Write 的完整画面：

```text
本地 CommMem ──读──► Channel（本地 Endpoint）══链路══► 对端 Endpoint ──写──► 远端 CommMem
                              └────────────── ChannelNotify（完成通知）──────┘
```

## 6. 自测题

1. 数据面的四个原语概念是什么，各自对应什么硬件？
2. Channel 的构成公式是什么？
3. 网络语义和内存语义的核心区别是什么？各适用哪些协议？
4. ThreadNotify 与 ChannelNotify 分别解决什么场景的同步？
5. 为什么说"数据面必须瘦"？

::: details 自测答案

1. 通信设备 Endpoint（NPU 网口/Host NIC）、通信通道 Channel（RoCE QP / UB Jetty）、通信内存 CommMem（NPU HBM / Host 内存）、通信引擎 CommEngine（AICPU_TS、CCU、AIV）。
2. Channel = 两端通信设备（Endpoint）+ 通信协议 + N 个 Notify。
3. 网络语义以 Channel 为核心，走 Write/Read/Notify，支持单边与双边操作，适用 RoCE、UB；内存语义以"通信设备 + 映射内存"为核心，像本地拷贝一样单边操作远端内存，适用 UB_MEM、HCCS。区别本质是"隔着通道收发"还是"直接够到对方内存"。
4. ThreadNotify 用于同一通信实体内不同 Thread 之间协调顺序（如先搬运后规约）；ChannelNotify 用于不同通信实体之间，通过 Channel 上的 Notify 与数据通道让远端 Thread 确认数据已到。
5. 数据面处于通信热路径，每层抽象都按次付费；资源与拓扑等复杂度留在控制面，数据面只保留少量动词（Write/Read/Reduce + Notify）反复执行。

:::

## 本单元小结

- **四原语**：Endpoint（设备）、Channel（通道）、CommMem（内存）、CommEngine（引擎，下一单元展开）；
- **两组动词**：网络语义（Write/Read/Notify）与内存语义（映射内存直写），由底层协议决定；
- **两种同步**：ThreadNotify（实体内）与 ChannelNotify（跨实体），Record/Wait 成对；
- **读码落点**：hcomm 仓 `base_comm/primitives/`（原语）与 `base_comm/resource/`（资源）；hccl 仓 executor 里编排的就是这些动词。

## 参考资料

- [HCCL & HCOMM 软件架构简介：基础通信（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [HCCL 性能分析：通信数据通路（usermem/hcclbuffer）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/perf_analysis/perf_data_analysis.md)

---

下一单元进入 **H01-5：通信引擎与任务执行**。

[返回专题导学 →](../hccl-source.md)
