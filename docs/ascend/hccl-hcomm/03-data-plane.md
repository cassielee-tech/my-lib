# 单元 3｜数据面：原语与资源

> 所属课程：[HCCL 与 HCOMM 源码学习](../hccl-hcomm.md) · 第 3 单元（共 12 单元）
> 精读对象：hcomm 仓 `base_comm/primitives/` 与 `base_comm/resources/`——`hcomm_primitives.h` 的 30 个数据面动词

::: info 本单元目标
读完后，你能够说出数据面的**四个原语名词（Endpoint / Channel / CommMem / CommEngine）**、**两组动词（网络语义与内存语义）**、**Thread 并发模型**和**两种同步（ThreadNotify / ChannelNotify）**；再进入源码，把 **30 个数据面动词**归入四个家族、解释命名规律（OnThread/Nbi/WithNotify），并用真实签名写出 Send/Receive 的同步时序。
:::

## 先记住 6 个结论

1. **基础通信由四个原语名词构成**：通信设备（Endpoint）、通信通道（Channel）、通信内存（CommMem）、通信引擎（CommEngine）；Channel = 两端 Endpoint + 协议 + N 个 Notify。
2. **数据面动词分两组**：网络语义（Write/Read/Notify，走 Channel）与内存语义（像操作本地内存一样远端读写，走映射内存）；选哪组取决于底层协议。
3. **并发模型 = Thread 抽象**：通信任务绑定 Thread，Thread 间用 Notify 同步；不同引擎下 Thread 落地不同——AI CPU+TS / Host CPU+TS 是 Stream + NPU Notify 寄存器，AIV 是 Vector Core + Device 内存。
4. **同步只有两种**：实体内 ThreadNotify、跨实体 ChannelNotify——所有看似复杂的集合通信同步，最终都由这两原语组合而成。
5. **数据面 = 四个家族的动词**：本地操作（LocalCopy/LocalReduce）、Thread 同步（ThreadNotify Record/Wait）、Channel 通信（Read/Write/ReadReduce/WriteReduce 及变体）、批量与缓存（BatchMode/TaskCache）；命名即语义。
6. **HCCL Buffer 是搬运的中转站**：每个通信域管理的 Device 锁页内存（默认 200 MB）——异步通信要求输入数据在执行时地址固定，所以先拷入 Buffer 再搬运。

## 第一部分｜概念地图：名词、动词与同步

### 1. 数据面为什么"瘦"

单元 1 说控制面/数据面分离，控制面"厚"（建域、建图、分配资源），数据面必须"瘦"：数据面在**通信热路径**上，每多一层抽象就多一份开销。所以它的接口设计目标可以概括为一句话：

> 资源由控制面备好，数据面只提供少量动词反复使用。

这个设计直接体现在 L3-prim（`hcomm_primitives.h`）：Write / Read / Reduce + Notify，几乎没有别的——第二部分逐个签名走读。

### 2. 四个原语名词

四个原语名词的组织关系如下图，它是本单元的"总纲"：

![基础通信模型：Endpoint、Channel、CommMem 与 CommEngine（图源：HCCL 官方文档）](/images/cann/hccl/official/base-comm-model.svg)

| 概念 | 一句话解释 | 关键细节 | 主要对应硬件 |
| --- | --- | --- | --- |
| **通信设备（Endpoint）** | 网络通信的逻辑接口/端口，包含协议与地址 | 一个对象可有多个 Endpoint；一个 Endpoint 映射一个物理端口，物理端口可为硬件控制的 Bonding 口（软件不感知） | NPU 网口 / Host NIC |
| **通信通道（Channel）** | 两端通信设备间的数据通道（含同步 Notify） | 一对 Endpoint 可建多个 Channel；创建需**两端同步调用**并交换内存信息；可查远端内存（地址与大小） | RoCE QP / UB Jetty 连接 |
| **通信内存（CommMem）** | 注册到通信域、可被域内成员访问的内存段 | 建 Channel 时两端交换内存信息；必须注册才能被通信操作使用 | NPU HBM / Host 内存 |
| **通信引擎（CommEngine）** | 执行通信任务的模块，含 Thread 与调度器，驱动通信硬件搬数据 | 引擎差异 = "谁来执行、谁来调度"（单元 4 展开） | AICPU_TS、CCU、AIV |

组合关系值得背下来：

```text
Channel = 两端通信设备（Endpoint）+ 通信协议 + N 个 Notify
```

与单元 2 的递进链对上：Channel 由 Link 实例化，Notify 内嵌在 Channel 上，是同步的载体。

### 3. 两组动词：网络语义 vs 内存语义

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

展开网络语义：以 RoCE 协议为例，Channel 打开后内部有三样东西——

- **本端 Channel ↔ 远端 Channel 一一对应**——通信的入口；
- **QP（Queue Pair）**：Channel 关联一个或多个 QP，Channel 建立时 QP 建链；
- **Notify 实例**：Channel 包含多个 Notify（创建时指定数量），本端可按**序号**向远端 Channel 的某个 Notify 发同步信号，也可以在本地某个 Notify 上等远端的信号。

内存语义有一个容易忽略的细节：**Channel 的建立只表示"内存映射启用"**，数据面不一定走这个 Endpoint（用哪个网络口由映射机制决定）；开发者拿到的是"远端内存映射到本地后的地址"，用本地拷贝接口就能跨节点搬数据。

::: tip 与 AI Infra 的知识连接
这就是通用课程里"单边/双边通信"概念在昇腾上的具体化：RoCE 的 Write 类似 RDMA one-sided；需要远端配合的握手流程则是双边模型。回扣 [《集合通信》01 章](../../collective/01-collective-semantics-cost.md#_01-2-六个核心原语的语义卡片)的网络/内存语义之分——HCOMM 把它们做成了同一套 Channel 的两种用法。选择哪种语义**取决于底层协议与场景需求**，由底层组件对上层屏蔽。
:::

### 4. Thread 并发模型与两种同步

通信算子由多个通信任务组成，无资源冲突的任务应并发执行。HCOMM 的并发抽象：

```text
并发单元：Thread —— 通信任务与 Thread 绑定，不同 Thread 的操作并发执行
同步手段：Thread 可含多个 Notify 实例 —— Thread 间收发同步信号
```

**Thread 的引擎落地**（本单元最重要的表之一）：

| 引擎 | 并发实体 | 同步实现 | Thread 是什么 |
| --- | --- | --- | --- |
| AI CPU+TS / Host CPU+TS | Stream | NPU Notify 寄存器 | Stream + Notify 的封装抽象 |
| AIV | Vector Core | Device 内存 | AIV 核 + Device 内存的封装抽象 |

**"Thread"不是 OS 线程的简单等价物**，而是引擎模型里的执行抽象——引擎与 Thread 的完整模型在单元 4 展开。

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

Record/Wait 成对出现，本质上就是一对轻量的"信号量记账"。单元 4 讲引擎时会看到：一个引擎可含多个 Thread，**Thread 间靠 ThreadNotify/ChannelNotify 协调执行顺序**——同步不是外挂能力，而是内嵌在引擎模型里。单元 7 精读模板代码时，会看到 `PreSyncInterThreads` / `PostSyncInterThreads` 正是这对原语的封装。

### 5. CommMem：被通信的内存

通信内存容易被忽视，但有两个读码要点：

1. **必须注册**：只有注册到通信域、可被 Endpoint 访问的内存段才能被通信操作使用——这就是为什么用户 buffer 进通信算子前，HCCL 需要确认/建立内存映射关系（回顾性能分析文档里"usermem → hcclbuffer → usermem"的数据通路）；
2. **两种位置**：NPU HBM 或 Host 内存。数据放在哪里、由哪个 Endpoint 访问，直接影响可用协议与性能路径。

结合 Channel 与 CommMem，一次网络语义 Write 的完整画面：

```text
本地 CommMem ──读──► Channel（本地 Endpoint）══链路══► 对端 Endpoint ──写──► 远端 CommMem
                               └────────────── ChannelNotify（完成通知）──────┘
```

## 第二部分｜源码走读：30 个动词

名词与同步的概念齐了。现在打开 `include/hcomm_primitives.h`，看数据面的全部"动词"长什么样、怎么用。

### 6. 原语四家族（真实签名）

| 家族 | 函数（节选，真实声明） | 用途 |
| --- | --- | --- |
| **本地操作** | `HcommLocalCopyOnThread(thread, dst, src, len)`、`HcommLocalReduceOnThread(...)` | 本地内存拷贝/归约（内存语义的跨节点访问也用它） |
| **Thread 同步** | `HcommThreadNotifyRecordOnThread(thread, dstThread, dstNotifyIdx)`、`HcommThreadNotifyWaitOnThread(...)` | 同一实体内 Thread 间同步（主从 Thread 握手） |
| **Channel 通信** | `HcommWriteOnThread(thread, channel, dst, src, len)`、`HcommReadOnThread(...)`、`HcommReadReduceOnThread` / `HcommWriteReduceOnThread`（边搬边算）、`HcommChannelNotifyRecord/WaitOnThread(...)`（跨实体同步） | 经 Channel 读写远端内存、与远端 Notify 握手 |
| **批量与缓存** | `HcommBatchModeStart/End(tag)`、`HcommAicpuTsTaskCacheStart/Execute/Lookup/Clear(tag)`、`HcommAcquireComm/ReleaseComm` | 批量提交模式、AICPU 任务缓存、通信域句柄复用 |

三个命名后缀的读法：

- **OnThread**：所有热路径操作都挂在某个 Thread 上（并发模型的落地）；
- **Nbi**（如 `HcommWriteNbi` / `HcommReadNbi`）：非阻塞立即返回，完成靠后续 Notify/Fence 确认；
- **WithNotify**（如 `HcommWriteWithNotifyOnThread`）：写远端的同时给对方发通知——**搬运+握手一次完成**，是编排里的高频组合。

### 7. 用原语写同步：Send/Receive 时序

官方 Send/Receive 样例的三行核心（AI CPU 侧任务编排的原子级视图）：

```c
// Send 端：
HcommLocalCopyOnThread(thread, localAddr, inputPtr, size);      // ① 数据拷入通信内存
HcommChannelNotifyRecordOnThread(thread, channel, 0);           // ② 通知对端：数据就绪
HcommChannelNotifyWaitOnThread(thread, channel, 1, 1800);       // ③ 等对端确认：已读完

// Receive 端：
HcommChannelNotifyWaitOnThread(thread, channel, 0, 1800);       // ① 等发送端就绪
HcommReadOnThread(thread, channel, outputPtr, remoteAddr, size);// ② 读远端数据
HcommChannelNotifyRecordOnThread(thread, channel, 1);           // ③ 通知发送端：读完了
```

三步握手即"生产者-消费者"：**Channel 的 Notify 序号（0/1）就是两条方向相反的信号线**。把它与 [《集合通信》02 章](../../collective/02-ring-allreduce.md#_02-2-第一阶段-reducescatter-轮转)手推的 Ring 轮转对照——**集合通信算法的每一"轮"，展开后就是若干组这样的三步握手**。

### 8. HCCL Buffer：为什么需要中转

通信任务是异步下发的——下发时用户输入指针还在，**真正执行时呢**？为避免悬空指针，数据先拷入通信域管理的**锁页内存**（HCCL Buffer，默认 200 MB）：

```text
用户输入 → [LocalCopy] → HCCL Buffer → [Read/Write/Reduce...] → 对端 HCCL Buffer → [LocalCopy] → 用户输出
```

超过 200 MB 的输入按块切分循环处理——这与 [《集合通信》05-3 的 Chunk](../../collective/05-topology-hierarchical-overlap.md#_05-3-chunk-与-channel-拆小、铺满) 是同一思想在框架侧的再现。

### 9. base_comm 源码落点

| 模块 | 内容 |
| --- | --- |
| `primitives/api_c_adpt/` | C 接口适配（`*_c_adpt.cc`），30 个动词的实现入口 |
| `primitives/aicpu/` | AICPU 侧原语与**任务缓存**（TaskCache：同 tag 任务描述复用，Host 免重复编排） |
| `resources/endpoints/`、`endpoint_pairs/` | Endpoint 与端对管理（控制面名词） |
| `resources/reged_mems/` | 注册内存（通信内存的家） |
| `resources/hccp/` | **HCCP 自研协议**：HCOMM 自己的集合通信协议栈 |
| `resources/ccu/`、`southbound_adpt/` | CCU 资源与南向（硬件/驱动）适配 |

hccl 仓侧对这些原语的调用（wrapper 层：`SendRecvWrite` / `LocalReduce` 等语义化封装）在单元 7 精读——模板编排的每一个"动词"，最终都落到本单元的这批函数上。

## 10. 自测题

1. 数据面的四个原语名词是什么，各自对应什么硬件？Channel 的构成公式是什么？
2. 网络语义和内存语义的核心区别是什么？各适用哪些协议？
3. RoCE 场景下 Channel 内部有哪三样东西？
4. 并发模型的并发单元和同步手段分别是什么？AIV 引擎下 Thread 对应什么物理实体？
5. ThreadNotify 与 ChannelNotify 分别解决什么场景的同步？
6. 数据面原语的四个家族是什么？`OnThread`、`Nbi`、`WithNotify` 三个后缀分别什么含义？
7. Send/Receive 三步握手中 Notify 序号 0 和 1 分别承担什么方向？
8. 为什么需要 HCCL Buffer？默认多大？
9. HCCP 住在源码哪个目录？TaskCache 解决什么问题？

::: details 自测答案

1. 通信设备 Endpoint（NPU 网口/Host NIC）、通信通道 Channel（RoCE QP / UB Jetty）、通信内存 CommMem（NPU HBM / Host 内存）、通信引擎 CommEngine（AICPU_TS、CCU、AIV）。Channel = 两端通信设备（Endpoint）+ 通信协议 + N 个 Notify。
2. 网络语义以 Channel 为核心，走 Write/Read/Notify，支持单边与双边操作，适用 RoCE、UB；内存语义以"通信设备 + 映射内存"为核心，像本地拷贝一样单边操作远端内存，适用 UB_MEM、HCCS。区别本质是"隔着通道收发"还是"直接够到对方内存"。
3. 一一对应的远端 Channel 关联、一个或多个 QP（建链）、多个 Notify 实例（按序号收发同步信号）。
4. 并发单元是 Thread（任务与 Thread 绑定）；同步手段是 Thread 内的多个 Notify 实例。AIV 下 Thread = AIV 核 + Device 内存的封装，同步用 Device 内存实现。
5. ThreadNotify 用于同一通信实体内不同 Thread 之间协调顺序（如先搬运后规约）；ChannelNotify 用于不同通信实体之间，通过 Channel 上的 Notify 与数据通道让远端 Thread 确认数据已到。
6. 本地操作（HcommLocalCopyOnThread）、Thread 同步（HcommThreadNotifyRecordOnThread）、Channel 通信（HcommReadOnThread）、批量与缓存（HcommBatchModeStart）。OnThread=操作绑定到某 Thread 执行；Nbi=非阻塞立即返回；WithNotify=搬运与通知原子组合。
7. 0 = 发送端→接收端的"数据就绪"线；1 = 接收端→发送端的"已读完"确认线。
8. 异步执行时用户指针可能失效，需先拷入固定地址的锁页内存；默认 200 MB，超限切块。
9. `src/base_comm/resources/hccp/`；TaskCache 让相同 tag 的任务描述免重复编排，降低 Host 开销。

:::

## 本单元小结

- **四名词**：Endpoint（设备）、Channel（通道）、CommMem（内存）、CommEngine（引擎，单元 4 展开）；
- **两组动词**：网络语义（Write/Read/Notify）与内存语义（映射内存直写），由底层协议决定；同一套 Channel 的两种用法；
- **并发与同步**：Thread 抽象（引擎决定落地）+ 两种 Notify（ThreadNotify 实体内 / ChannelNotify 跨实体），Record/Wait 成对；
- **30 个动词**：本地 / Thread 同步 / Channel 通信 / 批量缓存四家族，命名后缀即语义；
- **Send/Recv 三步握手** = 集合通信算法的最小组件；HCCL Buffer 解决异步悬空指针（200 MB 起切块）；
- **读码落点**：hcomm 仓 `base_comm/primitives/`（原语）与 `base_comm/resources/`（资源）；hccl 仓 executor/template 里编排的就是这些动词（单元 7 的 wrapper 层衔接）。

## 参考资料

- [HCCL & HCOMM 软件架构简介：基础通信（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [编程模型与概念：通信模型（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/prog_models_concepts/comm_model.md)
- [编程模型与概念：并发模型（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/prog_models_concepts/concurrency_model.md)
- [API 参考：数据面 API（cpu-cpu_ts-aicpu_ts，hcomm 仓）](https://gitcode.com/cann/hcomm/tree/master/docs/zh/api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts)
- [HCCL 性能分析：通信数据通路（usermem/hcclbuffer）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/perf_analysis/perf_data_analysis.md)

---

下一单元进入 **[4｜通信引擎：模型、瀑布与三引擎编排](04-comm-engines.md)**。

[返回课程导学 →](../hccl-hcomm.md)
