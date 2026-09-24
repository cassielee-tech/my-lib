# 第 1 章｜Runtime 与任务执行

> 本章目标：掌握 Host/Device 异构协作模型；会用 Runtime API 完成设备管理、内存申请与数据搬运；理解 Stream/Event 的任务编排与同步语义；最终能用"Host/Device 双时间线"解释一次算子调用从下发到完成的全过程。

## 本章导学

::: tip 本章只记住 3 件事
1. **Host 指挥、Device 干活**：CPU 负责控制和调度，NPU 负责大规模并行计算；Runtime（`aclrt*` API）是 Host 管理 Device 的把手。
2. **数据搬家是显式的**：Host 内存与 Device 内存是两个地址空间，`aclrtMemcpy` 是唯一的桥；搬运往往比计算更贵。
3. **下发 ≠ 执行**：算子调用只是把任务放进 Stream 队列就返回；Host 与 Device 各走各的时间线，同步（Event/Sync）是两条线的交汇点。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。从"谁在指挥谁"的异构模型开始，到内存与搬运，再到 Stream/Event 编排，最后把一次算子调用拆成双时间线全景。

- [ ] 我能写出最小的 Runtime 样板代码（init → setDevice → 干活 → reset → finalize）
- [ ] 我能解释 Host 指针与 Device 指针为什么不能混用，以及三种拷贝方向
- [ ] 我能用 Stream + Event 编排"搬运与计算重叠"，并说出每一步谁在等谁
- [ ] 我能画出一次 aclnn 算子调用的 Host/Device 双时间线

## 本章单元

- **1-0（约 15 分钟）**：[Host、Device 与异构计算](#_1-0-host、device-与异构计算)
- **1-1（约 15 分钟）**：[设备内存、数据搬运与生命周期](#_1-1-设备内存、数据搬运与生命周期)
- **1-2（约 15 分钟）**：[Stream、Event、Task 与异步执行](#_1-2-stream、event、task-与异步执行)
- **1-3（约 15 分钟）**：[一次算子调用怎样被下发和完成](#_1-3-一次算子调用怎样被下发和完成)
- **本章总结**：[用时间线解释计算、搬运与同步](#本章总结)

## 这一章在整条路线中的位置

```text
第 0 章（昇腾）：软件栈全景（Runtime 第一次露面）
          ↓
《单卡执行系统》第 7 章：Stream、Event 与异步执行的通用原理
          ↓
第 1 章（昇腾）：这些原理在昇腾上的 API 与执行模型 ← 你在这里
          ↓
HCCL 第 0 章：torch_npu → HCCL 的调用链（主线）
          ↓
HCCL 源码：通信任务与计算任务在 Stream 上的交汇
```

第 0 章反复按下不表的 **Stream、Event、Memory、任务下发与同步**，本章全部展开——它们是 HCCL 执行模型的直接前置。

---

[开始单元 1-0 →](#_1-0-host、device-与异构计算)

## 1-0｜Host、Device 与异构计算

::: info 本单元目标
读完后，你能够说出 **Host 与 Device 的分工、Runtime 在 CANN 栈中的位置与最小样板代码**（`aclInit` → `aclrtSetDevice` → 干活 → `aclrtResetDevice` → `aclFinalize`），并解释 Device、Context 与 `npu-smi` 编号的关系。
:::

### 先记住 3 个结论

1. **Host 是指挥，Device 是乐队**：CPU 负责控制流（发指令、备数据、等结果），NPU 负责大规模数据并行（Cube/Vector 一波算完）——这就是"异构计算"。
2. **Runtime 是 Host 管理 Device 的 API 层**：选设备、分内存、拷数据、建队列、下发任务，全部通过 `aclrt*` 接口完成，概念上与 CUDA Runtime 一一对应。
3. **一切从样板代码开始**：`aclInit` → `aclrtSetDevice` → 干活 → `aclrtResetDevice` → `aclFinalize`；顺序错了，后面每一步都会莫名其妙地失败。

### 1. 为什么要异构

第 0 章拆过 NPU 内部：几十上百个 AI Core、Cube/Vector 计算单元、巨大的数据搬运带宽。但擅长"一口气算一大片"的硬件，**不擅长**琐碎的控制流：分支、循环、对象管理、响应操作系统。

于是分工：

| 角色 | 硬件 | 擅长 | 在一次训练中的工作 |
| --- | --- | --- | --- |
| **Host** | CPU + 主机内存 | 控制流、逻辑、调度 | 解析模型、准备输入、下发算子任务、等待结果 |
| **Device** | NPU + HBM | 大规模数据并行 | 矩阵乘、向量运算、集合通信 |

**异构的代价**：两边内存独立、时钟独立——于是有了本章剩下三个单元的主题：数据怎么搬（02-2）、任务怎么排（02-3）、一次调用怎么走完（02-4）。

### 2. Runtime 在栈中的位置

回扣 01-2 的分层图：Runtime 位于算子库/GE 之下、Driver 之上，是 **CANN 的"设备管家"**：

| Runtime 管什么 | 对应 API（本章后续单元逐一展开） |
| --- | --- |
| Device（选哪台设备） | `aclrtSetDevice` / `aclrtResetDevice` / `aclrtGetDevice` |
| Context（执行上下文） | 隐式创建，随 `aclrtSetDevice` 绑定 |
| Memory（Host/Device 内存） | `aclrtMalloc` / `aclrtFree` / `aclrtMallocHost` / `aclrtMemcpy` |
| Stream（任务队列） | `aclrtCreateStream` / `aclrtSynchronizeStream` |
| Event（同步信号） | `aclrtCreateEvent` / `aclrtRecordEvent` / `aclrtStreamWaitEvent` |
| Kernel/算子（任务下发） | aclnn 两段式（02-4 展开） |

用过 CUDA 的话，概念完全对齐：

| 概念 | CANN（昇腾） | CUDA（NVIDIA） |
| --- | --- | --- |
| 初始化 | `aclInit` | `cuInit`（Runtime 隐式） |
| 选设备 | `aclrtSetDevice` | `cudaSetDevice` |
| 设备内存 | `aclrtMalloc` | `cudaMalloc` |
| 数据搬运 | `aclrtMemcpy` | `cudaMemcpy` |
| 任务队列 | `aclrtCreateStream` | `cudaStreamCreate` |
| 打点同步 | `aclrtRecordEvent` | `cudaEventRecord` |

### 3. 最小样板代码

```c
#include <acl/acl.h>
#include <stdio.h>

int main() {
    // 1. 全局初始化（进程生命周期内一次）
    int ret = aclInit(NULL);
    if (ret != ACL_ERROR_NONE) { printf("aclInit failed: %d\n", ret); return -1; }

    // 2. 指定当前线程操作的设备（0 号 NPU）
    ret = aclrtSetDevice(0);

    // 3. —— 在这里干活：建 Stream、分内存、拷数据、下发算子 ——
    //    （02-2 ~ 02-4 的全部内容）

    // 4. 释放设备资源（Stream/内存必须在此之前释放）
    ret = aclrtResetDevice(0);

    // 5. 全局去初始化
    ret = aclFinalize();
    return 0;
}
```

三条工程铁律：

1. **每个返回值都要检查**——Runtime 错误几乎总是级联的：`aclInit` 失败后面全军覆没，早检查早定位；
2. **`aclrtResetDevice` 之前必须释放该设备上申请的所有资源**（内存、Stream、Event），否则报资源泄漏错误；
3. **顺序不可颠倒**：先 Init 再 SetDevice，先释放资源再 ResetDevice，最后 Finalize。

::: warning 最常见的两种"莫名其妙"
- 忘了 `aclInit`，或在不同线程 `aclInit`/`aclFinalize` 次数不配对——接口返回初始化错误；
- 先干活后 `aclrtSetDevice`——干活时上下文还没绑定到设备，报"无有效 context"类错误。
:::

### 4. Device、Context 与设备编号

**deviceId 从哪来？** `aclrtSetDevice(0)` 的 0 是**逻辑编号**，与 `npu-smi info` 显示的设备编号对应。用 `ASCEND_RT_VISIBLE_DEVICES` 环境变量可以限制当前进程可见的 NPU（排障与多人共享机器时常用，见 [CANN 常用命令速查](cann常用命令.md)第 5 节）——**可见列表会重映射编号**，进程里的 0 号不一定是机器上的 0 号卡。

**Context 是什么？** 执行上下文——"当前线程在操作哪台设备、有哪些默认资源（Stream、内存池）"的隐式凭据：

- `aclrtSetDevice` 会（若不存在则）为当前线程绑定默认 Context；
- 后续 `aclrtMalloc`、任务下发等操作都作用在这个 Context 上；
- 多线程程序里每线程各自 SetDevice/绑定 Context，互不串台（HCCL 多进程通信域同理：每个 rank 一个进程，各自管理自己的设备上下文）。

**框架都替你做了什么？** 写 `torch_npu` 时从来没调过这些 API——`import torch_npu` 与 `.npu()` 背后，框架适配层已经完成了 Init/SetDevice/Context/Stream 的管理。学这一层不是为了手写，是为了**看懂框架和 HCCL 源码在做什么**。

### 5. 自测题

先用自己的话回答，再展开答案。

1. Host 与 Device 的分工是什么？"异构"异在哪里？
2. Runtime 在 CANN 栈中处于什么位置？它管哪六类资源？
3. 写出最小样板代码的五个步骤，并说出哪一步必须在最前、哪一步必须在最后。
4. `aclrtResetDevice` 之前必须完成什么？违反了会怎样？
5. `ASCEND_RT_VISIBLE_DEVICES=2,3` 时，`aclrtSetDevice(0)` 操作的是哪张卡？

::: details 自测答案

1. Host（CPU）负责控制流：解析模型、准备数据、下发任务、等待结果；Device（NPU）负责大规模数据并行计算。异构指两套处理器、两套内存、两条时间线协作完成同一任务。
2. 位于算子库/GE 之下、Driver 之上。管理 Device、Context、Memory、Stream、Event、Kernel/任务下发六类资源。
3. `aclInit` → `aclrtSetDevice` → 干活 → `aclrtResetDevice` → `aclFinalize`；`aclInit` 必须最前（其后所有接口依赖它），`aclFinalize` 必须最后。
4. 必须释放该设备上申请的全部资源（`aclrtFree` 内存、销毁 Stream/Event 等）；违反则报资源未释放/泄漏类错误。
5. 可见列表中的第一张——物理 2 号卡（可见列表会重映射逻辑编号）。

:::

### 本单元小结

- **分工**：Host 指挥、Device 干活；代价是内存独立、时间线独立；
- **Runtime**：`aclrt*` API 层，概念与 CUDA Runtime 一一对应；
- **样板**：Init → SetDevice → 干活 → ResetDevice → Finalize，顺序即铁律；
- **编号**：deviceId 是逻辑编号，受 `ASCEND_RT_VISIBLE_DEVICES` 重映射影响；
- **动机**：框架替你调这些 API——学它是为了读懂 torch_npu 与 HCCL 源码。

### 参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [CANN Learning Hub：quick_start 公共基础](https://gitcode.com/cann/cann-learning-hub/tree/master/quick_start/cann_basics)
- [《单卡执行系统》第 7 章：Stream、Event 与异步执行（通用原理）](../device/07-stream-event-async.md)
- [CANN 常用命令速查：限制可见 NPU、设备状态](cann常用命令.md)

---

下一单元将进入 **02-2：设备内存、数据搬运与生命周期**。

## 1-1｜设备内存、数据搬运与生命周期

::: info 本单元目标
读完后，你能够说出 **Host 与 Device 两个地址空间的边界、`aclrtMalloc`/`aclrtMemcpy` 一族 API 的分工与生命周期规则**（申请/释放配对、拷贝三方向、同步与异步搬运的区别），并解释 `.npu()`/`.cpu()` 背后发生了什么。
:::

### 先记住 3 个结论

1. **两个地址空间，指针不能混用**：Host 指针解引用不到 Device 内存，反之亦然——数据过桥只有一条路：`aclrtMemcpy`。
2. **搬运比计算贵**：HBM 带宽再高也是"过桥"；`H2D → 计算 → D2H` 的完整循环里，搬运常常占大头（回扣 [《单卡执行系统》第 3 章](../device/03-memory-hierarchy.md)）。
3. **同步拷贝卡 Host，异步拷贝挂 Stream**：`aclrtMemcpy` 阻塞到搬完为止；`aclrtMemcpyAsync` 只是入队——省下的时间要靠 Stream 同步兜底。

### 1. 两个地址空间

```text
   Host（CPU）                        Device（NPU）
 ┌──────────────┐                  ┌──────────────────┐
 │ malloc 的内存  │   ══aclrtMemcpy═▶ │ HBM（Global Mem） │
 │ hostPtr      │   ◀══aclrtMemcpy═ │ devPtr            │
 └──────────────┘                  └──────────────────┘
        CPU 直接解引用 devPtr → 崩溃/脏数据
```

Device 内存就是第 0 章说的 **Global Memory（HBM）**——算子的输入输出都住在这里；AI Core 干活时还要从 Global Memory 搬进片上 Local Memory（那是 Ascend C 篇的故事）。

::: tip 为什么设计成两个空间？
隔离带来高性能：HBM 带宽按大规模并行设计，Host 内存按通用性设计；DMA 引擎在两者之间异步搬运，搬运期间 CPU 可以继续干别的。隔离的代价就是"过桥必须显式"。
:::

### 2. 核心 API 一览

| API | 作用 | 关键细节 |
| --- | --- | --- |
| `aclrtMalloc(&devPtr, size, policy)` | 申请 Device 内存 | policy 常用 `ACL_MEM_MALLOC_HUGE_FIRST`（优先大页，性能更好） |
| `aclrtFree(devPtr)` | 释放 Device 内存 | 必须与 Malloc 严格配对 |
| `aclrtMallocHost(&hostPtr, size)` | 申请**锁页** Host 内存 | DMA 可直达，异步搬运更稳更快 |
| `aclrtFreeHost(hostPtr)` | 释放锁页内存 | 同样严格配对 |
| `aclrtMemcpy(dst, destMax, src, count, kind)` | **同步**搬运 | 阻塞 Host 直到搬完 |
| `aclrtMemcpyAsync(dst, destMax, src, count, kind, stream)` | **异步**搬运 | 挂到 Stream，立即返回 |

**三种方向（kind）**：

| kind | 方向 | 典型场景 |
| --- | --- | --- |
| `ACL_MEMCPY_HOST_TO_DEVICE` | Host → Device | 把输入数据送上卡（`.npu()`） |
| `ACL_MEMCPY_DEVICE_TO_HOST` | Device → Host | 把结果取回 CPU（`.cpu()`） |
| `ACL_MEMCPY_DEVICE_TO_DEVICE` | Device → Device | 卡内/跨卡数据重排（框架内部常用） |

方向写反 = 往错误地址空间解引用，结果是崩溃或"数据莫名其妙是垃圾"——这是新手前三大 bug 之一。

### 3. 生命周期与两条工程规则

**规则一：申请与释放严格配对。**

```c
void *devPtr = nullptr;
aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);   // 申请
// ……干活……
aclrtFree(devPtr);                                        // 释放
// 且 aclrtFree 必须发生在 aclrtResetDevice 之前（02-1 铁律）
```

Device 显存不是进程一退出就必然归还——训练挂掉后显存不释放，先查有没有残留进程（`npu-smi info` 看占用，配合 [常用命令速查](cann常用命令.md)第 4/7 节）。

**规则二：普通内存 vs 锁页内存。**

- `malloc` 的普通 Host 内存可能被操作系统换页；DMA 搬运时"页不在原地"会导致失败或退化成两段搬运；
- `aclrtMallocHost` 申请**锁页（page-locked）内存**，操作系统承诺不换页——异步搬运的安全选择，代价是占用 Host 物理内存且申请释放更贵；
- 经验法则：**频繁/异步搬运的缓冲区用锁页内存，一次性小数据无所谓**。

### 4. 同步拷贝 vs 异步拷贝

```c
// 同步：Host 在这一行等到搬完才继续
aclrtMemcpy(devPtr, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE);

// 异步：任务入队，Host 立即继续
aclrtMemcpyAsync(devPtr, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
// 危险：此刻数据还没到！任何依赖 devPtr 的操作必须排在同一 stream 的后面，
//      或先 aclrtSynchronizeStream(stream)
```

回扣 [《单卡执行系统》第 7 章](../device/07-stream-event-async.md)的核心命题——**Host 跑得比 Device 快**：异步拷贝省下的 Host 时间是真实的，但"数据何时就绪"的责任转移给了你。两条安全线：

1. **同 Stream 保序**：依赖 `devPtr` 的算子下发到**同一条 Stream**，天然排在拷贝之后；
2. **显式同步**：跨 Stream 依赖时，用 Event 或 `aclrtSynchronizeStream` 兜底（02-3 展开）。

### 5. 与 PyTorch 的关系

```python
x = torch.randn(1024, 1024)
y = x.npu()    # 背后：aclrtMalloc + aclrtMemcpyAsync(H2D, 当前 stream)
z = y.cpu()    # 背后：aclrtMemcpyAsync(D2H) + 同步当前 stream
```

框架替你选好了 policy、锁页内存池和默认 Stream——但 Profiling 里看到的那段 `H2D/D2H` 时间，就是本章这些 API 在时间线上的具象。

### 6. 自测题

先用自己的话回答，再展开答案。

1. 为什么 Host 指针和 Device 指针不能混用？"过桥"必须用哪个 API？
2. 列出三种拷贝方向及各自的典型场景。
3. 锁页内存（`aclrtMallocHost`）解决了什么问题？代价是什么？
4. `aclrtMemcpyAsync` 返回后，数据就绪了吗？说出两种保证正确性的手段。
5. `aclrtFree` 与 `aclrtResetDevice` 的先后顺序是什么？为什么？

::: details 自测答案

1. 两个地址空间互相不可见：CPU 解引用 Device 指针访问不到 HBM，反之亦然。必须用 `aclrtMemcpy`/`aclrtMemcpyAsync` 显式搬运。
2. H2D：输入上卡（`.npu()`）；D2H：结果回 CPU（`.cpu()`）；D2D：卡内/跨卡数据重排（框架内部）。
3. 解决普通内存可被换页、DMA 直达不可靠的问题；代价是长期占用 Host 物理内存、申请释放开销更大。
4. 没有——只是任务入了队。手段：① 依赖该数据的任务下发到同一条 Stream（保序）；② 跨 Stream 时用 Event/`aclrtSynchronizeStream` 显式同步。
5. 先 `aclrtFree` 后 `aclrtResetDevice`；ResetDevice 会校验资源是否全部释放，未释放则报错。

:::

### 本单元小结

- **边界**：Host/Device 两个地址空间，`aclrtMemcpy` 是唯一的桥，方向写反必炸；
- **生命周期**：Malloc/Free 严格配对，且在 ResetDevice 之前；锁页内存是异步搬运的安全垫；
- **同步语义**：同步拷贝卡 Host、异步拷贝挂 Stream，"就绪责任"随异步转移；
- **框架映射**：`.npu()`/`.cpu()` 就是 H2D/D2H 拷贝在 Profiling 里的样子；
- **性能直觉**：搬运常常比计算贵——能少过桥就少过桥。

### 参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [《单卡执行系统》第 3 章：存储层次与数据搬运](../device/03-memory-hierarchy.md)
- [《单卡执行系统》第 7 章：异步执行](../device/07-stream-event-async.md#_07-1-异步执行-host-跑得比-device-快)
- [CANN 常用命令速查：NPU 状态与显存](cann常用命令.md)

---

下一单元将进入 **02-3：Stream、Event、Task 与异步执行**。

## 1-2｜Stream、Event、Task 与异步执行

::: info 本单元目标
读完后，你能够说出 **Stream 的队列语义、Event 的打点/等待用法与四种同步层次**（SynchronizeStream/SynchronizeDevice/Event 同步/无同步），并用双 Stream + Event 编排"搬运与计算重叠"，说明每一步谁在等谁。
:::

### 先记住 3 个结论

1. **Stream 是 Device 侧的 FIFO 任务队列**：同一条 Stream 内任务严格保序；下发（launch）只是入队，Host 立即返回——"下发 ≠ 执行"。
2. **Event 是打点与等待的原语**：`aclrtRecordEvent` 在某条 Stream 上打一个时间点，`aclrtStreamWaitEvent` 让另一条 Stream 等到这个点再继续——跨 Stream 依赖的唯一正道。
3. **多 Stream 是并行的钥匙，也是数据竞争的入口**：两条 Stream 各自保序，但互相不知道对方在动哪块内存——共享数据必须用 Event 显式编排。

### 1. 从通用原理到昇腾 API

[《单卡执行系统》第 7 章](../device/07-stream-event-async.md)已经建立了通用模型（单行道、依赖、同步），本单元做平台落地。昇腾 Device 侧的"任务"有三类：

| 任务类型 | 例子 | 下发接口 |
| --- | --- | --- |
| 计算任务 | 算子 Kernel | aclnn 两段式（02-4） |
| 搬运任务 | H2D/D2H/D2D 拷贝 | `aclrtMemcpyAsync` |
| 通信任务 | HCCL 集合通信 | HCCL 接口（挂到通信 Stream） |

它们都是"任务"，都在 Stream 上排队——**这就是计算与通信能在 Runtime 层协同的原因**，也是本章在 HCCL 主线上的价值。

### 2. Stream：创建、销毁与默认流

```c
aclrtStream stream;
aclrtCreateStream(&stream);   // 创建
// ……向 stream 下发任务……
aclrtDestroyStream(stream);   // 销毁（之前必须保证任务都已完成）
```

- **同 Stream 保序**：先下发的搬运，一定在后面的算子之前完成——02-2 的"安全线一"；
- **跨 Stream 不保序**：Stream A 的任务与 Stream B 的任务可能乱序完成；
- **默认 Stream**：不传 Stream（或 NULL）时任务进默认流——单流程序够用，但**没有重叠可言**；性能场景显式建流。

::: warning HCCL 与 Stream
HCCL 集合通信调用同样接收一个 Stream 参数：通信任务排在该 Stream 上，与计算任务的依赖关系由 Stream/Event 表达。`torch_npu` 里通常为通信单独建 Stream，再与计算 Stream 用 Event 编排（HCCL 第 0 章展开）。
:::

### 3. Event：打点与等待

三个 API 构成完整闭环：

```c
aclrtEvent event;
aclrtCreateEvent(&event);

aclrtRecordEvent(event, computeStream);     // 在 computeStream 上打点：
                                           // "跑到这里时，event 完成"
aclrtStreamWaitEvent(copyStream, event);   // copyStream 等到 event 完成后才继续
aclrtSynchronizeEvent(event);              // （可选）Host 也来等这个点
aclrtDestroyEvent(event);
```

#### 3.1 经典编排：搬运与计算重叠

双 Stream 流水（通用原理见 [重叠的艺术](../device/07-stream-event-async.md#_07-4-重叠的艺术-让设备闲不下来)）：

```text
computeStream：  [算子 batch1]···········[等event2][算子 batch2]
copyStream：    [等event1]              [拷 batch2 数据 H2D][record event2]
                          ▲ record event1（batch1 数据已拷完）
```

每一方"先等对方上一轮的完成点，再动共享内存"——**依赖正确性与重叠执行同时成立**。把"算子"换成"HCCL 通信"、把"拷贝"换成"反向梯度计算"，就是训练循环里通信计算重叠的骨架（回扣 [《集合通信》第 5 章](../collective/05-topology-hierarchical-overlap.md#_05-4-通信计算重叠-从公式到实践)）。

### 4. 四种同步层次

| 层次 | API | 语义 | 用法 |
| --- | --- | --- | --- |
| Stream 同步 | `aclrtSynchronizeStream(stream)` | Host 阻塞到该队列全部完成 | 最常用：下发完一批任务后等结果 |
| Device 同步 | `aclrtSynchronizeDevice()` | Host 阻塞到所有队列全部完成 | 兜底/退出前 |
| Event 同步 | `aclrtSynchronizeEvent(event)` | Host 只等到某个点 | 精细化等待 |
| 无同步 | —— | Host 继续跑，Device 自己排队 | 需要正确性自负的异步编程 |

**异步的阴暗面**：Device 侧的错误（Kernel 崩溃、非法地址）在**发生时**不会立刻出现在调用点，而是在后续某个同步点才抛出——所以异步报错的堆栈往往"驴唇不对马嘴"（排障思路见 [常用命令速查：异步报错定位](cann常用命令.md)）。

### 5. 自测题

先用自己的话回答，再展开答案。

1. Stream 的保序规则是什么？"下发 ≠ 执行"怎么理解？
2. 默认 Stream 够用时为什么还要显式建 Stream？
3. 说出 Event 闭环的三个 API 及各自语义。
4. 双 Stream 重叠编排中，`aclrtStreamWaitEvent` 防的是什么问题？
5. 为什么异步报错的堆栈经常"驴唇不对马嘴"？

::: details 自测答案

1. 同一条 Stream 内任务严格 FIFO 保序；不同 Stream 之间不保序。下发只是把任务描述放进队列，Host 立即返回，Device 稍后执行。
2. 默认 Stream 只有单队列，任务只能串行；显式多 Stream 才能把搬运/计算/通信放到并行车道，实现重叠。
3. `aclrtRecordEvent`：在指定 Stream 上打时间点；`aclrtStreamWaitEvent`：让某 Stream 等到该点再继续；`aclrtSynchronizeEvent`：Host 阻塞等待该点。
4. 数据竞争：两条 Stream 各自闭眼跑，可能"还没写完就读"；用 WaitEvent 建立跨 Stream 的先后依赖，在动共享内存前对齐。
5. Device 错误在发生时刻无法同步回 Host，要等后续某个同步点才抛出——堆栈指向的是"报错浮出水面的地方"，不是"事故现场"。

:::

### 本单元小结

- **Stream**：Device 侧 FIFO 队列，同流保序、跨流不保序；通信任务也是 Stream 上的任务；
- **Event**：Record 打点 + StreamWait 等待 = 跨 Stream 依赖的正道；双 Stream 重叠是通用骨架；
- **同步四层次**：Stream / Device / Event / 无同步，按需选择；
- **阴暗面**：异步报错延迟浮出水面，排障要有"时间线思维"；
- **主线钩子**：HCCL 通信与计算任务的协同 = 本章 Stream/Event 语义的直接应用（HCCL 源码 3 的 Notify 同族）。

### 参考资料

- [《单卡执行系统》第 7 章：Stream、Event 与异步执行](../device/07-stream-event-async.md)
- [《集合通信》第 5 章：通信计算重叠](../collective/05-topology-hierarchical-overlap.md#_05-4-通信计算重叠-从公式到实践)
- [HCCL 源码 3：通信原语与同步机制](hccl-source/03-primitives-and-sync.md)
- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)

---

下一单元将进入 **02-4：一次算子调用怎样被下发和完成**。

## 1-3｜一次算子调用怎样被下发和完成

::: info 本单元目标
读完后，你能够说出 **aclnn 算子接口的两段式结构**（`GetWorkspaceSize` 准备段 + 执行段）及其设计动机，并独立画出一次算子调用的 **Host/Device 双时间线**，解释气泡、等待与重叠的来源。
:::

### 先记住 3 个结论

1. **一次算子调用 = Host 侧纯准备 + 一次轻量入队**：准备（查描述、算 workspace、组 executor）发生在 Host，执行段把任务挂上 Stream 后立即返回——全程异步。
2. **aclnn 两段式**：`aclnnXxxGetWorkspaceSize(...)` 做无副作用的准备工作（可缓存、可并发），`aclnnXxx(workspace, wsSize, executor, stream)` 才真正入队执行。
3. **双时间线读性能**：Host 时间线与 Device 时间线各画一条，任务在中间"投递"——90% 的性能现象（气泡、等待、重叠）都能在这张图上指认。

### 1. 两段式结构

以 `aclnnAdd` 为例，昇腾单算子接口（aclnn）的标准形态：

```c
// 第一段：准备（纯 Host 工作，不碰 Device）
aclnnStatus aclnnAddGetWorkspaceSize(
    const aclTensor *self, const aclTensor *other, float alpha,
    aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

// 第二段：执行（把 executor 绑定的任务挂上 Stream）
aclnnStatus aclnnAdd(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

调用方标准流程：

```c
uint64_t wsSize = 0;
aclOpExecutor *executor = nullptr;
aclnnAddGetWorkspaceSize(a, b, 1.0, out, &wsSize, &executor);  // ① 准备

void *ws = nullptr;
if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST); // ② 申请 workspace

aclnnAdd(ws, wsSize, executor, stream);                        // ③ 入队执行
```

#### 1.1 为什么要拆成两段？

| 设计点 | 收益 |
| --- | --- |
| 准备段**无副作用** | 同一组参数可以只准备一次，**executor 复用**（框架的算子调用大量重复） |
| 准备段不碰 Device | Host 侧可以**多线程并发**地为多个算子做准备 |
| workspace 大小显式给出 | 调用方统一管理设备内存（复用内存池），而不是每次暗地里 malloc |
| 执行段极轻 | 只是"把组好的任务挂上 Stream"，为整个 Eager 流水提速 |

::: tip workspace 是什么
算子执行时需要的**设备侧临时空间**（中间结果、tiling 缓冲等），大小由准备段算出。它是显存账本的一部分——《训练与推理系统》第 3 章 FSDP/显存优化的通用原理在这里同样适用。
:::

### 2. 双时间线全景

把 02-1 ~ 02-3 的所有零件装上，一次"上卡 → 计算 → 取回"的完整旅程：

```text
Host 时间线
 ├ aclInit / aclrtSetDevice（02-1）
 ├ aclrtCreateStream（02-3）
 ├ aclrtMallocHost / aclrtMalloc（02-2）
 ├ aclrtMemcpyAsync(H2D, stream) ──────────┐ 立即返回
 ├ aclnnAddGetWorkspaceSize ──┐ 纯 Host 计算│
 ├ aclrtMalloc(workspace)      │             │
 ├ aclnnAdd(executor, stream) ─┼─────────────┤ 立即返回
 ├ ……Host 继续干别的（准备下一批输入）        │
 ├ aclrtMemcpyAsync(D2H, stream) ────────────┤ 立即返回
 └ aclrtSynchronizeStream(stream) ◀═════════╧ Host 在此等待

Device 时间线（stream 队列，FIFO）
 [拷贝 H2D] → [Add Kernel] → [拷贝 D2H]
   ▲此刻才真正开始执行——Host 早就跑到后面去了
```

**读图三则**：

1. **气泡**：Host 准备慢（如 `GetWorkspaceSize` 串联执行太多），Device 队列空转——Eager 模式 Host 开销的来源（回扣 [《单卡执行系统》第 6 章](../device/06-eager-graph-compilation.md)：图模式下沉正是为了消灭它）；
2. **等待**：`aclrtSynchronizeStream` 之后 Host 才能安全读 D2H 的结果——同步点是两条时间线的交汇；
3. **重叠**：H2D（第 N+1 批）与 Kernel（第 N 批）若在同一条 Stream 上天然流水；跨 Stream 才需要 Event 编排（02-3）。

### 3. 闭环检查清单

一次正确的算子调用，按顺序核对：

- [ ] `aclInit` → `aclrtSetDevice` 已完成（02-1）；
- [ ] 输入数据已 H2D，且依赖它的算子在**同一条 Stream**（或已用 Event 对齐）；
- [ ] 准备段成功、workspace 已申请（大小 > 0 才需要）；
- [ ] 执行段挂上了正确的 Stream；
- [ ] 结果 D2H 之前，有 Stream/Event 同步兜底；
- [ ] 释放顺序：workspace/内存 Free → Stream/Event Destroy → `aclrtResetDevice` → `aclFinalize`。

::: warning 最经典的 bug
D2H 拷回结果后直接读——但忘了 `aclrtSynchronizeStream`，读到的是**还没写入的旧内存**：不崩溃、数值"看起来还行"，偶发且难查。记住：**异步世界里，读到结果之前必有同步点**。
:::

### 4. 从这里去向哪里

- **向上**：`torch_npu` 把这套样板包进了框架——`y = a + b`（npu tensor）背后就是"准备段 + 执行段"的循环（HCCL 第 0 章展开）；
- **向旁**：HCCL 的 AllReduce 一样是"准备 + 入队通信 Stream"的结构（[HCCL 源码 6 调用链走读](hccl-source/06-allreduce-call-chain.md)）；
- **向工具**：Profiling（msprof）呈现的正是 Device 时间线上每个任务的起止——学会本章的双时间线，Profiling 火焰图/时间轴就是它的放大版（[常用命令速查：msprof](cann常用命令.md)）。

### 5. 自测题

先用自己的话回答，再展开答案。

1. 一次 aclnn 调用的两段分别做什么？哪一段碰 Device？
2. 准备段"无副作用"带来了哪两个工程收益？
3. workspace 是什么？由谁算出、由谁申请？
4. 画出双时间线，标出 Host 开始等待的位置和 Device 开始执行的位置。
5. "D2H 后直接读，偶尔读到旧数据"的 bug 根源是什么？怎么修？

::: details 自测答案

1. `GetWorkspaceSize` 准备段：纯 Host 工作（参数校验、shape 推导、算 workspace、组 executor），不碰 Device；执行段：把任务挂上 Stream 入队。
2. executor 可复用（重复调用同一算子只准备一次）；Host 侧可多线程并发准备多个算子。
3. 算子执行所需的设备侧临时空间；由准备段算出大小，由调用方用 `aclrtMalloc` 申请（可走内存池复用）。
4. Host：Init → 拷贝入队 → 准备 → 执行入队 → …… → SynchronizeStream（此处开始等待）；Device：队列中的 [H2D] → [Kernel] → [D2H] 依次执行，起点在 Host 已往下跑之后。
5. 异步拷贝刚入队、尚未完成，Host 抢先读了目标缓冲；在读取前补 `aclrtSynchronizeStream`（或 Event 同步）。

:::

### 本单元小结

- **两段式**：准备段（无副作用、可缓存并发）+ 执行段（轻量入队）——昇腾单算子接口的统一形态；
- **双时间线**：Host 与 Device 各走各的，任务从 Host"投递"到 Device 队列，同步点是交汇；
- **气泡/等待/重叠**三大性能现象都能在双时间线图上指认；
- **经典 bug**：异步读结果不加同步——"读到旧内存"；
- **主线衔接**：torch_npu、HCCL、Profiling 全部建立在这张图上。

### 参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [《单卡执行系统》第 6 章：Eager 与图编译](../device/06-eager-graph-compilation.md)
- [HCCL 源码 6：AllReduce 调用链走读](hccl-source/06-allreduce-call-chain.md)
- [CANN 常用命令速查：msprof 性能采集](cann常用命令.md)

---

下一单元将进入 **本章总结：用时间线解释计算、搬运与同步**。

## 本章总结

四个单元走完，把异构模型、内存、Stream/Event 与算子调用拼成一张**双时间线全景图**，并核对本章开头立下的目标。

### 一张图收束整章

```text
Host 时间线（指挥）
 aclInit ─ aclrtSetDevice ─ aclrtCreateStream
    │
    ├ aclrtMallocHost ─ aclrtMalloc ─ aclrtMalloc(workspace)
    │
    ├ aclrtMemcpyAsync(H2D) ──┐
    ├ aclnnXxxGetWorkspaceSize │ 立即返回：Host 只负责"投递"
    ├ aclnnXxx(executor,stream)│
    │        ……Host 继续准备下一批（这就是重叠的空隙）……
    ├ aclrtMemcpyAsync(D2H) ──┤
    └ aclrtSynchronizeStream ◀╯ Host 在此等待
    │
    ├ 读结果 ─ aclrtFree × N ─ aclrtDestroyStream/Event
    └ aclrtResetDevice ─ aclFinalize

Device 时间线（干活，stream 队列 FIFO）
 [H2D 拷贝] → [算子 Kernel] → [D2H 拷贝]
      ▲ 双 Stream 时：copyStream 与 computeStream 用 Event 互等，编织重叠
```

这张图同时回答三件事：**谁在动数据**（拷贝任务）、**谁在算**（计算任务）、**谁在等谁**（同步点）——它们正是 HCCL 源码里"通信任务与计算任务协同"的全部素材。

### 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| Host / Device | 指挥（CPU/控制流）与乐队（NPU/数据并行） | 02-1 |
| Runtime 样板 | Init → SetDevice → 干活 → ResetDevice → Finalize | 02-1 |
| Context | "当前线程操作哪台设备"的隐式凭据 | 02-1 |
| 两个地址空间 | Host/Device 指针不能混用，Memcpy 是唯一的桥 | 02-2 |
| 三种拷贝方向 | H2D / D2H / D2D，方向写反必炸 | 02-2 |
| 锁页内存 | `aclrtMallocHost`：DMA 可直达的 Host 内存 | 02-2 |
| Stream | Device 侧 FIFO 队列：同流保序、跨流不保序 | 02-3 |
| Event 闭环 | Record 打点 + StreamWait 等待 + SyncEvent | 02-3 |
| 同步四层次 | Stream / Device / Event / 无同步 | 02-3 |
| 下发 ≠ 执行 | launch 只是入队，Host 立即返回 | 02-3/02-4 |
| aclnn 两段式 | 准备段（无副作用）+ 执行段（入队） | 02-4 |
| workspace | 算子的设备侧临时空间，准备段算大小 | 02-4 |
| 双时间线 | Host/Device 各走各的，同步点是交汇 | 02-4 |

### 章节自检清单

回到导学立下的目标，逐条核对：

- [ ] 我能写出最小样板代码（init → setDevice → 干活 → reset → finalize）
- [ ] 我能解释 Host 指针与 Device 指针为什么不能混用，以及三种拷贝方向
- [ ] 我能用 Stream + Event 编排"搬运与计算重叠"，说出每一步谁在等谁
- [ ] 我能画出一次 aclnn 调用的 Host/Device 双时间线

额外三条实操向检查：

- [ ] 我知道 `ASCEND_RT_VISIBLE_DEVICES` 怎么重映射设备编号
- [ ] 我能解释"异步报错堆栈驴唇不对马嘴"的原因与排查思路
- [ ] 我看到 Profiling 时间轴时，能把它对应到本章双时间线的各区段

### 与 HCCL 主线的接口

本章每个概念都在 HCCL 中有直接化身：

| 本章概念 | HCCL 中的化身 |
| --- | --- |
| Stream（任务队列） | 通信任务挂在通信 Stream 上，与计算 Stream 并行 |
| Event（跨流依赖） | 计算与通信的重叠编排；HCCL 的 Notify 同族机制（HCCL 源码 3） |
| 内存生命周期 | 通信收发缓冲的申请/复用/释放（HCCL 源码 8 资源管理） |
| 下发 ≠ 执行 | 集合通信同样是"准备 + 入队"，异步语义完全一致 |
| 双时间线 | 排障与性能分析的标准视图（Profiling） |

### 下一章预告

按学习顺序进入 **Ascend C 篇：算子开发基础**（二梯队）；按 HCCL 主线，可直接跳到 **HCCL 第 0 章：从 PyTorch 走向 HCCL**——把本章的 Runtime 世界观接到 `torch.distributed` → torch_npu → HCCL 的调用链上。

无论走哪条，本章的双时间线都是后面所有内容的底图。

### 最终参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [CANN Learning Hub：quick_start 公共基础](https://gitcode.com/cann/cann-learning-hub/tree/master/quick_start/cann_basics)
- [《单卡执行系统》第 7 章：Stream、Event 与异步执行](../device/07-stream-event-async.md)
- [HCCL 源码 3：通信原语与同步机制](hccl-source/03-primitives-and-sync.md)
- [CANN 常用命令速查](cann常用命令.md)
