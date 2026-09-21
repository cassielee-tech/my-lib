# 单元 02-1｜Host、Device 与异构计算

> 所属章节：[第 2 章｜Runtime 与任务执行](../02-runtime-task-execution.md)

::: info 本单元目标
读完后，你能够说出 **Host 与 Device 的分工、Runtime 在 CANN 栈中的位置与最小样板代码**（`aclInit` → `aclrtSetDevice` → 干活 → `aclrtResetDevice` → `aclFinalize`），并解释 Device、Context 与 `npu-smi` 编号的关系。
:::

## 先记住 3 个结论

1. **Host 是指挥，Device 是乐队**：CPU 负责控制流（发指令、备数据、等结果），NPU 负责大规模数据并行（Cube/Vector 一波算完）——这就是"异构计算"。
2. **Runtime 是 Host 管理 Device 的 API 层**：选设备、分内存、拷数据、建队列、下发任务，全部通过 `aclrt*` 接口完成，概念上与 CUDA Runtime 一一对应。
3. **一切从样板代码开始**：`aclInit` → `aclrtSetDevice` → 干活 → `aclrtResetDevice` → `aclFinalize`；顺序错了，后面每一步都会莫名其妙地失败。

## 1. 为什么要异构

第 1 章拆过 NPU 内部：几十上百个 AI Core、Cube/Vector 计算单元、巨大的数据搬运带宽。但擅长"一口气算一大片"的硬件，**不擅长**琐碎的控制流：分支、循环、对象管理、响应操作系统。

于是分工：

| 角色 | 硬件 | 擅长 | 在一次训练中的工作 |
| --- | --- | --- | --- |
| **Host** | CPU + 主机内存 | 控制流、逻辑、调度 | 解析模型、准备输入、下发算子任务、等待结果 |
| **Device** | NPU + HBM | 大规模数据并行 | 矩阵乘、向量运算、集合通信 |

**异构的代价**：两边内存独立、时钟独立——于是有了本章剩下三个单元的主题：数据怎么搬（02-2）、任务怎么排（02-3）、一次调用怎么走完（02-4）。

## 2. Runtime 在栈中的位置

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

## 3. 最小样板代码

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

## 4. Device、Context 与设备编号

**deviceId 从哪来？** `aclrtSetDevice(0)` 的 0 是**逻辑编号**，与 `npu-smi info` 显示的设备编号对应。用 `ASCEND_RT_VISIBLE_DEVICES` 环境变量可以限制当前进程可见的 NPU（排障与多人共享机器时常用，见 [CANN 常用命令速查](../cann常用命令.md)第 5 节）——**可见列表会重映射编号**，进程里的 0 号不一定是机器上的 0 号卡。

**Context 是什么？** 执行上下文——"当前线程在操作哪台设备、有哪些默认资源（Stream、内存池）"的隐式凭据：

- `aclrtSetDevice` 会（若不存在则）为当前线程绑定默认 Context；
- 后续 `aclrtMalloc`、任务下发等操作都作用在这个 Context 上；
- 多线程程序里每线程各自 SetDevice/绑定 Context，互不串台（HCCL 多进程通信域同理：每个 rank 一个进程，各自管理自己的设备上下文）。

**框架都替你做了什么？** 写 `torch_npu` 时从来没调过这些 API——`import torch_npu` 与 `.npu()` 背后，框架适配层已经完成了 Init/SetDevice/Context/Stream 的管理。学这一层不是为了手写，是为了**看懂框架和 HCCL 源码在做什么**。

## 5. 自测题

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

## 本单元小结

- **分工**：Host 指挥、Device 干活；代价是内存独立、时间线独立；
- **Runtime**：`aclrt*` API 层，概念与 CUDA Runtime 一一对应；
- **样板**：Init → SetDevice → 干活 → ResetDevice → Finalize，顺序即铁律；
- **编号**：deviceId 是逻辑编号，受 `ASCEND_RT_VISIBLE_DEVICES` 重映射影响；
- **动机**：框架替你调这些 API——学它是为了读懂 torch_npu 与 HCCL 源码。

## 参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [CANN Learning Hub：quick_start 公共基础](https://gitcode.com/cann/cann-learning-hub/tree/master/quick_start/cann_basics)
- [《单卡执行系统》第 7 章：Stream、Event 与异步执行（通用原理）](../../device/07-stream-event-async.md)
- [CANN 常用命令速查：限制可见 NPU、设备状态](../cann常用命令.md)

---

下一单元将进入 **02-2：设备内存、数据搬运与生命周期**。

[返回第 2 章 →](../02-runtime-task-execution.md)
