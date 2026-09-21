# 本章总结｜用时间线解释计算、搬运与同步

> 所属章节：[第 2 章｜Runtime 与任务执行](../02-runtime-task-execution.md)

四个单元走完，把异构模型、内存、Stream/Event 与算子调用拼成一张**双时间线全景图**，并核对本章开头立下的目标。

## 一张图收束整章

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

## 必须带走的概念清单

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

## 章节自检清单

回到导学立下的目标，逐条核对：

- [ ] 我能写出最小样板代码（init → setDevice → 干活 → reset → finalize）
- [ ] 我能解释 Host 指针与 Device 指针为什么不能混用，以及三种拷贝方向
- [ ] 我能用 Stream + Event 编排"搬运与计算重叠"，说出每一步谁在等谁
- [ ] 我能画出一次 aclnn 调用的 Host/Device 双时间线

额外三条实操向检查：

- [ ] 我知道 `ASCEND_RT_VISIBLE_DEVICES` 怎么重映射设备编号
- [ ] 我能解释"异步报错堆栈驴唇不对马嘴"的原因与排查思路
- [ ] 我看到 Profiling 时间轴时，能把它对应到本章双时间线的各区段

## 与 HCCL 主线的接口

本章每个概念都在 HCCL 中有直接化身：

| 本章概念 | HCCL 中的化身 |
| --- | --- |
| Stream（任务队列） | 通信任务挂在通信 Stream 上，与计算 Stream 并行 |
| Event（跨流依赖） | 计算与通信的重叠编排；HCCL 的 Notify 同族机制（H01-4） |
| 内存生命周期 | 通信收发缓冲的申请/复用/释放（H01-6 buffer 管理） |
| 下发 ≠ 执行 | 集合通信同样是"准备 + 入队"，异步语义完全一致 |
| 双时间线 | 排障与性能分析的标准视图（Profiling） |

## 下一章预告

按学习顺序进入 **第 4 章：Ascend C 算子基础**（二梯队）；按 HCCL 主线，可直接跳到 **第 3 章：从 PyTorch 走向 HCCL**——把本章的 Runtime 世界观接到 `torch.distributed` → torch_npu → HCCL 的调用链上。

无论走哪条，本章的双时间线都是后面所有内容的底图。

## 最终参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [CANN Learning Hub：quick_start 公共基础](https://gitcode.com/cann/cann-learning-hub/tree/master/quick_start/cann_basics)
- [《单卡执行系统》第 7 章：Stream、Event 与异步执行](../../device/07-stream-event-async.md)
- [HCCL 源码专题 H01-4：通信原语与同步机制](../hccl-source/04-primitives-and-sync.md)
- [CANN 常用命令速查](../cann常用命令.md)

---

[返回第 2 章 →](../02-runtime-task-execution.md) · [返回专栏目录 →](../index.md)
