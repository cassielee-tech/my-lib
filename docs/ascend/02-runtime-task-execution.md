# 第 2 章｜Runtime 与任务执行

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

- **02-1（约 15 分钟）**：[Host、Device 与异构计算](02-runtime-task-execution/01-host-device.md)
- **02-2（约 15 分钟）**：[设备内存、数据搬运与生命周期](02-runtime-task-execution/02-memory-copy-lifecycle.md)
- **02-3（约 15 分钟）**：[Stream、Event、Task 与异步执行](02-runtime-task-execution/03-stream-event-task.md)
- **02-4（约 15 分钟）**：[一次算子调用怎样被下发和完成](02-runtime-task-execution/04-op-call-lifecycle.md)
- **本章总结**：[用时间线解释计算、搬运与同步](02-runtime-task-execution/summary.md)

## 这一章在整条路线中的位置

```text
第 1 章：软件栈全景（Runtime 第一次露面）
          ↓
《单卡执行系统》第 7 章：Stream、Event 与异步执行的通用原理
          ↓
第 2 章：这些原理在昇腾上的 API 与执行模型 ← 你在这里
          ↓
第 3 章：torch_npu → HCCL 的调用链（主线）
          ↓
HCCL 源码：通信任务与计算任务在 Stream 上的交汇
```

第 1 章反复按下不表的 **Stream、Event、Memory、任务下发与同步**，本章全部展开——它们是 HCCL 执行模型的直接前置。

---

[开始单元 02-1 →](02-runtime-task-execution/01-host-device.md)
