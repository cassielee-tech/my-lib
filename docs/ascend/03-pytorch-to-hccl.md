# 第 3 章｜从 PyTorch 走向 HCCL

> 本章目标：打通 `torch.distributed` → torch_npu → HCCL 的完整调用链——说清 torch_npu 作为"桥"注册了什么、ProcessGroup 抽象怎样把框架的通信请求交给 HCCL、Stream/Notify 怎样桥接两个世界；最终能独立追踪一次 `dist.all_reduce` 从 Python 到通信库入口的每一步。

## 本章导学

::: tip 本章只记住 3 件事
1. **torch_npu 是一座三件套的桥**：注册 `npu` 设备（`.npu()`）、把 aten 算子映射到 aclnn 两段式（计算路径）、提供 `torch.npu.Stream/Event` 与 Runtime 的对应（异步路径）——第 1 章分层图里那一层的放大。
2. **ProcessGroup 是通信的门面**：`init_process_group(backend="hccl")` 创建 `ProcessGroupHCCL`，其内部建立 HCCL 通信域（HcclComm）；`dist.all_reduce` 沿 Python → c10d → ProcessGroupHCCL 下降到 HCCL 入口。
3. **Stream 是两个世界的缝合线**：框架把当前流（或专用通信流）传给 HCCL，集合通信任务挂上流异步执行——"下发 ≠ 执行"与重叠编排都建立在这条线上。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。本章是全课程主线的收官——把 [《并行策略》](../parallel/index.md)的需求侧与 [HCCL 源码专题](./hccl-source.md)的供给侧缝成一条完整链路。

- [ ] 我能画出 `y = a + b`（npu tensor）从 Python 到 aclnn 的路径
- [ ] 我能说出 `init_process_group` 到 `HcclComm` 建立之间发生了什么
- [ ] 我能独立追踪一次 `dist.all_reduce` 的完整旅程并指出每层的职责

## 本章单元

- **03-1（约 15 分钟）**：[torch_npu：框架与 CANN 之间的桥](./03-pytorch-to-hccl/01-torch-npu-bridge.md)
- **03-2（约 15 分钟）**：[torch.distributed 的门面：ProcessGroup](./03-pytorch-to-hccl/02-process-group.md)
- **03-3（约 15 分钟）**：[Stream 与 Notify 的桥接](./03-pytorch-to-hccl/03-stream-notify-bridge.md)
- **03-4（约 15 分钟）**：[一次 AllReduce 的完整旅程](./03-pytorch-to-hccl/04-allreduce-journey.md)
- **本章总结**：[追踪一次计算与集合通信的交汇点](./03-pytorch-to-hccl/summary.md)

## 这一章在整条路线中的位置

```text
《并行策略》：并行策略为什么产生通信（需求侧）
《昇腾与 HCCL》第 1-2 章：CANN 分层与 Runtime 执行模型（平台地基）
        ↓
本章：框架调用链——torch_npu / ProcessGroup / Stream 三座桥（缝合处）
        ↓
HCCL 源码专题 H01-7：AllReduce 在通信库内部的下潜（供给侧）
```

模型视角的同一条链在 [《模型全景》01-3：从 PyTorch 调用到 HCCL](../model/01-landscape/03-pytorch-to-hccl.md)——本章是它的工程完全体；读完后，[H01-7 调用链走读](./hccl-source/07-allreduce-call-chain.md) 的入口就亮了。

---

[开始单元 03-1 →](./03-pytorch-to-hccl/01-torch-npu-bridge.md)
