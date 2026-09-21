# 单元 03-3｜Stream 与 Notify 的桥接

> 所属章节：[第 3 章｜从 PyTorch 走向 HCCL](../03-pytorch-to-hccl.md)

::: info 本单元目标
读完后，你能够说出 **框架流怎样传进 HCCL、Event 与 Notify 各管哪一层同步、通信计算重叠的完整闭环**——两个世界的异步体系如何缝合。
:::

## 先记住 3 个结论

1. **流是缝合线**：`torch.npu.Stream` 底层就是 `aclrtStream`（[第 2 章 02-3](../02-runtime-task-execution/03-stream-event-task.md)）——ProcessGroupHCCL 把流对象原样传给 HCCL，通信任务挂上这条队列异步执行。
2. **两层同步，各管各的**：框架的 `Event`（record/wait）编排**计算流之间**的依赖；HCCL 内部的 **Notify**（[H01-4](../hccl-source/04-primitives-and-sync.md)）负责**设备间通信的握手**——名字不同、层次不同。
3. **重叠闭环三方各出一块**：框架分流（计算流/通信流）、通信库挂流（不占计算单元）、硬件并行（链路与 AI Core 独立）——缺一环都重不起来。

## 1. 流的传递：一条线穿三层

```text
torch.npu.Stream (Python)
  ↕ 同一对象
c10/npu 的 Stream 封装 (C++)
  ↕ 底层句柄
aclrtStream (Runtime)          ← 第 2 章的"任务单行道"
  ↕ 作为参数传入
HcclAllReduce(..., stream)     ← 通信任务挂上同一条单行道
```

关键推论：**`HcclAllReduce` 传哪个流，通信就排在哪个流的队列里**——

- 传**当前流**：通信与计算严格保序（同流 FIFO）——简单正确，但无重叠；
- 传**独立通信流**：通信与计算并行——需要 Event 编排依赖（[《单卡执行系统》07-4](../../device/07-stream-event-async/04-overlap.md) 的三条件）。

DDP 的选择是后者：反向 hook 触发时把桶的 AllReduce 发到通信流，与剩余层的反向计算重叠（[《并行策略》05-4](../../parallel/05-ddp/04-overlap.md)）——**那条"通信流"就是 torch_npu Stream，最终以 aclrtStream 的身份进入 HCCL**。

## 2. Event 与 Notify：两层同步不要混

| | 框架 Event | HCCL Notify |
| --- | --- | --- |
| 属于 | torch.npu / Runtime | HCCL 内部机制 |
| 编排什么 | **同一设备内**流与流的依赖（record/wait） | **跨设备**通信参与方的握手 |
| 你直接用吗 | 用（`record_stream`/`wait_stream`） | 不用（通信库内部） |

一次跨流 AllReduce 的完整同步图：

```text
计算流:  [反向算梯度]──record(e)──............[wait(e2) 用结果]
通信流:        └──wait(e)──[AllReduce on stream]──record(e2)
                                   └ HCCL 内部：各 rank 的 Notify 握手 → 数据交换
```

外圈（Event）是框架看得见的编排；内圈（Notify）是 HCCL 在设备侧让 N 个 rank 步调一致的机制（[H01-4](../hccl-source/04-primitives-and-sync.md)）——**外圈管"什么时候能开始"，内圈管"大家到齐没有"**。

## 3. 自测题

先用自己的话回答，再展开答案。

1. `torch.npu.Stream` 与 `aclrtStream` 是什么关系？
2. `HcclAllReduce` 的 stream 参数决定什么？传当前流与传独立流各有什么后果？
3. DDP 的通信流在三层各对应什么对象？
4. Event 与 Notify 分别编排什么？为什么说"层次不同"？
5. 重叠闭环的三方各贡献什么？

::: details 自测答案

1. 同一条队列的两个身份：Python 封装 ↔ Runtime 句柄——torch.npu.Stream 底层就是 aclrtStream。
2. 决定通信任务挂哪条队列：当前流 = 与计算保序但无重叠；独立通信流 = 可重叠但需 Event 编排依赖。
3. torch.npu.Stream（Python）→ C++ 流封装 → aclrtStream（传给 HCCL）。
4. Event 编排同设备内流间依赖（框架可见）；Notify 是 HCCL 跨设备 rank 间握手（库内部）——前者管开始时机，后者管参与方到齐。
5. 框架分流、通信库把任务挂流不占计算单元、硬件链路与 AI Core 独立。

:::

## 本单元小结

- 流是三层的缝合线：torch.npu.Stream = aclrtStream = HcclAllReduce 的 stream 参数；
- 传哪个流决定通信排在哪条队列——重叠的开关就在这个参数；
- Event（框架层）与 Notify（通信库层）各管一层同步，勿混用概念。

## 参考资料

- [第 2 章 02-3：Stream、Event 与异步执行](../02-runtime-task-execution/03-stream-event-task.md)
- [《单卡执行系统》07-4：重叠的艺术](../../device/07-stream-event-async/04-overlap.md)
- [HCCL 源码专题 H01-4：通信原语与同步机制](../hccl-source/04-primitives-and-sync.md)

---

下一单元将进入 **03-4：一次 AllReduce 的完整旅程**。

[返回第 3 章 →](../03-pytorch-to-hccl.md)
