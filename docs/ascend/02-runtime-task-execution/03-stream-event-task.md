# 单元 02-3｜Stream、Event、Task 与异步执行

> 所属章节：[第 2 章｜Runtime 与任务执行](../02-runtime-task-execution.md)

::: info 本单元目标
读完后，你能够说出 **Stream 的队列语义、Event 的打点/等待用法与四种同步层次**（SynchronizeStream/SynchronizeDevice/Event 同步/无同步），并用双 Stream + Event 编排"搬运与计算重叠"，说明每一步谁在等谁。
:::

## 先记住 3 个结论

1. **Stream 是 Device 侧的 FIFO 任务队列**：同一条 Stream 内任务严格保序；下发（launch）只是入队，Host 立即返回——"下发 ≠ 执行"。
2. **Event 是打点与等待的原语**：`aclrtRecordEvent` 在某条 Stream 上打一个时间点，`aclrtStreamWaitEvent` 让另一条 Stream 等到这个点再继续——跨 Stream 依赖的唯一正道。
3. **多 Stream 是并行的钥匙，也是数据竞争的入口**：两条 Stream 各自保序，但互相不知道对方在动哪块内存——共享数据必须用 Event 显式编排。

## 1. 从通用原理到昇腾 API

[《单卡执行系统》第 7 章](../../device/07-stream-event-async.md)已经建立了通用模型（单行道、依赖、同步），本单元做平台落地。昇腾 Device 侧的"任务"有三类：

| 任务类型 | 例子 | 下发接口 |
| --- | --- | --- |
| 计算任务 | 算子 Kernel | aclnn 两段式（02-4） |
| 搬运任务 | H2D/D2H/D2D 拷贝 | `aclrtMemcpyAsync` |
| 通信任务 | HCCL 集合通信 | HCCL 接口（挂到通信 Stream） |

它们都是"任务"，都在 Stream 上排队——**这就是计算与通信能在 Runtime 层协同的原因**，也是本章在 HCCL 主线上的价值。

## 2. Stream：创建、销毁与默认流

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
HCCL 集合通信调用同样接收一个 Stream 参数：通信任务排在该 Stream 上，与计算任务的依赖关系由 Stream/Event 表达。`torch_npu` 里通常为通信单独建 Stream，再与计算 Stream 用 Event 编排（第 3 章展开）。
:::

## 3. Event：打点与等待

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

### 3.1 经典编排：搬运与计算重叠

双 Stream 流水（通用原理见 [重叠的艺术](../../device/07-stream-event-async/04-overlap.md)）：

```text
computeStream：  [算子 batch1]···········[等event2][算子 batch2]
copyStream：    [等event1]              [拷 batch2 数据 H2D][record event2]
                          ▲ record event1（batch1 数据已拷完）
```

每一方"先等对方上一轮的完成点，再动共享内存"——**依赖正确性与重叠执行同时成立**。把"算子"换成"HCCL 通信"、把"拷贝"换成"反向梯度计算"，就是训练循环里通信计算重叠的骨架（回扣 [《集合通信》第 5 章](../../collective/05-topology-hierarchical-overlap/04-overlap-practice.md)）。

## 4. 四种同步层次

| 层次 | API | 语义 | 用法 |
| --- | --- | --- | --- |
| Stream 同步 | `aclrtSynchronizeStream(stream)` | Host 阻塞到该队列全部完成 | 最常用：下发完一批任务后等结果 |
| Device 同步 | `aclrtSynchronizeDevice()` | Host 阻塞到所有队列全部完成 | 兜底/退出前 |
| Event 同步 | `aclrtSynchronizeEvent(event)` | Host 只等到某个点 | 精细化等待 |
| 无同步 | —— | Host 继续跑，Device 自己排队 | 需要正确性自负的异步编程 |

**异步的阴暗面**：Device 侧的错误（Kernel 崩溃、非法地址）在**发生时**不会立刻出现在调用点，而是在后续某个同步点才抛出——所以异步报错的堆栈往往"驴唇不对马嘴"（排障思路见 [常用命令速查：异步报错定位](../cann常用命令.md)）。

## 5. 自测题

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

## 本单元小结

- **Stream**：Device 侧 FIFO 队列，同流保序、跨流不保序；通信任务也是 Stream 上的任务；
- **Event**：Record 打点 + StreamWait 等待 = 跨 Stream 依赖的正道；双 Stream 重叠是通用骨架；
- **同步四层次**：Stream / Device / Event / 无同步，按需选择；
- **阴暗面**：异步报错延迟浮出水面，排障要有"时间线思维"；
- **主线钩子**：HCCL 通信与计算任务的协同 = 本章 Stream/Event 语义的直接应用（H01-4 的 Notify 同族）。

## 参考资料

- [《单卡执行系统》第 7 章：Stream、Event 与异步执行](../../device/07-stream-event-async.md)
- [《集合通信》第 5 章：通信计算重叠](../../collective/05-topology-hierarchical-overlap/04-overlap-practice.md)
- [HCCL 源码专题 H01-4：通信原语与同步机制](../hccl-source/04-primitives-and-sync.md)
- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)

---

下一单元将进入 **02-4：一次算子调用怎样被下发和完成**。

[返回第 2 章 →](../02-runtime-task-execution.md)
