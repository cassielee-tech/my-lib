# 单元 3｜数据面走读：原语与资源

> 所属课程：[HCOMM 源码学习](../hcomm-source.md) · 第 3 单元（共 7 单元）

::: info 本单元目标
读完后，你能够把 **`hcomm_primitives.h` 的 30 个数据面动词**归入四个家族并解释命名规律（OnThread/Nbi/WithNotify），用真实签名写出 Send/Receive 的同步时序，并说出 `base_comm` 源码的模块落点。
:::

## 先记住 3 个结论

1. **数据面 = 四个家族的动词**：本地操作（LocalCopy/LocalReduce）、Thread 同步（ThreadNotify Record/Wait）、Channel 通信（Read/Write/ReadReduce/WriteReduce 及变体）、批量与缓存（BatchMode/TaskCache）。
2. **命名即语义**：`OnThread` 后缀 = 操作绑定到某 Thread 执行；`Nbi` = 非阻塞立即返回；`WithNotify` = 搬运与通知原子组合（省一次握手）。
3. **HCCL Buffer 是搬运的中转站**：每个通信域管理的 Device 锁页内存（默认 200 MB）——异步通信要求输入数据在执行时地址固定，所以先拷入 Buffer 再搬运。

## 1. 原语四家族（真实签名）

以 `include/hcomm_primitives.h` 为准：

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

## 2. 用原语写同步：Send/Receive 时序

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

## 3. HCCL Buffer：为什么需要中转

通信任务是异步下发的——下发时用户输入指针还在，**真正执行时呢**？为避免悬空指针，数据先拷入通信域管理的**锁页内存**（HCCL Buffer，默认 200 MB）：

```text
用户输入 → [LocalCopy] → HCCL Buffer → [Read/Write/Reduce...] → 对端 HCCL Buffer → [LocalCopy] → 用户输出
```

超过 200 MB 的输入按块切分循环处理——这与 [《集合通信》05-3 的 Chunk](../../collective/05-topology-hierarchical-overlap.md#_05-3-chunk-与-channel-拆小、铺满) 是同一思想在框架侧的再现。

## 4. base_comm 源码落点

| 模块 | 内容 |
| --- | --- |
| `primitives/api_c_adpt/` | C 接口适配（`*_c_adpt.cc`），30 个动词的实现入口 |
| `primitives/aicpu/` | AICPU 侧原语与**任务缓存**（TaskCache：同 tag 任务描述复用，Host 免重复编排） |
| `resources/endpoints/`、`endpoint_pairs/` | Endpoint 与端对管理（控制面名词） |
| `resources/reged_mems/` | 注册内存（通信内存的家） |
| `resources/hccp/` | **HCCP 自研协议**：HCOMM 自己的集合通信协议栈 |
| `resources/ccu/`、`southbound_adpt/` | CCU 资源与南向（硬件/驱动）适配 |

## 5. 自测题

1. 数据面原语的四个家族是什么？各举一函数。
2. `OnThread`、`Nbi`、`WithNotify` 三个命名后缀分别什么含义？
3. Send/Receive 三步握手中 Notify 序号 0 和 1 分别承担什么方向？
4. 为什么需要 HCCL Buffer？默认多大？
5. HCCP 住在源码哪个目录？TaskCache 解决什么问题？

::: details 自测答案

1. 本地操作（HcommLocalCopyOnThread）、Thread 同步（HcommThreadNotifyRecordOnThread）、Channel 通信（HcommReadOnThread）、批量与缓存（HcommBatchModeStart）。
2. OnThread=操作绑定到某 Thread 执行；Nbi=非阻塞立即返回；WithNotify=搬运与通知原子组合。
3. 0 = 发送端→接收端的"数据就绪"线；1 = 接收端→发送端的"已读完"确认线。
4. 异步执行时用户指针可能失效，需先拷入固定地址的锁页内存；默认 200 MB，超限切块。
5. `src/base_comm/resources/hccp/`；TaskCache 让相同 tag 的任务描述免重复编排，降低 Host 开销。

:::

## 本单元小结

- 30 个动词、四个家族，命名后缀即语义；
- Send/Receive 三步握手 = 集合通信算法的最小组件；
- HCCL Buffer 解决异步悬空指针，200 MB 起切块；
- base_comm = 原语适配 + 资源管理 + HCCP 协议 + 南向适配。

## 参考资料

- [API 参考：数据面 API（cpu-cpu_ts-aicpu_ts）](https://gitcode.com/cann/hcomm/tree/master/docs/zh/api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts)
- [HCOMM 仓库：base_comm 源码目录](https://gitcode.com/cann/hcomm/tree/master/src/base_comm)
- [HCCL 源码 3：通信原语与同步机制](../hccl-source/03-primitives-and-sync.md)

---

下一单元进入 **[4｜AICPU 算子开发：七步流程](04-aicpu-op-dev.md)**。

[返回课程导学 →](../hcomm-source.md)
