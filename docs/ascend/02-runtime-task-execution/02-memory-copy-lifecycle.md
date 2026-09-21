# 单元 02-2｜设备内存、数据搬运与生命周期

> 所属章节：[第 2 章｜Runtime 与任务执行](../02-runtime-task-execution.md)

::: info 本单元目标
读完后，你能够说出 **Host 与 Device 两个地址空间的边界、`aclrtMalloc`/`aclrtMemcpy` 一族 API 的分工与生命周期规则**（申请/释放配对、拷贝三方向、同步与异步搬运的区别），并解释 `.npu()`/`.cpu()` 背后发生了什么。
:::

## 先记住 3 个结论

1. **两个地址空间，指针不能混用**：Host 指针解引用不到 Device 内存，反之亦然——数据过桥只有一条路：`aclrtMemcpy`。
2. **搬运比计算贵**：HBM 带宽再高也是"过桥"；`H2D → 计算 → D2H` 的完整循环里，搬运常常占大头（回扣 [《单卡执行系统》第 3 章](../../device/03-memory-hierarchy.md)）。
3. **同步拷贝卡 Host，异步拷贝挂 Stream**：`aclrtMemcpy` 阻塞到搬完为止；`aclrtMemcpyAsync` 只是入队——省下的时间要靠 Stream 同步兜底。

## 1. 两个地址空间

```text
   Host（CPU）                        Device（NPU）
 ┌──────────────┐                  ┌──────────────────┐
 │ malloc 的内存  │   ══aclrtMemcpy═▶ │ HBM（Global Mem） │
 │ hostPtr      │   ◀══aclrtMemcpy═ │ devPtr            │
 └──────────────┘                  └──────────────────┘
        CPU 直接解引用 devPtr → 崩溃/脏数据
```

Device 内存就是第 1 章说的 **Global Memory（HBM）**——算子的输入输出都住在这里；AI Core 干活时还要从 Global Memory 搬进片上 Local Memory（那是 Ascend C 的故事，第 4 章）。

::: tip 为什么设计成两个空间？
隔离带来高性能：HBM 带宽按大规模并行设计，Host 内存按通用性设计；DMA 引擎在两者之间异步搬运，搬运期间 CPU 可以继续干别的。隔离的代价就是"过桥必须显式"。
:::

## 2. 核心 API 一览

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

## 3. 生命周期与两条工程规则

**规则一：申请与释放严格配对。**

```c
void *devPtr = nullptr;
aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);   // 申请
// ……干活……
aclrtFree(devPtr);                                        // 释放
// 且 aclrtFree 必须发生在 aclrtResetDevice 之前（02-1 铁律）
```

Device 显存不是进程一退出就必然归还——训练挂掉后显存不释放，先查有没有残留进程（`npu-smi info` 看占用，配合 [常用命令速查](../cann常用命令.md)第 4/7 节）。

**规则二：普通内存 vs 锁页内存。**

- `malloc` 的普通 Host 内存可能被操作系统换页；DMA 搬运时"页不在原地"会导致失败或退化成两段搬运；
- `aclrtMallocHost` 申请**锁页（page-locked）内存**，操作系统承诺不换页——异步搬运的安全选择，代价是占用 Host 物理内存且申请释放更贵；
- 经验法则：**频繁/异步搬运的缓冲区用锁页内存，一次性小数据无所谓**。

## 4. 同步拷贝 vs 异步拷贝

```c
// 同步：Host 在这一行等到搬完才继续
aclrtMemcpy(devPtr, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE);

// 异步：任务入队，Host 立即继续
aclrtMemcpyAsync(devPtr, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
// 危险：此刻数据还没到！任何依赖 devPtr 的操作必须排在同一 stream 的后面，
//      或先 aclrtSynchronizeStream(stream)
```

回扣 [《单卡执行系统》第 7 章](../../device/07-stream-event-async.md)的核心命题——**Host 跑得比 Device 快**：异步拷贝省下的 Host 时间是真实的，但"数据何时就绪"的责任转移给了你。两条安全线：

1. **同 Stream 保序**：依赖 `devPtr` 的算子下发到**同一条 Stream**，天然排在拷贝之后；
2. **显式同步**：跨 Stream 依赖时，用 Event 或 `aclrtSynchronizeStream` 兜底（02-3 展开）。

## 5. 与 PyTorch 的关系

```python
x = torch.randn(1024, 1024)
y = x.npu()    # 背后：aclrtMalloc + aclrtMemcpyAsync(H2D, 当前 stream)
z = y.cpu()    # 背后：aclrtMemcpyAsync(D2H) + 同步当前 stream
```

框架替你选好了 policy、锁页内存池和默认 Stream——但 Profiling 里看到的那段 `H2D/D2H` 时间，就是本章这些 API 在时间线上的具象。

## 6. 自测题

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

## 本单元小结

- **边界**：Host/Device 两个地址空间，`aclrtMemcpy` 是唯一的桥，方向写反必炸；
- **生命周期**：Malloc/Free 严格配对，且在 ResetDevice 之前；锁页内存是异步搬运的安全垫；
- **同步语义**：同步拷贝卡 Host、异步拷贝挂 Stream，"就绪责任"随异步转移；
- **框架映射**：`.npu()`/`.cpu()` 就是 H2D/D2H 拷贝在 Profiling 里的样子；
- **性能直觉**：搬运常常比计算贵——能少过桥就少过桥。

## 参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [《单卡执行系统》第 3 章：存储层次与数据搬运](../../device/03-memory-hierarchy.md)
- [《单卡执行系统》第 7 章：异步执行](../../device/07-stream-event-async/01-async-execution.md)
- [CANN 常用命令速查：NPU 状态与显存](../cann常用命令.md)

---

下一单元将进入 **02-3：Stream、Event、Task 与异步执行**。

[返回第 2 章 →](../02-runtime-task-execution.md)
