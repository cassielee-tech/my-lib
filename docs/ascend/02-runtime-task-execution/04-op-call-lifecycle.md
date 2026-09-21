# 单元 02-4｜一次算子调用怎样被下发和完成

> 所属章节：[第 2 章｜Runtime 与任务执行](../02-runtime-task-execution.md)

::: info 本单元目标
读完后，你能够说出 **aclnn 算子接口的两段式结构**（`GetWorkspaceSize` 准备段 + 执行段）及其设计动机，并独立画出一次算子调用的 **Host/Device 双时间线**，解释气泡、等待与重叠的来源。
:::

## 先记住 3 个结论

1. **一次算子调用 = Host 侧纯准备 + 一次轻量入队**：准备（查描述、算 workspace、组 executor）发生在 Host，执行段把任务挂上 Stream 后立即返回——全程异步。
2. **aclnn 两段式**：`aclnnXxxGetWorkspaceSize(...)` 做无副作用的准备工作（可缓存、可并发），`aclnnXxx(workspace, wsSize, executor, stream)` 才真正入队执行。
3. **双时间线读性能**：Host 时间线与 Device 时间线各画一条，任务在中间"投递"——90% 的性能现象（气泡、等待、重叠）都能在这张图上指认。

## 1. 两段式结构

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

### 1.1 为什么要拆成两段？

| 设计点 | 收益 |
| --- | --- |
| 准备段**无副作用** | 同一组参数可以只准备一次，**executor 复用**（框架的算子调用大量重复） |
| 准备段不碰 Device | Host 侧可以**多线程并发**地为多个算子做准备 |
| workspace 大小显式给出 | 调用方统一管理设备内存（复用内存池），而不是每次暗地里 malloc |
| 执行段极轻 | 只是"把组好的任务挂上 Stream"，为整个 Eager 流水提速 |

::: tip workspace 是什么
算子执行时需要的**设备侧临时空间**（中间结果、tiling 缓冲等），大小由准备段算出。它是显存账本的一部分——《训练与推理系统》第 3 章 FSDP/显存优化的通用原理在这里同样适用。
:::

## 2. 双时间线全景

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

1. **气泡**：Host 准备慢（如 `GetWorkspaceSize` 串联执行太多），Device 队列空转——Eager 模式 Host 开销的来源（回扣 [《单卡执行系统》第 6 章](../../device/06-eager-graph-compilation.md)：图模式下沉正是为了消灭它）；
2. **等待**：`aclrtSynchronizeStream` 之后 Host 才能安全读 D2H 的结果——同步点是两条时间线的交汇；
3. **重叠**：H2D（第 N+1 批）与 Kernel（第 N 批）若在同一条 Stream 上天然流水；跨 Stream 才需要 Event 编排（02-3）。

## 3. 闭环检查清单

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

## 4. 从这里去向哪里

- **向上**：`torch_npu` 把这套样板包进了框架——`y = a + b`（npu tensor）背后就是"准备段 + 执行段"的循环（第 3 章展开）；
- **向旁**：HCCL 的 AllReduce 一样是"准备 + 入队通信 Stream"的结构（[H01-7 调用链走读](../hccl-source/07-allreduce-call-chain.md)）；
- **向工具**：Profiling（msprof）呈现的正是 Device 时间线上每个任务的起止——学会本章的双时间线，Profiling 火焰图/时间轴就是它的放大版（[常用命令速查：msprof](../cann常用命令.md)）。

## 5. 自测题

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

## 本单元小结

- **两段式**：准备段（无副作用、可缓存并发）+ 执行段（轻量入队）——昇腾单算子接口的统一形态；
- **双时间线**：Host 与 Device 各走各的，任务从 Host"投递"到 Device 队列，同步点是交汇；
- **气泡/等待/重叠**三大性能现象都能在双时间线图上指认；
- **经典 bug**：异步读结果不加同步——"读到旧内存"；
- **主线衔接**：torch_npu、HCCL、Profiling 全部建立在这张图上。

## 参考资料

- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [《单卡执行系统》第 6 章：Eager 与图编译](../../device/06-eager-graph-compilation.md)
- [HCCL 源码专题 H01-7：AllReduce 调用链走读](../hccl-source/07-allreduce-call-chain.md)
- [CANN 常用命令速查：msprof 性能采集](../cann常用命令.md)

---

下一单元将进入 **本章总结：用时间线解释计算、搬运与同步**。

[返回第 2 章 →](../02-runtime-task-execution.md)
