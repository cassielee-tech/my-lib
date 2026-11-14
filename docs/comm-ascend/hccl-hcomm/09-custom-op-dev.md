# 单元 9｜自定义算子开发：AI CPU 七步流程

> 所属课程：[HCCL 与 HCOMM 源码学习](../hccl-hcomm.md) · 第 9 单元（共 12 单元）

::: info 本单元目标
读完后，你能够复述 **AI CPU 通信算子的七步开发/执行流程**，解释"**Kernel 下发在前、任务编排在后**"这句话的实现方式，并说出任务编排七步与资源复用机制（EngineCtx + Tag）——从本单元起，视角从"读内置算子"切换为"自己写算子"。
:::

前面八个单元回答的是"内置算子怎么跑"：selector 选名、executor 组装、template 落原语。但当内置算法在特定拓扑/数据量下不够快，或通算融合需要新的通信语义时，就需要**自己写一个扩展通信算子**——HCOMM 开放的 L2-res + L3-prim 接口正是为此准备（单元 1 的第一条开发路线）。本单元讲最通用的 AI CPU 路径，单元 10 讲 MC2 框架与官方样例，单元 11 讲实战路线图。

## 先记住 3 个结论

1. **两阶段分工**：Host 侧准备（构造参数 → 查拓扑 → 选算法 → 建/复用资源 → 下发 Kernel）+ AICPU 侧执行（Kernel 启动后反序列化上下文、动态编排任务）——与直觉相反，**编排在 Kernel 启动之后**。
2. **资源以"算子 + 算法 Tag"为粒度复用**：`HcclEngineCtxGet(tag, engine)` 命中则直接拿 Device Context，未命中才走完整创建（Thread/Notify/Channel/内存 + 序列化到 Device）——第二次调用起开销骤降。
3. **任务编排七步**：取 HCCL Buffer → 拷入 → 切分 → 前同步（主通知从）→ 数据搬运 → 后同步 → 拷出——每一步都对应单元 3 的某个原语家族。

## 1. 完整流程图（时序）

官方时序（文字化）：

```text
通信算子接口(Host)          HCOMM(Host API)           RTS                    AICPU Kernel / HCOMM(Device)
─────────────────────────────────────────────────────────────────────────────────────────────
构造算子参数
  │ 查询 rank、拓扑层级、可用链路 ──► 返回通信域与拓扑信息
  │ (可选) 按引擎/拓扑/数据量选算法
  │ HcclEngineCtxGet(tag, engine) ──► 命中：返回 Device Context
  │      └─ 未命中：创建 Context、申请控制/算法 Thread+Notify+Channel+内存、
  │         序列化资源上下文拷贝到 Device
  │ Host Thread 通知 AICPU 控制 Thread
  │ aclrtLaunchKernelWithConfig 下发 Kernel ──────► RTS 启动 Kernel
  │ Host Thread 等待 AICPU 完成通知                                    反序列化资源 Context
  │                                                                    HcommBatchModeStart(tag)
  │                                                                    控制Thread 等 Host 启动通知
  │                                                                    ExecOp(param, resCtx)
  │                                                                      编排 Thread 同步 / Channel 同步 / 数据搬运
  │                                                                    控制 Thread 通知 Host 完成
  │                                                                    HcommBatchModeEnd(tag)
◄─┴─ AICPU 执行完成
```

七个阶段职责（官方编号）：**①定义算子接口 ②查询拓扑信息 ③算法选择（单一实现可省）④创建资源 ⑤下发 Kernel ⑥任务编排 ⑦完成同步**。

## 2. 为什么"下发在前、编排在后"

因为 **AICPU 是通用核，任务编排本身是代码逻辑**：Kernel 启动后才能执行"按算法动态生成任务序列"的代码（数据量、rank 数在此时才知道）。对比：AIV 的编排是静态的（单元 4 的编排时机对比，展开反例）。

这个设计直接决定了**资源必须提前序列化**：Host 把 Thread/Notify/Channel/内存的上下文打包拷到 Device，Kernel 启动后反序列化——**控制面的产物（名词）打包快递给数据面（动词）使用**。

## 3. 资源复用：EngineCtx 与 Tag

```text
首次执行：创建 Context → 申请资源 → 序列化到 Device ──┐
                                                         ├ 同一个 (算子, 算法Tag) 后续执行
后续执行：HcclEngineCtxGet 直接命中 ───────────────────┘ 只需 lookup
```

配套的 `HcommBatchModeStart/End(tag)` 与 **TaskCache**（Lookup/Start/Execute/Clear）进一步把重复编排的开销摊薄——**热路径上，控制面几乎退化成查表**（[单元 1](01-architecture-layering.md)"控制面低频、数据面高频"的工程兑现；engineCtx 作为全系统缓存主干的完整机制——topoInfo/执行计划/回退记忆全挂其上——见 [单元 8](08-resources-dlsym.md)，HCCL 内置算子走的正是同一套）。

## 4. 任务编排七步（AICPU Kernel 内）

1. **取 HCCL Buffer**（本端通信内存，单元 3 的 200 MB 中转）；
2. **拷入**：`HcommLocalCopyOnThread` 把算子输入拷到 Buffer；
3. **切分**：输入超过 200 MB 则分块循环；
4. **前同步**：主 Thread `HcommThreadNotifyRecordOnThread` 通知从 Thread，从 Thread Wait；
5. **数据搬运**：`HcommReadOnThread` / `HcommReadReduceOnThread` 等按算法轮转；
6. **后同步**：从 Thread 通知主 Thread 完成；
7. **拷出**：Buffer 结果拷回算子输出内存。

每一步的原语都在单元 3 的四家族表里；主从 Thread 的同步正是 ThreadNotify（实体内），跨 rank 的搬运与确认是 ChannelNotify + Write/Read（跨实体）。

## 5. 快速上手：官方 Send/Receive 样例的调用序列

样例（[hccl 仓 examples/04_custom_ops_p2p](https://gitcode.com/cann/hccl/tree/master/examples/04_custom_ops_p2p)）的接口调用顺序，就是七步的落地：

```text
HcclGetRankId / HcclGetRankSize     → 步骤② 查拓扑
HcclThreadAcquire                   → 步骤④ 建 Thread 资源
HcclChannelAcquire                  → 步骤④ 建 Channel
HcclChannelGetHcclBuffer            → 步骤④ 取远端通信内存地址
HcommLocalCopyOnThread              → 步骤⑥ 编排：拷入
HcommChannelNotifyRecord/Wait...    → 步骤⑥ 编排：握手与搬运
```

这个样例的**源码级逐行走读**（五步法、建链的两种姿势）在 [单元 10](10-mc2-custom-ops.md) 展开——两单元对照阅读效果最好。

## 6. 自测题

1. 七步流程按顺序是什么？哪一步可以省略？
2. "Kernel 下发在前、任务编排在后"的原因是什么？
3. `HcclEngineCtxGet` 的参数与作用是什么？未命中时走什么路径？
4. 资源上下文为什么要"序列化到 Device"？
5. 任务编排七步中，前同步与后同步分别发生在谁和谁之间？

::: details 自测答案

1. 定义算子接口 → 查询拓扑 → 算法选择 → 创建资源 → 下发 Kernel → 任务编排 → 完成同步；算法选择在单一实现时可省。
2. AICPU 是通用核，编排逻辑是代码——只有 Kernel 启动后才能按运行时参数（数据量/rank 数）动态生成任务序列。
3. 参数 (tag, engine)：算子+算法 Tag 与引擎标识；命中返回 Device Context；未命中走完整创建（Thread/Notify/Channel/内存 + 序列化拷贝）。
4. Kernel 在 Device 侧执行，要用控制面创建的资源——把上下文打包带过去，启动后反序列化。
5. 前同步：主 Thread 通知从 Thread 启动；后同步：从 Thread 通知主 Thread 完成——同一实体内的 Thread 间 Notify。

:::

## 本单元小结

- 七步两阶段：Host 备好"名词"并下发，AICPU Kernel 启动后用"动词"动态编排；
- 编排在后的根因：AICPU 的编排是运行时代码；
- EngineCtx+Tag 与 TaskCache 把控制面摊薄成查表——与单元 8 的 engineCtx 主干同一套机制；
- 编排七步 = HCCL Buffer + 切分 + Thread/Channel 双同步 + 搬运；
- 官方 Send/Recv 样例 = 七步的最小可跑实现（源码走读见单元 10）。

## 参考资料

- [AI CPU 算子开发：总体流程（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/aicpu_comm_op_dev/overall_flow.md)
- [AI CPU 算子开发：任务编排（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/aicpu_comm_op_dev/task_sched.md)
- [快速上手：AI CPU 点对点算子（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/aicpu_quick_start.md)

---

下一单元进入 **[10｜MC2 自定义算子框架与官方样例](10-mc2-custom-ops.md)**。

[返回课程导学 →](../hccl-hcomm.md)
