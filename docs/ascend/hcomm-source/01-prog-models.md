# 单元 1｜编程模型：通信、并发与拓扑

> 所属课程：[HCOMM 源码学习](../hcomm-source.md) · 第 1 单元（共 7 单元）

::: info 本单元目标
读完后，你能够解释 HCOMM 的**三个编程模型**——通信模型（通信内存/Endpoint/Channel 与两种语义）、并发模型（Thread 及其引擎落地）、拓扑模型（rank graph 层级）——并说出它们如何互相咬合。
:::

## 先记住 3 个结论

1. **通信模型三概念**：通信内存（注册给通信域，可被域内成员访问）+ Endpoint（通信端口，含地址与协议）+ Channel（基于 Endpoint 对建立的通道，创建时两端同步并交换内存信息）。
2. **两种通信语义**：网络语义（经 Channel 读写远端/同步）与内存语义（远端内存映射到本地地址空间，本地拷贝直接访问）——同一对 Endpoint，两种用法。
3. **并发模型 = Thread 抽象**：通信任务绑定 Thread，Thread 间用 Notify 同步；不同引擎下 Thread 落地不同——AI CPU+TS/Host CPU+TS 是 Stream+NPU Notify 寄存器，AIV 是 Vector Core+Device 内存。

## 1. 通信模型：名词表

| 概念 | 定义 | 关键细节 |
| --- | --- | --- |
| **通信内存** | 通信成员上注册给通信域的一块内存 | 域内其他成员可访问；建 Channel 时两端交换内存信息 |
| **Endpoint** | 与其他通信对象通信时使用的端口（如网卡 NetDevice） | 含地址与协议属性；一个 Endpoint 可含多个物理口（Bonding）；一个对象可多个 Endpoint |
| **Channel** | 本端与远端基于特定 Endpoint 对建立的通道 | 一对 Endpoint 可建多个 Channel；创建需**两端同步调用**；可查远端内存（地址与大小） |

### 1.1 网络语义：Channel 打开后的样子

![网络语义通信模型（图源：HCCL 官方文档）](/images/cann/hccl/official/semantic-communication.png)

以 RoCE 协议为例，Channel 内部有三样东西：

- **本端 Channel ↔ 远端 Channel 一一对应**——通信的入口；
- **QP（Queue Pair）**：Channel 关联一个或多个 QP，Channel 建立时 QP 建链；
- **Notify 实例**：Channel 包含多个 Notify（创建时指定数量），本端可按**序号**向远端 Channel 的某个 Notify 发同步信号，也可以在本地某个 Notify 上等远端的信号。

### 1.2 内存语义：映射即通道

![内存语义通信模型（图源：HCCL 官方文档）](/images/cann/hccl/official/memory-semantic-model.png)

内存语义下，**Channel 的建立只表示"内存映射启用"**，数据面不一定走这个 Endpoint（用哪个网络口由映射机制决定）；开发者拿到的是"远端内存映射到本地后的地址"，用本地拷贝接口就能跨节点搬数据。两种语义怎么选——回扣 [《集合通信》01 章](../../collective/01-collective-semantics-cost.md#_01-2-六个核心原语的语义卡片)的网络/内存语义之分，HCOMM 把它们做成了同一套 Channel 的两种用法。

## 2. 并发模型：Thread 抽象

通信算子由多个通信任务组成，无资源冲突的任务应并发执行。HCOMM 的抽象：

```text
并发单元：Thread —— 通信任务与 Thread 绑定，不同 Thread 的操作并发执行
同步手段：Thread 可含多个 Notify 实例 —— Thread 间收发同步信号
```

**Thread 的引擎落地**（本单元最重要的表）：

| 引擎 | 并发实体 | 同步实现 | Thread 是什么 |
| --- | --- | --- | --- |
| AI CPU+TS / Host CPU+TS | Stream | NPU Notify 寄存器 | Stream + Notify 的封装抽象 |
| AIV | Vector Core | Device 内存 | AIV 核 + Device 内存的封装抽象 |

::: tip 三个模型怎么咬合
**拓扑模型**（rank graph，[HCCL 源码 2](../hccl-source/02-comm-domain-rank-graph.md) 已详述七概念）回答"谁和谁、经哪个 Endpoint 互连"——**通信模型**据此建 Channel——**并发模型**决定任务在哪个 Thread 上跑、靠什么同步。控制面查拓扑 → 建 Channel/Thread（名词），数据面用原语在 Thread 上操作 Channel（动词，单元 3）。
:::

## 3. CCU 通信模型（预告）

CCU（Collective Communication Unit）的通信模型类似网络语义，但交换的是 **CcuBuffer**（片上缓存分片）与对端内存；Channel 除 Notify（对应 CCU 同步寄存器）外还有 **Variable**（对应 CCU 通用寄存器，用于与对端同步数据）。完整展开在 [单元 5](05-engine-comparison.md)。

## 4. 自测题

1. 通信模型三概念分别是什么？Channel 创建时发生哪两件事？
2. 网络语义与内存语义的本质区别是什么？
3. RoCE 场景下 Channel 内部有哪三样东西？
4. 并发模型的并发单元和同步手段分别是什么？
5. AIV 引擎下 Thread 对应什么物理实体？

::: details 自测答案

1. 通信内存、Endpoint、Channel；创建时两端同步调用，且本端注册的内存信息与远端交换。
2. 网络语义经 Channel 读写远端；内存语义把远端内存映射进本地地址空间，用本地拷贝操作访问——前者面向通用协议，后者依赖硬件映射能力。
3. 一一对应的远端 Channel 关联、一个或多个 QP（建链）、多个 Notify 实例（按序号收发同步信号）。
4. 并发单元是 Thread（任务与 Thread 绑定）；同步手段是 Thread 内的多个 Notify 实例（Thread 间收发信号）。
5. Vector Core；同步用 Device 内存实现（Thread = AIV 核 + Device 内存的封装）。

:::

## 本单元小结

- 通信模型：内存/Endpoint/Channel 三名词 + 网络/内存两种语义；
- 并发模型：Thread 抽象，引擎决定落地（Stream+Notify 寄存器 或 Vector Core+Device 内存）；
- 拓扑模型：rank graph 供查询，驱动 Channel/Thread 的创建；
- 三模型咬合：拓扑（能连谁）→ 通信（怎么连）→ 并发（怎么跑）。

## 参考资料

- [编程模型与概念：通信模型](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/prog_models_concepts/comm_model.md)
- [编程模型与概念：并发模型](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/prog_models_concepts/concurrency_model.md)
- [编程模型与概念：拓扑模型](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/prog_models_concepts/topology_model.md)

---

下一单元进入 **[2｜控制面走读：通信域的一生](02-control-plane-communicator.md)**。

[返回课程导学 →](../hcomm-source.md)
