# 单元 H01-3｜通信域、Rank 与 RankGraph

> 所属专题：[HCCL 源码学习](../hccl-source.md)

::: info 本单元目标
读完后，你能够解释 **通信域、Rank、RankGraph** 三者的关系，记住 **Node / Endpoint / Edge / Link / netLayer / Fabric / TopoInstance** 七个拓扑概念，并理解"Edge → Link → Channel"的递进链。
:::

## 先记住 3 个结论

1. **通信域是执行上下文，Rank 是成员，RankGraph 是"谁和谁怎么连"的图模型**——三者合起来构成集合通信的控制面核心。
2. **RankGraph 用分层抽象适配真实集群**：Server 内是 Layer0（HCCS 直连，快），Server 间是 Layer1（RoCE 经交换机，慢），通信质量随层级增加而递减。
3. **Edge 描述"谁和谁连"，Link 描述"怎么建链"，Channel 才是"真正可用的数据通道"**——Channel 由 Link 实例化。

## 1. 从一个问题出发

AllReduce 的输入里没有任何网络信息——只有 buffer、count、数据类型和通信域句柄。那么 HCCL 必须自己回答：

> 这 64 个 rank 分布在 8 台服务器上，谁和谁之间有直连？走什么协议？带宽多少？

如果每次调用都临时探测，开销不可接受。所以答案是：**建域时一次性建图，之后所有算子共享这张图**。这张图就是 RankGraph，它由 HCOMM 的 `rank_graph/` 模块管理（H01-2 的域管理层）。

## 2. 三个核心概念

集合通信模型由三个核心概念构成，它们的关系如下图：

![集合通信模型：通信域、Rank 与 RankGraph 的关系（图源：HCCL 官方文档）](/images/cann/hccl/official/coll-comm-model.svg)

| 术语 | 一句话解释 | 主要对应硬件 |
| --- | --- | --- |
| **集合通信域** | 集合通信执行的上下文，管理参与通信的实体和资源 | 多个 NPU 组成 |
| **Rank** | 通信域中的成员，拥有唯一 Rank ID（从 0 开始） | 一个 NPU |
| **RankGraph** | Rank 间的通信关系图，描述"谁和谁怎么连" | 网络拓扑 |

通信域的生命周期决定了控制面的节奏：**创建域（建图、分配资源）→ 反复执行算子（只读图）→ 销毁域（释放资源）**。这也解释了 H01-2 的约束"同一 NPU 上需要串行创建多个通信域"——建图是重操作。

## 3. RankGraph 的七个概念

这是本单元最密的表格，建议先看官方的拓扑模型图建立整体印象，再配合下面的类比读三遍：

![RankGraph 拓扑模型：Node、Endpoint、Edge、Link 与分层结构（图源：HCCL 官方文档）](/images/cann/hccl/official/topo-concepts.svg)

| 概念 | 一句话解释 | 类比 |
| --- | --- | --- |
| **Node** | 图中的节点，分为通信实体和 Fabric | 通信实体 = 带网口的 NPU；Fabric = 交换机组 |
| **Endpoint** | Node 的通信设备（逻辑概念）；一个 Node 可有多个 Endpoint，一个 Endpoint 映射一个物理端口，物理端口可被多个 Endpoint 共享 | NPU 上的网卡口 |
| **Edge** | Node 间的连接关系，两端是 Endpoint | 网线，两端插在 NPU 网口上 |
| **Link** | 从 Edge 提取的两个通信实体间可建链信息（含两端 Endpoint + 协议） | 两个 NPU 之间可建链的路径描述 |
| **netLayer** | 拓扑层级；通信质量逐层增加而递减 | Server 内 HCCS 直连 = Layer0，Server 间 RoCE = Layer1 |
| **Fabric** | 网络交换/路由组的抽象；与它相连的通信实体两两互通；同一网络层次不存在相连的两个 Fabric | 一台交换机让插在上面的 NPU 互通 |
| **TopoInstance** | 每层内的拓扑实例 | 同机房 8 个 NPU 卡组成一个 1DMesh 实例 |

::: warning 两个官方强调的细节
1. **与 NCCL 的命名差异**：RankGraph 的 Edge/Link 与 NCCL 的 Link/Path 名称对应不同——HCCL 的 Edge ≈ NCCL 的 Link，HCCL 的 Link ≈ NCCL 的 Path。读 NCCL 资料做对照时不要直接画等号。
2. **Bonding 口对软件透明**：一个物理端口可以是 Bonding 口（硬件控制），软件不感知——这就是为什么 Endpoint 是"逻辑概念"而数量关系是"多对多"。
:::

### 3.1 层级结构图

把七个概念放进一个两层集群：

```text
            Layer 1（Server 间，经 Fabric/交换机，RoCE）
  ┌────────────────────────────────────────────────┐
  │   Server A                          Server B   │
  │  ┌─────────┐   Layer 0（Server 内） ┌─────────┐ │
  │  │ NPU 0-3 │◄────── HCCS 直连 ────►│ NPU 4-7 │ │
  │  └─────────┘      TopoInstance     └─────────┘ │
  └────────────────────────────────────────────────┘
```

- 集群天然分层（Layer），每层内有拓扑实例（TopoInstance）；
- 拓扑类型包括 **Fullmesh、1DMesh、CLOS、Ring** 等；
- 超节点组网（灵衢总线）同样被这套抽象覆盖。

### 3.2 递进链：Edge → Link → Channel

```text
Edge（谁和谁连）
  ↓ 提取可建链信息（两端 Endpoint + 协议）
Link（怎么建链）
  ↓ 实例化
Channel（怎么通信：两端 Endpoint + 协议 + N 个 Notify）
```

Channel 不再是纯拓扑概念，而是**真正可用的数据通道**——它属于 H01-4 的数据面。这条递进链是连接本单元（控制面建图）与下一单元（数据面搬运）的桥梁，也是理解 `op_common/topo/`（"rankGraph 拓扑信息适配"）存在的理由：算子层拿到的拓扑描述需要转换成自己可用的形态。

## 4. 算法视角：为什么拓扑决定算法

H01-6 会展开算法，这里先给一个直觉，说明为什么要把拓扑建得这么细：

| 场景 | 拓扑事实 | 算法倾向 |
| --- | --- | --- |
| 单机 8 卡 | 全互联 HCCS（Layer0 带宽极高） | Mesh / 两段式，Server 内重数据量 |
| 两机 16 卡 | 两台 8 卡机 + 8 条跨机链路 | 分级：机内 ReduceScatter → 机间 AllReduce → 机内 AllGather |
| 大规模非 2 幂 | 交换机胖树，链路同质 | Ring（大数据）/ NHR（小数据） |

一句话：**RankGraph 把"通信质量随层级递减"这个物理事实变成算法可消费的数据结构**。selector（H01-7）做选择时，读的就是它。

## 5. 高频术语速查

来自官方"相关概念"术语表，读码与看日志时最常撞见：

| 缩写 | 全称 | 说明 |
| --- | --- | --- |
| HCCS | Huawei Cache Coherence System | CPU/NPU 间高速互联（Server 内） |
| HCCP | Huawei Collective Communication adaptive Protocol | 集合通信适配协议，向上屏蔽具体通信协议差异 |
| RoCE | RDMA over Converged Ethernet | 跨以太网的 RDMA |
| QP | Queue Pair | RDMA 核心通信单元（SQ+RQ） |
| SDMA | System Direct Memory Access | 系统直接内存访问 |
| AIV | AI Core 中的 Vector Core | 也可作为通信引擎（H01-5） |
| TS | Task Scheduler | 任务调度器 |
| CCU | Collective Communication Unit | 集合通信加速单元（H01-5） |

## 6. 自测题

1. 通信域、Rank、RankGraph 分别解决什么问题？
2. Endpoint 为什么是逻辑概念？它和物理端口是什么关系？
3. Edge、Link、Channel 三者的递进关系是什么？
4. netLayer 表达的物理事实是什么？它如何影响算法选择？
5. HCCL 的 Edge/Link 与 NCCL 的什么概念对应？

::: details 自测答案

1. 通信域是执行上下文（管理成员与资源）；Rank 是域内拥有唯一 ID 的成员（通常一个 NPU）；RankGraph 是 rank 间通信关系图，描述"谁和谁怎么连"。
2. 因为物理端口可以是硬件控制的 Bonding 口，软件不感知；一个 Node 可有多个 Endpoint，一个 Endpoint 映射一个物理端口，而一个物理端口可被多个 Endpoint 共享。
3. Edge 描述"谁和谁连"（两端 Endpoint 的连接关系）；Link 从 Edge 提取"怎么建链"（两端 Endpoint + 协议）；Channel 由 Link 实例化，是含同步 Notify 的真正数据通道。
4. 通信质量逐层增加而递减（Server 内 Layer0 直连最快，Server 间 Layer1 经交换机较慢）；算法选择器据此把大数据量放在高带宽层级、把小消息高频交互放在低延迟路径，例如分级 AllReduce。
5. HCCL 的 Edge ≈ NCCL 的 Link；HCCL 的 Link ≈ NCCL 的 Path（名称对应不同，不能直接画等号）。

:::

## 本单元小结

- **控制面核心**：建域 = 建 RankGraph；算子执行只读图；
- **七个概念**：Node / Endpoint / Edge / Link / netLayer / Fabric / TopoInstance，用"网线—交换机—机房"类比记忆；
- **递进链**：Edge（谁连）→ Link（怎么建）→ Channel（怎么通信）；
- **价值**：拓扑的物理事实（分层、质量递减）被编码成数据结构，供算法选择消费；
- **代码落点**：hcomm 仓 `coll_communicator_mgr/rank_graph/`、hccl 仓 `src/ops/op_common/topo/`。

## 参考资料

- [HCCL & HCOMM 软件架构简介：集合通信模型（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [相关概念（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/concepts.md)
- [集群信息配置（rank table）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/cluster_info_config/README.md)

---

下一单元进入 **H01-4：通信原语与同步机制**。

[返回专题导学 →](../hccl-source.md)
