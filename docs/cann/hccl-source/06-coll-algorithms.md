# 单元 H01-6｜集合通信算法与代价模型

> 所属专题：[HCCL 源码学习](../hccl-source.md)

::: info 本单元目标
读完后，你能够用 **α-β-γ 代价模型**比较 **Ring / RHD / NHR / Pairwise** 等算法，解释**分级通信**为什么"先机内后机间"，并知道源码中算法选择发生在 selector、可由 `HCCL_ALG` 等环境变量干预。
:::

## 先记住 3 个结论

1. **算法差异的本质是通信步数与每步数据量的不同取舍**：Ring 步数 O(n−1) 但每步只传 1/n；RHD/NHR 步数 O(log n) 但每步数据量更大——前者吃带宽、后者吃延迟。
2. **分层拓扑必然导向分级通信**：把大数据量放在高带宽的 Server 内（Layer0），把跨 Server（Layer1）的传输压缩到最少。
3. **算法不是写死的**：selector 依据数据量、拓扑、规模自动选择，环境变量（如 `HCCL_ALG`）可人工干预——这是性能调优的第一抓手。

## 1. 代价模型：读算法文档的钥匙

官方算法文档的耗时公式全部基于 **α-β-γ 模型**：

| 符号 | 含义 | 对应什么 |
| --- | --- | --- |
| **α** | 每次通信的启动延迟 | 下发一次任务的固定开销 |
| **β** | 单位数据的传输时间（1/带宽） | 链路带宽 |
| **γ** | 单位数据的本地计算时间 | 本地 Reduce（如加法）开销 |

一个算法的总耗时 ≈ 步数 × α + 总数据量 × β +（规约类）总数据量 × γ。**比较算法就是比较这三个系数的加权**：小消息场景 α 主导（要少步数），大消息场景 β 主导（要总搬运量小、带宽利用高）。带着这把钥匙，下面的算法对比会非常自然。

## 2. Ring：环形接力

![Ring 拓扑：每张卡与左手卡、右手卡相连（图源：HCCL 官方文档）](/images/cann/hccl/official/ring-topology.png)

所有 NPU 连成环，每张卡有左手卡与右手卡，一个负责接收、一个负责发送：

![Ring 算法 AllReduce 流程：ReduceScatter 转一圈 + AllGather 转一圈（图源：HCCL 官方文档）](/images/cann/hccl/official/ring-principle.png)

- AllReduce 被拆成 **ReduceScatter（转一圈）+ AllGather（再转一圈）**；
- p 个节点需 **p−1 步**，每步交换 **1/p** 的数据；
- 时间复杂度 **O(n−1)**，适用于"星型"或"胖树"拓扑。

AllReduce 耗时：$2(p-1)\alpha + 2\frac{p-1}{p}n\beta + \frac{p-1}{p}n\gamma$

**读法**：步数 2(p−1) 很多——小消息时 α 被放大 p 倍，不划算；但每步只传 n/p，总搬运量 2n(p−1)/p ≈ 2n，且**任何时刻每张卡都在收发**，带宽利用充分——大数据量时接近链路极限。

## 3. RHD：递归折半-倍增

当规模增大（如 4K rank），Ring 的问题暴露：环太长、转太多次。RHD（Recursive Halving-Doubling）通过**递归折半及倍增**交换数据，以 5 个 rank（2²+1）为例的官方流程图：

![RHD 算法流程：先合并到 2 的整数次幂，再两两对半交换求和、两两拼接，最后还原（图源：HCCL 官方文档）](/images/cann/hccl/official/rhd.png)

- 通信对象每步翻倍/折半，步数 **⌈log₂N⌉**；
- 非 2 的整数次幂时先"合并"到最近 2 幂（例如 5 rank：先把 rank1 的数据并入 rank0 变成 4 rank，最后再还原回去），因此会引入额外通信步数；
- 适用于"星型"或"胖树"拓扑。

2 的整数次幂时 AllReduce 耗时：$2\log(p)\alpha + 2\frac{p-1}{p}n\beta + \frac{p-1}{p}n\gamma$

**与 Ring 对比**：α 项从 2(p−1) 降到 2log(p)，小消息大幅占优；代价是**每个通信阶段的对象在变化，链路也随之变化**——大流量场景可能引起交换机流量冲突，导致带宽下降。

## 4. NHR：非均衡层次环

RHD 的两个残余问题：非 2 幂规模时额外开销（出现"N−1 规模比 N 规模还慢"的怪现象）、链路变化引起冲突。NHR（Nonuniform Hierarchical Ring）针对这两点：

- 对 N 个节点构建 **N 棵生成树**，通过生成树构建最优通信关系；树深（步数）为 **⌈log₂N⌉**；
- 通过**重排数据片编号聚合发送**，保证地址连续的数据切片连续发送（2 幂时每步收发均为 1 份）；
- 最大通信流量集中在**物理位置相近**的节点间，减少流量冲突；
- 无论规模是否 2 幂都能充分利用链路；**小数据包场景进一步退化为只建 1 棵树**，减少网络数据包数量与芯片并发任务数。

rank size 为 4（2 的整数次幂）与 5（非 2 幂）时的通信过程对比：

![NHR 算法：rank size 为 4 时的通信过程，每步收发数据份数均为 1（图源：HCCL 官方文档）](/images/cann/hccl/official/nhr-4rank-flow.png)

![NHR 算法：rank size 为 5 时大部分数据切片可连续收发，仅少部分离散（图源：HCCL 官方文档）](/images/cann/hccl/official/nhr-5rank-flow.png)

**一句话记忆**：Ring 吃满带宽、RHD 压缩步数、NHR 兼顾步数与链路稳定性，且对非 2 幂友好。

## 5. 其他家族成员（混个脸熟）

| 算法 | 关键词 | 适用 |
| --- | --- | --- |
| **Mesh** | 全互联，一步完成（如 4 卡 HCCS 全互联） | 小规模全互联组网 |
| **NB**（Nonblocking） | 非阻塞分段 | 与流水/并行执行配合 |
| **Pairwise** | 两两成对交换，单卡一进一出 | **数据并行梯度同步**（等量交换场景最优） |
| **Pipeline** | 数据切块流水推进 | 与计算重叠 |
| **Star** | 中心节点聚合分发 | 特定拓扑/小规模 |
| **AHC** | 自研算法族 | 见官方算法文档 |

其中 Mesh 与 Pairwise 的拓扑最直观：

![Mesh 算法：所有 NPU 全互联，可一步完成数据交换（图源：HCCL 官方文档）](/images/cann/hccl/official/mesh.png)

![Pairwise 算法：两两成对交换，每张卡一进一出（图源：HCCL 官方文档）](/images/cann/hccl/official/pairwise.png)

不需要一次记住全部——读 `selector` 代码或排查性能时再回来查表。

## 6. 分级通信：分层拓扑的正确用法

H01-3 说"通信质量随层级递减"，算法层面的直接产物就是**分级通信**（hierarchical communication）。以 8 机 × 8 卡集群的 AllReduce 为例：

![AllReduce 分级通信流程：机内 ReduceScatter → 机间 AllReduce → 机内 AllGather（图源：HCCL 官方文档）](/images/cann/hccl/official/allreduce-hierarchical.png)

1. **Server 内执行 ReduceScatter**：先在 HCCS 高带宽域内归约切分；
2. **Server 间执行 AllReduce**：跨机只剩"每机一份"的小数据量，昂贵的 Layer1 链路被压到最少；
3. **Server 内执行 AllGather**：结果在机内高带宽扩散。

官方文档特别指出一个精妙处：AllReduce 的输出是完整归约结果，**不要求严格遵循 ReduceScatter + AllGather 的语义顺序**，因此可以把大数据量的过程放在带宽更高的 Server 内——语义允许的灵活性与拓扑约束在此互相成就。

| 算子 | 分级顺序 | 原因 |
| --- | --- | --- |
| ReduceScatter | 先机间、后机内 | 保证 Server 间通信数据块的连续性 |
| AllGather | 先机内、后机间 | 同上 |
| AllReduce | 机内 RS → 机间 AR → 机内 AG | 大数据量留在机内高带宽域 |

ReduceScatter 与 AllGather 的分级流程图（注意两者顺序恰好相反）：

![ReduceScatter 分级通信流程：先 Server 间、后 Server 内（图源：HCCL 官方文档）](/images/cann/hccl/official/reduce-scatter-hierarchical.png)

![AllGather 分级通信流程：先 Server 内、后 Server 间（图源：HCCL 官方文档）](/images/cann/hccl/official/allgather-hierarchical.png)

## 7. 算法选择：从文档到源码

算法的"选择权"在三层可见：

1. **自动选择**：`src/ops/<op>/selector/`（如 `all_reduce_auto_selector.cc`）依据数据量、rank 数、拓扑层级、芯片能力综合决策；
2. **环境变量干预**：`HCCL_ALG` 指定算法，`HCCL_ALGO_MULTIPLE_DIMENSION_SPLIT_RATIO` 等调节切分比例——调优时先读 `user_guide/hccl_env/`；
3. **耗时公式验证**：用本单元的 α-β-γ 公式手工估算，与 Profiling 数据对照（`perf_analysis/typical_op_behavior.md`）。

::: warning 注意
算法可用性与具体产品型号相关（如 CCU 引擎仅 950 系），跨平台结论必须回到当前环境的支持清单（`comm_ops_support_list/`）核对。
:::

## 8. 自测题

1. α、β、γ 分别建模什么？小消息与大数据量场景分别由谁主导？
2. Ring 的 AllReduce 耗时公式是什么？为什么大数据量场景它占优？
3. RHD 解决了 Ring 的什么问题？又引入了什么新问题？
4. NHR 相比 RHD 的两个改进是什么？
5. AllReduce 的分级通信为什么允许"不严格 RS + AG"？顺序怎么排？

::: details 自测答案

1. α = 每步启动延迟，β = 单位数据传输时间（带宽倒数），γ = 单位数据本地规约时间。小消息由 α（步数）主导；大数据量由 β（总搬运量与带宽利用）主导。
2. $2(p-1)\alpha + 2\frac{p-1}{p}n\beta + \frac{p-1}{p}n\gamma$。步数虽多（2(p−1)），但每步只传 n/p，且每张卡始终在收发，带宽利用充分，总传输量约 2n，适合大数据量。
3. RHD 把步数从 O(n−1) 压到 ⌈log₂N⌉，改善小消息延迟；但非 2 幂规模需先合并/还原引入额外步骤，且每步通信对象与链路变化，大流量时可能引发交换机流量冲突。
4. 一是对非 2 幂规模友好（N 棵生成树，任何规模都充分利用链路，无" N−1 比 N 慢"现象）；二是通过重排数据片编号聚合发送、流量集中在物理位置相近节点，减少链路变化与流量冲突。
5. 因为 AllReduce 的语义只要求每个 rank 最终拿到完整归约结果，不约束中间切分的语义顺序；因此把大数据量通信放在机内高带宽域：机内 ReduceScatter → 机间 AllReduce → 机内 AllGather。

:::

## 本单元小结

- **代价模型**：α-β-γ 是读一切算法耗时公式的钥匙；
- **三大算法**：Ring（带宽型，O(n−1) 步）、RHD（延迟型，log 步、非 2 幂有额外开销）、NHR（兼顾步数与链路稳定，非 2 幂友好）；
- **分级通信**：机内高带宽域消化大数据量，机间传输压到最少；
- **选择机制**：selector 自动选 + `HCCL_ALG` 等环境变量干预 + 公式估算验证；
- **知识连接**：这里的选择逻辑正是 H01-7 走读 `all_reduce_auto_selector.cc` 时要找的代码。

## 参考资料

- [集合通信算法介绍：Ring（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/Ring.md)
- [集合通信算法介绍：RHD（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/RHD.md)
- [集合通信算法介绍：NHR（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/NHR.md)
- [分级通信原理（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/hierarchical_comm_principle.md)
- [环境变量参考：HCCL_ALG](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_ALGO.md)

---

下一单元进入 **H01-7：源码走读——一次 AllReduce 的调用链**。

[返回专题导学 →](../hccl-source.md)
