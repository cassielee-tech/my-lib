# 第 1 章｜MoE、Router 与 Expert Parallel

> 本章目标：理解 Sparse MoE 怎样用稀疏激活扩大模型容量；看懂 Router、Top-k、Expert Capacity 和负载均衡；能够画出 Expert Parallel 中两次 Alltoall 的完整 Token 数据流。

## 本章导学

::: tip 本章核心结论
1. MoE 保存多套 Expert FFN，但每个 Token 只激活 Top-k 个，因此总参数大而单 Token 计算仍稀疏。
2. Router 失衡会同时造成模型训练问题、设备等待和通信热点。
3. Expert Parallel 需要 Dispatch Alltoall 把 Token 送到 Expert，再用 Combine Alltoall 把结果送回。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，围绕 Token 被谁处理、怎样移动展开。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](#⚡-速通-约-5-分钟)：一家分诊台医院类比讲完整章，再回来按单元深入。

**先看这几张核心图：** Dense vs MoE、Router Top-k、Expert Parallel Alltoall 三张图。

- [ ] 我能区分总参数与激活参数
- [ ] 我能解释为什么路由需要负载均衡
- [ ] 我能画出两次 Alltoall 的方向

## 本章单元

- **01-1（约 15 分钟）**：[MoE、Router 与 Top-k](#_01-1-moe、router-与-top-k)
- **01-2（约 15 分钟）**：[负载均衡、Capacity 与丢 Token](#_01-2-负载均衡、capacity-与丢-token)
- **01-3（约 15 分钟）**：[Token 重排与 Alltoall](#_01-3-token-重排与-alltoall)
- **01-4（约 15 分钟）**：[MoE 的并行组合与性能瓶颈](#_01-4-moe-的并行组合与性能瓶颈)

- **本章总结**：[串联知识、练习与检查](#本章总结)

## ⚡ 速通（约 5 分钟）

> 适用：时间紧，或完整版跟不动时。这页用一家医院讲完整章，读完抓住 80% 的主干。任何一节想深入，拉到文末点对应单元。

### 1. 分诊台医院：养一百个医生，每人只看两个病人

普通模型（Dense）像**全科诊所**：每个病人来了，全部医生都上手——知识多，但每个病人都贵。

**MoE 是分诊台医院**：

- **Router（分诊台护士）**：看一眼病人（Token），决定挂哪几个科；
- **Expert（专科医生）**：很多位，但每人只在自己的诊室等；
- **Top-k**：每个病人**只挂 k 个科**（常见 k=1 或 2）。

医院还通常**只把深加工工位（FFN/MLP）换成多科诊室**——会议室（Attention）等其他科室照旧，不是整座医院复制八份。

于是出现一个奇妙的账本：**全院养着 100 位医生（总参数巨大），但每个病人只惊动 2 位（单 Token 计算稀疏）**。这就是 MoE 的核心交易：**用显存买知识，用稀疏省计算**。

注意：没被挂号的医生也要发工资——**全部 Expert 的权重都占显存**，不管今天有没有病人。

::: details 小测 1：MoE 总参数量大，每个 Token 的计算量也大吗？
不一定。Router 让每个 Token 只激活 Top-k 个 Expert（比如 128 位医生里挑 2 位），单 Token 的计算量只看被激活的少数专家。总参数决定显存，激活参数决定计算——两个数字要分开看。
:::

### 2. 挂号不均的连锁反应

分诊台没有上帝视角，**挂号会冷热不均**：某几个科天天爆满，另一些科闲得喝茶。

这一件事同时引发三层灾难：

1. **模型层面**：爆满科室的病人被压缩、冷门科室的医生学不到东西（训练不充分）；
2. **设备层面**：排队严重的科成为**全院瓶颈**——所有人都等它（Straggler）；
3. **效率层面**：单个医生一次只看两三个病人，**诊室利用率极低**（小 GEMM 效率差）。

解法是**Capacity（接诊上限）**：给每个科设每日限额，超出限额的挂号被排队填充或丢弃。同时训练时加**辅助 Loss** 鼓励分诊台摊平负载。代价：限额意味着**可能丢病人**（Token 被丢弃/填充）——模型质量和负载均衡是一对要谈判的矛盾。

::: details 小测 2：为什么需要 Expert Capacity？
不设上限的话，热门 Expert 的输入张量大小完全随路由波动，无法预留稳定的显存和计算规划，还会让排队失控。设上限让每科接诊量可控、张量尺寸固定；代价是溢出的 Token 可能被填充或丢弃。
:::

### 3. 多院区：AlltoAll 登场

医生太多，一家院区放不下——**Expert Parallel（EP）把科室分布到多家分院**（多张卡）。

现在病人要跨院区看病了，前向需要**两次全院快递**：

```text
Dispatch AlltoAll：把每个 Token 送到「它挂的医生所在的分院」
  → 各地医生看诊
Combine AlltoAll：把诊断结果寄回「病人原来的医院」，按原顺序拼回病历
```

为什么用 AlltoAll 而不是 AllGather？**AllGather 是"每家医院都拿到所有病人"**——绝大部分病人和你没关系，纯浪费。AlltoAll 是"各家只收自己要看的病人"——精准的多对多交换。

和《模型全景》第 1 章的 AllReduce 对比一下语义：AllReduce 汇总的是**同一份东西的不同意见**（DP 的梯度）；AlltoAll 交换的是**不同的东西给不同的人**（MoE 的 Token）。

::: details 小测 3：MoE 前向为什么需要两次 AlltoAll？
Dispatch 把 Token 发到 Expert 所在的 Rank 完成计算；但最终输出必须回到 Token 原来的位置、按原顺序拼接，否则后面的层就乱了。所以一次"发过去看病"，一次"把结果寄回来"。反向传播还有对应的逆向交换。
:::

### 3 句话带走

1. MoE = **分诊台医院**：总参数（显存）大，单 Token 激活（计算）稀疏——两个账分开算；
2. **挂号不均**同时伤害模型质量、设备负载和 GEMM 效率，Capacity 是带代价的解药；
3. EP 的跨院区看病 = **Dispatch + Combine 两次 AlltoAll**，与 AllReduce 的"汇总意见"语义不同。

### 黑话小词典

| 术语 | 人话 |
| --- | --- |
| MoE / Dense | 分诊台医院 / 全科诊所 |
| Router / Expert | 分诊台护士 / 专科医生 |
| Top-k | 每个病人挂几个科 |
| 总参数 / 激活参数 | 全院医生数 / 被惊动的医生数 |
| Capacity | 每科接诊上限（超了丢号） |
| 负载均衡 / 辅助 Loss | 摊平挂号表的手段 |
| Expert Parallel（EP） | 科室分布到多分院 |
| AlltoAll Dispatch / Combine | 发病人去看诊 / 寄回诊断结果 |

### 想深入？

每节 15 分钟，按需点开，不必按顺序全读：

| 单元 | 讲什么（白话） | 什么时候需要它 |
| --- | --- | --- |
| [01-1 MoE、Router 与 Top-k](#_01-1-moe、router-与-top-k) | 分诊台的工作机制 | 想读懂 MoE 模型配置 |
| [01-2 负载均衡、Capacity 与丢 Token](#_01-2-负载均衡、capacity-与丢-token) | 挂号不均的连锁灾难 | 想分析 MoE 训练问题 |
| [01-3 Token 重排与 Alltoall](#_01-3-token-重排与-alltoall) | 跨院区快递的完整细节 | 想理解 EP 通信 |
| [01-4 MoE 的并行组合与性能瓶颈](#_01-4-moe-的并行组合与性能瓶颈) | MoE + 其他并行的组合拳 | 想做 MoE 性能分析 |
| [本章总结](#本章总结) | 动手练习 + 自测 | 想检验整章掌握程度 |

## 01-1｜MoE、Router 与 Top-k

::: info 本单元目标
围绕 **MoE、Router 与 Top-k** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

普通 Decoder Block 的每个 Token 都经过同一套 Attention 和 FFN。若把 FFN 扩大一倍，每个 Token 的计算量也大致随之增加。

MoE（Mixture of Experts）提供了另一种扩展思路：准备多个不同参数的 FFN Expert，但每个 Token 只选择少数几个 Expert 计算。

```text
Dense FFN：每个 Token → 同一个 FFN
Sparse MoE：每个 Token → Router → Top-k Experts
```

这样可以增加**总参数量**，同时让每个 Token 的**激活参数量和计算量**只与 Top-k Expert 有关。

![Dense FFN 与 Sparse MoE 的计算路径差异](/images/llm/dense-vs-moe.svg)

::: warning 总参数不等于激活参数
MoE 的所有 Expert 权重仍需要存储在设备集群中，只是单个 Token 不会经过全部 Expert。因此“计算稀疏”不等于“权重不用占显存”。
:::

### 2. MoE 通常替换 Decoder Block 的哪一部分

常见 Sparse MoE Transformer 保留 Attention、RMSNorm 和残差连接，只把部分或全部 Dense FFN 替换为 MoE Layer：

$$
h'=h+Attention(RMSNorm(h))
$$

$$
y=h'+MoE(RMSNorm(h'))
$$

每个 Expert 本身通常仍是一个独立 FFN，例如 SwiGLU：

$$
Expert_i(x)=W_{down}^{(i)}left(SiLU(W_{gate}^{(i)}x)\odot W_{up}^{(i)}x\right)
$$

所以“8 个 Expert”通常意味着这一层拥有 8 套不同的 FFN 参数，而不是 8 个完整 Transformer 模型。Attention 和其他公共层仍然共享。

有些架构还会同时使用：

- **Routed Experts**：由 Router 为每个 Token 动态选择；
- **Shared Experts**：所有 Token 都执行，用于承载公共能力。

### 3. Router 怎样为 Token 选择 Expert

设一层有 $E$ 个 Expert，Token Hidden State 为 $x\in\mathbb{R}^{H}$。Router 通常先做一个小型线性投影：

$$
z=W_rx,\qquad W_r\in\mathbb{R}^{E\times H}
$$

经过 Softmax 得到路由概率：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{E}e^{z_j}}
$$

再选择概率最高的 $k$ 个 Expert：

$$
\mathcal{T}(x)=TopK(p,k)
$$

输出是选中 Expert 结果的加权和：

$$
y=\sum_{i\in\mathcal{T}(x)}\tilde p_i Expert_i(x)
$$

$\tilde p_i$ 通常是对 Top-k 概率重新归一化后的权重，具体实现可能不同。

![Router 为每个 Token 产生不同的 Top-k 路径](/images/llm/moe-router-topk.svg)

需要注意：

- 不同 Token 可以选择不同 Expert；
- 同一个 Token 在不同 MoE 层也可能选择不同 Expert；
- Expert 是训练中自动形成的参数分支，不应简单假设它们分别固定代表“数学”“代码”“英语”；
- Top-1 只执行一个 Expert，Top-2 会执行两个并加权合并。

### 4. 从 Shape 看一次路由

把 Batch 和 Sequence 展平，设：

- Token 数 $T=B\times S$
- Hidden Size $H$
- Expert 数 $E$
- 每 Token 选择 $k$ 个 Expert

则主要张量为：

| 张量 | Shape | 含义 |
| --- | --- | --- |
| 输入 | `[T, H]` | 展平后的 Token Hidden State |
| Router Logits | `[T, E]` | 每个 Token 对所有 Expert 的分数 |
| Top-k Index | `[T, k]` | 被选中的 Expert ID |
| Top-k Weight | `[T, k]` | 合并 Expert 输出的权重 |
| Expert 输入 | 逻辑上 `[T × k, H]` | 每个 Token 复制到所选 Expert |
| MoE 输出 | `[T, H]` | 按原 Token 顺序恢复后的结果 |

当 $k=2$ 时，一个 Token 会产生两条 Expert Assignment，因此逻辑 Expert 输入数是 $2T$。实际实现还会根据 Expert 分组、容量 Padding 或 Dropless Routing 改变物理布局。

## 01-2｜负载均衡、Capacity 与丢 Token

::: info 本单元目标
围绕 **负载均衡、Capacity 与丢 Token** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

如果 Router 发现某个 Expert 在训练早期略好，更多 Token 会流向它；它获得更多训练样本后可能变得更强，于是继续吸引更多 Token。这会形成“富者愈富”的路由坍缩。

假设 $T=1024$、$E=8$、Top-2，总 Assignment 数为：

$$
T\times k=2048
$$

理想平均每个 Expert 接收：

$$
\frac{T\times k}{E}=256
$$

但真实路由可能是 `[620, 410, 300, 260, 180, 140, 90, 48]`。这会导致：

- 热门 Expert 成为 Straggler，其他设备等待；
- 冷门 Expert 训练不足；
- 跨设备流量集中到少数 Rank；
- 每个 Expert 的 GEMM Shape 差异大，执行效率下降。

![相同 Token 数下，路由失衡如何同时浪费计算与通信资源](/images/llm/moe-load-imbalance.svg)

### 6. Capacity 与 Token Dropping

早期 MoE 常为每个 Expert 设置最大容量：

$$
C=\left\lceil CapacityFactor\times\frac{T\times k}{E}\right\rceil
$$

若理想平均为 256、`CapacityFactor=1.25`，则每个 Expert 容量为 320。超过容量的 Assignment 可能被丢弃、转到备选 Expert，或通过其他策略处理。

- Capacity 太小：容易丢 Token，损伤训练质量；
- Capacity 太大：Padding 和预留空间增多，浪费计算与显存；
- Dropless MoE：不设固定丢弃上限，保留所有 Token，但需要支持变长 Expert Batch，并承担失衡带来的性能波动。

“Token Dropping”通常指该 Token 的某条 Expert 路径被跳过，并不是把原始训练 Token 从整个 Transformer 中删除。

### 7. 负载均衡 Loss 做什么

训练时通常加入辅助 Loss，鼓励 Router 更均匀地使用 Expert。一个常见直觉形式为：

$$
L_{aux}=\alpha E\sum_{i=1}^{E}f_iP_i
$$

- $f_i$：实际分配给 Expert $i$ 的 Token 比例；
- $P_i$：Router 给 Expert $i$ 的平均概率；
- $\alpha$：辅助 Loss 权重。

如果某个 Expert 同时获得高概率和大量 Token，对应项会变大，优化器会推动路由分布更加均衡。

还可能使用 Router Z-Loss、无辅助 Loss 的 Bias 调节或更细粒度的设备级均衡策略。均衡不是要求每个小 Batch 完全相等，而是在训练质量、专业化和硬件利用率之间折中。

## 01-3｜Token 重排与 Alltoall

::: info 本单元目标
围绕 **Token 重排与 Alltoall** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

即使所有 Expert 都在同一设备，也不能让原始 `[T,H]` 顺序直接执行多个 Expert。实现通常需要：

1. 根据 Top-k Expert ID 统计每个 Expert 的 Token；
2. 将 Token Permute，按 Expert 分组；
3. 对多个 Expert 执行 Grouped GEMM；
4. 将输出 Unpermute 回原 Token 顺序；
5. 按 Router Weight 合并 Top-k 输出。

这里的 Permute/Unpermute 是真实的数据移动或索引操作。Expert 很多、每个 Expert Token 很少时，如果逐个启动小 GEMM，计算单元利用率会很低，因此常把多个 Expert 计算组织成 Grouped GEMM。

### 9. Expert Parallel 为什么需要 Alltoall

Expert 权重很大时，可以把不同 Expert 放到不同 Rank：

```text
Rank 0：Expert 0、Expert 1
Rank 1：Expert 2、Expert 3
```

但 Rank 0 上的输入 Token 可能被 Router 分给 Expert 3，Rank 1 上的 Token 也可能需要 Expert 0。每个 Rank 都要把 Token 发给“拥有目标 Expert 的 Rank”。

这正符合 Alltoall：每个 Rank 给不同目标 Rank 发送不同内容，同时从所有 Rank 接收属于本地 Expert 的 Token。

![Expert Parallel 中 Dispatch Alltoall 与 Combine Alltoall 的完整往返](/images/llm/expert-parallel-alltoall.svg)

前向传播的完整路径是：

1. 每个 Rank 对本地 Token 运行 Router；
2. 按目标 Rank 和 Expert ID 对 Token 排列；
3. **Dispatch Alltoall**：Token Hidden State 发到 Expert 所在 Rank；
4. 接收后再按本地 Expert 分组；
5. 本地 Expert 执行 FFN / Grouped GEMM；
6. **Combine Alltoall**：Expert 输出发回 Token 原所属 Rank；
7. 恢复原 Token 顺序，并按 Router Weight 合并。

反向传播沿相反数据依赖传播，对应的激活梯度同样需要跨 Rank 交换。

### 10. Alltoall 发送的到底是什么

MoE Dispatch 发送的主要是 Token Hidden State，而不是 Expert 权重。忽略索引、权重、Padding 和协议开销，一次逻辑 Dispatch 数据规模约为：

$$
V_{dispatch}\approx T\times k\times H\times Bytes
$$

Combine 还要发送近似同规模的 Expert 输出。Top-2 相比 Top-1 产生两倍 Assignment，既增加 Expert 计算，也增加需要路由的数据量。

实际通信并不一定均匀：每个源 Rank 发往不同目标 Rank 的 Token 数由 Router 动态决定，因此常需要先统计 Split Size，再执行变长 Alltoall。

与 AllGather 的差别是：

- **AllGather**：每个 Rank 的数据被所有 Rank 收集；
- **Alltoall**：每个 Rank 为每个目标 Rank 准备不同分片，最终只把目标需要的内容发过去。

如果用 AllGather 实现路由，所有 Rank 会拿到大量不属于本地 Expert 的 Token，带来不必要的带宽和显存开销。

## 01-4｜MoE 的并行组合与性能瓶颈

::: info 本单元目标
围绕 **MoE 的并行组合与性能瓶颈** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

Expert Parallel 只描述 Expert 怎样分布，真实训练通常还会组合：

- **DP**：复制整套 MoE 模型，不同副本处理不同数据；
- **EP**：在一个 EP Group 中把不同 Expert 分给不同 Rank；
- **TP**：继续切分单个 Expert 的大矩阵；
- **PP**：把不同 Transformer 层放在不同 Stage；
- **CP/SP**：切分长序列或减少非 MoE 区域的重复激活。

如果同一个 Expert 在不同 Expert-Data-Parallel Group 中有副本，这些对应 Expert 的梯度仍需在副本之间做 AllReduce/ReduceScatter。EP 的 Token Alltoall 与 Expert 副本的梯度同步是两类不同通信。

### 12. MoE 的主要性能瓶颈

#### 12.1 Communication Wall

Dispatch 和 Combine Alltoall 位于 Expert 计算前后，形成严格依赖。跨节点时，网络拓扑、消息粒度和动态 Split 会显著影响性能。

#### 12.2 Compute Efficiency Wall

Expert 数越多，每个 Expert 分到的 Token 可能越少，形成许多小 GEMM。Grouped GEMM、Token Padding 和 Kernel Fusion 用于提高计算效率。

#### 12.3 Memory Wall

虽然单 Token 只激活 Top-k Expert，所有 Expert 权重仍需要驻留在设备集群中。训练还要保存 Expert 梯度和优化器状态。

#### 12.4 Load Imbalance

最慢 Expert / Rank 决定整个 MoE Layer 的完成时间。Router 均衡既是训练问题，也是系统问题。

常见工程优化包括：

- Router、Top-k、Permute 融合；
- Grouped GEMM；
- Alltoall 与 Shared Expert 或其他 Micro-batch 计算重叠；
- 分层 Alltoall，优先利用机内高速互联；
- Expert Placement 与动态负载感知路由；
- 更大的 Token Batch，提高每个 Expert 的 GEMM 尺寸。

### 13. MoE 推理有什么不同

Prefill 有较多 Token，可以聚合成较大的 Expert Batch。Decode 每个请求每步只有一个 Token，即使服务端做 Continuous Batching，路由后每个 Expert 获得的 Token 仍可能很少。

因此 MoE Decode 会同时面临：

- 动态路由导致的小 Expert Batch；
- 每层两次 Token 交换；
- 总 Expert 权重的存储压力；
- 每 Token 串行链路中的通信延迟；
- 不同请求路由分布造成的性能波动。

“激活参数少”说明理论计算量较低，不代表推理一定更快。能否把 Expert 计算做大、把 Alltoall 做快，决定了稀疏性的收益能否真正落到硬件上。

### 14. 与集合通信算子开发的关系

MoE 是理解 Alltoall 最典型的模型场景之一。阅读相关实现时，可以沿下面的问题检查：

1. Router 输出的 Expert ID 怎样映射到目标 Rank？
2. 每个 Rank 的 Send Count / Receive Count 如何计算和交换？
3. Token 在发送前怎样 Permute，元数据放在哪里？
4. 通信 Buffer 是定长 Padding 还是变长？
5. Dispatch 与 Combine 是否复用同一通信域和 Stream？
6. 通信能否与 Shared Expert、其他 Micro-batch 或权重梯度计算重叠？
7. 跨节点时是否采用分层调度？
8. 某个 Rank Token 数异常时，超时和性能诊断如何呈现？

以后进入 HCCL 源码，可以用这条链路把模型语义映射到通信算子的输入：**不是抽象地“做一次 Alltoall”，而是在重排和恢复 Token 的所有权。**

## 本章总结

> 返回：[第 1 章首页](#)

::: tip 本章核心结论
1. MoE 保存多套 Expert FFN，但每个 Token 只激活 Top-k 个，因此总参数大而单 Token 计算仍稀疏。
2. Router 失衡会同时造成模型训练问题、设备等待和通信热点。
3. Expert Parallel 需要 Dispatch Alltoall 把 Token 送到 Expert，再用 Combine Alltoall 把结果送回。
:::

### 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

### ⚠️ 易错点

1. **8 个 Expert 等于 8 个完整模型吗？** 通常不是，常见 MoE 只把 FFN 替换为多套 Expert。
2. **每个 Expert 是否对应一个可命名领域？** 不一定，Expert 的专业化由训练形成，可能是更复杂的语法或表示模式。
3. **总参数量大是否意味着每 Token FLOPs 同样大？** 不一定，Sparse MoE 每 Token 只激活 Top-k Expert。
4. **未选中的 Expert 还占显存吗？** 占，它们的权重仍要存储。
5. **Top-2 只是把两个结果平均吗？** 通常按 Router Weight 加权，且权重可能重新归一化。
6. **EP 只需要一次 Alltoall 吗？** 前向通常需要 Dispatch 和 Combine 两次，反向还有对应的数据交换。
7. **负载均衡只是为了模型质量吗？** 不是，它还直接影响设备等待、通信热点和 GEMM 效率。

### 16. 动手练习

#### 练习 1：计算 Assignment 和容量

设 $T=4096$、$E=16$、Top-2、`CapacityFactor=1.25`，计算总 Assignment、平均每 Expert Token 数和容量上限。

#### 练习 2：手工模拟 Dispatch

两个 Rank 各有 4 个 Token，共 4 个 Expert，Rank 0 持有 E0/E1，Rank 1 持有 E2/E3。任选一组 Top-1 路由结果，写出两个 Rank 的 Send Buffer、Receive Buffer 和恢复顺序。

#### 练习 3：估算通信量

设 $T=8192$、Top-2、$H=4096$、BF16，忽略额外开销，估算一次 Dispatch 和一次 Combine 的逻辑数据量。

### 17. 自测题

1. Sparse MoE 为什么能同时拥有较大总参数量和较低每 Token 激活量？
2. Expert 通常替换 Transformer 的哪一部分？
3. Router 的输入和输出分别是什么？
4. Top-2 时，$T$ 个 Token 会产生多少 Assignment？
5. 为什么需要 Expert Capacity？
6. 路由失衡会造成哪些模型和系统问题？
7. Expert Parallel 为什么使用 Alltoall 而不是 AllGather？
8. 前向 MoE 为什么通常有两次 Alltoall？
9. LoRA 的梯度 AllReduce 与 MoE 的 Token Alltoall 有什么语义差别？
10. 为什么 MoE 激活参数少却不保证 Decode 一定快？

::: details 自测答案

1. 它保存许多 Expert 参数，但 Router 让每个 Token 只执行 Top-k 个 Expert。
2. 常见做法是替换 Decoder Block 的 FFN/MLP，Attention 和残差等结构仍共享。
3. 输入是 Token Hidden State；输出是对所有 Expert 的分数/概率以及 Top-k Expert ID 和权重。
4. 逻辑上为 $2T$ 条 Assignment。
5. 它限制单个 Expert 接收的最大 Token 数，使张量尺寸和计算负载可控；代价是可能 Padding 或丢弃溢出路径。
6. 热门 Expert 过载、冷门 Expert 训练不足，同时造成 Straggler、通信热点和小 GEMM 低利用率。
7. 每个源 Rank 只需把不同 Token 发给拥有对应 Expert 的目标 Rank；AllGather 会让所有 Rank 收到大量无关 Token。
8. Dispatch 把 Token 发到 Expert 所在 Rank，Combine 把 Expert 输出送回 Token 原所属 Rank以恢复顺序。
9. LoRA 梯度 AllReduce 聚合相同参数副本的梯度；MoE Alltoall 根据动态路由交换不同 Token 激活。
10. Decode 的 Expert Batch 可能很小，还要读取分布式 Expert 权重并在每层执行延迟敏感的动态 Alltoall。

:::

### 18. 本章小结

- MoE 通常用多个 Expert FFN 替换 Dense FFN；
- Router 为每个 Token 选择 Top-k Expert，并加权合并输出；
- 稀疏激活降低每 Token 计算，但所有 Expert 权重仍需存储；
- Capacity、辅助 Loss 和路由策略共同处理负载均衡；
- Expert Parallel 将 Expert 分布到多个 Rank；
- 前向需要 Dispatch Alltoall 和 Combine Alltoall，反向还有对应交换；
- MoE 性能取决于通信、Expert GEMM、内存和负载均衡，而不只取决于理论 FLOPs；
- 从模型语义看，MoE Alltoall 的本质是动态转移并恢复 Token 的计算所有权。

### 参考资料

- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Megatron Core：Mixture of Experts](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron Core：Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

下一课将学习现代模型的效率设计，比较 MHA、MQA、GQA，理解量化如何减少权重和 KV Cache 的存储与搬运，并回顾长上下文优化。

---

<!-- chapter-navigation -->
