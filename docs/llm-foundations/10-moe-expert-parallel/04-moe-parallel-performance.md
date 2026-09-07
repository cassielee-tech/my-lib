# 单元 10-4｜MoE 的并行组合与性能瓶颈

> 所属章节：[第 10 章｜MoE 与 Expert Parallel](../10-moe-expert-parallel.md) · 预计用时：约 **15 分钟**

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

## 12. MoE 的主要性能瓶颈

### 12.1 Communication Wall

Dispatch 和 Combine Alltoall 位于 Expert 计算前后，形成严格依赖。跨节点时，网络拓扑、消息粒度和动态 Split 会显著影响性能。

### 12.2 Compute Efficiency Wall

Expert 数越多，每个 Expert 分到的 Token 可能越少，形成许多小 GEMM。Grouped GEMM、Token Padding 和 Kernel Fusion 用于提高计算效率。

### 12.3 Memory Wall

虽然单 Token 只激活 Top-k Expert，所有 Expert 权重仍需要驻留在设备集群中。训练还要保存 Expert 梯度和优化器状态。

### 12.4 Load Imbalance

最慢 Expert / Rank 决定整个 MoE Layer 的完成时间。Router 均衡既是训练问题，也是系统问题。

常见工程优化包括：

- Router、Top-k、Permute 融合；
- Grouped GEMM；
- Alltoall 与 Shared Expert 或其他 Micro-batch 计算重叠；
- 分层 Alltoall，优先利用机内高速互联；
- Expert Placement 与动态负载感知路由；
- 更大的 Token Batch，提高每个 Expert 的 GEMM 尺寸。

## 13. MoE 推理有什么不同

Prefill 有较多 Token，可以聚合成较大的 Expert Batch。Decode 每个请求每步只有一个 Token，即使服务端做 Continuous Batching，路由后每个 Expert 获得的 Token 仍可能很少。

因此 MoE Decode 会同时面临：

- 动态路由导致的小 Expert Batch；
- 每层两次 Token 交换；
- 总 Expert 权重的存储压力；
- 每 Token 串行链路中的通信延迟；
- 不同请求路由分布造成的性能波动。

“激活参数少”说明理论计算量较低，不代表推理一定更快。能否把 Expert 计算做大、把 Alltoall 做快，决定了稀疏性的收益能否真正落到硬件上。

## 14. 与集合通信算子开发的关系

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

---

[进入本章总结 →](./summary.md)
