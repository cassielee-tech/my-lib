# 单元 10-3｜Token 重排与 Alltoall

> 所属章节：[第 10 章｜MoE 与 Expert Parallel](../10-moe-expert-parallel.md) · 预计用时：约 **15 分钟**

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

## 9. Expert Parallel 为什么需要 Alltoall

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

## 10. Alltoall 发送的到底是什么

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

---

[继续单元 10-4 →](./04-moe-parallel-performance.md)
