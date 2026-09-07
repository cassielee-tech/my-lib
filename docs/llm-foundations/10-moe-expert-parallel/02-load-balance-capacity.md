# 单元 10-2｜负载均衡、Capacity 与丢 Token

> 所属章节：[第 10 章｜MoE 与 Expert Parallel](../10-moe-expert-parallel.md) · 预计用时：约 **15 分钟**

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

## 6. Capacity 与 Token Dropping

早期 MoE 常为每个 Expert 设置最大容量：

$$
C=\left\lceil CapacityFactor\times\frac{T\times k}{E}\right\rceil
$$

若理想平均为 256、`CapacityFactor=1.25`，则每个 Expert 容量为 320。超过容量的 Assignment 可能被丢弃、转到备选 Expert，或通过其他策略处理。

- Capacity 太小：容易丢 Token，损伤训练质量；
- Capacity 太大：Padding 和预留空间增多，浪费计算与显存；
- Dropless MoE：不设固定丢弃上限，保留所有 Token，但需要支持变长 Expert Batch，并承担失衡带来的性能波动。

“Token Dropping”通常指该 Token 的某条 Expert 路径被跳过，并不是把原始训练 Token 从整个 Transformer 中删除。

## 7. 负载均衡 Loss 做什么

训练时通常加入辅助 Loss，鼓励 Router 更均匀地使用 Expert。一个常见直觉形式为：

$$
L_{aux}=\alpha E\sum_{i=1}^{E}f_iP_i
$$

- $f_i$：实际分配给 Expert $i$ 的 Token 比例；
- $P_i$：Router 给 Expert $i$ 的平均概率；
- $\alpha$：辅助 Loss 权重。

如果某个 Expert 同时获得高概率和大量 Token，对应项会变大，优化器会推动路由分布更加均衡。

还可能使用 Router Z-Loss、无辅助 Loss 的 Bias 调节或更细粒度的设备级均衡策略。均衡不是要求每个小 Batch 完全相等，而是在训练质量、专业化和硬件利用率之间折中。

---

[继续单元 10-3 →](./03-token-alltoall.md)
