# 第 10 章｜MoE、Router 与 Expert Parallel

> 本章目标：理解 Sparse MoE 怎样用稀疏激活扩大模型容量；看懂 Router、Top-k、Expert Capacity 和负载均衡；能够画出 Expert Parallel 中两次 Alltoall 的完整 Token 数据流。

## 本章导学

::: tip 本章核心结论
1. MoE 保存多套 Expert FFN，但每个 Token 只激活 Top-k 个，因此总参数大而单 Token 计算仍稀疏。
2. Router 失衡会同时造成模型训练问题、设备等待和通信热点。
3. Expert Parallel 需要 Dispatch Alltoall 把 Token 送到 Expert，再用 Combine Alltoall 把结果送回。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，围绕 Token 被谁处理、怎样移动展开。

**先看这几张核心图：** Dense vs MoE、Router Top-k、Expert Parallel Alltoall 三张图。

- [ ] 我能区分总参数与激活参数
- [ ] 我能解释为什么路由需要负载均衡
- [ ] 我能画出两次 Alltoall 的方向

## 本章单元

- **10-1（约 15 分钟）**：[MoE、Router 与 Top-k](./10-moe-expert-parallel/01-moe-router-topk.md)
- **10-2（约 15 分钟）**：[负载均衡、Capacity 与丢 Token](./10-moe-expert-parallel/02-load-balance-capacity.md)
- **10-3（约 15 分钟）**：[Token 重排与 Alltoall](./10-moe-expert-parallel/03-token-alltoall.md)
- **10-4（约 15 分钟）**：[MoE 的并行组合与性能瓶颈](./10-moe-expert-parallel/04-moe-parallel-performance.md)

- **本章总结**：[串联知识、练习与检查](./10-moe-expert-parallel/summary.md)
