# 第 10 章总结｜MoE 与 Expert Parallel

> 返回：[第 10 章首页](../10-moe-expert-parallel.md)

::: tip 本章核心结论
1. MoE 保存多套 Expert FFN，但每个 Token 只激活 Top-k 个，因此总参数大而单 Token 计算仍稀疏。
2. Router 失衡会同时造成模型训练问题、设备等待和通信热点。
3. Expert Parallel 需要 Dispatch Alltoall 把 Token 送到 Expert，再用 Combine Alltoall 把结果送回。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

1. **8 个 Expert 等于 8 个完整模型吗？** 通常不是，常见 MoE 只把 FFN 替换为多套 Expert。
2. **每个 Expert 是否对应一个可命名领域？** 不一定，Expert 的专业化由训练形成，可能是更复杂的语法或表示模式。
3. **总参数量大是否意味着每 Token FLOPs 同样大？** 不一定，Sparse MoE 每 Token 只激活 Top-k Expert。
4. **未选中的 Expert 还占显存吗？** 占，它们的权重仍要存储。
5. **Top-2 只是把两个结果平均吗？** 通常按 Router Weight 加权，且权重可能重新归一化。
6. **EP 只需要一次 Alltoall 吗？** 前向通常需要 Dispatch 和 Combine 两次，反向还有对应的数据交换。
7. **负载均衡只是为了模型质量吗？** 不是，它还直接影响设备等待、通信热点和 GEMM 效率。

## 16. 动手练习

### 练习 1：计算 Assignment 和容量

设 $T=4096$、$E=16$、Top-2、`CapacityFactor=1.25`，计算总 Assignment、平均每 Expert Token 数和容量上限。

### 练习 2：手工模拟 Dispatch

两个 Rank 各有 4 个 Token，共 4 个 Expert，Rank 0 持有 E0/E1，Rank 1 持有 E2/E3。任选一组 Top-1 路由结果，写出两个 Rank 的 Send Buffer、Receive Buffer 和恢复顺序。

### 练习 3：估算通信量

设 $T=8192$、Top-2、$H=4096$、BF16，忽略额外开销，估算一次 Dispatch 和一次 Combine 的逻辑数据量。

## 17. 自测题

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

## 18. 本章小结

- MoE 通常用多个 Expert FFN 替换 Dense FFN；
- Router 为每个 Token 选择 Top-k Expert，并加权合并输出；
- 稀疏激活降低每 Token 计算，但所有 Expert 权重仍需存储；
- Capacity、辅助 Loss 和路由策略共同处理负载均衡；
- Expert Parallel 将 Expert 分布到多个 Rank；
- 前向需要 Dispatch Alltoall 和 Combine Alltoall，反向还有对应交换；
- MoE 性能取决于通信、Expert GEMM、内存和负载均衡，而不只取决于理论 FLOPs；
- 从模型语义看，MoE Alltoall 的本质是动态转移并恢复 Token 的计算所有权。

## 参考资料

- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Megatron Core：Mixture of Experts](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron Core：Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

下一课将学习现代模型的效率设计，比较 MHA、MQA、GQA，理解量化如何减少权重和 KV Cache 的存储与搬运，并回顾长上下文优化。

---

<!-- chapter-navigation -->
[进入第 11 章 →](../11-efficient-llm-design.md)
