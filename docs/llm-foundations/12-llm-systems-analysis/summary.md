# 第 12 章总结｜综合拆解一个现代 LLM

> 返回：[第 12 章首页](../12-llm-systems-analysis.md)

::: tip 本章核心结论
1. 模型资源账本从配置开始：先算参数，再乘精度，最后加入激活、KV Cache 和运行时开销。
2. $2P$、$6P$ 是前向与训练的数量级估算，不包含所有长序列和系统开销。
3. 并行策略切开不同张量，下一算子需要的数据布局决定使用哪种集合通信。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

1. **参数量等于运行显存吗？** 不等于，还要乘数据类型并加入其他状态。
2. **7.5B BF16 一定占 15 GiB 吗？** 15 是十进制 GB，约为 14 GiB。
3. **推理只需要权重吗？** 还需要 KV Cache、激活、Workspace 和框架内存。
4. **训练显存固定是 16 Byte/参数吗？** 不是，它取决于精度和优化器实现，而且没包含激活。
5. **$2P$ 包含全部 Attention 计算吗？** 没有完整包含随 $S^2$ 增长的部分。
6. **FlashAttention 会减少模型参数吗？** 不会，它主要优化 Attention 的 IO。
7. **多卡一定更快吗？** 不一定，切分降低单卡负担，也引入通信和同步。
8. **AllReduce 是模型自动产生的吗？** 它来自具体并行策略对数据依赖的改写。

## 15. 动手练习

### 练习 1：修改模型配置

把 Stack-7.5B 的 $F$ 从 14336 改为 11008，重新计算单层 FFN 参数量和总参数量。

### 练习 2：设计推理容量

假设单卡除权重外还有 24 GiB 可用于 KV Cache，使用 BF16 KV。估算 32K 请求最多能容纳几个；再计算 INT8 KV 的理论结果。实际部署为什么必须留余量？

### 练习 3：画出 TP 数据流

选择 Column Parallel 和 Row Parallel 切分一层 MLP，标出输入、局部矩阵、局部输出的 Shape，并判断在哪里需要 AllGather、AllReduce 或 ReduceScatter。

## 16. 自测题

1. Stack-7.5B 的 Embedding 参数量怎样计算？
2. 为什么 GQA 的 K/V Projection 参数少于 Q Projection？
3. 为什么这个模型的大多数 Block 参数位于 FFN？
4. 权重共享为什么能少一份 $V\times D$ 参数？
5. BF16 权重大小怎样从参数量估算？
6. KV Cache 为什么与 Layer、KV Head 和 Sequence Length 成正比？
7. 训练为什么还需要梯度和优化器状态？
8. $2P$ 和 $6P$ 分别用于什么粗略估算？
9. DP、TP、PP、CP、EP 分别切分什么？
10. 怎样从张量依赖推导需要哪一种集合通信？

::: details 自测答案

1. $V\times D=128000\times4096=524,288,000$，约 524.3M。
2. Q 有 32 个 Head，而共享后的 K/V 各只有 8 个 Head，投影输出宽度分别是 4096 与 1024。
3. SwiGLU 有两个 $D\to F$ 和一个 $F\to D$ 大矩阵，共 $3DF\approx176.16M$；Attention 约 41.94M。
4. 输入查表矩阵与输出 Logits 投影使用同一组参数，不再另外存储一个独立 LM Head。
5. 参数量乘每参数 2 Byte；7.50B 参数约为 15.0 GB，换算后约 14.0 GiB。
6. 每层都要为每个历史 Token 保存所有 KV Head 的 K 和 V，所以各维度都会乘入公式。
7. 反向传播产生梯度，AdamW 还要保存一阶、二阶矩等状态用于更新；这些在纯推理中不需要。
8. $2P$ 粗估 Dense 模型每 Token 前向参数相关 FLOPs，$6P$ 粗估前向加反向的训练 FLOPs。
9. DP 切 Batch；TP 切矩阵或 Head；PP 切 Layer；CP 切 Sequence；EP 切 Expert。
10. 先确定张量沿什么维度被分片，再判断下一算子需要拼接、求和、分片归约还是按目标重分发，分别映射到 AllGather、AllReduce、ReduceScatter 或 Alltoall 等语义。

:::

## 17. 大模型基础阶段总结

12 课形成了一条完整链路：

```text
文本与 Token
  → Embedding 与张量
  → Attention、RoPE、Decoder Block
  → 训练、反向传播与对齐
  → 自回归推理与 KV Cache
  → MoE、GQA、量化与长上下文
  → 参数、FLOPs、显存与通信估算
```

完成这一阶段后，面对一个算子或集合通信调用，不再只看到 API 名称，而能继续追问：它服务于模型中的哪段计算，张量 Shape 是什么，数据为什么必须移动，瓶颈可能在哪里。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA](https://arxiv.org/abs/2305.13245)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Megatron-LM](https://arxiv.org/abs/2104.04473)
- [ZeRO](https://arxiv.org/abs/1910.02054)

大模型基础 12 课至此完成。下一阶段进入 **AI Infra**：从加速器执行模型、存储层次、算术强度和 Roofline 开始，建立分析算子性能与集合通信性能的底层方法。

---

<!-- chapter-navigation -->
[进入 AI Infra →](../../ai-infra/index.md)
