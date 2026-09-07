# 第 11 章总结｜现代模型的效率设计

> 返回：[第 11 章首页](../11-efficient-llm-design.md)

::: tip 本章核心结论
1. GQA/MQA 共享 K/V Head，主要减少 KV Cache 和 Decode 搬运，不是减少 Q Head。
2. 量化降低存储位宽，但收益取决于量化对象、粒度、硬件和 Kernel。
3. RoPE Scaling、Sliding Window、FlashAttention 和 Paged Attention 解决的是四类不同问题。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

1. **GQA 会减少 Q Head 吗？** 通常不会，它主要减少 K/V Head。
2. **MQA 是否只产生一个 Attention Head？** 不是，多个 Q Head 仍产生多个输出，只是共享 K/V。
3. **KV Cache 是否保存 Q？** 通常不保存，未来只需要新的 Q 查询历史 K/V。
4. **INT4 模型的所有计算都是 4 bit 吗？** 不一定，常见方案只是 INT4 存权重。
5. **量化只影响显存容量吗？** 还影响带宽、Kernel、计算单元利用率和精度。
6. **FlashAttention 是稀疏注意力吗？** 不是，它保持精确全量 Attention，主要减少 HBM IO。
7. **Paged Attention 会减少 KV 的理论元素数吗？** 不会，它改善分配方式和碎片。
8. **RoPE Scaling 会让 Attention 变为线性复杂度吗？** 不会，它解决位置外推。

## 17. 动手练习

### 练习 1：计算 GQA KV Cache

设 $B=4$、$L_S=40$、$N_q=40$、$N_{kv}=8$、$d_h=128$、$S=16384$，KV 使用 BF16。计算 GQA 的 KV Cache，并与 MHA 比较。

### 练习 2：估算量化权重

一个 13B 模型分别以 BF16、INT8、INT4 保存权重，忽略元数据时各需要多少 GB？为什么真实显存占用会更高？

### 练习 3：为瓶颈选择方案

分别为以下问题选择优先方案：Prefill 中 $S\times S$ 中间矩阵造成大量 HBM 访问；Decode 时 KV 太大；动态请求造成显存碎片；模型无法理解超出训练长度的位置。

## 18. 自测题

1. MHA、GQA、MQA 的核心差别是什么？
2. KV Cache 为什么不保存历史 Q？
3. GQA 的 KV Cache 相对 MHA 缩小多少？
4. TP Size 大于 KV Head 数时为什么需要特别处理？
5. Scale 在量化中起什么作用？
6. Weight-Only、W8A8 和 KV Cache 量化分别压缩什么？
7. 为什么 INT4 权重不保证推理比 FP16 快 4 倍？
8. 长上下文对 Prefill 和 Decode 的压力有什么不同？
9. FlashAttention、Sliding Window 和 RoPE Scaling 分别解决什么？
10. Paged Attention 为什么不减少 KV Cache 的理论数据量？

::: details 自测答案

1. 区别在 K/V Head 的共享程度：MHA 每个 Q 对应独立 K/V，GQA 一组 Q 共享 K/V，MQA 所有 Q 共享一组 K/V。
2. 每个 Decode Step 只需当前 Token 的 Q 查询全部历史 K/V，过去的 Q 不会再次参与后续注意力。
3. 其他条件相同时，比例为 $N_{kv}/N_q$。例如 32 个 Q、8 个 KV 时降到 $1/4$。
4. KV Head 无法在所有 Rank 间一一均分，可能需要复制 KV、改变分片或引入通信。
5. Scale 建立低位整数与原始实数范围之间的映射，用于量化和近似反量化。
6. Weight-Only 压权重；W8A8 压权重和激活；KV Cache 量化压历史 K/V。
7. 还取决于硬件支持、Kernel 效率、反量化开销、Shape 和原始瓶颈；位宽只直接决定存储量。
8. Prefill 主要承担全量 Attention 的平方计算与中间数据压力；Decode 每步读取随历史长度线性增长的 KV，并对单步延迟敏感。
9. FlashAttention 优化精确 Attention 的 IO；Sliding Window 减少关注位置；RoPE Scaling 改善位置外推。
10. 它只是把逻辑连续 KV 映射到按需分配的物理 Block；每个有效 Token 的 K/V 仍需保存。

:::

## 19. 本章小结

- GQA/MQA 通过共享 K/V Head 减少 KV Cache 和 Decode 数据读取；
- GQA 在 MHA 表达能力与 MQA 效率之间折中，并影响 TP Head 分片；
- 量化需要结合量化对象、粒度和执行 Kernel 判断收益；
- Weight-Only、激活量化与 KV Cache 量化作用于不同数据；
- 长上下文同时涉及位置外推、Attention 计算、KV Cache 和服务调度；
- FlashAttention 优化 IO，Sliding Window 改变关注范围，Paged Attention 改善缓存管理；
- 效率技术最终都要落到计算量、存储量、搬运字节数和通信依赖上分析。

## 参考资料

- [Fast Transformer Decoding: One Write-Head is All You Need（MQA）](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [PagedAttention](https://arxiv.org/abs/2309.06180)

下一课将进行大模型基础综合实战：选取一个现代 Decoder-only LLM，估算参数量、训练与推理 FLOPs、权重和 KV Cache 显存，并把数据流映射到算子与通信。

---

<!-- chapter-navigation -->
[进入第 12 章 →](../12-llm-systems-analysis.md)
