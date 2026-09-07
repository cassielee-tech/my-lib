# 单元 11-3｜长上下文的四条优化路线

> 所属章节：[第 11 章｜现代模型的效率设计](../11-efficient-llm-design.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **长上下文的四条优化路线** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

序列从 $S$ 增长到 $2S$ 时，各类资源的变化不同：

| 项目 | 随序列长度变化 | 主要阶段 |
| --- | --- | --- |
| Attention Score 数量 | $O(S^2)$ | Prefill / 训练 |
| Q/K/V 和普通激活 | $O(S)$ | Prefill / 训练 |
| KV Cache 容量 | $O(S)$ | 推理 |
| 单步 Decode 读取历史 KV | $O(S)$ | Decode |
| 生成整段序列的 Attention 工作 | 近似 $O(S^2)$ | 完整生成 |

“支持 128K 上下文”至少包含三个不同问题：

1. 位置表示能否外推到这么远？
2. Attention 和显存能否处理这么长？
3. 模型是否真的能从远处找回关键信息？

上下文窗口很大，不代表模型在所有距离上都能同样有效地使用信息。

## 11. 长上下文优化的四条路线

![不同长上下文技术分别改变计算范围、数据搬运和缓存布局](/images/llm/long-context-techniques.svg)

### 11.1 改位置表示：RoPE Scaling

调整 RoPE 的位置映射或频率，使更长的真实位置落入模型可处理的范围。它解决位置外推，但不会消除全量 Attention 的 $O(S^2)$ 计算。

### 11.2 少看一部分：Sliding Window / Sparse Attention

每个 Token 只关注附近窗口或少量特殊位置。窗口 $W$ 固定时，Score 规模可从 $O(S^2)$ 降为 $O(SW)$。

代价是单层不能直接看到全部历史，需要多层传播、全局 Token 或混合全局 Attention 保留远距离信息。

### 11.3 精确计算但少搬数据：FlashAttention

FlashAttention 仍计算精确的全量 Attention，没有把数学复杂度从 $O(S^2)$ 改掉。它把 Q/K/V 分块搬入片上存储，在线维护 Softmax 统计量，避免显式将完整 $S\times S$ Score Matrix 写回 HBM。

它优化的是 **IO 和数据布局**，不是把注意力改成近似算法。

### 11.4 更高效地管理推理缓存

- GQA/MQA：减少 KV Head 数；
- KV Cache 量化：减少每个缓存元素的字节数；
- Paged Attention：按块管理 KV，减少连续空间预留和碎片；
- Chunked Prefill：把超长 Prompt 分块调度；
- Prefix Cache：为相同前缀复用已经计算的 KV。

这些方案可以组合使用。

## 12. Paged Attention 为什么像虚拟内存

传统做法为请求预留一段连续且足够大的 KV 空间，但最终生成长度未知：预留太大会浪费，太小又需要扩容搬迁。

Paged Attention 将逻辑连续的 Token 映射到多个固定大小的物理 Block：

```text
请求 A 的逻辑 Token： [0..15] [16..31] [32..47]
                          │       │        │
Block Table：            B7      B2       B9
                          │       │        │
物理显存：            [B2] ... [B7] ... [B9]
```

新 Token 到来时按需分配 Block，不要求整个请求的 KV 在物理显存中连续。它改善显存管理与动态批处理，但不会减少每个有效 Token 需要保存的 K/V。

## 13. 长序列怎样引出通信

单设备放不下长序列的激活和 KV 时，可以沿 Sequence 维切到多个设备，即 Sequence Parallel 或 Context Parallel。

在 Context Parallel 中，每个 Rank 只持有一段 Token，但某段 Query 若要执行全局 Attention，就必须获得其他 Rank 的 K/V，或通过 Ring 让 K/V 分块依次流过各 Rank。

问题于是从“一张卡怎样少读 HBM”，扩展为“多张卡怎样交换 K/V，并让通信与 Attention 计算重叠”。这可能涉及 AllGather、Alltoall 或 P2P Ring，具体代价会在 AI Infra 课程中展开。

---

[继续单元 11-4 →](./04-optimization-infra.md)
