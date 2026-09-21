# 单元 08-1｜长序列的墙：为什么是 CP

> 所属章节：[第 8 章｜Context Parallel（序列并行）](../08-context-parallel.md)

::: info 本单元目标
围绕 **"序列维是三把刀的盲区"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 1. 长上下文把谁逼到墙角

长上下文（128K+）时代，激活与 KV 的显存随序列线性膨胀（Attention 的 $\mathcal{O}(S^2)$ score 矩阵已被 FlashAttention 类算法解决——不物化、分块算）。真正的墙是：**B×S×h 的激活和 KV Cache，S 大到单卡放不下**。

回头看三把刀，**没有一把切序列**：

| 并行 | 切的维 | 每卡序列长度 |
| --- | --- | --- |
| DP | batch 维 | 全长 S |
| TP | head/特征维 | 全长 S |
| PP | 深度维 | 全长 S |

S=128K、h=8K 的场景，无论怎么组合前三把刀，**每一层内部的序列永远是全长的**——墙就在这。

## 2. CP 的直觉：切序列，通信只欠 Attention

**Context Parallel（CP）**：把序列均切成 N 段，每 rank 持 `[B, S/N, h]`。关键问题是——**哪些层被切坏了？**

逐层检查 Transformer 的一层（回扣 [《模型全景》05 章](../../model/05-decoder-block/01-norm-residual-block.md)）：

| 层 | 对序列的依赖 | 序列切开后 |
| --- | --- | --- |
| Norm / Dropout | **逐 token 独立** | ✅ 各算各的，零通信 |
| MLP | 逐 token 独立（$h \to 4h \to h$ 全在特征维） | ✅ 零通信 |
| **Attention** | 第 $i$ 个 token 要看 $0..i$ 的 **K/V**（causal，回扣 [《模型全景》04-2](../../model/04-self-attention/02-scaled-dot-product-mask.md)） | ❌ **本地只有 S/N 段的 K/V，看不全** |

结论收拢成一句话：

::: tip CP 的通信定律
序列切开后，**除 Attention 外的一切层零通信；Attention 必须想办法让每段 Q 看到全序列的 K/V**——CP 的全部通信设计，就是"K/V 怎么流动"这一道题。
:::

顺带认领 [第 6 章](../06-tensor-parallel/04-tp-sp-and-practice.md)的伏笔：那里的 SP 把 **Norm 层**沿序列切、通信只有出口的 RS/AG——正是 CP 定律的"半步版本"（只切了零通信的层，还没切 Attention）。本章补全另一步。

---

[继续单元 08-2 →](./02-attention-comm.md)
