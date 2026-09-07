# 单元 12-1｜读配置并估算参数量

> 所属章节：[第 12 章｜综合拆解一个现代 LLM](../12-llm-systems-analysis.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **读配置并估算参数量** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

| 配置 | 符号 | 数值 |
| --- | --- | ---: |
| Vocabulary Size | $V$ | 128,000 |
| Hidden Size | $D$ | 4,096 |
| Decoder Layer 数 | $L$ | 32 |
| Query Head 数 | $N_q$ | 32 |
| KV Head 数 | $N_{kv}$ | 8 |
| Head Dimension | $d_h$ | 128 |
| FFN Intermediate Size | $F$ | 14,336 |
| Norm | — | RMSNorm |
| FFN | — | SwiGLU |
| Position | — | RoPE |
| Embedding / LM Head | — | 权重共享 |

先做两个一致性检查：

$$
N_qd_h=32\times128=4096=D
$$

$$
N_{kv}d_h=8\times128=1024
$$

这说明 Q 投影宽度是 4096，K/V 投影宽度各为 1024；每 4 个 Q Head 共享一个 KV Head。

## 3. 参数量从哪里来

参数主要来自三个区域：Embedding、32 个 Decoder Block、输出 LM Head。

![Stack-7.5B 的参数账本](/images/llm/llm-parameter-ledger.svg)

### 3.1 Token Embedding

$$
P_{embed}=V\times D=128000\times4096=524,288,000
$$

约为 **524.3M** 参数。

### 3.2 一层 Attention

忽略 Bias：

$$
P_Q=D\times(N_qd_h)=4096\times4096
$$

$$
P_K=P_V=D\times(N_{kv}d_h)=4096\times1024
$$

$$
P_O=(N_qd_h)\times D=4096\times4096
$$

所以：

$$
P_{attn}=P_Q+P_K+P_V+P_O\approx41.94M
$$

GQA 不仅减少 KV Cache，也让 K/V Projection 的参数少于 Q/O Projection。

### 3.3 一层 SwiGLU FFN

SwiGLU 通常有 Gate、Up、Down 三个矩阵：

$$
P_{ffn}=D\times F+D\times F+F\times D=3DF
$$

$$
P_{ffn}=3\times4096\times14336\approx176.16M
$$

可见 FFN 参数约为 Attention 的 4.2 倍。对这个 Dense 模型来说，大部分参数在 FFN，而不是 Attention Score Matrix。

### 3.4 RMSNorm

每层两个 RMSNorm，每个只有 $D$ 个缩放参数：

$$
P_{norm}=2D=8192
$$

与数亿个矩阵参数相比，它在参数量估算中几乎可以忽略，但在执行链路中不能忽略。

### 3.5 总参数量

单层约为：

$$
P_{layer}\approx41.94M+176.16M=218.10M
$$

32 层加 Embedding：

$$
P\approx32\times218.10M+524.29M\approx7.50B
$$

因为 Embedding 与 LM Head 共享权重，输出层不再增加一份 $V\times D$。若不共享，总参数还要增加约 524.3M，达到约 8.03B。

::: tip 参数量估算顺序
先算大矩阵，再补 Norm 和 Bias。不要一开始陷入几千个参数的误差，却漏掉一整个 Embedding 或 LM Head。
:::

---

[继续单元 12-2 →](./02-model-memory.md)
