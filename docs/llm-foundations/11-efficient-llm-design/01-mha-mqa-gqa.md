# 单元 11-1｜MHA、MQA 与 GQA

> 所属章节：[第 11 章｜现代模型的效率设计](../11-efficient-llm-design.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **MHA、MQA 与 GQA** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

设 Hidden Size 为 $D$，Query Head 数为 $N_q$，每个 Head 的维度为 $d_h$，通常有：

$$
D=N_qd_h
$$

标准 Multi-Head Attention（MHA）中：

$$
N_q=N_k=N_v
$$

每个 Query Head 使用与它对应的一组 Key 和 Value Head。表达能力充分，但推理时必须为历史每个 Token 保存全部 K/V Head。

考虑 Batch Size $B$、层数 $L_S$、序列长度 $S$ 和每元素字节数 $b$，KV Cache 大小为：

$$
M_{KV}=2BL_SN_{kv}d_hSb
$$

这里的 $2$ 代表 Key 和 Value。KV Cache 会随 **Batch、层数、KV Head 数、序列长度和位宽**线性增长。

## 3. MQA 与 GQA 怎样共享 K/V

Multi-Query Attention（MQA）让所有 Query Head 共享同一组 K/V：

$$
N_{kv}=1
$$

Grouped-Query Attention（GQA）是 MHA 和 MQA 之间的折中：多个 Query Head 分成一组，共享一个 K/V Head。

$$
1<N_{kv}<N_q
$$

![MHA、GQA 与 MQA 中 Query Head 如何连接到 KV Head](/images/llm/mha-gqa-mqa-sharing.svg)

图中真正变化的是 **K/V 的份数**：

- MHA：8 个 Q Head 对应 8 组 K/V；
- GQA：8 个 Q Head 分成 2 组，只保存 2 组 K/V；
- MQA：8 个 Q Head 全部共享 1 组 K/V。

GQA/MQA 通常不会减少 Q Head，也不会让输出只剩一个 Head。每个 Q Head 仍独立生成注意力输出，只是查询的 K/V 被共享。

### 3.1 从 Shape 看 GQA

以 $N_q=32$、$N_{kv}=8$、$d_h=128$ 为例，每 4 个 Q Head 共享一组 K/V：

| 张量 | Shape |
| --- | --- |
| $Q$ | `[B, 32, S, 128]` |
| $K$ | `[B, 8, S, 128]` |
| $V$ | `[B, 8, S, 128]` |
| Attention 输出 | `[B, 32, S, 128]` |

高效实现不一定真的将 K/V 复制 4 份，而会利用广播、Stride 或专用 Kernel 完成 Head 映射。

### 3.2 一个 KV Cache 算例

假设模型有 32 层、32 个 Q Head，$d_h=128$，序列长度为 32768，Batch Size 为 1，KV 使用 BF16：

| 结构 | KV Head 数 | KV Cache |
| --- | ---: | ---: |
| MHA | 32 | 16 GiB |
| GQA | 8 | 4 GiB |
| MQA | 1 | 0.5 GiB |

GQA 相比 MHA 把缓存降至 $N_{kv}/N_q=1/4$。它通常比 MQA 保留更多 K/V 表达能力，又能明显降低 Decode 时的缓存读取量，因此成为常见折中。

## 4. GQA 对 Tensor Parallel 的影响

Tensor Parallel 常把 Attention Head 分给不同 Rank。MHA 的 Q/K/V Head 数相同，较容易平均切分；GQA 的 KV Head 更少，会产生新的约束：

- $N_{kv}$ 能被 TP Size 整除时，可以均匀分片；
- TP Size 大于 $N_{kv}$ 时，部分 Rank 可能复制同一 KV Head；
- 复制可以简化计算或减少通信，但会增加显存；
- 细切单个 KV Head 则可能引入额外通信和复杂 Kernel。

例如 $N_q=32$、$N_{kv}=8$：TP=8 时每个 Rank 可以持有 4 个 Q Head 和 1 个 KV Head；TP=16 时 KV Head 不够一一分配，通常需要复制或调整并行布局。

所以模型配置会直接限制并行策略。看到 `num_attention_heads` 和 `num_key_value_heads` 时，也要想到 Head 如何映射到 Rank。

---

[继续单元 11-2 →](./02-quantization.md)
