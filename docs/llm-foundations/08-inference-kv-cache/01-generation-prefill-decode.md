# 单元 08-1｜生成循环、Prefill 与 Decode

> 所属章节：[第 8 章｜自回归推理与 KV Cache](../08-inference-kv-cache.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **生成循环、Prefill 与 Decode** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

语言模型一次前向传播输出的是每个位置的下一个 Token 分布。生成答案时，程序只取最后一个位置的 Logits，选出一个 Token，把它追加到上下文末尾，再执行下一次前向传播。

假设输入是“天空为什么是蓝色”，生成过程可能是：

```text
天空为什么是蓝色 → 因
天空为什么是蓝色因 → 为
天空为什么是蓝色因为 → 瑞
天空为什么是蓝色因为瑞 → 利
……
```

每个新 Token 都依赖完整的已有上下文，因此这叫 **Autoregressive Generation，自回归生成**。

![Prompt 经过 Prefill 后进入逐 Token Decode 循环](/images/llm/autoregressive-inference-loop.svg)

完整过程可以分成：

1. Tokenize：文本变为 Token ID；
2. Prefill：并行处理全部 Prompt Token，建立 KV Cache；
3. 选择第一个输出 Token；
4. Decode：每次只输入最新 Token，复用历史 Cache；
5. Sampling：从当前概率分布选择下一个 Token；
6. 遇到停止条件后 Detokenize，返回文本。

## 2. Prefill 与 Decode 有什么不同

### 2.1 Prefill

Prompt 有 $S$ 个 Token 时，Prefill 会一次处理 `[B,S]` 的输入。每一层都为这 $S$ 个位置计算 Q、K、V，并将 K/V 写入 Cache。

Prefill 的特点：

- 一次处理许多 Token；
- 大矩阵乘法较多，并行度高；
- Attention 需要处理 Prompt 内部的 Token 关系；
- 产生第一个输出 Token；
- 直接影响 **TTFT（Time To First Token，首 Token 延迟）**。

### 2.2 Decode

进入 Decode 后，每轮只输入刚生成的一个 Token。模型为它计算新的 Q/K/V，再让当前 Q 查询全部历史 K，并用注意力权重汇聚历史 V。

Decode 的特点：

- 每轮、每个请求通常只新增一个 Token；
- 必须串行执行，下一轮依赖上一轮结果；
- 每轮都要读取模型权重和不断增长的 KV Cache；
- 单步并行度比 Prefill 小；
- 影响 **TPOT（Time Per Output Token）** 或 **ITL（Inter-Token Latency）**。

![Prefill 与 Decode 的输入规模和硬件行为不同](/images/llm/prefill-vs-decode.svg)

在许多常见部署条件下，Prefill 更容易体现计算密集特征；小 Batch Decode 更容易受模型权重和 KV Cache 的内存带宽限制。这不是绝对结论，具体瓶颈仍取决于 Batch、序列长度、模型结构、精度和硬件。

---

[继续单元 08-2 →](./02-kv-cache-memory.md)
