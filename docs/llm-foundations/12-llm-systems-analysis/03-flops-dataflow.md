# 单元 12-3｜FLOPs 与完整数据流

> 所属章节：[第 12 章｜综合拆解一个现代 LLM](../12-llm-systems-analysis.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **FLOPs 与完整数据流** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

矩阵乘法 $[M,K]\times[K,N]$ 大约包含：

$$
2MKN\ FLOPs
$$

因为一次乘加通常按一次乘法加一次加法，即 2 FLOPs 计算。

### 7.1 参数相关计算

Dense Transformer 每个 Token 的一次前向，粗略可用：

$$
F_{forward}\approx2P
$$

Stack-7.5B 约为 **15 GFLOPs / Token**。这个估算抓住了各权重矩阵参与一次乘法，但没有完整包含 Attention 的序列相关项、Norm、激活函数等。

### 7.2 Attention 的序列相关计算

每层的 $QK^T$ 和 $PV$ 大致为：

$$
F_{score}\approx4S^2D
$$

当序列很长时，这部分不能忽略。它解释了为什么同一个参数量的模型，长序列 Prefill 比短序列更昂贵。

### 7.3 训练计算

对 Dense 模型，前向加反向常粗略估算为：

$$
F_{train}\approx6P\quad /Token
$$

Stack-7.5B 约为 **45 GFLOPs / Token**。若一个优化 Step 处理 100 万个 Token，仅参数相关计算就约为 45 PFLOPs。

这仍是估算。Attention 长度项、Checkpointing 重计算、Padding 和 MoE 稀疏激活都会让真实计算偏离简单公式。

## 8. 一个 Token 的完整前向数据流

![从文本到 Loss 或下一个 Token 的完整数据流](/images/llm/llm-end-to-end-flow.svg)

### 8.1 输入

```text
文本 → Tokenizer → Token ID [B,S] → Embedding [B,S,D]
```

Tokenizer 不在 NPU/GPU 的 Transformer Kernel 中运行；Embedding 本质上是按 Token ID 查表。

### 8.2 进入每个 Decoder Block

```text
Hidden State
  ├─ RMSNorm → Q/K/V Projection → RoPE → Causal Attention → O Projection → 残差
  └─ RMSNorm → Gate/Up Projection → SwiGLU → Down Projection → 残差
```

主要算子包括 GEMM、RMSNorm、RoPE、Softmax、逐元素乘法和残差加法。训练时还需要保存或重算反向所需的信息。

### 8.3 输出

```text
Hidden State → Final RMSNorm → LM Head → Logits [B,S,V]
```

- 训练：Shift Label 后计算 Cross Entropy Loss，再反向传播；
- 推理：取最后位置 Logits，经过采样得到下一个 Token，并更新 KV Cache。

## 9. 反向传播时发生什么

反向传播沿计算图反向应用链式法则：

1. Cross Entropy 产生 Logits Gradient；
2. LM Head 产生 Hidden Gradient 和权重梯度；
3. 梯度倒序穿过 32 个 Decoder Block；
4. 每个 Linear 同时计算输入梯度与权重梯度；
5. Optimizer 使用最终梯度更新参数。

若使用数据并行，每个 Rank 只看到不同 Mini-batch，得到的本地梯度不同。为了让所有模型副本保持一致，需要在更新前聚合梯度。

---

[继续单元 12-4 →](./04-sharding-performance.md)
