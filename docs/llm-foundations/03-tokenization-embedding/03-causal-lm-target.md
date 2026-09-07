# 单元 03-3｜训练标签与下一个 Token 预测

> 所属章节：[第 3 章｜文本怎样成为模型输入](../03-tokenization-embedding.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **训练标签与下一个 Token 预测** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

大模型预训练不需要人工为每句话手写分类标签。对于 Causal Language Modeling，文本本身就可以通过“错开一位”产生监督信号。

![Decoder-only 语言模型的错位预测目标](/images/llm/causal-lm-shift.svg)

给定完整序列：

```text
[BOS] 我 喜欢 AI [EOS]
```

它提供四个训练样本：

```text
[BOS]          → 我
[BOS] 我       → 喜欢
[BOS] 我 喜欢  → AI
[BOS] 我 喜欢 AI → [EOS]
```

Transformer 可以并行计算所有位置，但 Causal Mask 保证第 `i` 个位置无法看到未来答案。这使训练阶段与逐 Token 生成阶段保持一致。

### 代码中的 labels 为什么经常与 input_ids 相同

许多 Causal LM 接口允许直接设置：

```python
labels = input_ids.clone()
```

这是因为模型内部会自动完成 logits 与 labels 的一位错位。概念上仍然是“当前位置预测下一个 Token”，不要误解为预测当前位置自己。

## 6. 从隐藏状态到词表预测

Transformer 为每个位置输出一个长度为 `H` 的隐藏向量：

```text
hidden_states: [B, S, H]
```

输出投影把它映射到整个词表：

```text
W_output: [V, H]
logits:   [B, S, V]
```

每个位置会得到 `V` 个 logits，即词表中每个 Token 的原始分数。

![从隐藏向量到 Cross Entropy Loss](/images/llm/logits-cross-entropy.svg)

### Logit、概率和预测 Token

- **Logit**：模型输出的任意实数分数，不要求为正或总和为 1；
- **Softmax 概率**：把全部 logits 变成和为 1 的分布；
- **预测 Token**：可以选最高概率的 Token，也可以按概率采样。

Softmax 定义为：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

其中 `z_i` 是 Token `i` 的 logit。

### Cross Entropy 在惩罚什么

如果正确 Token 的预测概率是 $p_y$，单个位置的 Loss 为：

$$
L=-\log p_y
$$

- 正确 Token 概率越接近 1，Loss 越接近 0；
- 正确 Token 概率越低，Loss 越大；
- 反向传播会提高正确 Token 的相对分数，降低错误 Token 的相对分数。

PyTorch 的 `CrossEntropyLoss` 直接接收 logits，不需要手动 Softmax。它会采用数值更稳定的方式合并 LogSoftmax 与负对数似然计算。

---

[继续单元 03-4 →](./04-shape-system-cost.md)
