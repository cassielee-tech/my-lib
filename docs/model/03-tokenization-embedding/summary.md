# 第 3 章总结｜文本怎样成为模型输入

> 返回：[第 3 章首页](../03-tokenization-embedding.md)

::: tip 本章核心结论
1. Tokenizer 把文本切成 Token 并映射为离散 Token ID。
2. Embedding 是按 ID 查表，把离散编号变成可计算的连续向量。
3. Causal LM 用当前位置预测下一个 Token，训练标签相对输入错开一位。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

### Token 不等于字或词

Token 是 Tokenizer 词表中的单元，可能是一个字、一个词、子词、字节或特殊符号。

### Token ID 不带语义距离

ID 是离散索引；语义关系来自学习到的向量和后续网络参数。

### Embedding 不是固定词典释义

Embedding 是训练得到的参数。真正结合上下文的表示会在 Transformer 各层中不断变化。

### Padding Mask 不等于 Causal Mask

前者忽略补齐位置，后者阻止看到未来位置，两者解决不同问题并可能同时存在。

### Logits 不等于概率

Logits 尚未归一化。只有经过 Softmax 后才能解释为概率分布。

### 训练时并不是一次只预测一个位置

Causal Mask 允许模型在一次前向中并行产生所有位置的下一个 Token 预测；自回归生成时才需要逐 Token 迭代。

## 11. 动手任务

### 必做 1：Shape 与参数量

给定：

```text
B = 2
S = 128
V = 50000
H = 4096
dtype = BF16
```

回答：

1. `input_ids`、Embedding 输出、logits 的 shape 分别是什么？
2. Embedding 表有多少参数？BF16 权重理论占多少 Byte？
3. logits 有多少个元素？BF16 下理论占多少 Byte？

::: details 必做 1 参考答案

Shape：

```text
input_ids:       [2, 128]
Embedding 输出: [2, 128, 4096]
logits:          [2, 128, 50000]
```

Embedding 参数量：

```text
V × H = 50,000 × 4,096 = 204,800,000 个参数
```

BF16 每个参数 2 Byte：

```text
204,800,000 × 2 = 409,600,000 Byte
约 390.625 MiB
```

logits 元素数量：

```text
B × S × V = 2 × 128 × 50,000 = 12,800,000
```

BF16 理论空间：

```text
12,800,000 × 2 = 25,600,000 Byte
约 24.414 MiB
```

:::

### 必做 2：运行最小代码

运行上面的 PyTorch 示例并打印：

```python
print(input_ids.shape)
print(hidden.shape)
print(logits.shape)
print(shift_logits.shape)
print(shift_labels.shape)
print(embedding.weight.grad.shape)
print(lm_head.weight.grad.shape)
```

先写出预期结果，再与实际输出比较。

### 选做：观察 Tokenizer

选择一个与你后续实验模型配套的 Hugging Face Tokenizer，比较以下文本的 Token 数量：

```text
集合通信
Huawei Collective Communication Library
HCCL实现AllReduce集合通信
```

观察中文、英文、缩写和混合文本怎样被切分。不要把访问令牌或内部文本放入公开在线工具。

## ✅ 理解检查

1. Token、Token ID、Embedding 三者有什么区别？
2. 为什么 Tokenizer 必须与模型权重配套？
3. `[B,S]` 的 input IDs 为什么经过 Embedding 后变成 `[B,S,H]`？
4. Padding Mask 和 Causal Mask 分别屏蔽什么？
5. 一条长度为 `S`、包含 BOS 和 EOS 的序列通常能产生多少个下一个 Token 预测目标？
6. logits `[B,S,V]` 的三个维度分别表示什么？
7. 为什么 `CrossEntropyLoss` 直接接收 logits，而不是手动 Softmax 后的概率？
8. 增大 vocabulary size 会影响哪些参数、计算和内存？
9. 为什么训练吞吐更适合用 tokens/s 衡量？

## 13. 参考答案

::: details 1. Token、Token ID、Embedding 三者有什么区别？

- Token 是 Tokenizer 切出的文本单元；
- Token ID 是该单元在固定词表中的整数索引；
- Embedding 是模型参数表中由该 ID 选出的浮点向量。

它们对应“文本符号 → 离散编号 → 可学习连续表示”三个阶段。

:::

::: details 2. 为什么 Tokenizer 必须与模型权重配套？

Embedding 的每一行在训练时已经绑定到固定 Token。如果换了 Tokenizer，同一 ID 可能代表不同文本，模型会查到错误向量。特殊 Token、规范化和切分方式不一致也会改变输入序列。

:::

::: details 3. 为什么 [B,S] 经过 Embedding 后变成 [B,S,H]？

`[B,S]` 中每个位置存放一个整数 ID。Embedding 用每个 ID 从 `[V,H]` 参数表中选择一行长度为 `H` 的向量，因此每个原位置都多出一个 hidden 维，得到 `[B,S,H]`。

:::

::: details 4. Padding Mask 和 Causal Mask 分别屏蔽什么？

- Padding Mask 屏蔽为了对齐 Batch 长度而补出的 `[PAD]` 位置；
- Causal Mask 屏蔽当前位置右侧的未来 Token，避免训练时提前看到答案。

它们可以同时应用。

:::

::: details 5. 长度为 S、包含 BOS 和 EOS 的序列能产生多少个目标？

通常是 `S-1` 个。第 1 个到第 `S-1` 个 Token 作为输入位置，分别预测后面的一个 Token。最后一个 EOS 后面若没有继续定义目标，就不再产生预测标签。

:::

::: details 6. logits [B,S,V] 的三个维度分别表示什么？

- `B`：Batch 中有多少条序列；
- `S`：每条序列有多少个位置；
- `V`：每个位置对词表中所有 Token 分别给出的分数。

所以固定一个 batch 和位置后，会得到长度为 `V` 的向量，用于预测下一个 Token。

:::

::: details 7. 为什么 CrossEntropyLoss 直接接收 logits？

交叉熵内部需要计算 LogSoftmax。把 Softmax 与对数等步骤合并实现，可以使用 log-sum-exp 等技巧避免指数溢出、概率下溢，也能减少不必要的中间张量和算子。因此应直接传入未归一化 logits。

:::

::: details 8. 增大 vocabulary size 会影响什么？

- Embedding 参数量 `V×H` 增大；
- 若 LM Head 不共享权重，其参数也随之增大；
- logits 最后一维增大，占用更多激活内存；
- 输出投影矩阵乘法、Softmax/Cross Entropy 的计算与访存增加；
- 但更大的词表有时能用更少 Token 表示同一文本，需要综合权衡。

:::

::: details 9. 为什么训练吞吐更适合用 tokens/s？

不同文本经过 Tokenizer 后的长度不同，“一条文本”包含的计算工作量并不固定。模型计算通常与实际 Token 数和序列长度直接相关，因此 tokens/s 比 samples/s 更容易比较真实处理能力。

还应区分总 Token 与非 Padding 的有效 Token，否则大量 Padding 可能让吞吐指标看起来很高但有效利用率很低。

:::

## 参考资料

- [Hugging Face LLM Course：Tokenizers](https://huggingface.co/learn/llm-course/en/chapter2/4)
- [Hugging Face：Padding 与 Truncation](https://huggingface.co/docs/transformers/pad_truncation)
- [PyTorch：Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch：CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

下一课将进入 Attention 前的最后一块基础：向量相似度、Softmax、Q/K/V，以及一个 Token 如何从上下文中选择并聚合信息。

---

<!-- chapter-navigation -->
[进入第 4 章 →](../04-self-attention.md)
