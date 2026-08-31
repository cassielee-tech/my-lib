# 03｜Tokenization、Embedding 与语言模型目标

## 这一课要解决的问题

上一课中，我们直接使用了数值张量。但大模型接收的是文字，硬件只能处理数字。这一课要串起完整的数据流：

```text
文本
→ Token
→ Token ID
→ Embedding 向量
→ Transformer 隐藏状态
→ 词表上的预测分数
→ 下一个 Token 的训练 Loss
```

学完后应当能够回答：

- 为什么不能直接把汉字的 Unicode 编码输入模型？
- Token、Token ID 和 Embedding 有什么区别？
- 一条文本怎样变成 `[batch, sequence, hidden]` 张量？
- Decoder-only 模型没有人工标签，怎样从原始文本产生训练目标？
- 词表大小和序列长度为什么会影响显存、计算与通信？

## 1. 文本为什么不能直接进入模型

文本由字符组成，但矩阵乘法需要固定形状的数值张量。我们需要一套稳定规则，把任意文本映射到有限的离散符号集合。

这个符号集合叫 **Vocabulary（词表）**，其中的每个符号叫 **Token**。每个 Token 在词表中有一个整数编号，即 **Token ID**。

### 三个概念不要混淆

| 概念 | 示例 | 本质 |
| --- | --- | --- |
| Token | `喜欢` | Tokenizer 切分得到的文本片段 |
| Token ID | `815` | Token 在词表中的整数索引 |
| Embedding | `[0.12, -0.08, ...]` | 模型为该 Token 学习的浮点向量 |

Token ID 只是查表编号，不包含连续数值意义。ID 815 不代表它比 ID 42“更重要”，也不代表两者语义距离是 773。

## 2. Tokenization：文本到整数序列

Tokenizer 不只是调用字符串的 `split()`。一个完整流程可能包含规范化、预切分、子词算法、ID 映射和特殊 Token 处理。

![文本经过 Tokenizer 变成模型输入](/images/llm/tokenization-pipeline.svg)

### 为什么现代大模型常使用子词或字节级 Token

有三种容易想到的切分粒度：

| 粒度 | 优点 | 问题 |
| --- | --- | --- |
| 整词 | 序列较短，一个 Token 语义较完整 | 词表巨大，容易出现未登录词 |
| 字符 | 词表小，几乎能覆盖任意文本 | 序列更长，单个字符语义有限 |
| 子词/字节 | 在词表大小与序列长度间折中 | 同一文本的切分不一定符合人类直觉 |

子词算法通常让高频片段保留为一个 Token，把低频词拆成更小片段。常见方案包括 BPE、WordPiece、Unigram 和 Byte-level BPE。

例如同一个英文词可能被切成：

```text
tokenization → token + ization
```

中文 Token 也不一定“一字一个”。具体怎样切分完全由对应 Tokenizer 的词表与算法决定。

### Tokenizer 是模型的一部分

模型训练时，Embedding 第 815 行对应某个固定 Token。若推理时换了另一套 Tokenizer，同样的 ID 可能代表完全不同的文本，输入语义就会错位。

因此模型权重必须与以下内容配套使用：

- vocabulary；
- Tokenizer 算法与规范化规则；
- 特殊 Token 及其 ID；
- 最大上下文长度等配置。

## 3. 特殊 Token、Padding 与 Attention Mask

常见特殊 Token 包括：

- `[BOS]`：序列开始；
- `[EOS]`：序列结束；
- `[PAD]`：为了组成 Batch 而补齐的空位置；
- `[UNK]`：无法被词表表示的内容，字节级方案通常很少需要；
- 模型还可能定义对话角色、工具调用等专用 Token。

不同文本的 Token 数量不同，但一个普通张量要求每行长度相同，所以短序列需要 Padding，过长序列需要 Truncation。

### Attention Mask 的作用

假设：

```text
input_ids:
[[1, 42, 815, 9, 2],
 [1, 73,   2, 0, 0]]
```

对应的 Padding Attention Mask 可以是：

```text
attention_mask:
[[1, 1, 1, 1, 1],
 [1, 1, 1, 0, 0]]
```

`1` 表示真实 Token，`0` 表示 Padding。模型应避免让 Padding 位置参与有效注意力或 Loss。

注意不要混淆两种 Mask：

- **Padding Mask**：屏蔽为了对齐长度而补出的 `[PAD]`；
- **Causal Mask**：屏蔽未来 Token，保证当前位置只能看见自己和左侧上下文。

## 4. Embedding：从离散 ID 到连续向量

Embedding 层是一张可训练的参数表：

$$
E \in \mathbb{R}^{V \times H}
$$

其中：

- `V` 是 vocabulary size；
- `H` 是 hidden size；
- 第 `i` 行就是 Token ID `i` 的向量。

![Token ID 通过 Embedding 查表变成向量](/images/llm/embedding-lookup.svg)

输入输出 shape 为：

```text
input_ids: [B, S]       整数
E:         [V, H]       可训练浮点参数
输出:      [B, S, H]    浮点激活
```

### Embedding 是查表，不是把 ID 当数字计算

对于 Token ID `815`，Embedding 前向操作等价于选择 `E[815]`。它在数学上也可以写成 one-hot 向量与矩阵相乘，但实际实现不会构造巨大的 one-hot 矩阵，而是直接 Gather 对应行。

### Embedding 如何学到语义

Embedding 初始通常是随机的。训练时，如果某个 Token 的向量导致后续预测产生 Loss，反向传播会计算该行参数的梯度，优化器再更新它。

出现在相似上下文中的 Token 会收到相似的训练约束，因此它们的向量可能逐渐形成有意义的几何关系。但模型中的“语义”不只存在于 Embedding 表，也分布在后续 Transformer 层中。

### 位置信息从哪里来

Token Embedding 只表示“这是什么 Token”，无法区分同一个 Token 出现在第 1 位还是第 100 位。模型还需要位置信息。

位置表示的具体方法包括：

- 可学习位置 Embedding；
- 正弦/余弦位置编码；
- RoPE（Rotary Position Embedding）等相对位置方法。

这一课只需记住：进入 Transformer 的表示必须同时包含 Token 内容与位置信息。RoPE 会在 Attention 课详细解释。

## 5. Decoder-only 模型怎样从文本获得训练标签

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

## 7. 完整 Shape 流程

设：

```text
B = 2       batch size
S = 4       sequence length
V = 32000   vocabulary size
H = 4096    hidden size
```

数据流如下：

| 阶段 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| Token IDs | `[2, 4]` | integer | 每个位置的词表编号 |
| Token Embedding | `[2, 4, 4096]` | float | 每个 Token 的隐藏向量 |
| Transformer 输出 | `[2, 4, 4096]` | float | 融合上下文后的隐藏状态 |
| LM Head logits | `[2, 4, 32000]` | float | 每个位置对整个词表的分数 |
| Labels | `[2, 4]` | integer | 每个位置的正确下一个 Token ID |
| Loss | `[]` | float | 所有有效位置 Loss 的平均值 |

这里可以看到：输出 logits 的最后一维是词表大小，可能远大于 hidden size。

## 8. 最小 PyTorch 实验

下面用很小的词表演示 Embedding、输出投影与 Loss：

```python
import torch
from torch import nn

B, S, V, H = 2, 4, 10, 8

input_ids = torch.tensor([
    [1, 3, 5, 2],
    [1, 4, 2, 0],
])

embedding = nn.Embedding(V, H, padding_idx=0)
lm_head = nn.Linear(H, V, bias=False)

hidden = embedding(input_ids)       # [B, S, H]
logits = lm_head(hidden)            # [B, S, V]

# 概念上：位置 0..S-2 预测位置 1..S-1
shift_logits = logits[:, :-1, :]    # [B, S-1, V]
shift_labels = input_ids[:, 1:]     # [B, S-1]

# 第二条序列最后一个标签是 PAD，不参与 Loss
shift_labels[shift_labels == 0] = -100

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(
    shift_logits.reshape(-1, V),
    shift_labels.reshape(-1),
)

loss.backward()

print(hidden.shape)  # torch.Size([2, 4, 8])
print(logits.shape)  # torch.Size([2, 4, 10])
print(loss.item())
```

这个示例没有 Transformer，所以各位置还没有融合左侧上下文。它只用于观察文本进入 Transformer 前后所需的 shape 与 Loss 接口。

## 9. 与 AI Infra 的关系

### 词表大小影响参数量

Embedding 参数量为：

$$
V \times H
$$

若 `V=32000、H=4096`：

```text
参数量 = 32,000 × 4,096 = 131,072,000
```

约 1.31 亿参数。仅按 BF16 权重计算约占 250 MiB。若输入 Embedding 与输出 LM Head 不共享权重，还要再保存一份类似大小的矩阵。

### 序列长度影响计算和显存

Token 越多：

- 激活张量 `[B,S,H]` 越大；
- 输出 logits `[B,S,V]` 越大；
- Attention 的计算与中间数据通常增长更快；
- 每轮训练处理的有效 Token 数发生变化。

因此训练吞吐常用 `tokens/s` 而不是“句子数/s”衡量。

### Padding 是无效工作

若同一 Batch 中序列长度差异很大，大量 `[PAD]` 位置虽然不计入 Loss，仍可能占用显存并参与部分计算。工程上会使用长度分桶、动态 Batch、Packing 等方法提高有效 Token 比例。

### Embedding 与输出层的性能特征不同

- Embedding 前向主要是按 ID 随机读取参数行，常更偏向访存；
- LM Head 是 `[B×S,H] × [H,V]` 的大矩阵乘法，计算量可能很高；
- 大词表还会增加 logits 的写出、Softmax/Cross Entropy 的读取和归约开销。

## 10. 容易混淆的点

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

## 12. 本课自测

1. Token、Token ID、Embedding 三者有什么区别？
2. 为什么 Tokenizer 必须与模型权重配套？
3. `[B,S]` 的 input IDs 为什么经过 Embedding 后变成 `[B,S,H]`？
4. Padding Mask 和 Causal Mask 分别屏蔽什么？
5. 一条长度为 `S`、包含 BOS 和 EOS 的序列通常能产生多少个下一个 Token 预测目标？
6. logits `[B,S,V]` 的三个维度分别表示什么？
7. 为什么 `CrossEntropyLoss` 直接接收 logits，而不是手动 Softmax 后的概率？
8. 增大 vocabulary size 会影响哪些参数、计算和内存？
9. 为什么训练吞吐更适合用 tokens/s 衡量？

## 13. 本课自测答案

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
