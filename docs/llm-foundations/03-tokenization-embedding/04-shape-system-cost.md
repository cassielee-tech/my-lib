# 单元 03-4｜完整 Shape 流程与系统代价

> 所属章节：[第 3 章｜文本怎样成为模型输入](../03-tokenization-embedding.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **完整 Shape 流程与系统代价** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

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

---

[进入本章总结 →](./summary.md)
