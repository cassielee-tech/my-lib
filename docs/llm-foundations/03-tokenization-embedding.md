# 第 3 章｜Tokenization、Embedding 与语言模型目标

## 本章导学

::: tip 本章核心结论
1. Tokenizer 把文本切成 Token 并映射为离散 Token ID。
2. Embedding 是按 ID 查表，把离散编号变成可计算的连续向量。
3. Causal LM 用当前位置预测下一个 Token，训练标签相对输入错开一位。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，依次跟踪文本、ID、向量和 Logits。

**先看这几张核心图：** 开头的数据流、Tokenization Pipeline、Embedding Lookup 和 Causal LM Shift 四张图。

- [ ] 我能解释 Token、Token ID、Embedding 的区别
- [ ] 我能画出文本到 Logits 的路径
- [ ] 我能解释训练标签为什么左移

## 本章要解决的问题

上一章中，我们直接使用了数值张量。但大模型接收的是文字，硬件只能处理数字。本章要串起完整的数据流：

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

## 本章单元

- **03-1（约 15 分钟）**：[Tokenization：文本变成整数序列](./03-tokenization-embedding/01-tokenization.md)
- **03-2（约 15 分钟）**：[特殊 Token、Mask 与 Embedding](./03-tokenization-embedding/02-mask-embedding.md)
- **03-3（约 15 分钟）**：[训练标签与下一个 Token 预测](./03-tokenization-embedding/03-causal-lm-target.md)
- **03-4（约 15 分钟）**：[完整 Shape 流程与系统代价](./03-tokenization-embedding/04-shape-system-cost.md)

- **本章总结**：[串联知识、练习与检查](./03-tokenization-embedding/summary.md)
