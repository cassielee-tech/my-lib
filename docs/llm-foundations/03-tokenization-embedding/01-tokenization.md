# 单元 03-1｜Tokenization：文本变成整数序列

> 所属章节：[第 3 章｜文本怎样成为模型输入](../03-tokenization-embedding.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **Tokenization：文本变成整数序列** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

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

---

[继续单元 03-2 →](./02-mask-embedding.md)
