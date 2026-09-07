# 第 6 章｜RoPE、位置编码与长上下文

> 本章目标：理解注意力为什么必须获得位置信息；看懂 RoPE 如何旋转 Query 和 Key；解释它为什么让注意力分数感知相对位置，并建立长上下文的基本认识。

## 本章导学

::: tip 本章核心结论
1. Attention 只比较内容，本身不知道 Token 的先后顺序。
2. RoPE 按位置旋转 Q 和 K，使它们的点积包含相对位置信息。
3. 长上下文不只受位置编码限制，还受 Attention 计算、KV Cache 和训练长度限制。
:::

**学习节奏：** 本章拆为 **3 个学习单元，每个约 15 分钟**。二维旋转公式第一次只看几何含义即可。

**先看这几张核心图：** “我爱于金凤”的顺序图、RoPE 相对旋转图和长上下文问题总结。

- [ ] 我能解释为什么交换词序会改变含义
- [ ] 我知道 RoPE 作用于 Q/K 而不是 V
- [ ] 我能说出长上下文的两个额外瓶颈

## 本章单元

- **06-1（约 15 分钟）**：[注意力为什么需要位置信息](./06-rope-long-context/01-position-information.md)
- **06-2（约 15 分钟）**：[二维旋转与相对位置](./06-rope-long-context/02-rotation-relative-position.md)
- **06-3（约 15 分钟）**：[RoPE 的 Shape 与长上下文限制](./06-rope-long-context/03-rope-shape-long-context.md)

- **本章总结**：[串联知识、练习与检查](./06-rope-long-context/summary.md)
