# 第 6 章总结｜RoPE 与长上下文

> 返回：[第 6 章首页](../06-rope-long-context.md)

::: tip 本章核心结论
1. Attention 只比较内容，本身不知道 Token 的先后顺序。
2. RoPE 按位置旋转 Q 和 K，使它们的点积包含相对位置信息。
3. 长上下文不只受位置编码限制，还受 Attention 计算、KV Cache 和训练长度限制。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

1. **RoPE 是给 Token Embedding 加一个位置向量吗？** 不是。它通常在得到 Q/K 后旋转其维度对。
2. **RoPE 会旋转 V 吗？** 主流用法通常不旋转 V；以具体模型实现为准。
3. **RoPE 有可训练参数吗？** 标准 RoPE 的频率按公式生成，通常没有可训练参数。
4. **相对位置是否意味着不需要绝对 position id？** 不是。每个 Q/K 仍先根据绝对 position id 旋转，点积后呈现相对关系。
5. **扩大 RoPE 缓存就能获得长上下文吗？** 不能。还涉及模型外推能力、训练/微调、Attention 计算量和 KV Cache 显存。

## 10. 动手练习

### 练习 1：验证长度保持不变

运行上面的最小实现，再比较旋转前后的向量二范数：

```python
print(torch.max(torch.abs(q.norm(dim=-1) - q_rotated.norm(dim=-1))))
```

结果应接近 0，因为二维旋转不改变向量长度。

### 练习 2：验证共同平移不改变相对点积

任选二维向量 $q,k$ 和位置 $m,n$，分别计算：

$$
[R(m\theta)q]^T[R(n\theta)k]
$$

以及同时平移 10 个位置后的：

$$
[R((m+10)\theta)q]^T[R((n+10)\theta)k]
$$

两者应近似相等。

## 11. 自测题

1. 没有位置编码的 Self-Attention 缺少什么信息？
2. RoPE 在 Attention 链路中的位置在哪里？
3. 为什么标准 RoPE 通常只作用于 Q 和 K？
4. $D=128$ 的一个注意力头包含多少个二维旋转对？
5. RoPE 旋转前后为什么不改变向量长度？
6. 如何从公式说明 RoPE 点积依赖相对位置？
7. 为什么不同维度对要使用不同频率？
8. 直接增大 position id 为什么不等于获得可靠长上下文？
9. RoPE 本身通常需要集合通信吗？
10. Context Parallel 中 position id 使用错误会发生什么？

::: details 自测答案

1. 缺少 Token 的顺序和距离信息；它只能根据内容进行匹配。
2. 在线性投影得到 Q/K 后、计算 $QK^T$ 前。
3. Q/K 决定匹配分数，旋转二者就能把位置关系写进分数；V 负责携带被取回的内容。
4. $128/2=64$ 个。
5. 旋转矩阵是正交矩阵，满足 $R^TR=I$，因此保持二范数不变。
6. $(R(m\theta)q)^T(R(n\theta)k)=q^TR((n-m)\theta)k$，位置通过差值 $n-m$ 出现。
7. 高频擅长区分近距离，低频能覆盖更长距离；多种尺度共同表达位置。
8. 超出训练长度后模型会遇到陌生相位；同时 Attention 的平方计算量和 KV Cache 显存问题仍然存在。
9. 通常不需要，它是本地逐元素计算。
10. 各 Rank 会把不同的全局 Token 当成相同位置，导致 Q/K 旋转错误，最终注意力结果错误。

:::

## 12. 本章小结

- Self-Attention 本身不能从内容计算中恢复 Token 顺序；
- RoPE 把注意力头维度两两配对，并按位置和频率旋转 Q/K；
- 两次旋转后的点积包含 $(n-m)$，因而能够表达相对位置；
- 多频率让模型同时观察近距离与远距离；
- 长上下文不仅是位置编码问题，还受训练分布、Attention 计算量和 KV Cache 限制；
- RoPE 通常没有通信，但并行切分必须保持旋转对和全局 position id 正确。

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Hugging Face Transformers：RoPE utilities](https://huggingface.co/docs/transformers/internal/rope_utils)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

下一课将进入训练循环，把 Loss、反向传播、AdamW、学习率、混合精度和梯度累积连成一次完整的参数更新。

---

<!-- chapter-navigation -->
[进入第 7 章 →](../07-training-loop.md)
