# 单元 05-1｜RMSNorm、残差与 Block 结构

> 所属章节：[第 5 章｜完整 Decoder Block](../05-decoder-block.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **RMSNorm、残差与 Block 结构** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

![Pre-Norm Decoder Block 数据流](/images/llm/decoder-block-flow.svg)

用公式表示：

$$
h=x+Attention(RMSNorm(x))
$$

$$
y=h+MLP(RMSNorm(h))
$$

输入与输出 shape 都是 `[B,S,H]`，所以可以连续堆叠很多层。

### 每个模块的分工

| 模块 | 主要作用 | 是否混合 Token 位置 |
| --- | --- | --- |
| RMSNorm | 稳定每个 Token 隐藏向量的尺度 | 否 |
| Self-Attention | 让当前位置读取上下文 | 是 |
| SwiGLU MLP | 对每个 Token 的 hidden 特征做非线性变换 | 否 |
| Residual | 保留主路信息并提供梯度捷径 | 不执行内容混合 |

## 2. 为什么需要归一化

网络堆叠很多层后，隐藏状态的数值尺度可能不断变化。尺度过大或过小都会让训练变得不稳定。归一化让每层接收到的输入处在较可控的范围内。

### RMSNorm

对一个长度为 H 的 Token 向量：

$$
RMS(x)=\sqrt{\frac{1}{H}\sum_{i=1}^{H}x_i^2+\epsilon}
$$

$$
RMSNorm(x)=\frac{x}{RMS(x)}\odot\gamma
$$

其中 `γ [H]` 是可训练缩放参数。

![RMSNorm 的处理过程](/images/llm/rmsnorm-process.svg)

输入 `[B,S,H]` 时，RMSNorm 对每个 `B、S` 位置独立沿最后一个 H 维计算，不会把不同 Token 混在一起。

### RMSNorm 与 LayerNorm

- LayerNorm 会减去均值，再除以标准差；
- RMSNorm 不做中心化，主要按均方根缩放；
- 两者具体性能与效果应以模型和实现为准，不能简单断言一个永远更好。

## 3. Pre-Norm 与 Post-Norm

区别是归一化相对于子层的位置。

Pre-Norm：

$$
y=x+F(Norm(x))
$$

Post-Norm：

$$
y=Norm(x+F(x))
$$

本章使用 Pre-Norm。它让残差主路从 x 到 y 保持直接的加法通道，深层模型通常更容易优化。不同架构仍可能使用 Post-Norm、夹心 Norm 或其他变体。

## 4. 残差连接为什么重要

若子层学习的变换是 `F(x)`，残差输出为：

$$
y=x+F(x)
$$

直观上，子层不必重新生成完整表示，只需学习在当前表示上增加什么修正。

反向传播时：

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}
$$

其中恒等项 `I` 提供一条直接的梯度路径。即使 `F` 的梯度较弱，梯度仍能沿残差主路传播。

### 残差相加要求什么

相加两侧 shape 必须相同。因此 Attention 最后需要输出投影恢复 H，MLP 最后也必须从中间维 I 降回 H：

```text
x:             [B,S,H]
Attention(...):[B,S,H]
MLP(...):      [B,S,H]
```

---

[继续单元 05-2 →](./02-mlp-swiglu.md)
