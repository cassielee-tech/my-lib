# 单元 05-2｜MLP 与 SwiGLU 数据流

> 所属章节：[第 5 章｜完整 Decoder Block](../05-decoder-block.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **MLP 与 SwiGLU 数据流** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

Attention 主要让不同 Token 位置交换信息；MLP 则在每个 Token 内部混合和变换 hidden 特征，并通过非线性提高模型表达能力。

普通两层 MLP 可以写成：

$$
MLP(x)=\phi(xW_{up})W_{down}
$$

现代模型常使用门控变体，例如 SwiGLU。

## 6. SwiGLU 的数据流

一种常见形式是：

$$
SwiGLU(x)=\left(SiLU(xW_{gate})\odot(xW_{up})\right)W_{down}
$$

其中：

$$
SiLU(z)=z\cdot sigmoid(z)
$$

![SwiGLU 的双分支门控](/images/llm/swiglu-flow.svg)

Shape：

```text
x:                    [B,S,H]
xW_gate, xW_up:       [B,S,I]
逐元素门控结果:        [B,S,I]
乘 W_down 后:         [B,S,H]
```

`I` 是 intermediate size，通常大于 H。MLP 在每个 Token 上独立使用同一组权重，不直接在 S 维做信息交换。

---

[继续单元 05-3 →](./03-parameters-flops-memory.md)
