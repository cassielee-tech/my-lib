# 第 5 章总结｜完整 Decoder Block

> 返回：[第 5 章首页](../05-decoder-block.md)

::: tip 本章核心结论
1. Attention 在 Token 之间混合信息，SwiGLU FFN 在每个 Token 内部混合特征。
2. RMSNorm 控制数值尺度，残差连接保留原信息并提供更直接的梯度路径。
3. 一个现代 Pre-Norm Block 是两次“Norm → 子层 → Residual”的串联。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

### MLP 不等于分类器

Block 内 MLP 是逐 Token 的非线性特征变换，不是最终 LM Head。

### Norm 不会改变 shape

它改变数值分布，但输入输出仍为 `[B,S,H]`。

### 残差不是简单“防止信息丢失”

它既保留主路表示，也提供更直接的梯度路径，并要求分支输出与主路 shape 一致。

### 参数最多的模块不一定最慢

运行时间还取决于 Batch、序列长度、算术强度、访存、并行策略和 Kernel 实现。

### 所有 LLM Block 并不完全相同

模型可能使用 LayerNorm/RMSNorm、GeLU/SwiGLU、MHA/GQA/MQA、不同 Norm 位置和不同 bias 设置。阅读源码必须先看配置。

## 14. 动手任务

### 必做 1：参数量

给定 `H=2048、I=5632`，使用标准 MHA、SwiGLU、两个 RMSNorm，忽略 bias：

1. Attention 参数量是多少？
2. MLP 参数量是多少？
3. 两个 RMSNorm 参数量是多少？
4. 单层合计是多少？

::: details 必做 1 参考答案

```text
Attention = 4H²
          = 4 × 2048²
          = 16,777,216

MLP = 3HI
    = 3 × 2048 × 5632
    = 34,603,008

RMSNorm = 2H
        = 4,096

单层合计 = 51,384,320 参数
```

:::

### 必做 2：Shape 跟踪

对 `x=[2,128,2048]、I=5632`，写出以下张量 shape：

- RMSNorm 输出；
- Attention 输出及第一次残差结果；
- Gate/Up 输出；
- 逐元素门控结果；
- Down 输出及 Block 最终输出。

::: details 必做 2 参考答案

```text
RMSNorm 输出:       [2,128,2048]
Attention 输出:     [2,128,2048]
第一次残差结果 h:   [2,128,2048]
Gate 输出:          [2,128,5632]
Up 输出:            [2,128,5632]
逐元素门控结果:     [2,128,5632]
Down 输出:          [2,128,2048]
Block 最终输出:     [2,128,2048]
```

:::

## ✅ 理解检查

1. 一个 Pre-Norm Decoder Block 的执行顺序是什么？
2. RMSNorm 沿哪个维度计算？会不会混合不同 Token？
3. 为什么残差分支输出必须恢复到 H？
4. Attention 与 MLP 在“混合什么”方面有何区别？
5. SwiGLU 为什么有 Gate、Up、Down 三个矩阵？
6. 标准 MHA 的投影参数量为什么约为 `4H²`？
7. SwiGLU 参数量为什么约为 `3HI`？
8. 参数量和训练显存为什么不能画等号？
9. 哪些 Block 操作更可能受内存带宽影响？
10. 张量并行切分 MLP 后，为什么 Down 输出可能需要 AllReduce？

## 16. 参考答案

::: details 1. Pre-Norm Decoder Block 的顺序是什么？

先对输入做 RMSNorm，再执行 Causal Self-Attention，并与原输入残差相加；然后对结果做第二次 RMSNorm，执行 SwiGLU MLP，再与 MLP 前的状态残差相加。

:::

::: details 2. RMSNorm 沿哪个维度计算？

通常沿 `[B,S,H]` 的最后一个 H 维，对每个 batch、每个 Token 独立计算 RMS。它不会在 S 维混合不同 Token。

:::

::: details 3. 为什么残差分支要恢复 H？

残差操作是逐元素相加，两个张量必须具有相同 shape。主路是 `[B,S,H]`，所以 Attention 和 MLP 分支最终也必须输出 `[B,S,H]`。

:::

::: details 4. Attention 与 MLP 分别混合什么？

Attention 通过 S×S 权重在不同 Token 位置间混合信息；MLP 对每个 Token 独立，在 hidden/intermediate 特征维中进行非线性变换。

:::

::: details 5. SwiGLU 为什么有三个矩阵？

Gate 将 H 投影到 I 并经 SiLU 生成门控，Up 将 H 投影到 I 生成候选内容，两者逐元素相乘；Down 再把 I 降回 H，以便接回残差主路。

:::

::: details 6. Attention 参数为什么约为 4H²？

标准 MHA 中 Q、K、V 和输出投影各有一个 `[H,H]` 矩阵。忽略 bias 后合计 `H²+H²+H²+H²=4H²`。

:::

::: details 7. SwiGLU 参数为什么约为 3HI？

Gate 和 Up 各有一个 `[H,I]` 权重，Down 有一个 `[I,H]` 权重，合计 `HI+HI+IH=3HI`。

:::

::: details 8. 为什么参数量不等于训练显存？

训练显存还包括参数梯度、优化器状态、前向激活、反向所需保存值、通信 Bucket、Kernel Workspace 和框架管理开销。激活还随 B、S 和重计算策略变化。

:::

::: details 9. 哪些操作更可能受内存带宽影响？

RMSNorm、残差相加、SiLU 和逐元素门控等操作每读取一个元素只执行少量计算，算术强度通常较低，更容易受 HBM 带宽与 Kernel 启动开销影响。具体结论仍应通过 Profile 验证。

:::

::: details 10. 为什么切分后的 Down 可能需要 AllReduce？

若各 Rank 只持有 intermediate 维的一部分，Down 矩阵乘法会产生对应分片对完整 H 输出的局部贡献。完整结果是这些局部贡献之和，因此需要跨 Rank 归约；也可根据下一层数据布局选择 ReduceScatter。

:::

## 参考资料

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [PyTorch：RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [PyTorch：Tensor Parallel 教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

下一课会解释 RoPE：位置信息如何进入 Q/K，旋转怎样让 Attention 分数感知相对距离，以及长上下文扩展为什么困难。

---

<!-- chapter-navigation -->
[进入第 6 章 →](../06-rope-long-context.md)
