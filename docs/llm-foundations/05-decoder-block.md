# 第 5 章｜RMSNorm、SwiGLU 与完整 Decoder Block

## 本章导学

::: tip 本章核心结论
1. Attention 在 Token 之间混合信息，SwiGLU FFN 在每个 Token 内部混合特征。
2. RMSNorm 控制数值尺度，残差连接保留原信息并提供更直接的梯度路径。
3. 一个现代 Pre-Norm Block 是两次“Norm → 子层 → Residual”的串联。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，分别理解结构、FFN、资源和并行。

**先看这几张核心图：** Decoder Block 总图、RMSNorm Process、SwiGLU Flow 三张图。

- [ ] 我能默画一个 Decoder Block
- [ ] 我能区分 Attention 与 FFN 的职责
- [ ] 我能解释残差连接为什么不是简单重复

## 本章要解决的问题

前四章已经得到 Token 表示并学会了 Attention，但 Attention 只是 Transformer Block 的一个子模块。现代 Decoder-only 模型通常还包含归一化、MLP 和残差连接。

本章以常见的 Llama 风格 Pre-Norm Block 为主线：

```text
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

学完后应能：

- 画出一个完整 Decoder Block；
- 解释 RMSNorm、残差和 MLP 各自解决什么问题；
- 区分 Attention 的 Token 混合与 MLP 的特征混合；
- 估算一层的参数量与主要 FLOPs；
- 从 Block 结构定位张量并行可能产生通信的位置。

## 本章单元

- **05-1（约 15 分钟）**：[RMSNorm、残差与 Block 结构](./05-decoder-block/01-norm-residual-block.md)
- **05-2（约 15 分钟）**：[MLP 与 SwiGLU 数据流](./05-decoder-block/02-mlp-swiglu.md)
- **05-3（约 15 分钟）**：[参数量、FLOPs 与显存对象](./05-decoder-block/03-parameters-flops-memory.md)
- **05-4（约 15 分钟）**：[张量并行怎样切分 Block](./05-decoder-block/04-tensor-parallel-block.md)

- **本章总结**：[串联知识、练习与检查](./05-decoder-block/summary.md)
