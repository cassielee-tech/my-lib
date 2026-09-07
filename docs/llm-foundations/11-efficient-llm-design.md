# 第 11 章｜现代模型的效率设计：GQA、量化与长上下文

> 本章目标：理解现代大模型怎样减少计算、显存和数据搬运；能够计算 MHA、GQA、MQA 的 KV Cache；分清权重量化、激活量化与 KV Cache 量化；理解长上下文优化究竟改变了算法、数据布局还是执行方式。

## 本章导学

::: tip 本章核心结论
1. GQA/MQA 共享 K/V Head，主要减少 KV Cache 和 Decode 搬运，不是减少 Q Head。
2. 量化降低存储位宽，但收益取决于量化对象、粒度、硬件和 Kernel。
3. RoPE Scaling、Sliding Window、FlashAttention 和 Paged Attention 解决的是四类不同问题。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。量化公式可留到第二遍阅读。

**先看这几张核心图：** MHA/GQA/MQA Sharing、Group Quantization、Long Context Techniques 三张图。

- [ ] 我能算出 GQA 相对 MHA 的 KV 缩减比例
- [ ] 我能区分权重、激活和 KV Cache 量化
- [ ] 我能说出 FlashAttention 没有改变什么

## 1. 为什么“模型能算”还不够

模型真正部署到硬件上，会遇到三个现实限制：

- **算不动**：计算量太大，首 Token 等待时间过长；
- **放不下**：权重、激活或 KV Cache 超出显存；
- **搬不快**：计算单元在等待 HBM 或设备间通信的数据。

| 资源 | 典型问题 | 常见优化 |
| --- | --- | --- |
| 计算 | Attention 随序列长度平方增长 | Sliding Window、稀疏注意力 |
| 显存容量 | 权重和 KV Cache 太大 | 量化、GQA/MQA、Paged Attention |
| 显存带宽 | Decode 反复读取权重和 KV | 量化、算子融合、GQA/MQA |
| 片上存储与访存 | 中间矩阵反复写回 HBM | FlashAttention、融合算子 |
| 设备间通信 | 单设备放不下长序列 | TP、CP、通信计算重叠 |

一个优化可能改善多种资源，也可能用计算换显存。例如量化减少存储与搬运，却增加了缩放和反量化操作。

## 本章单元

- **11-1（约 15 分钟）**：[MHA、MQA 与 GQA](./11-efficient-llm-design/01-mha-mqa-gqa.md)
- **11-2（约 15 分钟）**：[量化原理、粒度与显存收益](./11-efficient-llm-design/02-quantization.md)
- **11-3（约 15 分钟）**：[长上下文的四条优化路线](./11-efficient-llm-design/03-long-context.md)
- **11-4（约 15 分钟）**：[把模型优化映射到算子与通信](./11-efficient-llm-design/04-optimization-infra.md)

- **本章总结**：[串联知识、练习与检查](./11-efficient-llm-design/summary.md)
