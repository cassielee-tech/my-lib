# 大模型基础

从传统后端开发转向 AI Infra 与昇腾集合通信开发，这里记录沿途建立的知识体系。

## 开始学习

- [01｜全景：从模型到集合通信](./01-landscape.md)
- [02｜张量、矩阵乘法、计算图与反向传播](./02-tensor-autograd.md)
- [03｜Tokenization、Embedding 与语言模型目标](./03-tokenization-embedding.md)
- [04｜Self-Attention、QKV 与多头注意力](./04-self-attention.md)
- [05｜RMSNorm、SwiGLU 与完整 Decoder Block](./05-decoder-block.md)
- [06｜RoPE、位置编码与长上下文](./06-rope-long-context.md)

## 大模型基础：12 课目录

大模型基础共规划 **12 课**。前 6 课建立 Decoder-only Transformer 的结构认知，第 7～11 课补齐训练、推理、微调和现代架构，第 12 课进行综合拆解。完成后再系统进入 AI Infra。

| 课次 | 主题 | 学完能够回答的问题 |
| ---: | --- | --- |
| 01 | [全景：从模型到集合通信](./01-landscape.md) | 大模型从训练到推理涉及哪些软硬件层？ |
| 02 | [张量、计算图与反向传播](./02-tensor-autograd.md) | 数据和梯度如何沿计算图流动？ |
| 03 | [Tokenization、Embedding 与语言模型目标](./03-tokenization-embedding.md) | 文本如何变为张量，模型到底在预测什么？ |
| 04 | [Self-Attention、QKV 与多头注意力](./04-self-attention.md) | 一个 Token 如何从上下文中取回信息？ |
| 05 | [RMSNorm、SwiGLU 与完整 Decoder Block](./05-decoder-block.md) | 一个现代 Decoder Block 如何连接起来？ |
| 06 | [RoPE、位置编码与长上下文](./06-rope-long-context.md) | 没有位置的注意力缺少什么，RoPE 为什么表达相对距离？ |
| 07 | 大模型训练循环 | AdamW、学习率、混合精度、梯度累积分别做什么？ |
| 08 | 自回归推理与 KV Cache | Prefill、Decode、采样和 KV Cache 如何协作？ |
| 09 | 预训练、SFT、LoRA 与偏好对齐 | 模型能力如何通过不同训练阶段形成？ |
| 10 | MoE、Router 与专家模型 | MoE 如何用稀疏计算扩大参数量，为什么需要 AlltoAll？ |
| 11 | 现代模型的效率设计 | GQA/MQA、量化和长上下文方案在优化什么？ |
| 12 | 综合实战：拆解一个现代 LLM | 如何估算参数量、FLOPs、显存，并画出完整数据流？ |

第 10～12 课会提前指出通信、显存和算子问题，但具体的性能模型、并行策略与集合通信实现会留到 AI Infra 专题深入学习。
