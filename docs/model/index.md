# 模型全景

> 返回：[课程总览](../roadmap.md)

**专栏目标**：知道模型在算什么、并行从哪里来、为什么必然产生集合通信——为整条路线提供"为什么"的底层视角。

## 章节列表

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [从模型到集合通信](./01-landscape.md) | 模型训练的主线是什么？并行策略为什么产生通信？ | ✅ |
| 2 | [张量、计算图与反向传播](./02-tensor-autograd.md) | 张量 Shape 怎样流动？梯度为什么需要通信？ | ✅ |
| 3 | [模型输入](./03-tokenization-embedding.md) | 文本怎样变成张量？完整 Shape 流程的系统代价是什么？ | ✅ |
| 4 | [Self-Attention](./04-self-attention.md) | Q/K/V 在算什么？多头的 Shape 与性能代价？ | ✅ |
| 5 | [完整 Decoder Block](./05-decoder-block.md) | Block 怎样拼起来？参数量、FLOPs 与张量并行切分？ | ✅ |
| 6 | [RoPE 与长上下文](./06-rope-long-context.md)【选修】 | 位置信息怎样注入？长上下文的限制在哪？ | ✅ |

## 阅读提示

- 主线章为第 1-5 章；第 6 章为选修，可跳过不影响推进；
- 第 1 章是整门课的钥匙：先读 [⚡ 速通](./01-landscape/quick.md) 再进单元；
- 每章的"04 单元"（如 04-4 梯度与通信）是与后续专栏最相关的钩子，时间紧可优先精读。

## 验收清单

- [ ] 能画出"模型代码 → 框架 → HCCL → 硬件"分层图；
- [ ] 能解释 DP 为什么需要 AllReduce、TP 为什么需要拼结果；
- [ ] 能沿着 Shape → 计算 → 显存 解释一个 Decoder Block。

## 参考资料

- [【闪客】一小时从函数到 Transformer](https://www.bilibili.com/video/BV1NCgVzoEG9/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)

---

[返回课程总览 →](../roadmap.md) · [进入《训练与推理系统》 →](../systems/index.md)
