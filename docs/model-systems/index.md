# 模型与系统

> 返回：[首页](../index.md)

**专栏目标**：上篇（模型全景）回答"模型在算什么、并行从哪里来、为什么必然产生集合通信"；下篇（训练与推理系统）回答"训练怎样运转、显存花在哪、梯度通信何时发生"——两者合起来，为整条路线提供"为什么"的底层视角。

**时间紧？** 每篇各有速通：[⚡ 上篇速通｜模型全景](00-速通-上篇.md)（6 章各约 5 分钟）、[⚡ 下篇速通｜训练与推理](00-速通-下篇.md)。

## 上篇｜模型全景（第 1-6 章）

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [从模型到集合通信](01-landscape.md) | 模型训练的主线是什么？并行策略为什么产生通信？ | ✅ |
| 2 | [张量、计算图与反向传播](02-tensor-autograd.md) | 张量 Shape 怎样流动？梯度为什么需要通信？ | ✅ |
| 3 | [模型输入](03-tokenization-embedding.md) | 文本怎样变成张量？完整 Shape 流程的系统代价是什么？ | ✅ |
| 4 | [Self-Attention](04-self-attention.md) | Q/K/V 在算什么？多头的 Shape 与性能代价？ | ✅ |
| 5 | [完整 Decoder Block](05-decoder-block.md) | Block 怎样拼起来？参数量、FLOPs 与张量并行切分？ | ✅ |
| 6 | [RoPE 与长上下文](06-rope-long-context.md)【选修】 | 位置信息怎样注入？长上下文的限制在哪？ | ✅ |

- 主线章为第 1-5 章；第 6 章为选修，可跳过不影响推进；
- 第 1 章是整门课的钥匙：先读 [⚡ 上篇速通](00-速通-上篇.md) 再进单元；
- 每章的"04 单元"（如 04-4 梯度与通信）是与后续专栏最相关的钩子，时间紧可优先精读。

**验收**：能画出"模型代码 → 框架 → HCCL → 硬件"分层图；能解释 DP 为什么需要 AllReduce、TP 为什么需要拼结果；能沿着 Shape → 计算 → 显存 解释一个 Decoder Block。

## 下篇｜训练与推理系统（第 1-6 章）

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [训练循环](01-training-loop.md) | 一次训练 Step 发生什么？梯度累积、AdamW、混合精度各自做什么？ | ✅ 主线 |
| 2 | [训练显存与计算优化](02-training-memory-compute.md) | 训练显存的四个大件各占多少？Checkpointing/累积/Offload 分别省什么？ | ✅ 主线 |
| 3 | [推理与 KV Cache](03-inference-kv-cache.md)【选修】 | Prefill 与 Decode 有何不同？KV Cache 占多少显存？ | ✅ |
| 4 | [后训练与对齐](04-post-training-alignment.md)【选修】 | SFT/LoRA/RLHF/DPO 分别解决什么？ | ✅ |
| 5 | [推理引擎与 KV Cache 管理](05-inference-engine.md)【选修】 | Continuous Batching、Paged Attention 和 Prefix Cache 怎样提高并发？ | ✅ |
| 6 | [Prefill/Decode 调度与推理指标](06-pd-scheduling.md)【选修】 | TTFT、TPOT、吞吐为什么互相制约？PD 分离解决什么？ | ✅ |

- **主线只需第 1、2 章**：先看模型侧的训练循环（第 1 章），再用系统侧的显存账本（第 2 章）落地；
- 第 3-6 章为推理/后训练视角（选修）：第 3 章讲单请求生成，第 5 章讲多请求服务，第 6 章讲指标与 PD 分离——对 HCCL 岗位，推理通信是训练通信互补的另一半战场；
- 第 1 章的 01-4（显存与通信）与第 2 章是同一问题的两个视角，建议连读。

**验收**：算出 7.5B 模型的训练显存账本（16 字节/参数）；说出梯度 AllReduce 发生在反向与更新之间、为什么能与反向计算重叠；能为一次训练任务画出"计算-通信"时间线草图。

## 参考资料

- [【闪客】一小时从函数到 Transformer](https://www.bilibili.com/video/BV1NCgVzoEG9/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)

---

[返回首页 →](../index.md) · [进入《执行与并行》 →](/execution-parallel/)
