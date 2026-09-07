# 第 12 章｜综合实战：拆解一个现代 LLM

> 本章目标：把前 11 课连成一条完整主线。面对一份模型配置，能够估算参数量、权重与 KV Cache 显存、训练与推理计算量，并把模型数据流映射到算子和集合通信。

## 本章导学

::: tip 本章核心结论
1. 模型资源账本从配置开始：先算参数，再乘精度，最后加入激活、KV Cache 和运行时开销。
2. $2P$、$6P$ 是前向与训练的数量级估算，不包含所有长序列和系统开销。
3. 并行策略切开不同张量，下一算子需要的数据布局决定使用哪种集合通信。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，每次只完成一种资源估算。

**先看这几张核心图：** Parameter Ledger、Training vs Inference Memory、Model Sharding Communication 三张图。

- [ ] 我能从模型配置估算参数和 BF16 权重
- [ ] 我能写出 KV Cache 的主要乘数
- [ ] 我能从一种切分方式推导一种 Collective

## 1. 为什么最后一课要做“估算”

工程中常会遇到这样的需求：

- 一个模型能不能放进 8 张卡？
- 32K 上下文的 KV Cache 每个请求占多少显存？
- 扩大 Batch 后，是计算先满、显存先满，还是通信先满？
- 一次 `loss.backward()` 为什么会触发 AllReduce？
- GQA、量化或并行切分到底节省了什么？

回答这些问题不需要一开始就得到完全精确的数字。更重要的是先建立**数量级正确、假设明确、可以继续细化**的模型。

本章使用一个虚构的 Decoder-only 教学模型 **Stack-7.5B**。它不是某个真实产品，目的是避免被具体实现细节干扰。

## 本章单元

- **12-1（约 15 分钟）**：[读配置并估算参数量](./12-llm-systems-analysis/01-config-parameters.md)
- **12-2（约 15 分钟）**：[权重、KV Cache 与训练显存](./12-llm-systems-analysis/02-model-memory.md)
- **12-3（约 15 分钟）**：[FLOPs 与完整数据流](./12-llm-systems-analysis/03-flops-dataflow.md)
- **12-4（约 15 分钟）**：[模型切分、通信与性能模型](./12-llm-systems-analysis/04-sharding-performance.md)

- **本章总结**：[串联知识、练习与检查](./12-llm-systems-analysis/summary.md)
