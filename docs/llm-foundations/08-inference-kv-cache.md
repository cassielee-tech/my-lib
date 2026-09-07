# 第 8 章｜自回归推理与 KV Cache

> 本章目标：理解 Decoder-only 模型怎样逐 Token 生成文本；区分 Prefill 与 Decode；解释 KV Cache 缓存了什么、节省了什么、占用了什么；理解采样参数与推理性能指标。

## 本章导学

::: tip 本章核心结论
1. 自回归生成每次只产生一个新 Token，再把它追加到上下文继续计算。
2. Prefill 并行处理 Prompt，Decode 逐步生成；两者的计算 Shape 和性能瓶颈不同。
3. KV Cache 避免重复计算历史 K/V，但会随层数、序列长度、Batch 和 KV Head 数增长。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，从单次生成推进到推理服务。

**先看这几张核心图：** Autoregressive Loop、Prefill vs Decode、KV Cache Recompute 三张图。

- [ ] 我能画出逐 Token 生成循环
- [ ] 我能区分 TTFT 与 TPOT
- [ ] 我能解释 KV Cache 节省和消耗了什么

## 本章单元

- **08-1（约 15 分钟）**：[生成循环、Prefill 与 Decode](./08-inference-kv-cache/01-generation-prefill-decode.md)
- **08-2（约 15 分钟）**：[KV Cache 保存什么、占多少显存](./08-inference-kv-cache/02-kv-cache-memory.md)
- **08-3（约 15 分钟）**：[采样与停止条件](./08-inference-kv-cache/03-sampling-stop.md)
- **08-4（约 15 分钟）**：[推理指标、服务与通信](./08-inference-kv-cache/04-inference-service-metrics.md)

- **本章总结**：[串联知识、练习与检查](./08-inference-kv-cache/summary.md)
