# 单元 08-4｜推理指标、服务与通信

> 所属章节：[第 8 章｜自回归推理与 KV Cache](../08-inference-kv-cache.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **推理指标、服务与通信** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

![TTFT、TPOT 与端到端延迟位于生成时间线的不同区间](/images/llm/inference-latency-metrics.svg)

| 指标 | 含义 | 主要受什么影响 |
| --- | --- | --- |
| TTFT | 请求到第一个输出 Token 的时间 | 排队、Prompt 长度、Prefill、调度 |
| TPOT / ITL | 后续每个输出 Token 的平均间隔 | Decode、Batch、内存带宽、通信 |
| E2E Latency | 请求到完整响应结束 | TTFT、输出长度、每步 Decode |
| Token Throughput | 单位时间系统生成的 Token 数 | Batch、调度、并行、算子效率 |
| Request Throughput | 单位时间完成的请求数 | 输入输出长度分布、并发与资源 |

低延迟和高吞吐并不总能同时最大化。扩大 Batch 往往能提高总吞吐，却可能增加排队时间和单请求延迟。

## 11. 从单请求到推理服务

真实服务会同时面对许多长度不同、到达时间不同的请求：

### 11.1 Continuous Batching

传统静态 Batch 要等整批请求全部结束才能换入新请求。Continuous Batching 可以在每个 Decode 迭代边界移除已完成请求、加入新请求，提高设备利用率。

### 11.2 Paged KV Cache

为每个请求预留最大连续 Cache 会造成大量空闲和内存碎片。PagedAttention 将 KV Cache 划分为固定大小的物理块，通过块表把逻辑连续 Token 映射到非连续物理块，思想类似操作系统分页。

### 11.3 Prefix Caching

如果大量请求共享相同系统 Prompt，可以复用其已经计算好的 KV Cache，减少重复 Prefill。但复用必须保证模型、Token、位置和相关配置完全一致。

这些优化不改变语言模型“逐 Token 生成”的语义，改变的是请求如何调度、Cache 如何管理以及 Kernel 如何执行。

## 12. 与 AI Infra 和集合通信的关系

### 12.1 为什么 Decode 对延迟敏感

每生成一个 Token 都要依次通过所有层，还要等待采样结果才能开始下一步。这条依赖链很难跨 Token 并行，所以单次 Kernel 启动、同步和通信延迟都会累积到 TPOT。

### 12.2 Tensor Parallel 通信

模型放不进单卡或需要提高算力时，可以做 Tensor Parallel。每层的分片矩阵计算之间通常需要 AllReduce 或 ReduceScatter/AllGather。Prefill 的消息对应多个 Token，Decode 每步只有少量 Token，通信更容易呈现延迟敏感特征。

### 12.3 KV Cache 如何切分

KV Cache 可以随 KV Head 按 Tensor Parallel Rank 分片。每个 Rank 保存自己负责的 Head，减少单设备 Cache；但具体是否需要额外通信，取决于 Attention 与输出投影的切分方案。

### 12.4 推理优化的三个对象

后续学习 AI Infra 时，可以把优化归纳为：

1. **权重**：量化、分片、减少重复读取；
2. **KV Cache**：分页、量化、复用、卸载或减少 KV Head；
3. **调度**：Batching、请求优先级、Prefill/Decode 协同和通信计算重叠。

---

[进入本章总结 →](./summary.md)
