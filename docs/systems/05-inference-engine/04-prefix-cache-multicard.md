# 单元 05-4｜Prefix Cache 与多卡推理

> 所属章节：[第 5 章｜推理引擎与 KV Cache 管理](../05-inference-engine.md)

::: info 本单元目标
围绕 **"前缀复用与多卡推理通信"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

分页解决了"装得下"，本单元解决"算得少"与"多卡怎么办"。

## 8. Prefix Cache：共享前缀只算一次

真实服务的请求常常共享长前缀：系统提示词（几千 token）、few-shot 模板、多轮对话的历史。

**Prefix Cache 的规则**：前缀部分的 KV Cache 只依赖前缀 token 本身（causal attention 的性质——前面的 token 看不到后面，[《模型全景》04-2](../../model/04-self-attention/02-scaled-dot-product-mask.md)），**与后文无关**。于是：

```text
请求 1: [系统提示 4K + 问题 1]  → Prefill 一次，4K 的 KV 入缓存
请求 2: [系统提示 4K + 问题 2]  → 命中！4K 的 KV 直接复用，只 Prefill 问题部分
```

省两笔账：**计算**（Prefill 的 FLOPs 按命中长度免掉）与**显存**（分页块级复用，多个请求共享同一批物理块——只读共享，天然安全）。管理策略通常是 LRU：缓存满时淘汰最久未用的前缀块。

一句话定位：**Prefix Cache 是 KV Cache 上的"内容寻址缓存"**，命中率越高收益越大——系统提示越长、请求越同质，越是划算。

## 9. 多卡推理：通信挂回到熟悉的钩子上

单卡装不下（权重或 KV 或吞吐不够）时，推理并行登场——通信需求回到本课程的主线上：

| 手段 | 通信形态 | 回扣 |
| --- | --- | --- |
| **TP 推理** | 每层 2 次 AllReduce（前向）——与训练同构 | [《并行策略》06 章](../../parallel/06-tensor-parallel.md) |
| **EP 推理（MoE）** | 每层 dispatch/combine 的 AlltoAll | [《集合通信》04-4](../../collective/04-gather-scatter-alltoall/04-moe-imbalance.md) |
| **PD 分离的 KV 传输** | Prefill→Decode 实例间大块 P2P | [第 6 章](../06-pd-scheduling.md)（下一章主角） |

值得先记的一个差异：**训练的通信"大而低频"（梯度 AllReduce），推理的 Decode 通信"小而高频、延迟敏感"**——每生成一个 token 就要过一遍每层通信，任何一次通信的微秒级抖动都直接累积进每个字的延迟（[第 3 章 03-4](../03-inference-kv-cache/04-inference-service-metrics.md) 的伏笔）。这正是推理成为通信库另一半主战场的原因——算法选择、拓扑贴合的要求与训练侧同源（[H01-6](../../ascend/hccl-source/06-coll-algorithms.md)）。

::: tip 本章收束
推理引擎三件套到此集齐：**调度器**（连续批：批活起来）+ **显存管理器**（分页 + 前缀复用：装得下、算得少）+ **执行器**（多卡通信：挂回主线）。下一章把镜头对准"指标"——TTFT、TPOT 与吞吐的三角制约，以及它的终局解法 PD 分离。
:::

---

[进入本章总结 →](./summary.md)
