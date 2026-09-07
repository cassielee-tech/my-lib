# 单元 12-2｜权重、KV Cache 与训练显存

> 所属章节：[第 12 章｜综合拆解一个现代 LLM](../12-llm-systems-analysis.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **权重、KV Cache 与训练显存** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

只考虑 7.50B 参数本身：

| 格式 | 每参数字节 | 理论权重大小 |
| --- | ---: | ---: |
| FP32 | 4 | 30.0 GB |
| BF16 / FP16 | 2 | 15.0 GB，约 14.0 GiB |
| INT8 | 1 | 7.5 GB |
| INT4 | 0.5 | 3.75 GB |

GB 与 GiB 不相同：

$$
1\ GiB=2^{30}\ Bytes,\qquad1\ GB=10^9\ Bytes
$$

模型文件和监控工具可能采用不同单位。真实推理显存还包括量化 Scale、KV Cache、临时 Buffer、算子 Workspace 和框架开销。

## 5. KV Cache 显存

Stack-7.5B 使用 8 个 KV Head。一个请求的 KV Cache 为：

$$
M_{KV}=2L_SN_{kv}d_hSb
$$

BF16 下，每新增一个 Token，需要：

$$
2\times32\times8\times128\times2=131072\ Bytes=128\ KiB
$$

因此：

| 上下文长度 | 单请求 KV Cache |
| ---: | ---: |
| 2,048 | 256 MiB |
| 8,192 | 1 GiB |
| 32,768 | 4 GiB |
| 131,072 | 16 GiB |

如果 Batch 中有 8 个都占满 32K 的请求，仅 KV Cache 就需要约 32 GiB。

若改为 MHA，即 $N_{kv}=N_q=32$，缓存会再扩大 4 倍；若采用 INT8 KV Cache，则理论上可以减半。

## 6. 训练显存为什么远大于推理

混合精度 AdamW 训练常需要保存多份模型状态。一个便于估算的传统配置是：

| 状态 | 每参数字节 |
| --- | ---: |
| BF16 参数 | 2 |
| BF16 梯度 | 2 |
| FP32 Master Weight | 4 |
| FP32 Adam 一阶矩 | 4 |
| FP32 Adam 二阶矩 | 4 |
| 合计 | 16 Byte / 参数 |

所以仅模型状态约为：

$$
7.50B\times16\ Byte\approx120\ GB
$$

![推理和训练分别把显存花在哪里](/images/llm/training-inference-memory.svg)

这还没有计算激活、临时 Buffer、通信 Bucket 和碎片。不同框架可能没有独立 Master Weight，或采用低精度优化器，因此不能把 16 Byte 当成永远不变的常数。

### 6.1 激活为什么难用一个固定数字表示

激活与以下因素共同相关：

- Micro-batch Size；
- Sequence Length；
- Hidden Size 与 Layer 数；
- 保存哪些中间结果；
- 是否使用 FlashAttention；
- 是否使用 Activation Checkpointing；
- Tensor/Sequence/Context Parallel 的切分方式。

Activation Checkpointing 少保存中间结果，反向时重新做部分前向计算，本质上是**用计算换显存**。

---

[继续单元 12-3 →](./03-flops-dataflow.md)
