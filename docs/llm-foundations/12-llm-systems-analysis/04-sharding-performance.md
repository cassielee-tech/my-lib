# 单元 12-4｜模型切分、通信与性能模型

> 所属章节：[第 12 章｜综合拆解一个现代 LLM](../12-llm-systems-analysis.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **模型切分、通信与性能模型** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

单卡计算图只描述算子依赖；把数据或参数切到多卡后，原本位于同一设备的数据依赖变成通信。

| 并行方式 | 切分对象 | 典型通信 | 传输语义 |
| --- | --- | --- | --- |
| DP / DDP | Batch | AllReduce | 聚合相同参数的梯度 |
| ZeRO / FSDP | 参数、梯度、优化器状态 | AllGather、ReduceScatter | 临时收集参数、归约并分片梯度 |
| TP | 矩阵 / Head | AllReduce、AllGather、ReduceScatter | 拼接或求和被切开的层输出 |
| PP | Layer | P2P Send/Recv | Stage 间传递激活与梯度 |
| CP | Sequence | AllGather、Alltoall 或 P2P Ring | 交换长序列的 K/V 或中间结果 |
| EP | Expert | Alltoall | 将 Token 发给目标 Expert，再送回 |

![模型切分方式如何决定通信语义](/images/llm/model-sharding-communication.svg)

最关键的推导方法不是背表，而是问：

1. 原张量被沿哪个维度切开？
2. 当前 Rank 拥有哪一部分？
3. 下一算子需要完整张量、分片张量还是求和结果？
4. 哪种 Collective 正好完成这个数据变换？

## 11. 一个资源预算示例

假设用 8 张设备做 BF16 推理，采用 TP=8，暂时忽略额外开销：

### 11.1 权重

$$
15.0\ GB\div8\approx1.875\ GB/Rank
$$

实际并非所有参数都完全均分，Embedding、Norm、LM Head 可能采用不同分片或复制策略。

### 11.2 KV Cache

若 8 个 KV Head 均匀分给 8 个 Rank，则每个 Rank 保存 1 个 KV Head。32K 单请求的 4 GiB KV Cache 可降至约 0.5 GiB/Rank。

### 11.3 通信

显存下降不代表免费：每个 TP Layer 的部分输出需要聚合。随着 Batch、Sequence 和 Hidden Size 增长，激活通信也会增长。最终吞吐取决于 GEMM 与通信能否高效执行和重叠。

因此，“能放下”只回答了容量问题，还没有回答“跑得快不快”。

## 12. 从估算走向性能模型

可以用三个上限判断瓶颈：

### 12.1 计算上限

$$
T_{compute}\approx\frac{实际FLOPs}{设备有效FLOPs/s}
$$

### 12.2 HBM 上限

$$
T_{memory}\approx\frac{HBM搬运字节数}{有效HBM带宽}
$$

### 12.3 通信上限

$$
T_{comm}\approx\alpha\times通信轮次+\beta\times通信字节数
$$

$\alpha$ 代表启动与同步延迟，$\beta$ 代表每字节传输成本。实际执行还要考虑拓扑、协议、并发和计算通信重叠。

粗略地看，总时间至少受最慢资源制约；存在严格依赖时，各段时间还会串行累加。AI Infra 的核心工作，就是让计算、访存和通信尽可能接近各自上限，并尽量重叠。

## 13. 一套通用的模型拆解清单

拿到任何模型配置时，按以下顺序分析：

### 模型结构

1. Vocabulary、Hidden Size、Layer、Q/KV Head、Head Dimension、FFN Size 是多少？
2. 使用 MHA、GQA、MQA 还是 MoE？
3. Embedding 与 LM Head 是否共享？

### 参数和计算

4. Embedding、Attention、FFN 各有多少参数？
5. 每 Token 前向和训练 FLOPs 是什么数量级？
6. 长序列 Attention 项什么时候不可忽略？

### 显存

7. 权重采用什么精度？
8. 训练还保存哪些梯度和优化器状态？
9. KV Cache 每 Token、每请求、每 Batch 多大？
10. 激活是否 Checkpoint、分片或重计算？

### 并行与通信

11. 哪个维度使用 DP、TP、PP、CP 或 EP？
12. 每次 Collective 之前和之后，张量 Shape 如何变化？
13. 通信是否跨节点，能否与计算重叠？

### 服务指标

14. 关注 TTFT、TPOT、吞吐还是并发数？
15. Prefill 与 Decode 是否采用不同调度和资源策略？

---

[进入本章总结 →](./summary.md)
