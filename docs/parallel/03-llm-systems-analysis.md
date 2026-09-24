# 第 3 章｜综合实战：拆解一个现代 LLM

> 本章目标：把前 11 课连成一条完整主线。面对一份模型配置，能够估算参数量、权重与 KV Cache 显存、训练与推理计算量，并把模型数据流映射到算子和集合通信。

## 本章导学

::: tip 本章核心结论
1. 模型资源账本从配置开始：先算参数，再乘精度，最后加入激活、KV Cache 和运行时开销。
2. $2P$、$6P$ 是前向与训练的数量级估算，不包含所有长序列和系统开销。
3. 并行策略切开不同张量，下一算子需要的数据布局决定使用哪种集合通信。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，每次只完成一种资源估算。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](#⚡-速通-约-5-分钟)：一份"开店预算"类比讲完整章，再回来按单元深入。

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

- **03-1（约 15 分钟）**：[读配置并估算参数量](#_03-1-读配置并估算参数量)
- **03-2（约 15 分钟）**：[权重、KV Cache 与训练显存](#_03-2-权重、kv-cache-与训练显存)
- **03-3（约 15 分钟）**：[FLOPs 与完整数据流](#_03-3-flops-与完整数据流)
- **03-4（约 15 分钟）**：[模型切分、通信与性能模型](#_03-4-模型切分、通信与性能模型)

- **本章总结**：[串联知识、练习与检查](#本章总结)

## ⚡ 速通（约 5 分钟）

> 适用：时间紧，或完整版跟不动时。这页用"一家店的开业预算"讲完整章，读完抓住 80% 的主干。任何一节想深入，拉到文末点对应单元。

### 1. 看图纸，算设备清单：从配置到参数量

拿到一个模型的配置文件（图纸），就能**心算出它的家底**：

```text
H（隐藏宽度）、I（中间宽度）、L（层数）、V（词表大小）、头数……
  → 每层：Attention ≈ 4H² + MLP ≈ 3HI
  → 全部层 × L，再加词表查表 V×H
  → 总参数量 P ≈ ……
```

两个省钱设计要认得：

- **GQA**：K/V 投影比 Q 投影窄（柜子少），每层再省一笔；
- **权重共享**：进门查表和出门打分（Embedding 与 LM Head）**共用同一份 V×H 参数**——直接省一个大件。

最后乘上每个参数的字节数（BF16 = 2 字节），就得到**权重显存**：7.5B 参数 ≈ 15 GB。

::: details 小测 1：为什么 GQA 模型的 K/V 投影参数比 Q 投影少？
Q 有 32 个头（投影输出宽 4096），共享后 K/V 各只有 8 个头（输出宽 1024）。头少 → 投影矩阵窄 → 参数少。这正是第 2 章"档案柜"在参数账本上的体现。
:::

### 2. 仓库和电费：显存与算力的两本账

**仓库账（显存）**——推理时：权重 + 笔记（KV Cache）+ 工作台（激活/Workspace）。**训练时还要再添三样**：

```text
权重 + 梯度（同样大小）+ 优化器两本账（AdamW 的 m 和 v，又是两份）
→ 常说"训练显存 ≈ 16 字节/参数"的量级，还没算激活
```

这就是为什么"推理 14 GiB 的模型，训练可能要 100+ GiB"。

**电费账（FLOPs）**——两句粗估口诀：

- **推理一次前向 ≈ 2P**（P 是参数量，每 Token 过一遍参数）；
- **训练一步 ≈ 6P**（前向 + 反向 + 更新，反向约是前向两倍）。

注意这是"数量级"口径：长序列时 Attention 里随 S² 增长的部分没算全，但**选卡、估时间、做预算**足够用了。

::: details 小测 2：为什么训练比推理多吃这么多显存？
训练多了：梯度（和权重一样大）+ 优化器状态（AdamW 每个参数两本历史账 m/v）+ 必须保存的前向激活（反向要用）。推理的 KV Cache 只是"笔记"，训练这三样才是大头。
:::

### 3. 开分店：五种切法一张表

单卡装不下/太慢，就要"开分店"（多卡并行）。**五种切法切的东西完全不同**：

| 切法 | 切什么 | 一句话 |
| --- | --- | --- |
| **DP** | 切**数据** | 每店同一本菜谱，接待不同顾客 |
| **TP** | 切**矩阵/头** | 一台大机器拆给多家合用 |
| **PP** | 切**层** | 按楼层流水线分工 |
| **CP** | 切**序列** | 长会议拆到多间会议室接力 |
| **EP** | 切**专家** | 科室分布到多分院 |

**用哪种通信，由"下一道工序需要什么形状的数据"决定**：下一层要完整数据 → AllGather 拼回来；要按份求和 → AllReduce / ReduceScatter；要把 Token 按去向重分发 → AlltoAll。这不是背出来的，是**顺着张量切分方向推出来的**——《模型全景》第 1 章那句话在收官处闭环：**切法决定对账方式**。

::: details 小测 3：TP 把 MLP 的 Down 投影按行切开，各卡算出"部分和"，下一步要完整输出——需要什么通信？
AllReduce（或 ReduceScatter）：完整输出 = 各卡部分和相加。这就是"下一道工序需要什么形状"决定通信原语的直接例子——要拼完整就归约，要分片就 ReduceScatter。
:::

### 3 句话带走

1. **看图纸算家底**：每层 4H²+3HI，乘层数加词表，GQA 和权重共享是两个常见折扣；
2. **两本账**：训练显存 ≈ 权重×4 以上（梯度+优化器），FLOPs 口诀 **2P 前向 / 6P 训练**；
3. **五种切法切五种东西**（DP 数据/TP 矩阵/PP 层/CP 序列/EP 专家），**通信由下一道工序需要的形状决定**。

### 黑话小词典

| 术语 | 人话 |
| --- | --- |
| 配置（H/I/L/V） | 图纸上的关键尺寸 |
| 参数量 P / 权重共享 | 设备清单 / 进出门共用一个大件 |
| 优化器状态 | AdamW 的两本历史账（m/v） |
| 激活 | 工作台上的半成品 |
| 2P / 6P | 每 Token 前向 / 训练一步的电费口诀 |
| DP / TP / PP / CP / EP | 切数据 / 切矩阵 / 切层 / 切序列 / 切专家 |
| AllGather / AllReduce / ReduceScatter / AlltoAll | 拼完整 / 求和 / 求和后分着拿 / 按去向重分发 |

### 想深入？

每节 15 分钟，按需点开，不必按顺序全读：

| 单元 | 讲什么（白话） | 什么时候需要它 |
| --- | --- | --- |
| [03-1 读配置并估算参数量](#_03-1-读配置并估算参数量) | 从图纸到清单的完整算例 | 想亲手拆一个模型配置 |
| [03-2 权重、KV Cache 与训练显存](#_03-2-权重、kv-cache-与训练显存) | 仓库账的精算 | 想做部署容量规划 |
| [03-3 FLOPs 与完整数据流](#_03-3-flops-与完整数据流) | 电费账 + 全链路数据流 | 想估训练/推理时间 |
| [03-4 模型切分、通信与性能模型](#_03-4-模型切分、通信与性能模型) | 分店方案与通信推导 | 想设计并行策略 |
| [本章总结](#本章总结) | 阶段总结 + 自测 | 检验大模型基础整阶段 |

## 03-1｜读配置并估算参数量

::: info 本单元目标
围绕 **读配置并估算参数量** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

| 配置 | 符号 | 数值 |
| --- | --- | ---: |
| Vocabulary Size | $V$ | 128,000 |
| Hidden Size | $D$ | 4,096 |
| Decoder Layer 数 | $L$ | 32 |
| Query Head 数 | $N_q$ | 32 |
| KV Head 数 | $N_{kv}$ | 8 |
| Head Dimension | $d_h$ | 128 |
| FFN Intermediate Size | $F$ | 14,336 |
| Norm | — | RMSNorm |
| FFN | — | SwiGLU |
| Position | — | RoPE |
| Embedding / LM Head | — | 权重共享 |

先做两个一致性检查：

$$
N_qd_h=32\times128=4096=D
$$

$$
N_{kv}d_h=8\times128=1024
$$

这说明 Q 投影宽度是 4096，K/V 投影宽度各为 1024；每 4 个 Q Head 共享一个 KV Head。

### 3. 参数量从哪里来

参数主要来自三个区域：Embedding、32 个 Decoder Block、输出 LM Head。

![Stack-7.5B 的参数账本](/images/llm/llm-parameter-ledger.svg)

#### 3.1 Token Embedding

$$
P_{embed}=V\times D=128000\times4096=524,288,000
$$

约为 **524.3M** 参数。

#### 3.2 一层 Attention

忽略 Bias：

$$
P_Q=D\times(N_qd_h)=4096\times4096
$$

$$
P_K=P_V=D\times(N_{kv}d_h)=4096\times1024
$$

$$
P_O=(N_qd_h)\times D=4096\times4096
$$

所以：

$$
P_{attn}=P_Q+P_K+P_V+P_O\approx41.94M
$$

GQA 不仅减少 KV Cache，也让 K/V Projection 的参数少于 Q/O Projection。

#### 3.3 一层 SwiGLU FFN

SwiGLU 通常有 Gate、Up、Down 三个矩阵：

$$
P_{ffn}=D\times F+D\times F+F\times D=3DF
$$

$$
P_{ffn}=3\times4096\times14336\approx176.16M
$$

可见 FFN 参数约为 Attention 的 4.2 倍。对这个 Dense 模型来说，大部分参数在 FFN，而不是 Attention Score Matrix。

#### 3.4 RMSNorm

每层两个 RMSNorm，每个只有 $D$ 个缩放参数：

$$
P_{norm}=2D=8192
$$

与数亿个矩阵参数相比，它在参数量估算中几乎可以忽略，但在执行链路中不能忽略。

#### 3.5 总参数量

单层约为：

$$
P_{layer}\approx41.94M+176.16M=218.10M
$$

32 层加 Embedding：

$$
P\approx32\times218.10M+524.29M\approx7.50B
$$

因为 Embedding 与 LM Head 共享权重，输出层不再增加一份 $V\times D$。若不共享，总参数还要增加约 524.3M，达到约 8.03B。

::: tip 参数量估算顺序
先算大矩阵，再补 Norm 和 Bias。不要一开始陷入几千个参数的误差，却漏掉一整个 Embedding 或 LM Head。
:::

## 03-2｜权重、KV Cache 与训练显存

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

### 5. KV Cache 显存

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

### 6. 训练显存为什么远大于推理

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

#### 6.1 激活为什么难用一个固定数字表示

激活与以下因素共同相关：

- Micro-batch Size；
- Sequence Length；
- Hidden Size 与 Layer 数；
- 保存哪些中间结果；
- 是否使用 FlashAttention；
- 是否使用 Activation Checkpointing；
- Tensor/Sequence/Context Parallel 的切分方式。

Activation Checkpointing 少保存中间结果，反向时重新做部分前向计算，本质上是**用计算换显存**。

## 03-3｜FLOPs 与完整数据流

::: info 本单元目标
围绕 **FLOPs 与完整数据流** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

矩阵乘法 $[M,K]\times[K,N]$ 大约包含：

$$
2MKN\ FLOPs
$$

因为一次乘加通常按一次乘法加一次加法，即 2 FLOPs 计算。

#### 7.1 参数相关计算

Dense Transformer 每个 Token 的一次前向，粗略可用：

$$
F_{forward}\approx2P
$$

Stack-7.5B 约为 **15 GFLOPs / Token**。这个估算抓住了各权重矩阵参与一次乘法，但没有完整包含 Attention 的序列相关项、Norm、激活函数等。

#### 7.2 Attention 的序列相关计算

每层的 $QK^T$ 和 $PV$ 大致为：

$$
F_{score}\approx4S^2D
$$

当序列很长时，这部分不能忽略。它解释了为什么同一个参数量的模型，长序列 Prefill 比短序列更昂贵。

#### 7.3 训练计算

对 Dense 模型，前向加反向常粗略估算为：

$$
F_{train}\approx6P\quad /Token
$$

Stack-7.5B 约为 **45 GFLOPs / Token**。若一个优化 Step 处理 100 万个 Token，仅参数相关计算就约为 45 PFLOPs。

这仍是估算。Attention 长度项、Checkpointing 重计算、Padding 和 MoE 稀疏激活都会让真实计算偏离简单公式。

### 8. 一个 Token 的完整前向数据流

![从文本到 Loss 或下一个 Token 的完整数据流](/images/llm/llm-end-to-end-flow.svg)

#### 8.1 输入

```text
文本 → Tokenizer → Token ID [B,S] → Embedding [B,S,D]
```

Tokenizer 不在 NPU/GPU 的 Transformer Kernel 中运行；Embedding 本质上是按 Token ID 查表。

#### 8.2 进入每个 Decoder Block

```text
Hidden State
  ├─ RMSNorm → Q/K/V Projection → RoPE → Causal Attention → O Projection → 残差
  └─ RMSNorm → Gate/Up Projection → SwiGLU → Down Projection → 残差
```

主要算子包括 GEMM、RMSNorm、RoPE、Softmax、逐元素乘法和残差加法。训练时还需要保存或重算反向所需的信息。

#### 8.3 输出

```text
Hidden State → Final RMSNorm → LM Head → Logits [B,S,V]
```

- 训练：Shift Label 后计算 Cross Entropy Loss，再反向传播；
- 推理：取最后位置 Logits，经过采样得到下一个 Token，并更新 KV Cache。

### 9. 反向传播时发生什么

反向传播沿计算图反向应用链式法则：

1. Cross Entropy 产生 Logits Gradient；
2. LM Head 产生 Hidden Gradient 和权重梯度；
3. 梯度倒序穿过 32 个 Decoder Block；
4. 每个 Linear 同时计算输入梯度与权重梯度；
5. Optimizer 使用最终梯度更新参数。

若使用数据并行，每个 Rank 只看到不同 Mini-batch，得到的本地梯度不同。为了让所有模型副本保持一致，需要在更新前聚合梯度。

## 03-4｜模型切分、通信与性能模型

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

### 11. 一个资源预算示例

假设用 8 张设备做 BF16 推理，采用 TP=8，暂时忽略额外开销：

#### 11.1 权重

$$
15.0\ GB\div8\approx1.875\ GB/Rank
$$

实际并非所有参数都完全均分，Embedding、Norm、LM Head 可能采用不同分片或复制策略。

#### 11.2 KV Cache

若 8 个 KV Head 均匀分给 8 个 Rank，则每个 Rank 保存 1 个 KV Head。32K 单请求的 4 GiB KV Cache 可降至约 0.5 GiB/Rank。

#### 11.3 通信

显存下降不代表免费：每个 TP Layer 的部分输出需要聚合。随着 Batch、Sequence 和 Hidden Size 增长，激活通信也会增长。最终吞吐取决于 GEMM 与通信能否高效执行和重叠。

因此，“能放下”只回答了容量问题，还没有回答“跑得快不快”。

### 12. 从估算走向性能模型

可以用三个上限判断瓶颈：

#### 12.1 计算上限

$$
T_{compute}\approx\frac{实际FLOPs}{设备有效FLOPs/s}
$$

#### 12.2 HBM 上限

$$
T_{memory}\approx\frac{HBM搬运字节数}{有效HBM带宽}
$$

#### 12.3 通信上限

$$
T_{comm}\approx\alpha\times通信轮次+\beta\times通信字节数
$$

$\alpha$ 代表启动与同步延迟，$\beta$ 代表每字节传输成本。实际执行还要考虑拓扑、协议、并发和计算通信重叠。

粗略地看，总时间至少受最慢资源制约；存在严格依赖时，各段时间还会串行累加。AI Infra 的核心工作，就是让计算、访存和通信尽可能接近各自上限，并尽量重叠。

### 13. 一套通用的模型拆解清单

拿到任何模型配置时，按以下顺序分析：

#### 模型结构

1. Vocabulary、Hidden Size、Layer、Q/KV Head、Head Dimension、FFN Size 是多少？
2. 使用 MHA、GQA、MQA 还是 MoE？
3. Embedding 与 LM Head 是否共享？

#### 参数和计算

4. Embedding、Attention、FFN 各有多少参数？
5. 每 Token 前向和训练 FLOPs 是什么数量级？
6. 长序列 Attention 项什么时候不可忽略？

#### 显存

7. 权重采用什么精度？
8. 训练还保存哪些梯度和优化器状态？
9. KV Cache 每 Token、每请求、每 Batch 多大？
10. 激活是否 Checkpoint、分片或重计算？

#### 并行与通信

11. 哪个维度使用 DP、TP、PP、CP 或 EP？
12. 每次 Collective 之前和之后，张量 Shape 如何变化？
13. 通信是否跨节点，能否与计算重叠？

#### 服务指标

14. 关注 TTFT、TPOT、吞吐还是并发数？
15. Prefill 与 Decode 是否采用不同调度和资源策略？

## 本章总结

> 返回：[第 3 章首页](#)

::: tip 本章核心结论
1. 模型资源账本从配置开始：先算参数，再乘精度，最后加入激活、KV Cache 和运行时开销。
2. $2P$、$6P$ 是前向与训练的数量级估算，不包含所有长序列和系统开销。
3. 并行策略切开不同张量，下一算子需要的数据布局决定使用哪种集合通信。
:::

### 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

### ⚠️ 易错点

1. **参数量等于运行显存吗？** 不等于，还要乘数据类型并加入其他状态。
2. **7.5B BF16 一定占 15 GiB 吗？** 15 是十进制 GB，约为 14 GiB。
3. **推理只需要权重吗？** 还需要 KV Cache、激活、Workspace 和框架内存。
4. **训练显存固定是 16 Byte/参数吗？** 不是，它取决于精度和优化器实现，而且没包含激活。
5. **$2P$ 包含全部 Attention 计算吗？** 没有完整包含随 $S^2$ 增长的部分。
6. **FlashAttention 会减少模型参数吗？** 不会，它主要优化 Attention 的 IO。
7. **多卡一定更快吗？** 不一定，切分降低单卡负担，也引入通信和同步。
8. **AllReduce 是模型自动产生的吗？** 它来自具体并行策略对数据依赖的改写。

### 15. 动手练习

#### 练习 1：修改模型配置

把 Stack-7.5B 的 $F$ 从 14336 改为 11008，重新计算单层 FFN 参数量和总参数量。

#### 练习 2：设计推理容量

假设单卡除权重外还有 24 GiB 可用于 KV Cache，使用 BF16 KV。估算 32K 请求最多能容纳几个；再计算 INT8 KV 的理论结果。实际部署为什么必须留余量？

#### 练习 3：画出 TP 数据流

选择 Column Parallel 和 Row Parallel 切分一层 MLP，标出输入、局部矩阵、局部输出的 Shape，并判断在哪里需要 AllGather、AllReduce 或 ReduceScatter。

### 16. 自测题

1. Stack-7.5B 的 Embedding 参数量怎样计算？
2. 为什么 GQA 的 K/V Projection 参数少于 Q Projection？
3. 为什么这个模型的大多数 Block 参数位于 FFN？
4. 权重共享为什么能少一份 $V\times D$ 参数？
5. BF16 权重大小怎样从参数量估算？
6. KV Cache 为什么与 Layer、KV Head 和 Sequence Length 成正比？
7. 训练为什么还需要梯度和优化器状态？
8. $2P$ 和 $6P$ 分别用于什么粗略估算？
9. DP、TP、PP、CP、EP 分别切分什么？
10. 怎样从张量依赖推导需要哪一种集合通信？

::: details 自测答案

1. $V\times D=128000\times4096=524,288,000$，约 524.3M。
2. Q 有 32 个 Head，而共享后的 K/V 各只有 8 个 Head，投影输出宽度分别是 4096 与 1024。
3. SwiGLU 有两个 $D\to F$ 和一个 $F\to D$ 大矩阵，共 $3DF\approx176.16M$；Attention 约 41.94M。
4. 输入查表矩阵与输出 Logits 投影使用同一组参数，不再另外存储一个独立 LM Head。
5. 参数量乘每参数 2 Byte；7.50B 参数约为 15.0 GB，换算后约 14.0 GiB。
6. 每层都要为每个历史 Token 保存所有 KV Head 的 K 和 V，所以各维度都会乘入公式。
7. 反向传播产生梯度，AdamW 还要保存一阶、二阶矩等状态用于更新；这些在纯推理中不需要。
8. $2P$ 粗估 Dense 模型每 Token 前向参数相关 FLOPs，$6P$ 粗估前向加反向的训练 FLOPs。
9. DP 切 Batch；TP 切矩阵或 Head；PP 切 Layer；CP 切 Sequence；EP 切 Expert。
10. 先确定张量沿什么维度被分片，再判断下一算子需要拼接、求和、分片归约还是按目标重分发，分别映射到 AllGather、AllReduce、ReduceScatter 或 Alltoall 等语义。

:::

### 17. 大模型基础阶段总结

12 课形成了一条完整链路：

```text
文本与 Token
  → Embedding 与张量
  → Attention、RoPE、Decoder Block
  → 训练、反向传播与对齐
  → 自回归推理与 KV Cache
  → MoE、GQA、量化与长上下文
  → 参数、FLOPs、显存与通信估算
```

完成这一阶段后，面对一个算子或集合通信调用，不再只看到 API 名称，而能继续追问：它服务于模型中的哪段计算，张量 Shape 是什么，数据为什么必须移动，瓶颈可能在哪里。

### 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA](https://arxiv.org/abs/2305.13245)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Megatron-LM](https://arxiv.org/abs/2104.04473)
- [ZeRO](https://arxiv.org/abs/1910.02054)

大模型基础 12 课至此完成。下一阶段进入 **AI Infra**：从加速器执行模型、存储层次、算术强度和 Roofline 开始，建立分析算子性能与集合通信性能的底层方法。

---

<!-- chapter-navigation -->
