# 第 2 章｜现代模型的效率设计：GQA、量化与长上下文

> 本章目标：理解现代大模型怎样减少计算、显存和数据搬运；能够计算 MHA、GQA、MQA 的 KV Cache；分清权重量化、激活量化与 KV Cache 量化；理解长上下文优化究竟改变了算法、数据布局还是执行方式。

## 本章导学

::: tip 本章核心结论
1. GQA/MQA 共享 K/V Head，主要减少 KV Cache 和 Decode 搬运，不是减少 Q Head。
2. 量化降低存储位宽，但收益取决于量化对象、粒度、硬件和 Kernel。
3. RoPE Scaling、Sliding Window、FlashAttention 和 Paged Attention 解决的是四类不同问题。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。量化公式可留到第二遍阅读。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](#⚡-速通-约-5-分钟)：一个"给出版社省钱"类比讲完整章，再回来按单元深入。

**先看这几张核心图：** MHA/GQA/MQA Sharing、Group Quantization、Long Context Techniques 三张图。

- [ ] 我能算出 GQA 相对 MHA 的 KV 缩减比例
- [ ] 我能区分权重、激活和 KV Cache 量化
- [ ] 我能说出 FlashAttention 没有改变什么

## 1. 为什么“模型能算”还不够

模型真正部署到硬件上，会遇到三个现实限制：

- **算不动**：计算量太大，首 Token 等待时间过长；
- **放不下**：权重、激活或 KV Cache 超出显存；
- **搬不快**：计算单元在等待 HBM 或设备间通信的数据。

| 资源 | 典型问题 | 常见优化 |
| --- | --- | --- |
| 计算 | Attention 随序列长度平方增长 | Sliding Window、稀疏注意力 |
| 显存容量 | 权重和 KV Cache 太大 | 量化、GQA/MQA、Paged Attention |
| 显存带宽 | Decode 反复读取权重和 KV | 量化、算子融合、GQA/MQA |
| 片上存储与访存 | 中间矩阵反复写回 HBM | FlashAttention、融合算子 |
| 设备间通信 | 单设备放不下长序列 | TP、CP、通信计算重叠 |

一个优化可能改善多种资源，也可能用计算换显存。例如量化减少存储与搬运，却增加了缩放和反量化操作。

## 本章单元

- **02-1（约 15 分钟）**：[MHA、MQA 与 GQA](#_02-1-mha、mqa-与-gqa)
- **02-2（约 15 分钟）**：[量化原理、粒度与显存收益](#_02-2-量化原理、粒度与显存收益)
- **02-3（约 15 分钟）**：[长上下文的四条优化路线](#_02-3-长上下文的四条优化路线)
- **02-4（约 15 分钟）**：[把模型优化映射到算子与通信](#_02-4-把模型优化映射到算子与通信)

- **本章总结**：[串联知识、练习与检查](#本章总结)

## ⚡ 速通（约 5 分钟）

> 适用：时间紧，或完整版跟不动时。这页用"给出版社省钱"讲完整章，读完抓住 80% 的主干。任何一节想深入，拉到文末点对应单元。

### 1. 记者们共享档案柜：MHA、MQA、GQA

还记得《训练与推理系统》第 3 章的"桌上笔记"（KV Cache）吗？长会话+高并发时，**笔记比课本还占地方**。怎么省？

先看现状（**MHA**）：编辑部有 32 个记者（Q 头），**每人一个专属档案柜**（自己的 K/V 头）——柜子总共 32 组。

省钱思路：**柜子共享，记者不减**。

| 方案 | 档案柜 | 效果 |
| --- | --- | --- |
| **MHA** | 每人一个柜 | 效果最好，柜子最贵 |
| **GQA** | 每 4 人共用一个柜 | 柜子降到 1/4，效果只掉一点点（现代主流） |
| **MQA** | 全编辑部共用一个柜 | 柜子最少，效果掉得多些 |

关键澄清：**减的是档案柜（K/V 头），不是记者（Q 头）**。省下来的直接对应：KV Cache 显存 ↓、Decode 时每字要读的笔记量 ↓。

::: details 小测 1：GQA 会减少 Q 头（记者）吗？
不会。GQA 只共享 K/V 头（档案柜）——40 个 Q 头配 8 个 KV 头是常见配置，KV Cache 直接降到 MHA 的 8/40。Q 头数量和表达能力基本保留。
:::

### 2. 口袋书工程：量化

**量化 = 把精装书印成口袋书**：原来每个数字用 16 位存（BF16），现在 8 位（INT8）甚至 4 位（INT4）——**书架（显存）直接省一半或四分之三**。

三个必须知道的细节：

1. **印什么书分三档**：只压**权重**（Weight-Only，书压薄）、权重和**便签**（激活）一起压（W8A8）、压**档案柜**（KV Cache 量化）——压的东西不同，收益位置不同；
2. **要配换算表**（Scale）：低位整数和原浮点范围之间要建映射，读的时候还得换算回来（反量化）——**换算本身有开销**；
3. **省书架 ≠ 省时间**：读口袋书还是要换算，速度快慢取决于**印刷厂设备**（硬件和 Kernel 是否支持低位计算）。INT4 的书不保证读得快 4 倍——**位宽直接决定的只有存储量**。

::: details 小测 2：权重从 BF16 换成 INT4，推理一定快 4 倍吗？
不一定。存储量确实变成 1/4，但实际速度还取决于：硬件有没有 4-bit 计算单元、反量化的开销、Kernel 写得好不好、以及原来的瓶颈是不是显存带宽。如果原瓶颈是别的，压书架根本帮不上忙。
:::

### 3. 四件不同的工具，别拿锤子拧螺丝

长上下文和推理优化领域，四个名字经常被混为一谈，其实**各治一种病**：

| 工具 | 治什么病 | 机理 |
| --- | --- | --- |
| **FlashAttention** | 打分表来回搬运太贵 | 精确全量计算，只省搬运（IO），**不改数学** |
| **Sliding Window** | 全场打分开不起 | 干脆**只跟附近 W 个座位开会**（近似，改了关注范围） |
| **RoPE Scaling** | 座位号没见过大角度 | 改转速/缩放角度，治**位置外推**，不改计算量 |
| **PagedAttention** | 笔记本分配浪费碎片 | 按需分页租格子，治**管理**，不减少笔记内容本身 |

诊断口诀：先分清病根是**搬运、范围、外推还是碎片**，再选工具。

::: details 小测 3：FlashAttention 和 Sliding Window 都让长文本变快，本质区别是什么？
FlashAttention 算的还是精确的全量注意力，只是换了执行方式省 IO——结果不变；Sliding Window 直接砍掉了远距离的关注——计算量真的变少，但这是近似，模型行为也变了。一个是"跑得聪明"，一个是"少跑路"。
:::

### 3 句话带走

1. GQA/MQA 是**记者共享档案柜**：减 K/V 头不减 Q 头，直接省 KV Cache；
2. 量化是**口袋书工程**：位宽直接决定的只有存储，速度还看硬件与 Kernel；
3. 四件工具各治一病：**FlashAttention 治搬运、Window 砍范围、RoPE Scaling 治外推、Paged 治碎片**。

### 黑话小词典

| 术语 | 人话 |
| --- | --- |
| MHA / GQA / MQA | 人手一柜 / 几人一柜 / 全员一柜 |
| Q 头 / KV 头 | 记者 / 档案柜 |
| 量化 / Scale / 反量化 | 印口袋书 / 换算表 / 读时换算回来 |
| Weight-Only / W8A8 | 只压书 / 书和便签一起压 |
| 外推 / RoPE Scaling | 没见过的大角度 / 调转速来适应 |
| Sliding Window | 只跟附近几个座位开会 |
| PagedAttention | 笔记本按需分页，治碎片 |

### 想深入？

每节 15 分钟，按需点开，不必按顺序全读：

| 单元 | 讲什么（白话） | 什么时候需要它 |
| --- | --- | --- |
| [02-1 MHA、MQA 与 GQA](#_02-1-mha、mqa-与-gqa) | 档案柜共享的完整机制 | 想算 KV Cache 收益 |
| [02-2 量化原理、粒度与显存收益](#_02-2-量化原理、粒度与显存收益) | 口袋书工程的全部细节 | 想做量化部署 |
| [02-3 长上下文的四条优化路线](#_02-3-长上下文的四条优化路线) | 四件工具的深入对比 | 想做长文本方案选型 |
| [02-4 把模型优化映射到算子与通信](#_02-4-把模型优化映射到算子与通信) | 从模型层下到系统层 | 想衔接 AI Infra 视角 |
| [本章总结](#本章总结) | 动手练习 + 自测 | 想检验整章掌握程度 |

## 02-1｜MHA、MQA 与 GQA

::: info 本单元目标
围绕 **MHA、MQA 与 GQA** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

设 Hidden Size 为 $D$，Query Head 数为 $N_q$，每个 Head 的维度为 $d_h$，通常有：

$$
D=N_qd_h
$$

标准 Multi-Head Attention（MHA）中：

$$
N_q=N_k=N_v
$$

每个 Query Head 使用与它对应的一组 Key 和 Value Head。表达能力充分，但推理时必须为历史每个 Token 保存全部 K/V Head。

考虑 Batch Size $B$、层数 $L_S$、序列长度 $S$ 和每元素字节数 $b$，KV Cache 大小为：

$$
M_{KV}=2BL_SN_{kv}d_hSb
$$

这里的 $2$ 代表 Key 和 Value。KV Cache 会随 **Batch、层数、KV Head 数、序列长度和位宽**线性增长。

### 3. MQA 与 GQA 怎样共享 K/V

Multi-Query Attention（MQA）让所有 Query Head 共享同一组 K/V：

$$
N_{kv}=1
$$

Grouped-Query Attention（GQA）是 MHA 和 MQA 之间的折中：多个 Query Head 分成一组，共享一个 K/V Head。

$$
1<N_{kv}<N_q
$$

![MHA、GQA 与 MQA 中 Query Head 如何连接到 KV Head](/images/llm/mha-gqa-mqa-sharing.svg)

图中真正变化的是 **K/V 的份数**：

- MHA：8 个 Q Head 对应 8 组 K/V；
- GQA：8 个 Q Head 分成 2 组，只保存 2 组 K/V；
- MQA：8 个 Q Head 全部共享 1 组 K/V。

GQA/MQA 通常不会减少 Q Head，也不会让输出只剩一个 Head。每个 Q Head 仍独立生成注意力输出，只是查询的 K/V 被共享。

#### 3.1 从 Shape 看 GQA

以 $N_q=32$、$N_{kv}=8$、$d_h=128$ 为例，每 4 个 Q Head 共享一组 K/V：

| 张量 | Shape |
| --- | --- |
| $Q$ | `[B, 32, S, 128]` |
| $K$ | `[B, 8, S, 128]` |
| $V$ | `[B, 8, S, 128]` |
| Attention 输出 | `[B, 32, S, 128]` |

高效实现不一定真的将 K/V 复制 4 份，而会利用广播、Stride 或专用 Kernel 完成 Head 映射。

#### 3.2 一个 KV Cache 算例

假设模型有 32 层、32 个 Q Head，$d_h=128$，序列长度为 32768，Batch Size 为 1，KV 使用 BF16：

| 结构 | KV Head 数 | KV Cache |
| --- | ---: | ---: |
| MHA | 32 | 16 GiB |
| GQA | 8 | 4 GiB |
| MQA | 1 | 0.5 GiB |

GQA 相比 MHA 把缓存降至 $N_{kv}/N_q=1/4$。它通常比 MQA 保留更多 K/V 表达能力，又能明显降低 Decode 时的缓存读取量，因此成为常见折中。

### 4. GQA 对 Tensor Parallel 的影响

Tensor Parallel 常把 Attention Head 分给不同 Rank。MHA 的 Q/K/V Head 数相同，较容易平均切分；GQA 的 KV Head 更少，会产生新的约束：

- $N_{kv}$ 能被 TP Size 整除时，可以均匀分片；
- TP Size 大于 $N_{kv}$ 时，部分 Rank 可能复制同一 KV Head；
- 复制可以简化计算或减少通信，但会增加显存；
- 细切单个 KV Head 则可能引入额外通信和复杂 Kernel。

例如 $N_q=32$、$N_{kv}=8$：TP=8 时每个 Rank 可以持有 4 个 Q Head 和 1 个 KV Head；TP=16 时 KV Head 不够一一分配，通常需要复制或调整并行布局。

所以模型配置会直接限制并行策略。看到 `num_attention_heads` 和 `num_key_value_heads` 时，也要想到 Head 如何映射到 Rank。

## 02-2｜量化原理、粒度与显存收益

::: info 本单元目标
围绕 **量化原理、粒度与显存收益** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

量化使用更低位宽的数值近似表示原本的浮点数据。例如把 BF16/FP16 权重转换为 INT8 或 INT4，以减少存储和搬运。

简单的对称量化可以写成：

$$
q=clip\left(round\left(\frac{x}{s}\right),q_{min},q_{max}\right)
$$

反量化近似恢复：

$$
\hat{x}=s\cdot q
$$

- $x$：原始浮点数；
- $s$：Scale，描述一个整数刻度代表多大的实数范围；
- $q$：低位宽整数；
- $\hat{x}$：恢复出的近似值。

![分组量化如何用 Scale 将浮点权重映射为低位整数](/images/llm/group-quantization.svg)

同一组权重共享一个 Scale。Group 越小，Scale 越能贴合局部数值范围，误差通常越小；代价是需要更多 Scale，数据布局和 Kernel 也更复杂。

### 6. 量化粒度

| 粒度 | Scale 数量 | 特点 |
| --- | ---: | --- |
| Per-Tensor | 整个张量 1 个 | 元数据少，但容易被极端值拉大范围 |
| Per-Channel | 每个输出通道 1 个 | 更适应不同通道的数值分布 |
| Group-Wise | 每若干元素 1 个 | INT4 权重量化常见的精度与开销折中 |

还要区分：

- **对称量化**：通常围绕 0 使用正负对称范围，主要保存 Scale；
- **非对称量化**：额外使用 Zero Point，使整数零点对应某个实数位置。

因此，不能只说“模型是 INT4”，还应问：什么张量被量化、按什么粒度、Scale 使用什么精度、计算时采用什么 Kernel。

### 7. 权重、激活和 KV Cache 量化不是一回事

#### 7.1 Weight-Only Quantization

只把权重压到 INT8/INT4，激活仍使用 FP16/BF16。计算时由 Kernel 在片上反量化，或使用支持混合类型的矩阵乘法。

- 减少权重显存；
- Decode 每步读取权重更少，可缓解带宽瓶颈；
- 不必处理所有动态激活的低位表示，通常更容易保持精度。

#### 7.2 Weight-Activation Quantization

权重和激活都使用低位宽，例如 W8A8。它有机会使用整数计算单元提高吞吐，但激活分布会随输入变化，校准和异常值处理更加困难。

#### 7.3 KV Cache Quantization

把缓存的 K/V 从 FP16/BF16 压缩到 FP8、INT8 或更低位宽。它不减少模型权重，却能降低长上下文和大 Batch 下的缓存容量与读取带宽。

```text
模型加载： [量化权重]
               │
输入 ──▶ [低位/高精度激活] ──▶ Attention ──▶ [量化 KV Cache]
               │                                  │
               └──── 当前 Token 的计算 ───────────┴── 历史 Token 的存储
```

### 8. 量化能节省多少显存

忽略 Scale、Zero Point、对齐和运行时 Buffer，$P$ 个参数使用 $b$ bit 时：

$$
M_{weight}=\frac{P\times b}{8}\ Bytes
$$

以 70 亿参数为例：

| 权重格式 | 理论大小 |
| --- | ---: |
| FP16/BF16 | 约 14 GB |
| INT8 | 约 7 GB |
| INT4 | 约 3.5 GB |

真实占用通常更高，因为还要保存 Scale、元数据、未量化层和对齐空间。推理总显存还包含 KV Cache、工作区和框架开销。

::: warning 4 bit 不等于必然快 4 倍
性能还取决于硬件支持、Kernel 的反量化效率、矩阵 Shape、Batch Size 和真正的瓶颈。如果系统原本受计算而非带宽限制，显存下降不会自动变成同比例加速。
:::

### 9. PTQ 与 QAT

**Post-Training Quantization（PTQ）**在模型训练完成后，根据权重或少量校准数据确定量化参数。成本低，是部署已有模型时的常见选择。

**Quantization-Aware Training（QAT）**在训练或微调过程中模拟量化误差，让模型主动适应低精度表示。它通常能提高低位精度，但训练成本更高。

两者都不意味着计算中的每个张量始终以整数存在。真实系统常采用“低位存储 + 片上反量化 + 高精度累加”的混合执行方式。

## 02-3｜长上下文的四条优化路线

::: info 本单元目标
围绕 **长上下文的四条优化路线** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

序列从 $S$ 增长到 $2S$ 时，各类资源的变化不同：

| 项目 | 随序列长度变化 | 主要阶段 |
| --- | --- | --- |
| Attention Score 数量 | $O(S^2)$ | Prefill / 训练 |
| Q/K/V 和普通激活 | $O(S)$ | Prefill / 训练 |
| KV Cache 容量 | $O(S)$ | 推理 |
| 单步 Decode 读取历史 KV | $O(S)$ | Decode |
| 生成整段序列的 Attention 工作 | 近似 $O(S^2)$ | 完整生成 |

“支持 128K 上下文”至少包含三个不同问题：

1. 位置表示能否外推到这么远？
2. Attention 和显存能否处理这么长？
3. 模型是否真的能从远处找回关键信息？

上下文窗口很大，不代表模型在所有距离上都能同样有效地使用信息。

### 11. 长上下文优化的四条路线

![不同长上下文技术分别改变计算范围、数据搬运和缓存布局](/images/llm/long-context-techniques.svg)

#### 11.1 改位置表示：RoPE Scaling

调整 RoPE 的位置映射或频率，使更长的真实位置落入模型可处理的范围。它解决位置外推，但不会消除全量 Attention 的 $O(S^2)$ 计算。

#### 11.2 少看一部分：Sliding Window / Sparse Attention

每个 Token 只关注附近窗口或少量特殊位置。窗口 $W$ 固定时，Score 规模可从 $O(S^2)$ 降为 $O(SW)$。

代价是单层不能直接看到全部历史，需要多层传播、全局 Token 或混合全局 Attention 保留远距离信息。

#### 11.3 精确计算但少搬数据：FlashAttention

FlashAttention 仍计算精确的全量 Attention，没有把数学复杂度从 $O(S^2)$ 改掉。它把 Q/K/V 分块搬入片上存储，在线维护 Softmax 统计量，避免显式将完整 $S\times S$ Score Matrix 写回 HBM。

它优化的是 **IO 和数据布局**，不是把注意力改成近似算法。

#### 11.4 更高效地管理推理缓存

- GQA/MQA：减少 KV Head 数；
- KV Cache 量化：减少每个缓存元素的字节数；
- Paged Attention：按块管理 KV，减少连续空间预留和碎片；
- Chunked Prefill：把超长 Prompt 分块调度；
- Prefix Cache：为相同前缀复用已经计算的 KV。

这些方案可以组合使用。

### 12. Paged Attention 为什么像虚拟内存

传统做法为请求预留一段连续且足够大的 KV 空间，但最终生成长度未知：预留太大会浪费，太小又需要扩容搬迁。

Paged Attention 将逻辑连续的 Token 映射到多个固定大小的物理 Block：

```text
请求 A 的逻辑 Token： [0..15] [16..31] [32..47]
                          │       │        │
Block Table：            B7      B2       B9
                          │       │        │
物理显存：            [B2] ... [B7] ... [B9]
```

新 Token 到来时按需分配 Block，不要求整个请求的 KV 在物理显存中连续。它改善显存管理与动态批处理，但不会减少每个有效 Token 需要保存的 K/V。

### 13. 长序列怎样引出通信

单设备放不下长序列的激活和 KV 时，可以沿 Sequence 维切到多个设备，即 Sequence Parallel 或 Context Parallel。

在 Context Parallel 中，每个 Rank 只持有一段 Token，但某段 Query 若要执行全局 Attention，就必须获得其他 Rank 的 K/V，或通过 Ring 让 K/V 分块依次流过各 Rank。

问题于是从“一张卡怎样少读 HBM”，扩展为“多张卡怎样交换 K/V，并让通信与 Attention 计算重叠”。这可能涉及 AllGather、Alltoall 或 P2P Ring，具体代价会在 AI Infra 课程中展开。

## 02-4｜把模型优化映射到算子与通信

::: info 本单元目标
围绕 **把模型优化映射到算子与通信** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

假设服务遇到“长上下文下 Batch 开不大、Decode 又很慢”：

1. **GQA** 减少 KV Head 数，缓存容量和每步读取量一起下降；
2. **权重量化**减少每步 Decode 读取的模型权重；
3. **KV Cache 量化**进一步压缩历史 K/V；
4. **Paged Attention**减少动态请求造成的显存碎片；
5. **FlashAttention**提高 Prefill 的 Attention 执行效率；
6. **Chunked Prefill**改善长 Prompt 与短请求共同服务时的调度。

它们解决的是不同层的问题。性能优化应先找到瓶颈，再选择对应技术。

### 15. 与昇腾算子和集合通信开发的关系

阅读实现时，可以沿以下问题建立模型语义与底层执行的联系：

1. `num_attention_heads`、`num_key_value_heads` 怎样映射到每个 Rank？
2. GQA 的 K/V 是物理复制、广播读取，还是 Kernel 内完成 Head 映射？
3. 量化数据、Scale 和 Zero Point 采用什么布局与对齐？
4. 反量化发生在 HBM、片上 Buffer 还是矩阵计算流水中？
5. KV Cache 的 Block Table 由 Host 还是 Device 管理？
6. FlashAttention 的分块大小怎样受片上存储容量限制？
7. Context Parallel 交换的是 Q、K/V、中间结果还是输出？
8. 通信能否与当前分块的矩阵计算和 Softmax 重叠？

这些问题会让“GQA”“INT4”“长上下文”从产品参数变成可落到 Shape、内存地址、Kernel 和通信量上的工程对象。

## 本章总结

> 返回：[第 2 章首页](#)

::: tip 本章核心结论
1. GQA/MQA 共享 K/V Head，主要减少 KV Cache 和 Decode 搬运，不是减少 Q Head。
2. 量化降低存储位宽，但收益取决于量化对象、粒度、硬件和 Kernel。
3. RoPE Scaling、Sliding Window、FlashAttention 和 Paged Attention 解决的是四类不同问题。
:::

### 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

### ⚠️ 易错点

1. **GQA 会减少 Q Head 吗？** 通常不会，它主要减少 K/V Head。
2. **MQA 是否只产生一个 Attention Head？** 不是，多个 Q Head 仍产生多个输出，只是共享 K/V。
3. **KV Cache 是否保存 Q？** 通常不保存，未来只需要新的 Q 查询历史 K/V。
4. **INT4 模型的所有计算都是 4 bit 吗？** 不一定，常见方案只是 INT4 存权重。
5. **量化只影响显存容量吗？** 还影响带宽、Kernel、计算单元利用率和精度。
6. **FlashAttention 是稀疏注意力吗？** 不是，它保持精确全量 Attention，主要减少 HBM IO。
7. **Paged Attention 会减少 KV 的理论元素数吗？** 不会，它改善分配方式和碎片。
8. **RoPE Scaling 会让 Attention 变为线性复杂度吗？** 不会，它解决位置外推。

### 17. 动手练习

#### 练习 1：计算 GQA KV Cache

设 $B=4$、$L_S=40$、$N_q=40$、$N_{kv}=8$、$d_h=128$、$S=16384$，KV 使用 BF16。计算 GQA 的 KV Cache，并与 MHA 比较。

#### 练习 2：估算量化权重

一个 13B 模型分别以 BF16、INT8、INT4 保存权重，忽略元数据时各需要多少 GB？为什么真实显存占用会更高？

#### 练习 3：为瓶颈选择方案

分别为以下问题选择优先方案：Prefill 中 $S\times S$ 中间矩阵造成大量 HBM 访问；Decode 时 KV 太大；动态请求造成显存碎片；模型无法理解超出训练长度的位置。

### 18. 自测题

1. MHA、GQA、MQA 的核心差别是什么？
2. KV Cache 为什么不保存历史 Q？
3. GQA 的 KV Cache 相对 MHA 缩小多少？
4. TP Size 大于 KV Head 数时为什么需要特别处理？
5. Scale 在量化中起什么作用？
6. Weight-Only、W8A8 和 KV Cache 量化分别压缩什么？
7. 为什么 INT4 权重不保证推理比 FP16 快 4 倍？
8. 长上下文对 Prefill 和 Decode 的压力有什么不同？
9. FlashAttention、Sliding Window 和 RoPE Scaling 分别解决什么？
10. Paged Attention 为什么不减少 KV Cache 的理论数据量？

::: details 自测答案

1. 区别在 K/V Head 的共享程度：MHA 每个 Q 对应独立 K/V，GQA 一组 Q 共享 K/V，MQA 所有 Q 共享一组 K/V。
2. 每个 Decode Step 只需当前 Token 的 Q 查询全部历史 K/V，过去的 Q 不会再次参与后续注意力。
3. 其他条件相同时，比例为 $N_{kv}/N_q$。例如 32 个 Q、8 个 KV 时降到 $1/4$。
4. KV Head 无法在所有 Rank 间一一均分，可能需要复制 KV、改变分片或引入通信。
5. Scale 建立低位整数与原始实数范围之间的映射，用于量化和近似反量化。
6. Weight-Only 压权重；W8A8 压权重和激活；KV Cache 量化压历史 K/V。
7. 还取决于硬件支持、Kernel 效率、反量化开销、Shape 和原始瓶颈；位宽只直接决定存储量。
8. Prefill 主要承担全量 Attention 的平方计算与中间数据压力；Decode 每步读取随历史长度线性增长的 KV，并对单步延迟敏感。
9. FlashAttention 优化精确 Attention 的 IO；Sliding Window 减少关注位置；RoPE Scaling 改善位置外推。
10. 它只是把逻辑连续 KV 映射到按需分配的物理 Block；每个有效 Token 的 K/V 仍需保存。

:::

### 19. 本章小结

- GQA/MQA 通过共享 K/V Head 减少 KV Cache 和 Decode 数据读取；
- GQA 在 MHA 表达能力与 MQA 效率之间折中，并影响 TP Head 分片；
- 量化需要结合量化对象、粒度和执行 Kernel 判断收益；
- Weight-Only、激活量化与 KV Cache 量化作用于不同数据；
- 长上下文同时涉及位置外推、Attention 计算、KV Cache 和服务调度；
- FlashAttention 优化 IO，Sliding Window 改变关注范围，Paged Attention 改善缓存管理；
- 效率技术最终都要落到计算量、存储量、搬运字节数和通信依赖上分析。

### 参考资料

- [Fast Transformer Decoding: One Write-Head is All You Need（MQA）](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [PagedAttention](https://arxiv.org/abs/2309.06180)

下一课将进行大模型基础综合实战：选取一个现代 Decoder-only LLM，估算参数量、训练与推理 FLOPs、权重和 KV Cache 显存，并把数据流映射到算子与通信。

---

<!-- chapter-navigation -->
