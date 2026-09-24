# 第 5 章｜RMSNorm、SwiGLU 与完整 Decoder Block

## 本章导学

::: tip 本章核心结论
1. Attention 在 Token 之间混合信息，SwiGLU FFN 在每个 Token 内部混合特征。
2. RMSNorm 控制数值尺度，残差连接保留原信息并提供更直接的梯度路径。
3. 一个现代 Pre-Norm Block 是两次“Norm → 子层 → Residual”的串联。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，分别理解结构、FFN、资源和并行。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](#⚡-速通-约-5-分钟)：一条标准生产车间类比讲完整章，再回来按单元深入。

**先看这几张核心图：** Decoder Block 总图、RMSNorm Process、SwiGLU Flow 三张图。

- [ ] 我能默画一个 Decoder Block
- [ ] 我能区分 Attention 与 FFN 的职责
- [ ] 我能解释残差连接为什么不是简单重复

## 本章要解决的问题

前四章已经得到 Token 表示并学会了 Attention，但 Attention 只是 Transformer Block 的一个子模块。现代 Decoder-only 模型通常还包含归一化、MLP 和残差连接。

本章以常见的 Llama 风格 Pre-Norm Block 为主线：

```text
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

学完后应能：

- 画出一个完整 Decoder Block；
- 解释 RMSNorm、残差和 MLP 各自解决什么问题；
- 区分 Attention 的 Token 混合与 MLP 的特征混合；
- 估算一层的参数量与主要 FLOPs；
- 从 Block 结构定位张量并行可能产生通信的位置。

## 本章单元

- **05-1（约 15 分钟）**：[RMSNorm、残差与 Block 结构](#_05-1-rmsnorm、残差与-block-结构)
- **05-2（约 15 分钟）**：[MLP 与 SwiGLU 数据流](#_05-2-mlp-与-swiglu-数据流)
- **05-3（约 15 分钟）**：[参数量、FLOPs 与显存对象](#_05-3-参数量、flops-与显存对象)
- **05-4（约 15 分钟）**：[张量并行怎样切分 Block](#_05-4-张量并行怎样切分-block)

- **本章总结**：[串联知识、练习与检查](#本章总结)

## ⚡ 速通（约 5 分钟）

> 适用：时间紧，或完整版跟不动时。这页用一条标准生产车间讲完整章，读完抓住 80% 的主干。任何一节想深入，拉到文末点对应单元。

### 1. 一条标准车间流水线

上一章的会议室（Attention）只是一个工位。把工位们串起来，就得到大模型的基本重复单元——**Decoder Block**。一层 Block 是一条标准车间流水线，**每个工件（Token）从头走到尾**：

```text
工件进车间
  ① 整理工装（RMSNorm：把数值尺度调均匀）
  ② 会议工位（Attention：工件之间交换信息） → 结果叠回主路
  ③ 再整理工装（第二次 RMSNorm）
  ④ 深加工工位（MLP：每个工件独立深加工） → 结果叠回主路
工件出车间（进入下一层，同样的车间再走一遍，走 N 遍）
```

两个设计巧思：

- **整理工装（RMSNorm）**：不改变工件尺寸，只把数值"整理服帖"，防止越走越爆/越走越没声；
- **主路不动（残差连接）**：传送带上的原件永远保留，工位的加工结果**加上去**而不是替换掉。原件永远直达，学习时信号也有捷径可走——这就是为什么现代模型能堆几十层。

::: details 小测 1：Pre-Norm Block 的执行顺序？
整理（Norm）→ 会议（Attention）→ 叠回主路（残差相加）→ 再整理（Norm）→ 深加工（MLP）→ 再叠回（残差相加）。注意顺序是"先整理再进工位"（Pre-Norm），而且残差相加要求工位输出必须恢复到和主路一样的尺寸。
:::

### 2. 两个工位，干完全然不同的活

| 工位 | 混合什么 | 类比 |
| --- | --- | --- |
| **Attention（会议工位）** | **不同 Token 之间**交换信息 | 工件们开会互通有无 |
| **MLP（深加工工位）** | **每个 Token 内部**的特征变换 | 每个工件独立进机器精雕 |

MLP 工位内部是三台机器的**闸门结构（SwiGLU）**：

```text
原件 → [闸门机 Gate]：哪些特征放行？
     → [候选机 Up]：准备候选内容
     → 两者逐位相乘（闸门决定放行多少）
     → [收回机 Down]：压回原尺寸，好接回主路
```

::: details 小测 2：Attention 和 MLP 分别负责什么？
Attention 沿序列维混合——让不同位置的信息互相流动（"它"知道上下文）；MLP 逐 Token 独立——在特征维做非线性深加工（"它"自己变聪明）。一个管"交流"，一个管"内功"。
:::

### 3. 车间的账本：参数、显存和拆机器

**参数量两句话就能算**（H 是隐藏宽度，I 是中间宽度）：

- Attention：Q/K/V/输出四个 `[H,H]` 投影 ≈ **4H²**；
- MLP：闸门/候选/收回三个 `[H,I]` 级矩阵 ≈ **3HI**。

通常 `I ≈ 2.75H`，所以 **MLP 才是参数大户**。

两个反直觉的账：

1. **参数多 ≠ 慢**。整理工装（Norm）、闸门相乘这类"每读一个数只算一下"的轻活，反而常是时间杀手——它们天生是搬运瓶颈（AI Infra 会展开）；
2. **参数量 ≠ 显存**。训练时还要存梯度、优化器的两本账、所有中间半成品——实测常是参数的十几倍。

最后是**拆机器（张量并行 TP）**：一台大机器拆给多个车间各拿一半，各干各的再**拼结果**——这一"拼"，就是 AllReduce 通信的由来。

::: details 小测 3：为什么参数量不等于训练显存？
训练显存 = 权重 + 梯度 + 优化器状态（AdamW 每个参数还要两本历史账）+ 前向激活（随 Batch 和序列长度增长）+ 通信装箱等运行开销。参数只是四项中的一项。
:::

### 3 句话带走

1. 一层 Block = **两次"整理→工位→叠回主路"**：先是会议（Attention），再是深加工（MLP）；
2. 分工明确：**Attention 管 Token 之间交流，MLP 管每个 Token 内部加工**（参数大户是 MLP：3HI）；
3. **参数多≠慢、参数量≠显存**；拆机器（TP）干活的"拼结果"就是 AllReduce。

### 黑话小词典

| 术语 | 人话 |
| --- | --- |
| Decoder Block | 一条标准车间流水线，整个模型重复 N 条 |
| RMSNorm | 进工位前整理工装（调数值尺度，不改尺寸） |
| 残差连接 | 主路原件保留，加工结果"加上去" |
| Pre-Norm | 先整理再进工位（现代主流顺序） |
| MLP / FFN | 深加工工位（逐 Token 特征变换） |
| SwiGLU / Gate/Up/Down | 闸门式三台机器：放行、候选、收回 |
| H / I | 隐藏宽度 / 中间宽度（账本的两个关键数字） |
| 张量并行（TP） | 把大机器拆给多个车间，拼结果=AllReduce |

### 想深入？

每节 15 分钟，按需点开，不必按顺序全读：

| 单元 | 讲什么（白话） | 什么时候需要它 |
| --- | --- | --- |
| [05-1 RMSNorm、残差与 Block 结构](#_05-1-rmsnorm、残差与-block-结构) | 流水线的骨架与顺序 | 想画对 Block 结构图 |
| [05-2 MLP 与 SwiGLU 数据流](#_05-2-mlp-与-swiglu-数据流) | 三台机器的尺寸流 | 想跟踪 SwiGLU 的张量 |
| [05-3 参数量、FLOPs 与显存对象](#_05-3-参数量、flops-与显存对象) | 车间账本的精算 | 想自己估参数和显存 |
| [05-4 张量并行怎样切分 Block](#_05-4-张量并行怎样切分-block) | 拆机器方案与通信 | 进入 TP/多卡话题前 |
| [本章总结](#本章总结) | 动手任务 + 理解检查 | 想检验整章掌握程度 |

## 05-1｜RMSNorm、残差与 Block 结构

::: info 本单元目标
围绕 **RMSNorm、残差与 Block 结构** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

![Pre-Norm Decoder Block 数据流](/images/llm/decoder-block-flow.svg)

用公式表示：

$$
h=x+Attention(RMSNorm(x))
$$

$$
y=h+MLP(RMSNorm(h))
$$

输入与输出 shape 都是 `[B,S,H]`，所以可以连续堆叠很多层。

#### 每个模块的分工

| 模块 | 主要作用 | 是否混合 Token 位置 |
| --- | --- | --- |
| RMSNorm | 稳定每个 Token 隐藏向量的尺度 | 否 |
| Self-Attention | 让当前位置读取上下文 | 是 |
| SwiGLU MLP | 对每个 Token 的 hidden 特征做非线性变换 | 否 |
| Residual | 保留主路信息并提供梯度捷径 | 不执行内容混合 |

### 2. 为什么需要归一化

网络堆叠很多层后，隐藏状态的数值尺度可能不断变化。尺度过大或过小都会让训练变得不稳定。归一化让每层接收到的输入处在较可控的范围内。

#### RMSNorm

对一个长度为 H 的 Token 向量：

$$
RMS(x)=\sqrt{\frac{1}{H}\sum_{i=1}^{H}x_i^2+\epsilon}
$$

$$
RMSNorm(x)=\frac{x}{RMS(x)}\odot\gamma
$$

其中 `γ [H]` 是可训练缩放参数。

![RMSNorm 的处理过程](/images/llm/rmsnorm-process.svg)

输入 `[B,S,H]` 时，RMSNorm 对每个 `B、S` 位置独立沿最后一个 H 维计算，不会把不同 Token 混在一起。

#### RMSNorm 与 LayerNorm

- LayerNorm 会减去均值，再除以标准差；
- RMSNorm 不做中心化，主要按均方根缩放；
- 两者具体性能与效果应以模型和实现为准，不能简单断言一个永远更好。

### 3. Pre-Norm 与 Post-Norm

区别是归一化相对于子层的位置。

Pre-Norm：

$$
y=x+F(Norm(x))
$$

Post-Norm：

$$
y=Norm(x+F(x))
$$

本章使用 Pre-Norm。它让残差主路从 x 到 y 保持直接的加法通道，深层模型通常更容易优化。不同架构仍可能使用 Post-Norm、夹心 Norm 或其他变体。

### 4. 残差连接为什么重要

若子层学习的变换是 `F(x)`，残差输出为：

$$
y=x+F(x)
$$

直观上，子层不必重新生成完整表示，只需学习在当前表示上增加什么修正。

反向传播时：

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}
$$

其中恒等项 `I` 提供一条直接的梯度路径。即使 `F` 的梯度较弱，梯度仍能沿残差主路传播。

#### 残差相加要求什么

相加两侧 shape 必须相同。因此 Attention 最后需要输出投影恢复 H，MLP 最后也必须从中间维 I 降回 H：

```text
x:             [B,S,H]
Attention(...):[B,S,H]
MLP(...):      [B,S,H]
```

## 05-2｜MLP 与 SwiGLU 数据流

::: info 本单元目标
围绕 **MLP 与 SwiGLU 数据流** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

Attention 主要让不同 Token 位置交换信息；MLP 则在每个 Token 内部混合和变换 hidden 特征，并通过非线性提高模型表达能力。

普通两层 MLP 可以写成：

$$
MLP(x)=\phi(xW_{up})W_{down}
$$

现代模型常使用门控变体，例如 SwiGLU。

### 6. SwiGLU 的数据流

一种常见形式是：

$$
SwiGLU(x)=\left(SiLU(xW_{gate})\odot(xW_{up})\right)W_{down}
$$

其中：

$$
SiLU(z)=z\cdot sigmoid(z)
$$

![SwiGLU 的双分支门控](/images/llm/swiglu-flow.svg)

Shape：

```text
x:                    [B,S,H]
xW_gate, xW_up:       [B,S,I]
逐元素门控结果:        [B,S,I]
乘 W_down 后:         [B,S,H]
```

`I` 是 intermediate size，通常大于 H。MLP 在每个 Token 上独立使用同一组权重，不直接在 S 维做信息交换。

## 05-3｜参数量、FLOPs 与显存对象

::: info 本单元目标
围绕 **参数量、FLOPs 与显存对象** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

![Decoder Block 参数构成](/images/llm/block-cost-flow.svg)

#### 标准 MHA Attention

忽略 bias，Q、K、V、O 四个 `[H,H]` 投影：

$$
P_{attn}\approx4H^2
$$

#### SwiGLU MLP

Gate 和 Up 各为 `[H,I]`，Down 为 `[I,H]`：

$$
P_{mlp}\approx3HI
$$

#### RMSNorm

每个 RMSNorm 有一个 `[H]` 的 γ，一层两个 Norm：

$$
P_{norm}=2H
$$

因此单层近似参数量：

$$
P_{block}\approx4H^2+3HI+2H
$$

#### 数值示例

若 `H=4096、I=11008`：

```text
Attention: 4 × 4096²       = 67,108,864
SwiGLU:    3 × 4096 × 11008 = 135,266,304
RMSNorm:   2 × 4096          = 8,192
单层合计:                      202,383,360
```

本例 MLP 参数约是 Attention 投影参数的 2.02 倍。

注意：GQA/MQA 会减少 K/V 参数；不同模型的 I、bias 和结构也不同。

### 8. 主要 FLOPs 估算

矩阵乘法 `[M,K]×[K,N]` 约为 `2MKN FLOPs`。对完整序列，忽略 Norm、激活、Softmax 等相对小项：

#### Attention 线性投影

四个 `[H,H]` 投影：

$$
F_{attn\_linear}\approx8BSH^2
$$

#### Attention 核心

`QKᵀ` 与 `AV`：

$$
F_{attn\_core}\approx4BS^2H
$$

#### SwiGLU 三个投影

$$
F_{mlp}\approx6BSHI
$$

单层前向主要 FLOPs：

$$
F_{block}\approx8BSH^2+4BS^2H+6BSHI
$$

这只是近似前向计算。训练还包含反向传播，实际算子、重计算、稀疏性和并行方式都会改变端到端成本。

### 9. 参数、激活和状态不是一回事

估算显存时至少区分：

- **参数**：WQ/WK/WV/WO、MLP 权重、Norm γ；
- **梯度**：与可训练参数对应；
- **优化器状态**：例如 Adam 的一阶、二阶矩；
- **激活**：前向产生、反向可能需要的 X、Q/K/V、MLP 中间值等；
- **临时 Workspace**：Kernel 或通信执行使用的缓冲区。

参数量只由模型结构决定；激活量强烈依赖 `B、S` 和是否使用重计算。不能用“参数大小”直接代表训练总显存。

### 10. 一个最小 Decoder Block 骨架

```python
import torch
from torch import nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x, attn_mask):
        norm_x = self.attn_norm(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=True,
        )
        h = x + attn_out
        y = h + self.mlp(self.mlp_norm(h))
        return y
```

这是教学骨架，不等同于具体 Llama 源码：它未实现 RoPE、GQA、KV Cache、Dropout 策略或专用融合 Kernel。

### 11. 与 AI Infra 的关系

#### MLP 可能是计算大户

MLP 有三个大矩阵乘法，且 I 通常明显大于 H。在许多常见配置下，它的参数和矩阵计算量会超过 Attention 的线性投影。

#### Norm、激活和残差更偏向带宽

RMSNorm、SiLU、逐元素乘法与残差相加的单元素计算较少，常更受内存带宽和 Kernel 启动开销影响，因此适合研究算子融合。

例如可以把 RMSNorm 与后续线性层、SwiGLU 的激活与乘法、残差与其他操作按条件融合，减少中间张量写回 HBM。

#### Activation Checkpointing

训练时不保存部分中间激活，在反向阶段重新计算，可用额外 FLOPs 换显存。这种计算换存储策略会改变 Kernel 执行次数和性能分析结果。

## 05-4｜张量并行怎样切分 Block

::: info 本单元目标
围绕 **张量并行怎样切分 Block** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

#### Attention 投影切分

QKV 常按输出特征列切分，每个 Rank 计算部分 Head；输出投影再按相配方式切分，局部结果可能需要 AllReduce 或 ReduceScatter。

#### MLP 切分

Gate/Up 可以按 intermediate 维切分，每个 Rank 只计算一部分 I；Down 投影使用相反方向切分，局部 `[B,S,H]` 贡献需要聚合。

```text
Gate/Up：各 Rank 获得不同 intermediate 分片
逐元素 SwiGLU：完全本地完成
Down：各 Rank 产生 H 维局部和
AllReduce / ReduceScatter：合成后续需要的结果
```

这正是 Megatron 风格张量并行的重要模式。后续会用矩阵分块严格推导通信量。

## 本章总结

> 返回：[第 5 章首页](#)

::: tip 本章核心结论
1. Attention 在 Token 之间混合信息，SwiGLU FFN 在每个 Token 内部混合特征。
2. RMSNorm 控制数值尺度，残差连接保留原信息并提供更直接的梯度路径。
3. 一个现代 Pre-Norm Block 是两次“Norm → 子层 → Residual”的串联。
:::

### 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

### ⚠️ 易错点

#### MLP 不等于分类器

Block 内 MLP 是逐 Token 的非线性特征变换，不是最终 LM Head。

#### Norm 不会改变 shape

它改变数值分布，但输入输出仍为 `[B,S,H]`。

#### 残差不是简单“防止信息丢失”

它既保留主路表示，也提供更直接的梯度路径，并要求分支输出与主路 shape 一致。

#### 参数最多的模块不一定最慢

运行时间还取决于 Batch、序列长度、算术强度、访存、并行策略和 Kernel 实现。

#### 所有 LLM Block 并不完全相同

模型可能使用 LayerNorm/RMSNorm、GeLU/SwiGLU、MHA/GQA/MQA、不同 Norm 位置和不同 bias 设置。阅读源码必须先看配置。

### 14. 动手任务

#### 必做 1：参数量

给定 `H=2048、I=5632`，使用标准 MHA、SwiGLU、两个 RMSNorm，忽略 bias：

1. Attention 参数量是多少？
2. MLP 参数量是多少？
3. 两个 RMSNorm 参数量是多少？
4. 单层合计是多少？

::: details 必做 1 参考答案

```text
Attention = 4H²
          = 4 × 2048²
          = 16,777,216

MLP = 3HI
    = 3 × 2048 × 5632
    = 34,603,008

RMSNorm = 2H
        = 4,096

单层合计 = 51,384,320 参数
```

:::

#### 必做 2：Shape 跟踪

对 `x=[2,128,2048]、I=5632`，写出以下张量 shape：

- RMSNorm 输出；
- Attention 输出及第一次残差结果；
- Gate/Up 输出；
- 逐元素门控结果；
- Down 输出及 Block 最终输出。

::: details 必做 2 参考答案

```text
RMSNorm 输出:       [2,128,2048]
Attention 输出:     [2,128,2048]
第一次残差结果 h:   [2,128,2048]
Gate 输出:          [2,128,5632]
Up 输出:            [2,128,5632]
逐元素门控结果:     [2,128,5632]
Down 输出:          [2,128,2048]
Block 最终输出:     [2,128,2048]
```

:::

### ✅ 理解检查

1. 一个 Pre-Norm Decoder Block 的执行顺序是什么？
2. RMSNorm 沿哪个维度计算？会不会混合不同 Token？
3. 为什么残差分支输出必须恢复到 H？
4. Attention 与 MLP 在“混合什么”方面有何区别？
5. SwiGLU 为什么有 Gate、Up、Down 三个矩阵？
6. 标准 MHA 的投影参数量为什么约为 `4H²`？
7. SwiGLU 参数量为什么约为 `3HI`？
8. 参数量和训练显存为什么不能画等号？
9. 哪些 Block 操作更可能受内存带宽影响？
10. 张量并行切分 MLP 后，为什么 Down 输出可能需要 AllReduce？

### 16. 参考答案

::: details 1. Pre-Norm Decoder Block 的顺序是什么？

先对输入做 RMSNorm，再执行 Causal Self-Attention，并与原输入残差相加；然后对结果做第二次 RMSNorm，执行 SwiGLU MLP，再与 MLP 前的状态残差相加。

:::

::: details 2. RMSNorm 沿哪个维度计算？

通常沿 `[B,S,H]` 的最后一个 H 维，对每个 batch、每个 Token 独立计算 RMS。它不会在 S 维混合不同 Token。

:::

::: details 3. 为什么残差分支要恢复 H？

残差操作是逐元素相加，两个张量必须具有相同 shape。主路是 `[B,S,H]`，所以 Attention 和 MLP 分支最终也必须输出 `[B,S,H]`。

:::

::: details 4. Attention 与 MLP 分别混合什么？

Attention 通过 S×S 权重在不同 Token 位置间混合信息；MLP 对每个 Token 独立，在 hidden/intermediate 特征维中进行非线性变换。

:::

::: details 5. SwiGLU 为什么有三个矩阵？

Gate 将 H 投影到 I 并经 SiLU 生成门控，Up 将 H 投影到 I 生成候选内容，两者逐元素相乘；Down 再把 I 降回 H，以便接回残差主路。

:::

::: details 6. Attention 参数为什么约为 4H²？

标准 MHA 中 Q、K、V 和输出投影各有一个 `[H,H]` 矩阵。忽略 bias 后合计 `H²+H²+H²+H²=4H²`。

:::

::: details 7. SwiGLU 参数为什么约为 3HI？

Gate 和 Up 各有一个 `[H,I]` 权重，Down 有一个 `[I,H]` 权重，合计 `HI+HI+IH=3HI`。

:::

::: details 8. 为什么参数量不等于训练显存？

训练显存还包括参数梯度、优化器状态、前向激活、反向所需保存值、通信 Bucket、Kernel Workspace 和框架管理开销。激活还随 B、S 和重计算策略变化。

:::

::: details 9. 哪些操作更可能受内存带宽影响？

RMSNorm、残差相加、SiLU 和逐元素门控等操作每读取一个元素只执行少量计算，算术强度通常较低，更容易受 HBM 带宽与 Kernel 启动开销影响。具体结论仍应通过 Profile 验证。

:::

::: details 10. 为什么切分后的 Down 可能需要 AllReduce？

若各 Rank 只持有 intermediate 维的一部分，Down 矩阵乘法会产生对应分片对完整 H 输出的局部贡献。完整结果是这些局部贡献之和，因此需要跨 Rank 归约；也可根据下一层数据布局选择 ReduceScatter。

:::

### 参考资料

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [PyTorch：RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [PyTorch：Tensor Parallel 教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

下一课会解释 RoPE：位置信息如何进入 Q/K，旋转怎样让 Attention 分数感知相对距离，以及长上下文扩展为什么困难。

---

<!-- chapter-navigation -->
