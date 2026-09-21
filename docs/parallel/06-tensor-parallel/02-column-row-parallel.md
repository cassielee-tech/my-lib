# 单元 06-2｜Column 与 Row：两种切法

> 所属章节：[第 6 章｜Tensor Parallel](../06-tensor-parallel.md)

::: info 本单元目标
围绕 **"ColP 与 RowP 的对称性及经典组合"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

切矩阵只有两个方向——沿输出维（列）或沿输入维（行）——两种切法恰好互补。

## 3. Column Parallel：切输出

$W \in \mathbb{R}^{h_{in} \times h_{out}}$ 沿 $h_{out}$ 切成 $[W_1 \mid \cdots \mid W_N]$：

- **输入**：每卡需要**完整** $X$（复制或 AllGather）；
- **计算**：$Y_i = XW_i$，各卡独立；
- **输出**：$Y$ 的**列分片** $[Y_1 \mid \cdots \mid Y_N]$。

## 4. Row Parallel：切输入

$W$ 沿 $h_{in}$ **横**切成 $\begin{bmatrix} W_1 \\ \vdots \\ W_N \end{bmatrix}$，$X$ 对应列切成 $[X_1 \mid \cdots \mid X_N]$：

$$
Y = XW = \sum_{i=1}^{N} X_i W_i
$$

- **输入**：每卡恰好持有 $X_i$ 的**列分片**（天然分片，无需通信！）；
- **计算**：各卡算部分积 $X_iW_i$；
- **输出**：各卡是**部分和**——必须**求和**才得到完整 $Y$ → **AllReduce**（或 ReduceScatter，见 §6）。

### 4.1 经典组合：Megatron MLP = ColP → GeLU → RowP

Transformer 的 MLP（[《模型全景》05-2](../../model/05-decoder-block/02-mlp-swiglu.md)）恰好是这套组合的完美舞台：

```text
              W₁ [h,4h] 列切          GeLU          W₂ [4h,h] 行切
X [B,S,h] ──► 各卡: X·W₁ᵢ ──► GeLU(逐元素,分片上做!) ──► 各卡部分和 ──► AllReduce ──► Z [B,S,h]
(完整输入)    [B,S,4h/N]           [B,S,4h/N]          [B,S,h] 部分      完整输出
```

三步读出设计之美：

1. **ColP 的输出列分片，恰好是 RowP 需要的输入列分片**——中间**零通信**天然衔接；
2. **GeLU 是逐元素操作**——在列分片上各自做，照样正确，**零通信**；
3. 全部通信只有**首尾两次**：入口 X 的获得方式（复制/AG）与出口的部分和求和（AllReduce）。

### 4.2 Attention 的 TP：多头即天然切分

QKV 投影按 **head 维**做 ColP（每卡分到 $H/N$ 个头，回扣 [《模型全景》04-3](../../model/04-self-attention/03-multi-head-shape.md)：头与头独立）→ 每卡的注意力计算**在自己的头上独立完成，零通信** → 输出投影 RowP → **AllReduce**。

于是每个 Transformer 层的账（TP=8）：

| 位置 | 前向 | 反向 |
| --- | --- | --- |
| Attention 出口 | 1 × AllReduce | 1 × AllReduce |
| MLP 出口 | 1 × AllReduce | 1 × AllReduce |

**每层 4 次 AllReduce**——量小，但每一步推理/训练都发生，这就是下一单元的位置账要回答的问题。

---

[继续单元 06-3 →](./03-comm-cost.md)
