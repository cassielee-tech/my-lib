# 单元 05-3｜参数量、FLOPs 与显存对象

> 所属章节：[第 5 章｜完整 Decoder Block](../05-decoder-block.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **参数量、FLOPs 与显存对象** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

![Decoder Block 参数构成](/images/llm/block-cost-flow.svg)

### 标准 MHA Attention

忽略 bias，Q、K、V、O 四个 `[H,H]` 投影：

$$
P_{attn}\approx4H^2
$$

### SwiGLU MLP

Gate 和 Up 各为 `[H,I]`，Down 为 `[I,H]`：

$$
P_{mlp}\approx3HI
$$

### RMSNorm

每个 RMSNorm 有一个 `[H]` 的 γ，一层两个 Norm：

$$
P_{norm}=2H
$$

因此单层近似参数量：

$$
P_{block}\approx4H^2+3HI+2H
$$

### 数值示例

若 `H=4096、I=11008`：

```text
Attention: 4 × 4096²       = 67,108,864
SwiGLU:    3 × 4096 × 11008 = 135,266,304
RMSNorm:   2 × 4096          = 8,192
单层合计:                      202,383,360
```

本例 MLP 参数约是 Attention 投影参数的 2.02 倍。

注意：GQA/MQA 会减少 K/V 参数；不同模型的 I、bias 和结构也不同。

## 8. 主要 FLOPs 估算

矩阵乘法 `[M,K]×[K,N]` 约为 `2MKN FLOPs`。对完整序列，忽略 Norm、激活、Softmax 等相对小项：

### Attention 线性投影

四个 `[H,H]` 投影：

$$
F_{attn\_linear}\approx8BSH^2
$$

### Attention 核心

`QKᵀ` 与 `AV`：

$$
F_{attn\_core}\approx4BS^2H
$$

### SwiGLU 三个投影

$$
F_{mlp}\approx6BSHI
$$

单层前向主要 FLOPs：

$$
F_{block}\approx8BSH^2+4BS^2H+6BSHI
$$

这只是近似前向计算。训练还包含反向传播，实际算子、重计算、稀疏性和并行方式都会改变端到端成本。

## 9. 参数、激活和状态不是一回事

估算显存时至少区分：

- **参数**：WQ/WK/WV/WO、MLP 权重、Norm γ；
- **梯度**：与可训练参数对应；
- **优化器状态**：例如 Adam 的一阶、二阶矩；
- **激活**：前向产生、反向可能需要的 X、Q/K/V、MLP 中间值等；
- **临时 Workspace**：Kernel 或通信执行使用的缓冲区。

参数量只由模型结构决定；激活量强烈依赖 `B、S` 和是否使用重计算。不能用“参数大小”直接代表训练总显存。

## 10. 一个最小 Decoder Block 骨架

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

## 11. 与 AI Infra 的关系

### MLP 可能是计算大户

MLP 有三个大矩阵乘法，且 I 通常明显大于 H。在许多常见配置下，它的参数和矩阵计算量会超过 Attention 的线性投影。

### Norm、激活和残差更偏向带宽

RMSNorm、SiLU、逐元素乘法与残差相加的单元素计算较少，常更受内存带宽和 Kernel 启动开销影响，因此适合研究算子融合。

例如可以把 RMSNorm 与后续线性层、SwiGLU 的激活与乘法、残差与其他操作按条件融合，减少中间张量写回 HBM。

### Activation Checkpointing

训练时不保存部分中间激活，在反向阶段重新计算，可用额外 FLOPs 换显存。这种计算换存储策略会改变 Kernel 执行次数和性能分析结果。

---

[继续单元 05-4 →](./04-tensor-parallel-block.md)
