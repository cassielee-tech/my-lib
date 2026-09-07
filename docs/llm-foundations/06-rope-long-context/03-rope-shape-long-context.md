# 单元 06-3｜RoPE 的 Shape 与长上下文限制

> 所属章节：[第 6 章｜RoPE 与长上下文](../06-rope-long-context.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **RoPE 的 Shape 与长上下文限制** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

假设：

- Batch Size：$B=2$
- 注意力头数：$N=4$
- 序列长度：$S=128$
- 每头维度：$D=64$

则 $Q$、$K$ 的 Shape 都是：

$$
[B,N,S,D]=[2,4,128,64]
$$

每头的 64 维被组成 32 个二维对。预计算的角度表可以看成 `[S, D/2] = [128, 32]`，再沿 Batch 和 Head 维广播。旋转前后 Shape 完全不变。

最小实现如下：

```python
import torch

def apply_rope(x, cos, sin):
    # x: [B, N, S, D]；cos/sin: [1, 1, S, D/2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

B, N, S, D = 2, 4, 128, 64
q = torch.randn(B, N, S, D)

base = 10_000
inv_freq = 1.0 / (base ** (torch.arange(0, D, 2) / D))
angles = torch.outer(torch.arange(S), inv_freq)
cos = angles.cos()[None, None, :, :]
sin = angles.sin()[None, None, :, :]
q_rotated = apply_rope(q, cos, sin)

assert q_rotated.shape == q.shape
```

这里的代码用于理解数学过程。生产实现通常会缓存角度表，并把 RoPE 与相邻算子融合，以减少中间张量和内存访问。

## 7. 长上下文为什么仍然困难

假设模型只在最大长度 $L_{train}$ 内训练。直接输入远超该长度的位置时，会出现几个问题：

1. 模型遇到训练阶段没有充分学习过的旋转相位组合；
2. 高频维度已经旋转许多圈，远距离位置可能出现难以区分的周期现象；
3. 即使位置编码能够外推，标准 Attention 的计算量仍随 $S^2$ 增长；
4. 推理时 KV Cache 通常随序列长度 $S$ 线性增长。

因此“支持更大的 position id”不等于“模型已经可靠、高效地支持长上下文”。

常见扩展方法包括线性位置缩放、Dynamic NTK、YaRN、LongRoPE 和分段频率缩放等。它们大体是在压缩位置或调整各频率的变化速度，但效果与模型训练方式、原始上下文长度和缩放参数强相关。使用已有模型时，应采用该模型配置声明的 RoPE 类型，不要随意替换。

## 8. 与 AI Infra 和集合通信的关系

### 8.1 算子与访存

RoPE 主要是逐元素乘加，计算量不大，但需要读取 $Q/K$ 和正余弦表、再写回结果。它可能更受内存带宽和 Kernel 启动开销影响，因此常被融合进 Q/K 投影后的处理或 Attention Kernel。

### 8.2 KV Cache

自回归推理中，常见实现会在写入 Cache 前把 $K$ 按它的绝对位置旋转。之后 Decode 只需旋转当前 Token 的 $Q$，就能与缓存中的 $K'$ 计算注意力。具体缓存布局仍要以框架实现为准。

### 8.3 并行执行

RoPE 本身通常不触发集合通信，但分布式实现仍有正确性约束：

- Tensor Parallel 切分 Head 时，一个二维旋转对的两个元素必须留在一起；
- Sequence/Context Parallel 下，各 Rank 必须使用全局位置编号，而不是都从 0 重新开始；
- 不同 Rank 若使用了不同的 `base`、缩放规则或 position offset，结果会静默出错。

这类问题很适合用集合通信算子开发中的思路检查：不仅看张量 Shape，还要追踪每个分片对应的全局语义。

---

[进入本章总结 →](./summary.md)
