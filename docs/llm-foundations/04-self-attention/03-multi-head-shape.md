# 单元 04-3｜多头注意力与 Shape

> 所属章节：[第 4 章｜Self-Attention](../04-self-attention.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **多头注意力与 Shape** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

单头 Attention 只在一组投影空间中计算关系。多头注意力把隐藏维度切成多个 Head，让每个 Head 使用独立投影参数计算注意力，然后拼接结果。

![多头注意力的切分与合并](/images/llm/multi-head-attention.svg)

设隐藏维 `H=512`、头数 `N=4`，通常每头维度：

$$
D=H/N=128
$$

多头 Shape 常写为：

```text
投影前: [B,S,H]
拆头后: [B,N,S,D]
每头分数: [B,N,S,S]
每头输出: [B,N,S,D]
合并后: [B,S,H]
```

最终还要乘输出投影 `WO [H,H]`，让不同 Head 的信息重新混合。

### 多头不是把完整 H 维计算复制 N 次

经典实现中，总隐藏维 H 被切成 N 个大小为 D 的 Head，且 `N×D=H`。增加 Head 数量通常会减小单头维度，而不是按头数成倍增加总表示宽度。

## 7. 一个完整的 Shape 例子

给定：

```text
B = 2
S = 128
H = 512
N = 8
D = H/N = 64
```

| 张量 | Shape | 含义 |
| --- | --- | --- |
| X | `[2,128,512]` | 输入隐藏状态 |
| Q/K/V 投影后 | `[2,128,512]` | 尚未显式拆头 |
| Q/K/V 拆头后 | `[2,8,128,64]` | 8 个独立 Head |
| Attention scores | `[2,8,128,128]` | 每头的所有位置关系 |
| Attention weights | `[2,8,128,128]` | Mask、Softmax 后权重 |
| 每头输出 | `[2,8,128,64]` | 聚合后的 Value |
| 合并输出 | `[2,128,512]` | 恢复隐藏维 |

## 8. 用 PyTorch 手写单头 Attention

```python
import math
import torch
from torch import nn

torch.manual_seed(0)

B, S, H, D = 2, 4, 8, 8
x = torch.randn(B, S, H)

wq = nn.Linear(H, D, bias=False)
wk = nn.Linear(H, D, bias=False)
wv = nn.Linear(H, D, bias=False)

q = wq(x)                              # [B,S,D]
k = wk(x)                              # [B,S,D]
v = wv(x)                              # [B,S,D]

scores = q @ k.transpose(-2, -1)       # [B,S,S]
scores = scores / math.sqrt(D)

causal_mask = torch.triu(
    torch.ones(S, S, dtype=torch.bool),
    diagonal=1,
)
scores = scores.masked_fill(causal_mask, float('-inf'))

weights = torch.softmax(scores, dim=-1) # [B,S,S]
output = weights @ v                    # [B,S,D]

print(q.shape)
print(scores.shape)
print(weights[0])
print(weights[0].sum(dim=-1))           # 每行都是 1
print(output.shape)
```

生产代码应优先使用框架提供的 `scaled_dot_product_attention` 等优化实现，而不是物化所有中间张量的朴素版本。

---

[继续单元 04-4 →](./04-attention-performance.md)
