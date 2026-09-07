# 单元 02-3｜链式法则与反向传播

> 所属章节：[第 2 章｜张量、计算图与反向传播](../02-tensor-autograd.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **链式法则与反向传播** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

如果一个变量经过多步运算影响 Loss，它对 Loss 的梯度等于沿路径上各个局部导数的乘积。

![反向传播与链式法则](/images/llm/backprop-chain.svg)

对线性模型：

$$
\frac{\partial L}{\partial \hat{y}}=2(\hat{y}-y)
$$

$$
\frac{\partial \hat{y}}{\partial w}=x,\qquad
\frac{\partial \hat{y}}{\partial b}=1
$$

因此：

$$
\frac{\partial L}{\partial w}=2(\hat{y}-y)x
$$

$$
\frac{\partial L}{\partial b}=2(\hat{y}-y)
$$

重要的不是背公式，而是理解反向传播的执行模式：

1. 从标量 Loss 出发，上游梯度初始化为 1；
2. 每个节点根据自己的局部导数，把梯度传给输入；
3. 同一个变量通过多条路径影响 Loss 时，各路径梯度相加；
4. 最终得到每个可训练参数的梯度。

### 导数和梯度的直观含义

先不考虑严格数学定义。把参数 `w` 稍微增加一点，如果 Loss 随之：

- 增大：梯度为正；为了减小 Loss，应让 `w` 变小；
- 减小：梯度为负；为了减小 Loss，应让 `w` 变大；
- 基本不变：梯度接近 0，说明当前 Loss 对 `w` 不敏感。

优化器最基础的更新形式是梯度下降：

$$
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
$$

其中 `η` 是学习率。减号表示沿着梯度的反方向移动，从而尝试降低 Loss。

### 为什么需要链式法则

参数通常不会直接产生 Loss，而是经过多步运算：

```text
w → z → ŷ → L
```

要知道 `w` 对 `L` 的影响，需要把每一小段的影响连乘起来。这与后端排查“上游输入变化如何经过多层服务影响最终指标”类似，只是这里每一层传递的是精确的局部导数。

## 6. 用 PyTorch 验证一次反向传播

```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(10.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y_hat = w * x + b
loss = (y_hat - y) ** 2
loss.backward()

print(y_hat.item())  # 7.0
print(loss.item())   # 9.0
print(w.grad.item()) # -18.0
print(b.grad.item()) # -6.0
```

手工核对：

```text
ŷ = 2 × 3 + 1 = 7
L = (7 - 10)² = 9
∂L/∂w = 2 × (7 - 10) × 3 = -18
∂L/∂b = 2 × (7 - 10) = -6
```

### `requires_grad=True` 做了什么

它告诉 Autograd：后续需要计算这个张量相对于 Loss 的梯度。反向传播完成后，叶子参数的梯度累积在 `.grad` 中。

注意梯度默认是**累积**的。标准训练循环每轮都要在反向传播前调用：

```python
optimizer.zero_grad()
```

否则新梯度会继续加到旧梯度上。梯度累积训练正是有意利用了这一行为。

---

[继续单元 02-4 →](./04-gradient-communication.md)
