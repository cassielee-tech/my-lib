# 单元 07-1｜一次训练 Step 与梯度

> 所属章节：[第 7 章｜训练循环](../07-training-loop.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **一次训练 Step 与梯度** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

前六课解释了一个 Decoder-only Transformer 如何把 Token 变成下一个 Token 的概率。模型刚初始化时，这些概率几乎没有意义。训练的任务，就是反复调整参数，使正确 Token 的概率逐渐增大。

一次最小参数更新包含六步：

1. 从数据集中取出一批 Token；
2. 前向传播得到 Logits；
3. 用标签计算 Loss；
4. 反向传播得到每个参数的梯度；
5. 优化器根据梯度更新参数；
6. 清空旧梯度，开始下一轮。

![一次参数更新中数据、激活、梯度和参数的流向](/images/llm/training-step-lifecycle.svg)

先区分三个容易混淆的计数单位：

- **Micro-batch**：一次前向和反向实际送入设备的数据；
- **Optimizer Step**：优化器真正修改一次参数；
- **Training Step**：有时指 Micro-batch，有时指 Optimizer Step，阅读日志和源码时必须确认定义。

## 2. 从 Loss 到梯度

设当前参数为 $\theta$，一个 Batch 的平均 Loss 为：

$$
L(\theta)=\frac{1}{N}\sum_{j=1}^{N}\ell_j(\theta)
$$

执行 `loss.backward()` 后，每个可训练参数会得到：

$$
g_t=\nabla_{\theta}L(\theta_t)
$$

梯度不是“参数应该变成多少”，而是 Loss 对参数变化的局部敏感程度。最简单的梯度下降为：

$$
\theta_{t+1}=\theta_t-\eta_t g_t
$$

$\eta_t$ 是第 $t$ 步的学习率。负号表示沿着使 Loss 下降的方向移动。

在 PyTorch 中，梯度默认会**累加**到 `parameter.grad`，而不是自动覆盖。因此标准循环必须显式清空梯度：

```python
for input_ids, labels in dataloader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
```

`zero_grad()` 放在循环开头或上一次 `step()` 之后都可以，关键是一次参数更新所需的梯度不能和上一轮意外混在一起。

---

[继续单元 07-2 →](./02-gradient-accumulation-adamw.md)
