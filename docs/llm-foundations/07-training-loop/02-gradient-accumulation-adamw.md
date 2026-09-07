# 单元 07-2｜梯度累积与 AdamW 状态

> 所属章节：[第 7 章｜训练循环](../07-training-loop.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **梯度累积与 AdamW 状态** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

设备显存可能只容纳较小的 Micro-batch，但训练又希望使用较大的有效 Batch。梯度累积会连续执行 $K$ 次前向和反向，先把梯度累加起来，最后只更新一次参数。

$$
B_{global}=B_{micro}\times K_{acc}\times N_{data\ parallel}
$$

其中：

- $B_{micro}$：每个设备每次前向的样本数；
- $K_{acc}$：梯度累积步数；
- $N_{data\ parallel}$：数据并行副本数。

![四个 Micro-batch 如何合成一次参数更新](/images/llm/gradient-accumulation.svg)

若每个 Micro-batch 的 Loss 都已经取平均，应将 Loss 除以累积步数，使最终梯度等价于大 Batch 的平均梯度：

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)

for micro_step, (input_ids, labels) in enumerate(dataloader, start=1):
    logits = model(input_ids)
    loss = loss_fn(logits, labels) / accum_steps
    loss.backward()

    if micro_step % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

梯度累积降低的是一次前向/反向所需的激活显存，并不会减少完成同样 Token 数所需的总计算量。累积步数过大还会减少单位时间内的参数更新次数。

## 4. AdamW 在保存什么

大模型训练通常不直接使用普通 SGD，而会使用 AdamW。它为每个参数维护两份状态：

- 一阶矩 $m_t$：梯度的指数移动平均，可理解为带平滑的方向；
- 二阶矩 $v_t$：梯度平方的指数移动平均，用来估计各参数方向的尺度。

忽略偏置修正后的核心形式为：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

其中 $\lambda$ 是 Weight Decay。AdamW 把参数衰减与 Loss 梯度更新分开，因此名字中的 `W` 指的是 decoupled weight decay。

![AdamW 如何利用梯度、动量和尺度更新参数](/images/llm/adamw-state-update.svg)

AdamW 的代价也很直观：除参数和梯度外，还要保存 $m$、$v$ 等优化器状态。若状态使用 FP32，仅两份矩状态就约为每参数 8 Bytes。后续学习 ZeRO/FSDP 时，会看到这些状态为什么值得跨设备分片。

实践中通常不会对所有参数做 Weight Decay。Bias 和归一化层的缩放参数经常放入 `weight_decay=0` 的参数组，具体规则以模型训练方案为准。

---

[继续单元 07-3 →](./03-lr-mixed-precision-clipping.md)
