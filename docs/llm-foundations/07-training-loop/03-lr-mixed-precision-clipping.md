# 单元 07-3｜学习率、混合精度与梯度裁剪

> 所属章节：[第 7 章｜训练循环](../07-training-loop.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **学习率、混合精度与梯度裁剪** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

固定学习率很少贯穿整个大模型预训练过程。常见策略分为两个阶段：

### 5.1 Warmup

训练开始时，参数和 AdamW 的矩估计都不稳定。Warmup 会让学习率从较小值逐步升至峰值，避免一开始更新过猛。

线性 Warmup 可写为：

$$
\eta_t=\eta_{max}\frac{t}{T_{warmup}},\qquad t\le T_{warmup}
$$

### 5.2 Decay

Warmup 后逐渐降低学习率，让训练后期的更新更细致。常见方案有线性衰减和余弦衰减。余弦衰减示意为：

$$
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi(t-T_{warmup})}{T-T_{warmup}}\right)
$$

![Warmup 与余弦衰减控制不同训练阶段的更新幅度](/images/llm/learning-rate-schedule.svg)

Scheduler 应按 **Optimizer Step** 计数，而不是按 Micro-batch 计数，否则使用梯度累积后学习率曲线会被错误加速。

## 6. 混合精度解决什么问题

训练全部使用 FP32 通常会占用更多显存和内存带宽。混合精度的基本思想是：

- 适合低精度的矩阵乘法等算子使用 FP16 或 BF16；
- 对精度敏感的归一化、归约或部分状态保留 FP32；
- 由 `autocast` 根据算子选择合适精度，而不是简单把整个模型强制转换为同一种类型。

低精度通常能减少张量存储和搬运，并使用加速器的低精度计算单元。但低精度表示范围有限：

- **Underflow**：很小的梯度被舍入为 0；
- **Overflow**：数值太大，变成 `inf` 或 `nan`。

FP16 训练常使用 Loss Scaling：先把 Loss 乘以较大的缩放因子 $S$，反向得到放大的梯度；更新前再除以 $S$。动态 GradScaler 在检测到 `inf/nan` 时会跳过更新并减小缩放因子。BF16 的指数范围与 FP32 接近，通常不需要 GradScaler，但仍要监控数值稳定性。

## 7. 梯度裁剪为什么放在更新前

某一步的梯度可能突然变得很大，导致参数更新剧烈甚至训练发散。全局范数裁剪先计算所有梯度组成的整体范数：

$$
\lVert g\rVert_2=\sqrt{\sum_i\lVert g_i\rVert_2^2}
$$

若它超过阈值 $c$，则统一缩放：

$$
g_i\leftarrow g_i\cdot\frac{c}{\lVert g\rVert_2}
$$

裁剪不会分别把每个元素截断到 $[-c,c]$；它保持整体方向，只缩短梯度向量。使用 Loss Scaling 时，必须先 `unscale_`，再按真实梯度范数裁剪。

## 8. 一份顺序正确的训练骨架

下面用 FP16 展示关键顺序。实际 NPU/CUDA 设备类型、支持精度和 AMP 接口应以运行环境为准。

```python
import torch

accum_steps = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")
optimizer.zero_grad(set_to_none=True)

for micro_step, (input_ids, labels) in enumerate(dataloader, start=1):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids)
        loss = loss_fn(logits, labels) / accum_steps

    scaler.scale(loss).backward()

    if micro_step % accum_steps == 0:
        scaler.unscale_(optimizer)                 # 先还原真实梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)                     # inf/nan 时可能跳过更新
        scaler.update()
        scheduler.step()                           # 按 Optimizer Step 前进
        optimizer.zero_grad(set_to_none=True)
```

真实训练还要处理最后不足 `accum_steps` 的 Batch、Checkpoint 恢复、日志、评估以及分布式同步。不要只复制代码，要先确认每个计数器表示 Micro-step 还是 Optimizer Step。

---

[继续单元 07-4 →](./04-training-memory-communication.md)
