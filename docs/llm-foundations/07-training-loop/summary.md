# 第 7 章总结｜训练循环

> 返回：[第 7 章首页](../07-training-loop.md)

::: tip 本章核心结论
1. 一个训练 Step 是数据、前向、Loss、反向、梯度同步和参数更新的闭环。
2. 梯度累积扩大有效 Batch，混合精度减少资源，AdamW 根据梯度和历史统计更新参数。
3. 数据并行中，同一参数的梯度必须在更新前聚合。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

1. **一个 `backward()` 就会更新参数吗？** 不会，它只计算并累积梯度；`optimizer.step()` 才修改参数。
2. **梯度累积能减少计算量吗？** 不能，它主要用更多时间换取较低的峰值激活显存。
3. **Loss 为什么除以累积步数？** 为了让累加结果对应有效大 Batch 的平均梯度。
4. **AdamW 的 Weight Decay 等于把 L2 项塞进 Loss 吗？** 对自适应优化器而言不等价；AdamW 将衰减与梯度更新解耦。
5. **混合精度就是整个模型都用 FP16 吗？** 不是，不同算子和状态可使用不同精度。
6. **裁剪应该在 GradScaler 反缩放之前吗？** 不应该，否则裁剪的是被放大的梯度。
7. **Scheduler 每个 Micro-step 都执行吗？** 通常按 Optimizer Step 执行。

## 12. 动手练习

### 练习 1：观察梯度累加

连续执行两次 `loss.backward()`，中间不调用 `zero_grad()`，打印同一参数的 `.grad.norm()`；再清空梯度重复实验，观察差异。

### 练习 2：计算有效 Batch

假设每个设备 Micro-batch 为 2，梯度累积 8 步，数据并行 16 个 Rank：

$$
B_{global}=2\times8\times16=256
$$

如果每条样本序列固定为 4096 Token，则每次参数更新处理：

$$
256\times4096=1,048,576\ Tokens
$$

### 练习 3：记录一次训练时间线

为每个 Micro-step 打印 `loss`、梯度范数、学习率和是否执行 `optimizer.step()`。确认 Scheduler 只在参数更新时前进。

## 13. 自测题

1. 一次最小参数更新包含哪些步骤？
2. `backward()` 和 `optimizer.step()` 分别做什么？
3. 为什么每轮需要清空梯度？
4. 有效 Batch Size 如何计算？
5. 梯度累积主要节省哪部分显存？
6. AdamW 为每个参数维护哪两类主要状态？
7. Warmup 和 Decay 分别解决什么问题？
8. FP16 为什么常需要 Loss Scaling？
9. 使用 GradScaler 时，梯度裁剪前必须做什么？
10. 数据并行为什么需要 AllReduce？

::: details 自测答案

1. 取数据、前向、计算 Loss、反向得到梯度、优化器更新参数、清空梯度。
2. `backward()` 沿计算图计算并累积梯度；`optimizer.step()` 根据梯度和优化器状态修改参数。
3. PyTorch 默认把新梯度加到 `.grad`，不清空会意外混入上一轮梯度。
4. $B_{global}=B_{micro}\times K_{acc}\times N_{data\ parallel}$。
5. 它允许减小 Micro-batch，主要降低峰值激活显存；参数和优化器状态不会因此消失。
6. 梯度的一阶矩 $m$ 和平方梯度的二阶矩 $v$。
7. Warmup 避免训练初期更新过猛；Decay 让后期更新逐渐变小、更加细致。
8. FP16 的表示范围有限，小梯度可能 Underflow；先放大 Loss 能让反向梯度落入可表示范围。
9. 先调用 `unscale_` 恢复真实梯度，再计算范数并裁剪。
10. 各 Rank 使用不同数据得到不同本地梯度，AllReduce 将它们聚合，使各副本用一致梯度更新出一致参数。

:::

## 14. 本章小结

- 训练循环的核心闭环是前向、Loss、反向和参数更新；
- 梯度累积通过多个 Micro-batch 合成一次 Optimizer Step；
- AdamW 用一阶矩和二阶矩调节更新，并将 Weight Decay 解耦；
- Warmup 与 Decay 控制训练不同阶段的更新幅度；
- 混合精度降低存储与搬运成本，FP16 常配合动态 Loss Scaling；
- 正确顺序是反向、梯度同步、反缩放、裁剪、更新、清梯度；
- 数据并行训练中，梯度 AllReduce 位于反向传播与参数更新之间，并可与反向计算重叠。

## 参考资料

- [PyTorch：Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch：Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch：AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch：clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

下一课将进入自回归推理，拆解 Prefill、Decode、KV Cache 和采样如何协作生成一个 Token。

---

<!-- chapter-navigation -->
[进入第 8 章 →](../08-inference-kv-cache.md)
