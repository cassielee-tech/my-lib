# 第 2 章总结｜张量、计算图与反向传播

> 返回：[第 2 章首页](../02-tensor-autograd.md)

::: tip 本章核心结论
1. 张量是多维数据，Shape 描述每个维度的含义和大小。
2. 矩阵乘法能否执行，先检查左矩阵最后一维是否等于右矩阵倒数第二维。
3. 反向传播沿计算图反向应用链式法则，同时产生输入梯度和参数梯度。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

### Tensor 和 parameter 不是一回事

参数是需要被优化器更新的张量；输入和激活值也是张量，但通常不是模型参数。

### 前向 shape 和梯度 shape 不同路径、同形返回

前向中 `X → Y`，反向中梯度从 `dY → dX`。方向相反，但 `dX.shape == X.shape`。

### `backward()` 不是更新参数

`loss.backward()` 只计算并累积梯度；`optimizer.step()` 才使用梯度更新参数。

### 多卡不一定都用 AllReduce

是否使用 AllReduce 取决于张量怎样切分。数据并行常用 AllReduce；FSDP、张量并行和专家并行会产生其他 Collective。

## 10. 动手任务

### 必做 1：手工 shape 推导

给定：

```text
X: [2, 4, 8]
W: [8, 16]
b: [16]
Y = XW + b
```

回答：

1. `Y` 的 shape 是什么？
2. `Y` 一共有多少个元素？FP16 下占多少 Byte？
3. 这次矩阵乘法大约需要多少 FLOPs？
4. `dX`、`dW`、`db` 的 shape 分别是什么？

::: details 必做 1 参考答案与推导

已知：

```text
X: [2, 4, 8]
W: [8, 16]
b: [16]
```

前两个维度表示一共有 `2×4=8` 个 Token，每个 Token 是长度为 8 的向量。每个 Token 与 `[8,16]` 的权重相乘，输出长度变成 16：

```text
Y: [2, 4, 16]
```

元素数量：

```text
2 × 4 × 16 = 128 个元素
```

FP16 每个元素占 2 Byte：

```text
128 × 2 = 256 Byte
```

计算 FLOPs 时把前两维合并成 `M=2×4=8`，并令 `K=8、N=16`：

```text
FLOPs ≈ 2MKN
      = 2 × 8 × 8 × 16
      = 2048 FLOPs
```

这里仅计算矩阵乘法，不包含偏置加法。梯度必须与对应变量 shape 相同：

```text
dX: [2, 4, 8]
dW: [8, 16]
db: [16]
```

`db` 会把 batch 和 sequence 两个维度上的梯度进行求和，因为同一个偏置 `b` 被所有 Token 共享。

:::

### 必做 2：修改最小代码

修改示例中的 `x、y、w、b`，在运行前先手算 `y_hat、loss、w.grad、b.grad`，再用 PyTorch 验证。

### 选做：观察梯度累积

连续调用两次 `loss.backward()`。第一次加上 `retain_graph=True`，观察 `.grad` 如何变化，并解释原因。

## ✅ 理解检查

1. `[M, K] × [K, N]` 为什么得到 `[M, N]`？
2. FP16 张量 `[2, 4, 8]` 占多少 Byte？
3. 计算图在前向阶段保存什么，反向阶段使用它做什么？
4. 为什么 `dW.shape` 一定与 `W.shape` 相同？
5. `backward()` 与 `optimizer.step()` 分别做什么？
6. 两个数据并行 Rank 为什么不能直接用各自的本地梯度更新？
7. DDP 为什么有机会让梯度通信与反向计算重叠？

## 参考答案

::: details 1. [M,K] × [K,N] 为什么得到 [M,N]？

结果矩阵需要为左矩阵的每一行、右矩阵的每一列计算一个点积。左矩阵有 `M` 行，右矩阵有 `N` 列，所以结果有 `M` 行和 `N` 列。

两个参与点积的向量长度都必须是 `K`，这个维度在点积求和后被消去。

:::

::: details 2. FP16 张量 [2,4,8] 占多少 Byte？

元素数量是 `2×4×8=64`。FP16 每个元素占 2 Byte，所以张量数据占：

```text
64 × 2 = 128 Byte
```

这是张量数据的理论大小，不包含框架对象、内存对齐和 Allocator 额外开销。

:::

::: details 3. 计算图在前向阶段保存什么，反向阶段使用它做什么？

前向阶段记录张量经过了哪些运算、运算之间的依赖关系，以及反向计算局部梯度所需的中间值。

反向阶段从 Loss 出发，按照计算图的反方向依次调用各运算的反向规则，用链式法则把梯度传回输入和参数。

:::

::: details 4. 为什么 dW.shape 一定与 W.shape 相同？

`dW` 的每个元素描述 `W` 中对应参数对 Loss 的影响。`W` 有多少个可调参数，就必须有多少个对应的梯度，因此两者 shape 相同。

矩阵公式也能验证：`Xᵀ [K,M] × dY [M,N] → dW [K,N]`，与 `W [K,N]` 一致。

:::

::: details 5. backward() 与 optimizer.step() 分别做什么？

- `loss.backward()`：沿计算图计算梯度，并把结果累积到叶子参数的 `.grad`；
- `optimizer.step()`：读取这些梯度，根据 SGD、Adam 等优化规则真正修改参数；
- `optimizer.zero_grad()`：清空旧梯度，避免无意累积到下一轮。

:::

::: details 6. 两个数据并行 Rank 为什么不能直接用各自的本地梯度更新？

因为两个 Rank 处理不同的数据，本地梯度通常不同。如果各自更新，两个模型副本会产生不同参数，下一轮便不再是同一个模型。

DDP 用 AllReduce 聚合所有 Rank 的梯度，使每个 Rank 得到同一个全局梯度，再执行相同的优化器更新。

:::

::: details 7. DDP 为什么能让梯度通信与反向计算重叠？

反向传播按层从后向前生成梯度，不必等所有层梯度都生成后再一次性通信。DDP 可以把若干梯度装入 Bucket：

```text
后面层的 Bucket 就绪 → 启动 AllReduce
与此同时 → Autograd 继续计算前面层的梯度
```

如果通信由独立 Stream 异步执行，便能与剩余反向计算重叠。重叠效果取决于 Bucket 大小、层的计算时间、通信耗时以及各 Rank 到达时间是否一致。

:::

## 参考资料

- [PyTorch：Tensor 基础教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch：Autograd 基础教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch：Autograd 机制](https://docs.pytorch.org/docs/stable/notes/autograd.html)

下一课将进入 Embedding、Tokenization 与语言模型训练目标，回答“文本如何变成张量，模型又如何学会预测下一个 Token”。

---

<!-- chapter-navigation -->
[进入第 3 章 →](../03-tokenization-embedding.md)
