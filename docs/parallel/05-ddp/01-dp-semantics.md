# 单元 05-1｜DP 语义：为什么必须 AllReduce

> 所属章节：[第 5 章｜数据并行与 DDP](../05-ddp.md)

::: info 本单元目标
围绕 **"梯度取平均的数学必然性"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 1. 数据并行在切什么

与后面所有章节不同，DP **不切模型**：每张卡一份完整副本，切的是**数据**——global batch 均分为 N 份，每卡各喂一份。

```text
rank 0: 模型副本 + batch[0:B/N]     rank 1: 模型副本 + batch[B/N:2B/N]
rank 2: 模型副本 + batch[...]        rank 3: 模型副本 + batch[...]
```

前向各算各的 loss，反向各算各的梯度——到这里为止**零通信**。通信发生在哪？反向结束、更新之前。

## 2. 为什么不做 AllReduce 就会发散

关键事实：**各卡数据不同 → 各卡梯度不同**。如果各自直接更新：

```text
θ⁽⁰⁾ ← θ - lr·g⁽⁰⁾        ← rank 0 只往自己的数据方向走
θ⁽¹⁾ ← θ - lr·g⁽¹⁾        ← rank 1 走另一个方向
……
```

一步之后，N 张卡上出现 **N 个不同的模型**；下一步的数据再进来，差异继续放大——这不是"变慢"，是**训练在数学上已经错了**。

正确语义等价于"大 batch SGD"：

$$
\bar{g} = \frac{1}{N}\sum_{i=0}^{N-1} g^{(i)},\qquad \theta \leftarrow \theta - \text{lr}\cdot\bar{g}
$$

$\bar g$ 的求解动作正是 **AllReduce(avg)**（回扣 [《集合通信》01-2 原语卡片](../../collective/01-collective-semantics-cost/02-primitive-cards.md)）——所以 DP 的通信不是"性能问题"，是**算法本身的一步**。

### 2.1 每步通信账

用 Ring 账本（[《集合通信》02-4](../../collective/02-ring-allreduce/04-ledger-and-variants.md)）给 DDP 每步记账：

- 通信对象：**全量梯度** $S = 2\times$参数量（BF16 字节）；
- 每卡流量：$2S\cdot\frac{N-1}{N}$——**与 batch 大小无关**；
- 扩展性来源：batch 越大 → 反向计算越久 → 同样的通信量被越厚的计算摊薄。

**算例**：7.5B 模型、BF16、8 卡（机间 25 GB/s）：$S = 15$ GB，每卡流量 $2\times15\times\frac78 \approx 26$ GB，通信时间 $\approx 1.05$ s；batch 32×2048 token 的反向约 10 s（@300 TFLOPS）——**通信约占一步的 10%**，重叠后几乎可藏（05-4 展开）。

::: tip 位置回顾
"梯度 AllReduce 发生在反向与 `optimizer.step()` 之间"——这句在 [《训练与推理系统》01-4](../../systems/01-training-loop/04-training-memory-communication.md) 出现过的话，现在有了完整的推导与账本。
:::

---

[继续单元 05-2 →](./02-naive-to-ddp.md)
