# 单元 04-2｜ReduceScatter：规约后分着拿

> 所属章节：[第 4 章｜AllGather、ReduceScatter 与 AlltoAll](../04-gather-scatter-alltoall.md)

::: info 本单元目标
围绕 **ReduceScatter 的布局变化与使用场景** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 3. 语义：全员求和，结果各持一段

```text
开始：rank 0 有 [A₀|B₀|C₀|D₀]   rank 1 有 [A₁|B₁|C₁|D₁]   … （每人都有全长的输入）
结束：rank 0 有 [A*]   rank 1 有 [B*]   rank 2 有 [C*]   rank 3 有 [D*]
      （A* = A₀+A₁+A₂+A₃，即"逐块求和后按块分发"）
```

- 输入每人 **S**（N 份待规约），输出每人 **S/N**——**Reduce 的"人人有份"换成了"各拿一段"**；
- 这是第 1 章 ReduceScatter 卡片的落地，也是 Ring AllReduce 的**上半场**（RS 阶段原封不动）。

### 3.1 算法与账本

Ring RS：N−1 轮"累积块右移、边传边加"（第 2 章手推过）。账本与 AG 对称：

$$
T_{RS} \approx (N-1)\alpha + \frac{S(N-1)}{N}\cdot\frac{1}{BW}
$$

::: warning 一个容易混的记号
公式里的 S 指**输入全长**。RS 和 AG 各自传 S(N−1)/N 字节——**AllReduce = 两者相加 = 2S(N−1)/N**，与第 2 章账本完全一致。恒等式在账本层面也成立。
:::

## 4. 谁在用 ReduceScatter

| 场景 | 用在哪 | 为什么 |
| --- | --- | --- |
| **TP 反向** | 梯度规约 | 各 rank 算出的梯度要先求和；求完和再分片，每人正好持有自己负责的参数段梯度 |
| **SP 反向** | 与 TP 前向 AG 对偶 | 前向拼装多少，反向散回多少 |
| **FSDP** | backward 梯度规约 | 各 rank 累完本地梯度，RS 后每人只规约自己参数段的梯度，随后更新 |

### 4.1 与 FSDP 的省钱逻辑

FSDP 反向若用 **AllReduce** 规约梯度：每人拿全长梯度（通信 2S(N−1)/N）。用 **RS**：只拿自己那段（S(N−1)/N，**省一半**）——因为参数本来就是分片存的，别人的梯度段对我没用。**"语义上够用的最小原语"是通信优化的第一直觉**（呼应 [《训练与推理系统》第 1 章：训练中的通信](../../systems/01-training-loop/04-training-memory-communication.md)）。

---

[继续单元 04-3 →](03-alltoall.md)
