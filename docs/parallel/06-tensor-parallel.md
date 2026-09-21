# 第 6 章｜Tensor Parallel

> 本课目标：掌握 Column/Row Parallel 的切分推导与经典组合（Megatron MLP/Attention）；算清 TP 的通信账与位置账，解释"为什么 TP 通信量小却必须住在机内"；理解 TP 通信不可重叠的原因与 TP+SP 的减半魔法。

## 本章导学

::: tip 本课只记住 3 件事
1. **两种切法**：Column Parallel 切输出维度（输出天然分片，输入需复制）；Row Parallel 切输入维度（输入天然分片，输出是部分和求和）——**ColP 接 RowP 的经典组合，中间逐元素层零通信**。
2. **TP 的账本**：每卡三大件 ÷N（显存省），但通信**每层 4 次、发生在关键路径、不可重叠**——量小、频率高、延迟敏感。
3. **TP 组必须住机内**：每层都集合的通信只能贴最快的链路（HCCS 全互联）；TP 度 = 每机卡数，再往上扩靠 DP/PP。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。Shape 推导是主角——建议拿纸跟着画矩阵分块图，每一步都问"这一块在谁手里"。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./06-tensor-parallel/quick.md)：一个"多人共卷一场大考试"的类比讲完整章，再回来按单元深入。

- [ ] 我能独立推导 ColP→GeLU→RowP 的 shape 流转与两次通信的位置
- [ ] 我能说出 TP 与 DP 在通信对象/频率/可重叠性上的三点差异
- [ ] 我能算出给定模型 TP=N 时每卡显存与每步通信量

## 本章单元

- **06-1（约 15 分钟）**：[为什么是 TP：切层不切数据](./06-tensor-parallel/01-why-tp.md)
- **06-2（约 15 分钟）**：[Column 与 Row：两种切法](./06-tensor-parallel/02-column-row-parallel.md)
- **06-3（约 15 分钟）**：[通信账与位置账](./06-tensor-parallel/03-comm-cost.md)
- **06-4（约 15 分钟）**：[TP+SP 与工程要点](./06-tensor-parallel/04-tp-sp-and-practice.md)

- **本章总结**：[练习、自测与串联](./06-tensor-parallel/summary.md)

## 这一课在整条路线中的位置

```text
第 5 章 DDP：切数据 · world group · 步间通信（可重叠）
        ↓
本章 TP：切权重 · tp_group（机内）· 层内通信（不可重叠）
        ↓
第 7 章 PP / 第 8 章 CP / 第 9 章 ZeRO
```

模型视角的 TP 在 [《模型全景》05-4](../model/05-decoder-block/04-tensor-parallel-block.md)（Block 怎样切）已速览过——本章补全 shape 推导、通信账与工程约束；《集合通信》04 章的 AG/RS 原语在这里找到最大的需求方。

---

[开始单元 06-1 →](./06-tensor-parallel/01-why-tp.md)
