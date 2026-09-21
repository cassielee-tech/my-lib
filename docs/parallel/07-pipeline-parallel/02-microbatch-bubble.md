# 单元 07-2｜Micro-batch 与 Bubble

> 所属章节：[第 7 章｜Pipeline Parallel](../07-pipeline-parallel.md)

::: info 本单元目标
围绕 **"Bubble 从哪来、怎么摊"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 3. 朴素流水线：为什么空转

**朴素做法**：一个完整 batch 整体流过 4 个 stage——前向全过一遍，再反向全过一遍：

```text
Stage 0: F ········ ········ ········ B        ← 只有首尾在干活
Stage 1: · F ······ ········ ····· B ·
Stage 2: · · F ···· ······ ···· B · ·
Stage 3: · · · F ·· ···· ·· B · · · ·
时间 ──────────────────────────────────►
         F = 整个 batch 的前向      B = 整个 batch 的反向
```

Stage 3 在等数据到达时全程空转，Stage 0 发出数据后全程空等反向回来——**同一时刻只有一台机器在干活**。

**解法：micro-batch**。把 batch 切成 M 份，像流水线上的零件一样**连续注入**（GPipe 风格：全部前向，再全部反向）：

```text
M=4，f/b = 单个 micro-batch 的前向/反向时长
Stage 0: F1 F2 F3 F4 ·· ·· ·· ·· ·· ·· ·· B4 B3 B2 B1
Stage 1: · F1 F2 F3 F4 ·· ·· ·· ·· ·· B4 B3 B2 B1 ·
Stage 2: · · F1 F2 F3 F4 ·· ·· ·· B4 B3 B2 B1 · ·
Stage 3: · · · F1 F2 F3 F4 B4 B3 B2 B1 · · · · · ·
         └──── fill（填充）────┘└── 稳态 ──┘└─ drain（排空）─┘
```

稳态时段四个 stage 全部满负荷；**空转只剩两头**：开头等第一个 micro-batch 流到最深（fill）、结尾等最后一个反向流回最浅（drain）——这段统称 **Bubble**。

## 4. Bubble 率公式

理想时间（无 Bubble）：$M\times(f+b)$。实际时间：

$$
T = (M + S - 1)\times(f+b),\qquad
\text{Bubble 率} = \frac{(S-1)(f+b)}{T} = \frac{S-1}{M+S-1}
$$

**直觉版**：流水线多了 $S-1$ 个"零件在途"的节拍，零件总数越多（M 越大），这笔固定开销摊得越薄。

| M（S=4） | Bubble 率 |
| --- | --- |
| 4 | 3/7 ≈ **43%** |
| 16 | 3/19 ≈ 16% |
| 64 | 3/67 ≈ **4.5%** |

::: warning 公式里的矛盾
M 越大 Bubble 越小——但 GPipe 要**暂存全部 M 个 micro-batch 的激活**等反向（回扣 [《训练与推理系统》02-1](../../systems/02-training-memory-compute/01-memory-ledger.md) 的激活账）：M=64 的激活内存是 M=4 的 16 倍。**"用大 M 压 Bubble" 撞上 "激活放不下"**——这个矛盾正是 1F1B 的出场理由（下一单元）。
:::

---

[继续单元 07-3 →](./03-1f1b.md)
