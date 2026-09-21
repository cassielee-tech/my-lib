# 第 7 章｜Pipeline Parallel

> 本课目标：理解按深度切分的流水线模型与它极小的 P2P 通信；掌握 micro-batch 流水时间线与 Bubble 率公式；说清 1F1B 为什么不减 Bubble 却是必需品（内存 O(M)→O(S)）；能算 PP 的三本账并把它放进 3D 混合并行。

## 本章导学

::: tip 本课只记住 3 件事
1. **PP = 按层分段接力**：模型沿深度切成 S 个 stage，数据像流水线一样流过——通信只有**相邻 stage 间的 P2P 激活**（三把刀里唯一没有集合通信的），所以**天生跨机友好**。
2. **Bubble 公式**：$\frac{S-1}{M+S-1}$——流水线的启动/排空空转，靠**加大 micro-batch 数 M** 摊薄；但 GPipe 做法下 M 越大、激活内存越爆，矛盾由此产生。
3. **1F1B 解内存不解时间**：交替做前向/反向，把在途激活从 O(M) 压到 O(S)——Bubble 不变，但允许更大的 M，**间接**再压 Bubble。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。时间线图是主角——建议拿方格纸画一遍 GPipe 与 1F1B 的调度，画过就不会忘。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./07-pipeline-parallel/quick.md)：一个"洗车流水线"的类比讲完整章，再回来按单元深入。

- [ ] 我能画出 S=4、M=4 的流水时间线并标出 Bubble 段
- [ ] 我能推导并使用 Bubble 率公式
- [ ] 我能说清 1F1B 稳态下每 stage 在途激活的数量级

## 本章单元

- **07-1（约 15 分钟）**：[为什么是 PP：按深度分段](./07-pipeline-parallel/01-why-pp.md)
- **07-2（约 15 分钟）**：[Micro-batch 与 Bubble](./07-pipeline-parallel/02-microbatch-bubble.md)
- **07-3（约 15 分钟）**：[1F1B：解内存，不解时间](./07-pipeline-parallel/03-1f1b.md)
- **07-4（约 15 分钟）**：[PP 的三本账与 3D 组合](./07-pipeline-parallel/04-pp-engineering.md)

- **本章总结**：[练习、自测与串联](./07-pipeline-parallel/summary.md)

## 这一课在整条路线中的位置

```text
第 5 章 DDP：切数据 · 大通信 · 可重叠
第 6 章 TP：切层内 · 小通信 · 高频不可重叠
        ↓
本章 PP：切深度 · 极小 P2P 通信 · 新代价是 Bubble
        ↓
第 8 章 CP / 第 9 章 ZeRO 与混合并行收口
```

PP 的通信原语是 **Send/Recv（P2P）而非集合通信**——[《集合通信》01 章](../collective/01-collective-semantics-cost/02-primitive-cards.md)的原语家族在 HCCL 中同样包含这对兄弟（[H01-4](../ascend/hccl-source/04-primitives-and-sync.md)）；而激活内存的问题在 [《训练与推理系统》02-2](../systems/02-training-memory-compute/02-activation-checkpointing.md) 已有铺垫。

---

[开始单元 07-1 →](./07-pipeline-parallel/01-why-pp.md)
