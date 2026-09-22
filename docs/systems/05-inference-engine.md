# 第 5 章｜推理引擎与 KV Cache 管理

> 本课目标：理解从训练循环到推理服务的三重转变；掌握 Continuous Batching 的迭代级调度为什么带来量级吞吐提升；算清 KV Cache 的显存账并用 Paged Attention 与 Prefix Cache 两件套管理它；说出多卡推理的通信形态。

## 本章导学

::: tip 本课只记住 3 件事
1. **推理不是"没有反向的训练"**：自回归生成使请求耗时不可预知、batch 维度随时变化——按训练思路（静态 batch、一次前向）做服务必然低效，**推理引擎的全部设计都在对抗"不可预知"**。
2. **Continuous Batching = 迭代级调度**：每生成一步就重组 batch——完成的请求立刻退出、排队的新请求立刻补位，GPU 永远满负荷。静态批"整批等最慢"的浪费就此消灭。
3. **KV Cache 两件套**：**Paged Attention** 用分页消灭预留浪费（碎片 60-80% → <10%），**Prefix Cache** 让共享前缀只算一次——显存账本的两把省刀。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。本章是[第 3 章（推理与 KV Cache·选修）](./03-inference-kv-cache.md)的工程续篇——第 3 章讲"一个请求怎样生成"，本章讲"一千个请求怎样服务"。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./05-inference-engine/quick.md)：一个"奶茶店高峰期"的类比讲完整章，再回来按单元深入。

- [ ] 我能说出推理相对训练的三重转变及其后果
- [ ] 我能解释静态批的浪费来源与连续批的重组时机
- [ ] 我能算 KV Cache 账并解释分页与前缀复用各省什么

## 本章单元

- **05-1（约 15 分钟）**：[从训练循环到推理服务](./05-inference-engine/01-from-training-to-serving.md)
- **05-2（约 15 分钟）**：[Continuous Batching：迭代级调度](./05-inference-engine/02-continuous-batching.md)
- **05-3（约 15 分钟）**：[Paged Attention：KV Cache 分页](./05-inference-engine/03-paged-attention.md)
- **05-4（约 15 分钟）**：[Prefix Cache 与多卡推理](./05-inference-engine/04-prefix-cache-multicard.md)

- **本章总结**：[练习、自测与串联](./05-inference-engine/summary.md)

## 这一课在整条路线中的位置

```text
第 3 章（选修）：单请求视角——Prefill/Decode 与 KV Cache 账本
        ↓
本章：多请求视角——调度、显存管理与并发
        ↓
第 6 章：调度与指标——TTFT/TPOT 的制约与 PD 分离
```

对 HCCL 岗位而言，推理是通信需求的**另一半场景**：多卡推理的逐层集合通信、PD 分离的大块 KV 传输，与训练通信互补（[《集合通信》](../collective/index.md)与 [H01-5](../ascend/hccl-source/05-comm-engines.md) 的推理侧战场）。

---

[开始单元 05-1 →](./05-inference-engine/01-from-training-to-serving.md)
