# 第 8 章｜Context Parallel（序列并行）

> 本课目标：理解长序列为什么把 DP/TP/PP 三把刀都逼到墙角；掌握 Attention 的两种序列并行通信方案（AllGather 与 Ring 轮转）及各自的显存/重叠权衡；理解 causal mask 的工作量三角与均衡切法；把 CP 放进 4D 混合并行收口。

## 本章导学

::: tip 本课只记住 3 件事
1. **序列是最后一个维度**：Norm/MLP 逐 token 独立，序列切开**零通信**；唯独 Attention 要"看到全序列"——**K/V 必须流动**，这就是 CP 的全部通信。
2. **两条路线**：**AllGather**（收全长 K/V，简单，但显存峰值高、通信集中不可重叠）vs **Ring**（K/V 块沿环轮转、边传边算，峰值低、通信被计算覆盖，代价是 causal 工作量三角需要均衡）。
3. **CP 与 TP 正交**：TP 切 head、CP 切序列——可以叠加，凑出 **TP×CP×PP×DP 的 4D 并行**。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。先想清楚"哪些层序列独立、哪些不独立"，通信方案就只剩必然选择——建议边读边画 Q/K/V 的分块图。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./08-context-parallel/quick.md)：一个"分组看监控查案"的类比讲完整章，再回来按单元深入。

- [ ] 我能说出 CP 下哪些层零通信、哪一层产生全部通信
- [ ] 我能对比 AllGather 与 Ring 两种方案的三项权衡
- [ ] 我能解释 causal 工作量三角与 zigzag 均衡的思路

## 本章单元

- **08-1（约 15 分钟）**：[长序列的墙：为什么是 CP](./08-context-parallel/01-why-cp.md)
- **08-2（约 15 分钟）**：[Attention 的通信：AllGather 还是 Ring](./08-context-parallel/02-attention-comm.md)
- **08-3（约 15 分钟）**：[Ring Attention 与负载均衡](./08-context-parallel/03-ring-attention.md)
- **08-4（约 15 分钟）**：[CP 的账与 4D 组合](./08-context-parallel/04-cp-engineering.md)

- **本章总结**：[练习、自测与串联](./08-context-parallel/summary.md)

## 这一课在整条路线中的位置

```text
第 5-7 章：切数据 / 切层内 / 切深度 —— 序列维始终完整
        ↓
本章 CP：切序列 —— 长上下文时代的主角
        ↓
第 9 章 ZeRO/FSDP：切训练状态，混合并行收口
```

[第 6 章](./06-tensor-parallel.md)的 TP+SP 已埋下伏笔（Norm 沿序列切的"半步 CP"）；[《并行策略》02-3](./02-efficient-llm-design/03-long-context.md) 给过长上下文的模型视角四条路——本章补上系统视角的第五条：**把序列本身切开**。

---

[开始单元 08-1 →](./08-context-parallel/01-why-cp.md)
