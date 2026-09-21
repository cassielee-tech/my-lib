# 第 7 章总结｜Pipeline Parallel

> 返回：[第 7 章首页](../07-pipeline-parallel.md)

## 把本章串成一条主线

请先合上各单元正文，沿"分段 → 流水 → 1F1B → 组合"复述整章；遇到断点时，再回到对应单元查阅。

```text
动机（07-1）：切深度 · 通信只有相邻 stage 的 P2P 激活（MB 级、与模型大小无关）
   三把刀：DP 切数据 / TP 切层内 / PP 切深度 —— 跨机友好的那一把
        ↓
流水（07-2）：朴素串行 → micro-batch 连续注入
   Bubble = fill + drain：(S-1)/(M+S-1)
   矛盾：大 M 压 Bubble ↔ GPipe 激活内存 O(M) 爆
        ↓
1F1B（07-3）：warmup → 一前向一反向交替 → cooldown
   内存 O(M)→O(S)，Bubble 不变 → 放心加大 M → 间接压 Bubble
   Interleaved：虚拟段把空转再除 v（代价：P2P 更碎）
        ↓
组合（07-4）：通信 ✅ 极小 / 显存 三大件÷S+O(S) 激活 / 负载均衡 ⚠️ 独有软肋
   3D = TP(机内) × PP(跨机) × DP(扩展) —— 三种通信性格凑齐光谱
```

## 9. 动手练习

### 练习 1：画时间线

S=4、M=4：分别画出 GPipe（全前向后全反向）与 1F1B 的时间线，标出 fill/steady/drain 段，数一数 Stage 2 各有几个空转节拍。

### 练习 2：算 Bubble

S=8，单 micro-batch 前向 40 ms、反向 80 ms。M=32 时一步的流水线总时间与 Bubble 率各是多少？若想把 Bubble 率压到 10% 以下，M 至少取多少？

### 练习 3：切分方案

64 台 8 卡机器训练 175B 模型。给出一种 TP×PP×DP 切分（写明三组的大小与驻留位置），并说明：为什么 TP 不跨机、PP 放在哪、DP 用剩下的什么。

## 10. 自测题

1. 三把刀分别切什么？PP 的切分单位是什么？
2. PP 每步传什么、多大、在谁和谁之间？为什么说与模型大小无关？
3. 为什么说 PP 是三把刀里唯一"没有集合通信"的？
4. 朴素流水线为什么空转严重？micro-batch 怎么解决？
5. 写出 Bubble 率公式。S=4、M=12 时是多少？
6. GPipe 的"大 M"矛盾是什么？
7. 1F1B 的稳态调度是什么？在途激活从多少降到多少？
8. 为什么说 1F1B"解内存、不解时间，最终解时间"？
9. Interleaved 1F1B 用什么换什么？代价是什么？
10. PP 独有的负载均衡问题出在哪？常见对策是什么？

::: details 自测答案

1. DP 切数据、TP 切层内矩阵、PP 切深度（层段）；PP 的单位是 stage（连续若干层）。
2. stage 边界的激活（一个 micro-batch 的 B×S×h×2，MB 级）；只在相邻两个 stage（两台机器）之间 P2P；传的是"层的输出"而非参数/梯度，模型再大这一层输出的大小不变。
3. 它的通信原语是 Send/Recv（点对点），不做 AllGather/Reduce 类的全员规约。
4. 整 batch 串行流过时同一时刻只有一台机器干活；切成 M 个 micro-batch 连续注入后，稳态时段所有 stage 满负荷。
5. $(S-1)/(M+S-1) = 3/15 = 20\%$。
6. M 越大 Bubble 越小，但 GPipe 需暂存全部 M 个 micro-batch 的激活等反向——内存 O(M)，M 大即爆。
7. warmup 做 S−1 个前向，之后每来一个新前向就完成一个旧反向；在途激活从 O(M) 降到 O(S)。
8. Bubble 公式不变所以总时间不变；但内存解放后 M 可以加大，Bubble 率随 M 下降——间接压缩。
9. stage 内再切 v 个虚拟段，调度更细，空转从 (S−1) 拍降到约 (S−1)/v；代价是 P2P 次数 ×v、每次激活更碎。
10. 流水节拍由最慢 stage 决定；embedding/loss 集中在首尾段导致天然不均——首尾 stage 有意少放几层配平。

:::

## 11. 本课小结

- PP 切深度：P2P 边界激活极小且跨机友好，补上 TP 只能住机内的短板；
- Bubble 是 PP 的固有税：$(S-1)/(M+S-1)$，靠加大 M 摊薄；
- GPipe 内存 O(M) 与大 M 矛盾 → 1F1B 交替调度降到 O(S)；
- 三本账：通信最省、显存 ÷S+O(S) 激活、负载均衡是独有软肋；
- 3D 组合（TP 机内 × PP 跨机 × DP 扩展）与三种通信性格光谱收官。

## 参考资料

- [GPipe 论文](https://arxiv.org/abs/1811.06965) 与 [PipeDream-Flush / Megatron 1F1B](https://arxiv.org/abs/2104.04473)
- [《集合通信》01-2：原语家族（含 P2P）](../../collective/01-collective-semantics-cost/02-primitive-cards.md)
- [《训练与推理系统》02-1/02-2：显存账与 Activation Checkpointing](../../systems/02-training-memory-compute/02-activation-checkpointing.md)
- [HCCL 源码专题 H01-4：通信原语与同步](../../ascend/hccl-source/04-primitives-and-sync.md)
- [第 4 章：Process Group 与三重组身份](../04-distributed-basics/02-process-groups.md)

下一章 **Context Parallel**：长序列把最后一个维度也逼到墙角——单层激活沿序列维放不下，Attention 里 K/V 要交换、Load-Balanced 切分怎么挑最重的活——序列维的并行攻略。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
