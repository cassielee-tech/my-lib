# 第 8 章总结｜Context Parallel（序列并行）

> 返回：[第 8 章首页](../08-context-parallel.md)

## 把本章串成一条主线

请先合上各单元正文，沿"墙 → 定律 → 两条路线 → 均衡 → 4D"复述整章；遇到断点时，再回到对应单元查阅。

```text
动机（08-1）：激活/KV 随 S 膨胀，DP/TP/PP 都不切序列
   CP 定律：Norm/MLP 逐 token 独立零通信 · Attention 要看全长 K/V
        ↓
路线（08-2）：A. AllGather K/V —— 简单，峰值高，不可重叠
             B. Ring 轮转 —— 边传边算，峰值低，可重叠
        ↓
均衡（08-3）：causal 块计算量 = 三角形
   朴素轮转 → 有人忙死有人闲死（木桶效应）
   解法：zigzag/斜切，按块权相等分派
        ↓
账本（08-4）：显存 ✅ ÷N · 通信 ⚠️ 与 S 成正比（Ring 可藏）
   TP 切 head × CP 切序列 = 正交叠加 → 4D 混合并行完全体
```

## 9. 动手练习

### 练习 1：通信账

h=4096、S=32768、BF16、CP=4。计算每卡的 K/V 总流量（AllGather 与 Ring 各自表达），以及 AllGather 方案下每卡需同时持有的全长 K/V 显存。

### 练习 2：三角形数块

N=4 的 causal 块矩阵（对角块计 0.5、满量块计 1）：写出朴素轮转下每 rank 的工作量当量，再设计一种 zigzag 分派使四卡尽量均衡。

### 练习 3：方案选择

两个场景各选 AllGather 还是 Ring 并说明理由：① S=8K、机内 8 卡；② S=256K、跨 8 机。判断依据是什么？

## 10. 自测题

1. 长序列场景下，三把刀分别为什么救不了激活/KV 显存？
2. CP 切序列后，哪些层零通信？为什么？
3. 为什么说"CP 的全部通信就是 K/V 怎么流动这一道题"？
4. AllGather 路线的前向步骤是什么？输出为什么无需再通信？
5. AllGather 的两个代价是什么？
6. Ring 路线怎样做到通信与计算重叠？与哪一章的机制同构？
7. 在线 softmax 在 Ring 中起什么作用？
8. causal 三角问题是什么？为什么会演变成木桶效应？
9. zigzag 均衡的核心思想是什么？
10. CP 与 TP 各切 Attention 的哪个维？为什么能叠加？4D 并行是哪四维？

::: details 自测答案

1. DP 切 batch（每卡序列全长）、TP 切 head/特征（每卡序列全长）、PP 切深度（每层内序列全长）——序列维无人切。
2. Norm/Dropout/MLP：逐 token 独立，序列分段各自算结果不变。
3. 因为除 Attention 外零通信；Attention 需要本地 Q 看到全长 K、V——唯一的通信需求就是让 K/V 到位。
4. AG 收全长 K/V → Q_local×K_full×V_full → 输出只涉及本地行（softmax 与加权求和都在本地行内完成），天然是最终结果。
5. 显存峰值：每卡同时持全长 K/V（借回 CP 省下的显存）；不可重叠：AG 传完才能开算，通信裸奔在关键路径。
6. K/V 切块沿环轮转，算第 i 块的同时传第 i+1 块；与《集合通信》05-3 的 Chunk 流水同构。
7. 跨块累加 partial attention：保存 running max 与 running sum，每来一块先修正再合并——让分块的 softmax 结果可以正确合并成全长结果。
8. causal 使块 (i,j) 仅 j≤i 有计算量且对角半量——总工作量呈三角形；朴素轮转下各 rank 分到的块权不等，轮转节拍由最慢者决定。
9. 块所有权不沿对角线、沿"等权斜线"分派，使每 rank 的总块权当量相等（各约 (N+1)/2）。
10. TP 切 head 维、CP 切序列维，正交；4D = TP × CP × PP × DP（head / 序列 / 深度 / batch）。

:::

## 11. 本课小结

- 序列是三把刀的盲区：CP 补上最后一块拼图；
- CP 定律：一切层零通信，Attention 独扛全部 K/V 流动；
- AllGather 简单但峰值高、不可重叠；Ring 峰值低、可重叠、但要解 causal 三角的均衡题；
- 通信量与序列长成正比——Ring 的"藏"依赖块计算量足够重；
- TP×CP 正交叠加 → 4D 混合并行完全体；切什么取决于什么放不下。

## 参考资料

- [Ring Attention 论文](https://arxiv.org/abs/2310.01889) 与 [Megatron Context Parallelism](https://arxiv.org/abs/2405.05317)
- [《模型全景》04-2：Causal Mask](../../model/04-self-attention/02-scaled-dot-product-mask.md)
- [《并行策略》02-3：长上下文的四条优化路线（模型视角）](../02-efficient-llm-design/03-long-context.md)
- [《集合通信》04-1：AllGather](../../collective/04-gather-scatter-alltoall/01-allgather.md)
- [《集合通信》05-3：Chunk 流水](../../collective/05-topology-hierarchical-overlap/03-chunk-channel.md)

下一章收官混合并行：**ZeRO 与 FSDP**——不再切计算，而是把"训练状态的存储"切到极致，统一回答"参数、梯度、优化器状态怎样分片"与"DP/TP/PP/CP/EP 怎样组合"的系统方法论。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
