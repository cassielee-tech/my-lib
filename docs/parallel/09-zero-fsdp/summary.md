# 第 9 章总结｜ZeRO、FSDP 与混合并行

> 返回：[第 9 章首页](../09-zero-fsdp.md)

## 把本章串成一条主线

请先合上各单元正文，沿"冗余 → 递进 → 工程化 → 组合 → 会师"复述整章；遇到断点时，再回到对应单元查阅。

```text
洞察（09-1）：DP 三大件 8 份复制 · 更新逐参数独立 → 分片合法
   Z1 切 O（白省）→ Z2 加切 G（AR→RS，通信减半）
   → Z3 加切 P（用时 AG，通信 1.5×，显存 ÷N）
        ↓
FSDP（09-2）：Z3 的工程化身 —— 逐层"用时拼、用完还"
   计算完整不切 · 激活不省 · prefetch 预取重叠
   Offload 兜底 · 与 TP 的"切不切计算"之辨
        ↓
组合（09-3）：先诊断再开方 —— 放不下的是参数/序列/还是都放得下？
   Megatron 3D（极限吞吐）vs FSDP 系（简单弹性）· 五把刀总表
        ↓
会师（09-4）：需求清单 ↔ 《集合通信》供给表逐行对上 —— 专栏收官
```

## 9. 动手练习

### 练习 1：三级账本

7.5B（P=15、G=15、O=60 GB）、N=16。填写 DP/Z1/Z2/Z3 的每卡显存与每步通信量（RS=G(N-1)/N、AG(P) 按前向+反向两趟计），并指出哪一级开始"通信变贵"。

### 练习 2：生命周期排序

把 FSDP 中一个 Transformer 层在一步内的事件按时间排序：AG(Pᵢ)前向、释放、AG(Pᵢ)反向、RS(Gᵢ)、本段更新、预取 AG(Pᵢ₊₁)——标出预取与哪段计算重叠。

### 练习 3：开方

三个场景各配一套方案：① 7B 微调、8 卡单机、显存够但想快；② 70B 预训练、64 机 8 卡、追求极限吞吐；③ 128K 长序列推理服务、KV Cache 撑爆单卡。写出刀的选择与理由。

## 10. 自测题

1. 三大件各多大？为什么说优化器状态是大头？
2. "优化器更新逐参数独立"这一观察为什么让分片合法？
3. Z1 为什么说几乎免费？
4. Z2 的通信为什么减半？用到什么原语替换？
5. Z3 的通信代价是什么？换来什么？
6. FSDP 一层的参数生命周期四个阶段是什么？
7. FSDP 与 TP 在"切不切计算"上的区别带来哪两项显存差异？
8. prefetch 重叠的是什么与什么？
9. 决策树的第一问是什么？三个分支各指向哪些刀？
10. 两种经典配方各自的取舍是什么？

::: details 自测答案

1. 7.5B：P=15 GB、G=15 GB、O=60 GB；O（fp32 m+v）占 2/3。
2. 更新 θᵢ 只依赖自己的 mᵢ、vᵢ、gᵢ——N 卡各更新 1/N 参数互不干扰，完整复制没有必要。
3. 显存砍掉 (N−1)O/N，而梯度的 AllReduce 与计算完全不变——没有代价。
4. 梯度分片后每人只要自己那段：AllReduce(G)（2G(N−1)/N）换成 ReduceScatter(G)（G(N−1)/N）——"最小够用原语"。
5. 每层前向/反向都要 AG 拼回参数，通信从 0.5× 涨到约 1.5×；换来任意大模型放得下（每卡 90/N GB）。
6. 分片存放 → AG 拼齐（前向）→ 释放 → AG 拼齐（反向）→ RS 梯度 → 释放 → 更新自己段。
7. FSDP 计算完整：激活不省（完整保存）；TP 计算分片：激活随分块 ÷N。
8. 第 i+1 层参数的 AG 与第 i 层的计算——参数预取藏进当前层前向/反向。
9. "模型放得下单卡吗？"放得下→DP(+Z1/Z2)；单层放不下→TP；总体放不下→FSDP/PP；序列放不下→CP；MoE→EP。
10. Megatron 3D：通信最优、吞吐极限，但配置复杂、负载要人工配平；FSDP 系：工程简单、弹性好，通信与吞吐略逊。

:::

## 11. 专栏验收清单

《并行策略》九讲收官。对照验收：

- [ ] 能画出 2 机 16 卡的 rank 编号表并为 TP×DP 写出组身份（[第 4 章](../04-distributed-basics/01-process-rank.md)）
- [ ] 能推导 DP 梯度 AllReduce 的必要性并算每步通信（[第 5 章](../05-ddp/01-dp-semantics.md)）
- [ ] 能独立推导 ColP→GeLU→RowP 的 shape 流转（[第 6 章](../06-tensor-parallel/02-column-row-parallel.md)）
- [ ] 能画 1F1B 时间线并用 Bubble 公式估算（[第 7 章](../07-pipeline-parallel/02-microbatch-bubble.md)）
- [ ] 能对比 CP 的 AG 与 Ring 路线并解释 causal 三角（[第 8 章](../08-context-parallel/02-attention-comm.md)）
- [ ] 能填 Z1/Z2/Z3 的显存与通信账并按决策树开方（本章）
- [ ] 能把六种并行的通信需求对到《集合通信》的原语供给表（本章 09-4）

## 参考资料

- [ZeRO 论文](https://arxiv.org/abs/1910.02054) 与 [FSDP 论文](https://arxiv.org/abs/2304.11277)
- [《训练与推理系统》02 章：显存账本与 Offload](../../systems/02-training-memory-compute/01-memory-ledger.md)
- [《集合通信》04-2：ReduceScatter 的省钱逻辑](../../collective/04-gather-scatter-alltoall/02-reduce-scatter.md)
- [《集合通信》05-4：重叠的条件与边界](../../collective/05-topology-hierarchical-overlap/04-overlap-practice.md)
- [HCCL 源码专题 H01-7：AllReduce 调用链](../../ascend/hccl-source/07-allreduce-call-chain.md)

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
