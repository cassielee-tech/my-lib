# 第 6 章总结｜Prefill/Decode 调度与推理指标

> 返回：[第 6 章首页](../06-pd-scheduling.md)

## 把本章串成一条主线

请先合上各单元正文，沿"指标 → 干扰 → 缓解 → 根治 → 版图"复述整章；遇到断点时，再回到对应单元查阅。

```text
指标（06-1）：TTFT（排队+Prefill）· TPOT（Decode 步进）· 吞吐
   两种负载性格相反：Prefill compute-bound ⇄ Decode memory-bound
        ↓
干扰（06-2）：混跑双向伤害——Prefill 堵 TPOT 毛刺 / Decode 挤 TTFT 排队
   Chunked Prefill：大单切块穿插——毛刺压平，矛盾仍在
        ↓
根治（06-3）：PD 分离——两个池独立扩容、独立配置
   代价 = KV 传输（0.5MB/token × 上下文，32K ≈ 16GB 点对点）
        ↓
版图（06-4）：推理通信两类流量——每层小集合（α 主导）+ KV 大块 P2P（β 主导）
   与训练通信（梯度 AllReduce、可重叠）互补 → HCCL 完整战场
```

## 9. 动手练习

### 练习 1：毛刺测算

Decode 步长 20 ms，混入一个 32K Prefill（约 400 ms）。计算该请求群的 TPOT 长尾毛刺（未切块 / 切成 2048 token 块两种情况），并说明块大小旋钮的两个方向各牺牲什么。

### 练习 2：KV 传输账

70B 模型（80 层、h=8192、BF16），请求上下文 16K。PD 分离时每请求的 KV 传输量是多少？若机间链路有效带宽 20 GB/s，传输时间是多少，占一个 200 ms TPOT 预算的多少倍？

### 练习 3：架构决策

三个场景各选"单池混跑 + Chunked Prefill"还是"PD 分离"：① 小团队 8 卡服务内部工具，负载轻；② 对外 API，TTFT SLO 500 ms 且 TPOT SLO 50 ms，长系统提示；③ 多租户大集群，负载波动大。

## 10. 自测题

1. 三个指标各由哪个阶段决定？用户体感分别是什么？
2. 用 Roofline 解释 Prefill 与 Decode 的负载性格。
3. "制约的根源"一句话是什么？
4. 混跑干扰怎样双向伤害指标？
5. Chunked Prefill 的机制是什么？削平了什么、没解决什么？
6. 块大小旋钮的两个方向各牺牲什么？
7. PD 分离的三个结构性收益是什么？
8. KV 传输账的公式是什么？为什么说"用 KV 传输买指标解耦"？
9. PD 分离适合什么场景？小集群为什么不划算？
10. 推理通信的两类流量各是什么性格？与训练通信如何互补？

::: details 自测答案

1. TTFT = 排队 + Prefill（"多久开始回我"）；TPOT = Decode 步进（"回复流得多快"）；吞吐 = 并发 × 步进速率的聚合。
2. Prefill 是大矩阵 GEMM、算术强度高 → 屋顶区 compute-bound；Decode 是逐 token GEMV、每个权重只复用一次 → 山坡区 memory-bound（每步读全部权重）。
3. 两种性格相反的负载共享同一硬件——一山二虎。
4. 长 Prefill 堵住 Decode 步进（TPOT 长尾毛刺）；偏向 Decode 的调度又让 Prefill 排不上队（TTFT 飙升）。
5. 把长 Prefill 切成固定块穿插进 Decode 步群之间；削平 TPOT 毛刺，但两种负载仍共享硬件与配置，矛盾只是被时间切片掩盖。
6. 块大：Prefill 算力利用率高但毛刺大；块小：节拍平滑但 GEMM 变小、利用率下降。
7. 指标解耦（各自扩容）；配置各得其所（两山二虎）；Decode 节拍纯净。
8. ≈ 0.5 MB/token × 上下文长度；用这笔大块 P2P 传输的代价，换 TTFT 与 TPOT 的完全独立优化。
9. 两个指标都有严格 SLO、集群规模大到两个池都养得起；小集群单池 + Chunked Prefill 资源利用率更高。
10. ① TP/EP 每层小集合：MB 级、每 token、α 主导（延迟敏感、关键路径）；② PD 的 KV 大块 P2P：GB 级、每请求、β 主导（带宽饥渴、延迟宽容）——与训练的"大而低频、可重叠"互补。

:::

## 11. 专栏验收清单

《训练与推理系统》六讲收官。对照验收：

- [ ] 能算 7.5B 的训练显存账并说明梯度 AllReduce 的时机与重叠（[第 1-2 章](../01-training-loop.md)）
- [ ] 能解释 Prefill/Decode 差异与 KV Cache 账（[第 3 章](../03-inference-kv-cache.md)）
- [ ] 能说清连续批、分页、前缀复用各自消灭什么（[第 5 章](../05-inference-engine.md)）
- [ ] 能用 Roofline 解释两种负载并给出调度/分离决策（本章）
- [ ] 能把训练与推理的通信需求对到《集合通信》的原语与算法版图（本章 06-4）

## 参考资料

- [DistServe 论文：Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670)
- [Sarathi-Serve 论文：Chunked Prefill 调度](https://arxiv.org/abs/2403.02310)
- [第 3 章：推理与 KV Cache](../03-inference-kv-cache.md) 与 [第 5 章：推理引擎](../05-inference-engine.md)
- [《单卡执行系统》05 章：Roofline](../../device/05-flops-bandwidth-roofline/02-roofline-model.md)
- [《集合通信》05 章：分层与重叠](../../collective/05-topology-hierarchical-overlap/02-hierarchical.md)

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
