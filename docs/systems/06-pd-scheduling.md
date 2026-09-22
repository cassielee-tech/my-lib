# 第 6 章｜Prefill/Decode 调度与推理指标

> 本课目标：掌握 TTFT/TPOT/吞吐三指标的定义与制约根源（两种负载共享同一硬件）；理解 Chunked Prefill 的穿插调度与 PD 分离的架构逻辑；算清 PD 分离的 KV 传输账，说出推理通信与训练通信互补的性格差异。

## 本章导学

::: tip 本课只记住 3 件事
1. **制约的根源是"一山二虎"**：Prefill 是 compute-bound（大 GEMM 吃满算力），Decode 是 memory-bound（每步读全部权重，算术强度极低）——两种性格相反的负载挤在同一张卡上，谁都是对方的干扰源。
2. **两个调度手段递进**：**Chunked Prefill**（大单切小、穿插执行——缓解干扰）→ **PD 分离**（Prefill 池与 Decode 池分开部署、KV 传输衔接——根治干扰，代价是通信）。
3. **PD 分离的本质是"用 KV 传输买指标解耦"**：TTFT 归 Prefill 池、TPOT 归 Decode 池，各自独立扩容与优化——通信库因此多了一类大块点对点流量。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。本章是推理两讲的收官，也是《训练与推理系统》专栏的收官——把 [第 3 章](./03-inference-kv-cache.md)的指标伏笔和 [第 5 章](./05-inference-engine.md)的混批伏笔一并收口。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./06-pd-scheduling/quick.md)：一个"备菜间与出菜间"的类比讲完整章，再回来按单元深入。

- [ ] 我能用 Roofline 解释 Prefill 与 Decode 的负载性格差异
- [ ] 我能说出混跑干扰如何同时伤害 TTFT 与 TPOT
- [ ] 我能算 PD 分离的 KV 传输账并解释指标为什么解耦

## 本章单元

- **06-1（约 15 分钟）**：[三指标与两种负载](./06-pd-scheduling/01-metrics-and-workloads.md)
- **06-2（约 15 分钟）**：[混跑干扰与 Chunked Prefill](./06-pd-scheduling/02-interference-chunked-prefill.md)
- **06-3（约 15 分钟）**：[PD 分离：分而治之](./06-pd-scheduling/03-pd-disaggregation.md)
- **06-4（约 15 分钟）**：[推理通信的账本与专栏收官](./06-pd-scheduling/04-inference-comm-wrapup.md)

- **本章总结**：[练习、自测与专栏收官](./06-pd-scheduling/summary.md)

## 这一课在整条路线中的位置

```text
第 3 章（选修）：TTFT/TPOT 定义与 Decode 的延迟敏感
第 5 章：连续批——两种负载开始同批混跑
        ↓
本章：混跑干扰 → Chunked Prefill → PD 分离（终局）
        ↓
推理通信：与训练通信互补的另一半场景（回扣《集合通信》《昇腾与 HCCL》）
```

---

[开始单元 06-1 →](./06-pd-scheduling/01-metrics-and-workloads.md)
