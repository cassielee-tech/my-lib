# 单元 06-3｜PD 分离：分而治之

> 所属章节：[第 6 章｜Prefill/Decode 调度与推理指标](../06-pd-scheduling.md)

::: info 本单元目标
围绕 **"用 KV 传输买指标解耦"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 5. 架构：两个池 + 一条传输带

**PD 分离（Prefill/Decode Disaggregation，DistServe 等提出）**：把两种负载部署到**独立的实例池**：

```text
          ┌── Prefill 池 ──┐         ┌── Decode 池 ──┐
请求 ───► │ compute-bound  │  KV 传输 │ memory-bound  │ ───► 逐 token 流出
          │ 配置：大算力    │ ──────► │ 配置：大带宽/显存│
          │ 独立扩容       │  P2P 大块 │ 独立扩容       │
          └────────────────┘         └───────────────┘
        TTFT 在这里决定              TPOT 在这里决定
```

三个结构性收益：

1. **指标解耦**：TTFT 由 Prefill 池的排队+计算决定，TPOT 由 Decode 池的步进决定——**各自独立扩容**（TTFT 高就加 Prefill 卡，TPOT 高就加 Decode 卡），不再互相拆台；
2. **配置各得其所**：Prefill 池按 compute-bound 优化（大 batch、大 GEMM 算子），Decode 池按 memory-bound 优化（KV 显存配比、低延迟 kernel）——"一山二虎"变"两山二虎"；
3. **解码批更纯**：Decode 池里没有 Prefill 插队，节拍天然平稳。

## 6. 代价：KV 传输的账

Prefill 算完的 KV Cache 必须**搬到** Decode 实例——这不是小账（用 [第 5 章](../05-inference-engine/03-paged-attention.md)的公式）：

$$
\text{KV 传输量} \approx 0.5\ \text{MB/token} \times \text{上下文长度}
$$

一个 32K 上下文的请求 ≈ **16 GB 的点对点搬运**。传输要足够快（不能吃掉分离带来的收益），通常走**机内/机间高速链路的大块 P2P**，或落到 GPU/NPU 直连存储（分层卸载）。

::: tip 通信视角的转折点
PD 分离把推理通信从"每层小消息集合通信"（TP/EP 推理）扩展出**第二类形态：大块 KV 点对点传输**——延迟要求不敏感（可与新请求 Prefill 并行），但带宽要求高。这正是 [《昇腾与 HCCL》01 章](../../ascend/01-ascend-cann/02-cann-stack-execution-path.md)提过的 **HIXL（单边通信）**与 RDG 类链路的用武之地——训练与推理两类流量，构成通信库的完整版图。
:::

**什么时候值得分离**：负载 TTFT/TPOT 要求都高（同时长 Prefill 与严格 TPOT SLO）、集群规模够大（两个池都养得起）——小集群单池 + Chunked Prefill 通常更省。

---

[继续单元 06-4 →](./04-inference-comm-wrapup.md)
