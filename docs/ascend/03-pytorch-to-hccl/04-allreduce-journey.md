# 单元 03-4｜一次 AllReduce 的完整旅程

> 所属章节：[第 3 章｜从 PyTorch 走向 HCCL](../03-pytorch-to-hccl.md)

::: info 本单元目标
读完后，你能够**独立追踪 `dist.all_reduce` 从 DDP 的 backward hook 到 HCCL 入口的每一步**，说出每层职责与源码地图，为 H01-7 的下潜做好准备。
:::

## 先记住 3 个结论

1. **旅程六站**：DDP hook 触发 → `dist.all_reduce` → ProcessGroupHCCL（检查+取流）→ `HcclAllReduce` 入口 → HCCL 内部（选算法→编排→传输）→ 任务挂流异步返回。
2. **每层只做自己的事**：框架管时机与对象、门面管检查与取流、HCCL 管执行——追踪调用链的过程就是反复问"这一步是谁的职责"。
3. **本章止步于入口**：入口之后的 selector/executor/template/Transport 下潜，全部属于 [HCCL 源码专题](../hccl-source.md)（H01-6/H01-7）。

## 1. 总装：六站时序

以 DDP 训练一步中的一桶梯度为例（回扣 [《并行策略》05-3](../../parallel/05-ddp/03-buckets.md)）：

```text
① DDP backward hook：某桶梯度就绪 → 触发该桶通信
        ↓
② dist.all_reduce(bucket, op=SUM)          # Python（c10d）
        ↓
③ ProcessGroupHCCL::allreduce              # 检查 dtype/contiguous/设备
   取流：当前流或通信流（03-3 的缝合线）
        ↓
④ HcclAllReduce(dst, src, count, dtype,    # HCCL 入口：comm + stream
                op, comm, stream)
        ↓
⑤ HCCL 内部（黑盒预览）：                    #
   通信域 → selector 选算法（Ring/Tree/...，H01-6）
   → executor 编排任务 → transport 走链路 → 引擎执行（H01-5）
        ↓
⑥ 任务挂上 stream 异步返回                    # Host 继续跑（第 2 章双时间线）
   后续：框架 Event 同步 / stream sync 兜底
```

追踪的技巧：**每一站问三个问题**——数据从哪来（参数）、交给谁（下一站）、挂在哪条时间线上（流）。

## 2. 源码阅读地图

本章（框架侧）与源码专题（库内侧）的分界线就在第 ④⑤ 站之间：

| 想看什么 | 去哪 |
| --- | --- |
| torch_npu 适配层、ProcessGroupHCCL | torch_npu 仓库（[框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)） |
| HCCL 公开 API（HcclAllReduce 等） | [HCCL API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html) |
| 入口之后的下潜 | **[H01-7：AllReduce 调用链走读](../hccl-source/07-allreduce-call-chain.md)** |
| 算法选择的依据 | [H01-6：集合通信算法与代价模型](../hccl-source/06-coll-algorithms.md) |
| 通信域与 RankGraph | [H01-3](../hccl-source/03-comm-domain-rank-graph.md) |
| 引擎与执行 | [H01-5](../hccl-source/05-comm-engines.md) |

## 3. 自测题

先用自己的话回答，再展开答案。

1. 写出六站时序，并指出每站的"数据从哪来、交给谁"。
2. 第 ③ 站的 ProcessGroupHCCL 做哪两件事？
3. 追踪调用链时每站要问的三个问题是什么？
4. 框架侧与库内侧的分界线在哪？两侧的源码分别住在哪两个仓库？
5. 第 ⑥ 站之后，框架怎样知道通信完成了？

::: details 自测答案

1. ① hook 触发（梯度桶就绪）→ ② dist.all_reduce → ③ PGHCCL（检查、取流）→ ④ HcclAllReduce 入口 → ⑤ HCCL 内部（选算法/编排/传输/引擎）→ ⑥ 挂流异步返回。
2. 合法性检查（dtype/contiguous/设备一致）；确定流（当前流或通信流）。
3. 数据从哪来（参数）、交给谁（下一站）、挂在哪条时间线上（流）。
4. 分界线在 HcclAllReduce 入口：之前是 torch_npu 仓库，之后是 HCCL 仓库（H01-7）。
5. Event 同步（通信流 record，消费流 wait）或 stream sync 兜底——第 2 章的双时间线收口。

:::

## 本单元小结

- 六站时序 = 框架五站 + 库内一站（黑盒预览）；
- 追踪技巧：每站问"数据从哪来、交给谁、挂哪条流"；
- 本章与 H01-7 的分界线即两个仓库的分界线——继续下潜的路标已立好。

## 参考资料

- [HCCL 官方 API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)
- [HCCL 源码专题 H01-7：AllReduce 调用链走读](../hccl-source/07-allreduce-call-chain.md)
- [《并行策略》05-3：Bucket 分桶](../../parallel/05-ddp/03-buckets.md)

---

下一单元将进入 **本章总结：追踪一次计算与集合通信的交汇点**。

[返回第 3 章 →](../03-pytorch-to-hccl.md)
