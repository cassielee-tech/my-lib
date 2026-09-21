# 单元 07-4｜PP 的三本账与 3D 组合

> 所属章节：[第 7 章｜Pipeline Parallel](../07-pipeline-parallel.md)

::: info 本单元目标
围绕 **"PP 的账本与它在混合并行中的位置"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 7. PP 的三本账

**通信账** ✅ 最省：相邻 stage 间 P2P 边界激活，MB 级、与模型大小无关、跨机走网卡即可。

**显存账**：三大件 ÷S（每 stage 只存自己那段层）；激活经 1F1B 后只需 O(S) 个 micro-batch 在途——必要时还能叠加 Activation Checkpointing（[《训练与推理系统》02-2](../../systems/02-training-memory-compute/02-activation-checkpointing.md)）再省一档。

**负载均衡账** ⚠️ PP 独有的软肋：流水线的节拍由**最慢的 stage** 决定——Transformer 层数均匀时好切，但 embedding 只在首段、loss 只在尾段，**首尾 stage 常被有意少放几层**来配平；切分不均 = 全流水线陪最慢的 stage 空转。

## 8. 3D 混合并行：三把刀合体

生产级大模型训练的标准形态——三把刀各就其位：

```text
TP（机内 8 卡）    → 层内高速通信贴 HCCS
  × PP（跨机分段）  → 极小 P2P 通信跨机器
    × DP（数据复制）→ 剩余卡组做吞吐扩展
```

回扣 [第 4 章](../04-distributed-basics/02-process-groups.md)的身份系统：每个 rank 同时属于三个组——`tp_group`（机内 8 卡）、`pp_group`（相邻机器的同位卡）、`dp_group`（其余机器的对应卡）。

三种并行的**通信性格**至此凑成完整光谱：

| | DP | TP | PP |
| --- | --- | --- | --- |
| 通信量 | 大（梯度全量） | 小（激活） | 极小（边界激活） |
| 频率 | 低（每步一轮） | 高（每层 4 次） | 中（每 micro-batch 每边界） |
| 形态 | AllReduce（集合） | AG/RS/AllReduce（集合） | **Send/Recv（P2P）** |
| 可重叠 | ✅ | ❌ 关键路径 | ❌ 但量小 |
| 独有代价 | 通信占比 | 延迟敏感 | **Bubble + 负载均衡** |
| 驻留位置 | 任意 | 机内 | **跨机** |

::: tip 本章收束
PP 用"流水线空转"换来了极小且跨机友好的通信——当模型大到 TP+DP 放不下、或集群跨多机时，PP 是承重墙。下一章的 **Context Parallel** 处理三把刀都棘手的最后一个维度：**序列太长**——单层激活沿序列维也放不下时，切序列。
:::

---

[进入本章总结 →](./summary.md)
