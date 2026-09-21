# 单元 09-2｜FSDP：用时拼、用完还

> 所属章节：[第 9 章｜ZeRO、FSDP 与混合并行](../09-zero-fsdp.md)

::: info 本单元目标
围绕 **"参数的生命周期：分片存放、按层拼回、用完释放"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

FSDP（Fully Sharded Data Parallel）= ZeRO-3 思想的 PyTorch 工程化。核心问题是：参数分片了，前向/反向要完整参数怎么办？

## 3. 洋葱式包一层，一次只拼一层

FSDP 的答案：**不拼整个模型，逐层"用时拼、用完还"**。

对每个 Transformer 层包一层 FSDP（nested wrap），它的参数生命周期：

```text
平时：    参数分片存放（每卡 1/N），不占整层显存
前向该层：AllGather(Pᵢ) → 拼齐 → 计算 → 立刻释放，只留分片
反向该层：AllGather(Pᵢ) → 拼齐 → 算梯度 → ReduceScatter(Gᵢ) → 释放
更新：    各卡只更新自己那段（Z1 的逻辑）
```

显存峰值 = **当前拼齐的那一层 + 全部激活**（激活不省——这是"计算不切、只切存储"的代价，与 TP 的本质区别见下表）。

| | TP（[第 6 章](../06-tensor-parallel.md)） | FSDP（本章） |
| --- | --- | --- |
| 参数存放 | 分片 | 分片 |
| **计算** | **分片**（每卡算自己的块） | **完整**（拼齐了整层算） |
| 激活 | 随分片 ÷N | 完整（不省） |
| 通信 | 激活级、每层 4 次、不可重叠 | 参数级、每层 3 次、**可预取重叠** |

## 4. Prefetch：把拼参数藏进计算

逐层 AG 的通信在关键路径上吗？——**不在，因为可以预取**：算第 $i$ 层时，第 $i{+}1$ 层的参数已经在后台 AG——**通信与当前层计算重叠**（与 [第 5 章 DDP](../05-ddp/03-buckets.md) 的桶预取同一个思想，不同对象）。

FSDP 的使用体验（对照 [第 4 章](../04-distributed-basics/04-launch-and-debug.md)的骨架）：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = Transformer(...)                        # 每卡先有完整定义
model = FSDP(model, auto_wrap_policy=...)       # 包洋葱 → 参数即刻分片
# 训练循环与 DDP 几乎一样：loss.backward() 内部自动完成
#   AG(P)→算→RS(G) 的逐层接力，optimizer.step() 各更新各段
```

::: warning 三条工程备注
1. **`no_sync()` + 梯度累积**：累积步内不 RS，进一步省通信（回扣 [第 5 章 05-4](../05-ddp/04-overlap.md)）；
2. **Offload 兜底**：分片还能再甩到 CPU/NVMe（ZeRO-Offload/Infinity，[《训练与推理系统》02-3](../../systems/02-training-memory-compute/03-accumulation-offload.md) 的 Offload 思想 × 分片思想的乘法）；
3. **选择直觉**：中小模型/想要简单 → FSDP（无 stage、无调度心智）；超大模型/极限吞吐 → TP+PP 的 Megatron 路线（通信更省但工程更重）——两者也可组合。
:::

---

[继续单元 09-3 →](./03-combination.md)
