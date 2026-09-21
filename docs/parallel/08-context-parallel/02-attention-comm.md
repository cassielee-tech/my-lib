# 单元 08-2｜Attention 的通信：AllGather 还是 Ring

> 所属章节：[第 8 章｜Context Parallel（序列并行）](../08-context-parallel.md)

::: info 本单元目标
围绕 **"K/V 流动的两条路线"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

设定：N 卡序列并行，每卡持 Q/K/V 的本地段 $[B, S/N, h]$。Attention 需要本地 Q 乘以**全长** K、V。

## 3. 路线 A：AllGather K/V（先凑齐再算）

最直白的做法——把全序列的 K、V **拼齐**（[《集合通信》04-1](../../collective/04-gather-scatter-alltoall/01-allgather.md)）：

```text
每卡：Q_local (S/N)  +  AG 得到 K_full, V_full (S)
     → local attention: Q_local × K_full → scores [S/N, S] → × V_full
     → 输出 O_local —— 只涉及本地行，无需再通信 ✅
```

- **优点**：结构简单，输出天然就是最终结果（零收尾通信）；
- **代价 1（显存峰值）**：每卡要**同时持有全长 K/V**——CP 省下的序列显存，在这里又借回来一大块；
- **代价 2（不可重叠）**：AG 是一次性的大块集合通信，必须**先传完才能开算**——通信整段裸奔在关键路径上（和 TP 的处境一样，回扣 [《集合通信》05-4](../../collective/05-topology-hierarchical-overlap/04-overlap-practice.md)）。

## 4. 路线 B：Ring 轮转（边传边算）

换一种思路：**不凑齐全长，让 K/V 的小块像流水一样流过**。

- K/V 沿序列切成 N 块，沿逻辑环轮转（第 0 轮传给左邻，收到右邻的块）；
- 每一轮：本地 Q 与**手里这块** K/V 算一次 partial attention（在线 softmax 累加），同时**下一块正在路上**；
- N−1 轮后，Q 看完了所有块——**通信与计算天然重叠**（算块 $i$ 时传块 $i{+}1$，与 [《集合通信》05-3 Chunk 流水](../../collective/05-topology-hierarchical-overlap/03-chunk-channel.md) 同构）。

### 4.1 两条路线的对照表

| | AllGather | Ring 轮转 |
| --- | --- | --- |
| 总通信量（每卡） | ~$\frac{N-1}{N}\times 2S h$（K+V） | **相同** |
| 显存峰值 | 全长 K/V 同时在卡 | **同时只有一块** |
| 可重叠 | ❌ 先传后算 | ✅ 边传边算 |
| 实现复杂度 | 低 | 高（在线 softmax、块调度） |

::: tip 选择的直觉
短序列/机内高速域 → AG 的简单粗暴够用；**长序列/跨机** → 通信量本身巨大，必须用 Ring 把它藏进计算。下一单元处理 Ring 留下的最后一个难题：causal mask 的工作量三角。
:::

---

[继续单元 08-3 →](./03-ring-attention.md)
