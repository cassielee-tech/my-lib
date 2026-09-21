# 第 5 章总结｜数据并行与 DDP

> 返回：[第 5 章首页](../05-ddp.md)

## 把本章串成一条主线

请先合上各单元正文，沿"语义 → 实现 → 分桶 → 重叠"复述整章；遇到断点时，再回到对应单元查阅。

```text
语义（05-1）：完整模型 × 数据切片 → 梯度必然不同
   不 AllReduce = N 个各自漂移的模型（数学错误，不是慢）
   AllReduce(avg) ⇔ 大 batch SGD · 每卡流量 2S(N-1)/N 与 batch 无关
        ↓
实现（05-2）：朴素版 = 反向后全量 all_reduce（串行 N+M）
   DDP 三件事：Broadcast 起点对齐 / 梯度就绪即通信 / Bucket 组织
        ↓
分桶（05-3）：单参数（α 主导）与全量一次（无法重叠）的折中
   按反向顺序装桶 · 首桶小先点燃 · 桶满即发（25 MB 旋钮）
        ↓
重叠（05-4）：双流 + Event = max(N, M)
   藏不住的：通信>反向 / step 前的最后同步 / （可免的）梯度累积
```

## 9. 动手练习

### 练习 1：算一步的账

7.5B 模型、BF16、batch 64×2048 token/卡、单机 8 卡（机内 200 GB/s）、反向约 10 s。计算：梯度总量、每卡通信流量、通信时间、重叠后一步的近似时长。

### 练习 2：桶策略设计

某模型反向完成顺序为 `A→B→C→D→E→F`，梯度大小均为 10 MB。若桶容量 25 MB，给出一种装桶方案使第一桶尽早发出；若把首桶缩为 10 MB，对流水线点燃时间有什么影响？

### 练习 3：瓶颈判断

Profiling 显示：反向 4 s、通信流忙 9 s 且从第 1 s 起就在飞、一步总长 10 s。判断瓶颈在哪，给出至少两条不改模型的优化方向。

## 10. 自测题

1. DP 切的是什么？每张卡上有什么？
2. 不做梯度 AllReduce 会发生什么？为什么说这是"数学错误"而非"性能问题"？
3. 写出 DP 等价于大 batch SGD 的公式。
4. DDP 每步的通信对象与每卡流量公式是什么？为什么与 batch 大小无关？
5. 朴素 DDP 的性能硬伤是什么？
6. 生产级 DDP 的三件事是什么？
7. 按单参数通信和全量一次通信各有什么问题？Bucket 如何折中？
8. DDP 按什么顺序装桶？为什么首桶要小？
9. DDP 重叠的三个通用条件分别对应什么？
10. 哪些通信是重叠藏不住的？各有什么出路？

::: details 自测答案

1. 切数据（global batch 均分）；每卡一份完整模型副本 + 自己的数据分片。
2. N 张卡出现 N 个不同模型且差异逐步放大；因为缺失的是算法本身的取平均步骤，正确性缺失而非效率缺失。
3. $\bar g=\frac1N\sum g^{(i)}$，$\theta\leftarrow\theta-\text{lr}\cdot\bar g$——等价于用整个 global batch 算一次梯度再更新。
4. 全量梯度 $S=2\times$参数量（BF16 字节）；每卡 $2S\frac{N-1}{N}$；通信量只由模型决定，batch 增大只增加反向时长、摊薄通信占比。
5. 通信必须等反向全部结束才发起，一步 = 反向 N + 通信 M 串行相加。
6. 构造时 Broadcast rank 0 参数对齐起点；梯度就绪即触发通信（backward hook）；Bucket 组织通信粒度。
7. 单参数：次数巨量、α 主导；全量一次：无法与反向重叠。Bucket 攒成 25 MB 级：大到带宽高效、小到反向早期即可发出。
8. 按反向传播完成顺序（先算完先进桶）；首桶小使其尽早凑满发出、尽快点燃流水线，后续桶均衡。
9. 资源独立 = 计算/通信双流；依赖允许 = 通信已就绪梯度、消费者是 step；显式编排 = Event 同步 + step 前等最后一桶。
10. 通信时长超过反向的部分（出路：压缩/换算法/减量——ZeRO）；step 前的最后同步（语义下界，靠排布提前）；梯度累积可整段免掉。

:::

## 11. 本课小结

- DP 的通信是算法的一部分：AllReduce(avg) ⇔ 大 batch SGD；
- 每卡流量 $2S\frac{N-1}{N}$ 与 batch 无关——DP 的扩展性来自计算摊薄通信；
- DDP 三件事：起点对齐、就绪即发、分桶组织；
- Bucket 解决"α 主导 vs 无法重叠"的两难，按反向顺序装填、首桶小先点燃；
- 双流重叠把 N+M 压到 max(N,M)；藏不住的部分各有各的出路。

## 参考资料

- [PyTorch DDP 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [《模型全景》02-4：梯度与通信](../../model/02-tensor-autograd/04-gradient-communication.md)
- [《训练与推理系统》01-4：显存与通信](../../systems/01-training-loop/04-training-memory-communication.md)
- [《单卡执行系统》07-4：重叠的艺术](../../device/07-stream-event-async/04-overlap.md)
- [《集合通信》02 章：Ring AllReduce](../../collective/02-ring-allreduce/01-why-ring.md)
- [HCCL 源码专题 H01-7：AllReduce 调用链](../../ascend/hccl-source/07-allreduce-call-chain.md)

下一章进入 **Tensor Parallel**：模型本体登场——Column/Row Parallel 怎样切矩阵、为什么一层之内就产生 AllGather、AllReduce 或 ReduceScatter，以及为什么 TP 组必须住在一台机器里。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
