# 单元 05-2｜从朴素 DDP 到生产级 DDP

> 所属章节：[第 5 章｜数据并行与 DDP](../05-ddp.md)

::: info 本单元目标
围绕 **"生产级 DDP 比朴素版多做的三件事"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 3. 朴素版：十行写对 DP

语义清楚后，朴实现呼之欲出——在 [第 4 章](../04-distributed-basics/04-launch-and-debug.md)的骨架上加三行：

```python
model = ...
for data in loader:                      # 每卡各自的数据分片
    loss = model(data).loss
    loss.backward()                      # 各算各的梯度
    for p in model.parameters():         # ★ 全量梯度取平均
        dist.all_reduce(p.grad)          #   （world group 上）
        p.grad /= world_size
    optimizer.step()                     # 同步更新 → 副本保持一致
    optimizer.zero_grad()
```

数学上完全正确。但性能上有个硬伤：**反向全部结束才发起第一次通信**——通信 M 秒 + 反向 N 秒 串行相加，一步 = N+M。生产级 DDP 的全部工作就是消灭这个串行。

::: warning 别用错 DP
`torch.nn.DataParallel`（单进程多线程）受 GIL 与线程调度拖累，早已不推荐；本章的 `DistributedDataParallel`（DDP）= 多进程 + 通信库，与第 4 章"一进程一卡"对齐。
:::

## 4. 生产级 DDP 多做的三件事

`DistributedDataParallel(model, device_ids=[local_rank])` 包一层，它做了：

1. **起点对齐**：构造时把 rank 0 的参数 **Broadcast** 给全员——保证 N 个副本从同一起点出发（不然第 5.1 节的"同步"从第一步就不成立）；
2. **梯度就绪即通信**：给每个参数注册反向 hook——**某个参数的梯度一算完，立刻发起它的通信**，不等反向结束。通信与剩余层的反向并行，这是重叠的入口；
3. **Bucket 组织**：就绪即发的粒度不是"单参数"而是"桶"——按反向完成顺序攒桶、桶满即发（05-3 展开）。

从本章往下钻一层：DDP 发起的每次 AllReduce 最终走进 [HCCL 的 AllReduce 调用链](../../ascend/hccl-source/07-allreduce-call-chain.md)（H01-7）——框架侧"什么时候发"，通信库侧"怎么跑"，正好接成一条线。

```text
DDP（本章）：何时发 → 逐桶、就绪即发
HCCL（H01-7）：怎么跑 → 算法选择 → 引擎执行 → 环形流水
```

---

[继续单元 05-3 →](./03-buckets.md)
