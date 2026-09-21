# 第 7 章总结｜Stream、Event 与异步执行

> 返回：[第 7 章首页](../07-stream-event-async.md)

## 把本章串成一条主线

请先合上各单元正文，沿"异步模型 → Stream 语义 → 同步工具 → 重叠模式"复述整章；遇到断点时，再回到对应单元查阅。

```text
Host 提交、Device 排队（07-1）
   调用返回 ≠ 执行完成；隐式同步点；报错延迟、计时要用 Event
        ↓
Stream：同流严格按序，跨流才可能并行（07-2）
   计算流 / 搬运流 / 通信流；默认流陷阱；跨流数据不安全
        ↓
同步四层次（07-3）：synchronize · Event · Notify · Barrier
   能用 Event 不用 synchronize；漏同步=数据竞争，滥同步=隐形减速
        ↓
三种重叠模式（07-4）：备料 · 对账 · 接力
   假并行（闸门全关）与错并行（缺签认单）是两大敌人
        ↓
第 8 章：数据类型、布局与融合——改写账本数字的三个旋钮
```

## 9. 动手练习

### 练习 1：找出隐式同步点

审查这段训练循环，列出所有会掐断异步流水的位置：

```python
for x, y in loader:
    x = x.npu(non_blocking=True)
    loss = model(x, y).mean()
    loss.backward()
    print("loss:", loss.item())        # ①
    if step % 10 == 0:
        print("grad norm:", torch.norm(p.grad).item())   # ②
    optimizer.step(); optimizer.zero_grad()
```

哪些可以移出循环或降低频率？分别解释代价。

### 练习 2：设计跨流依赖

数据加载在搬运流 S1，训练在计算流 S2，每轮循环：S1 执行 `H2D(batch i+1)`，S2 用 `batch i` 训练。写出 Event 的 record/wait 插入位置，保证：S2 第 i+1 轮开始前 `batch i+1` 已就位，且 S1 复用的缓冲区不被提前覆盖。

### 练习 3：诊断两种病

某多流程序出现两种症状：A 版本——结果偶发错误，跑十次错一次；B 版本——结果永远正确但总时长与单流串行几乎一样。分别给出最可能的原因和验证方法。

## 10. 自测题

1. 为什么框架必须采用异步提交模型？
2. `.item()` 和 `.cpu()` 这类操作为什么是隐式同步点？
3. 设备报错时 Python 堆栈为什么不可信？排障用什么手段？
4. 同一条 Stream 上的任务有什么顺序保证？
5. 所有任务都挤在默认流上，实际执行形态是什么？
6. 三种典型车道是什么？通信流的价值在哪里？
7. Event 的 record 和 wait 分别做什么？
8. 四个层次的同步对象分别适用什么场景？
9. 漏同步和滥同步各自的症状是什么？
10. DDP 通信与反向计算重叠的前提条件有哪些？

::: details 自测答案

1. Host Launch 一次只要微秒级，Kernel 执行常为毫秒级；同步等待会让 CPU 绝大部分时间空转。异步让 Host 连续提交，设备流水执行。
2. 它们需要"具体的数值"而不只是"任务的句柄"，Host 必须等设备把前面的任务真正算完——被迫阻塞。
3. 设备报错时 Host 已提交到更后面的任务、跑到无关代码行。排障用 `ASCEND_LAUNCH_BLOCKING=1` 临时改为同步执行，让报错当场暴露。
4. 严格按提交顺序执行，不乱序、不并行——无论任务是拷贝、计算还是通信。
5. 彻底串行：搬运等计算、计算等通信，任何重叠都不存在。
6. 计算流、搬运流（H2D/D2H）、通信流。通信流让梯度聚合与剩余反向计算并行（DDP 重叠）。
7. record 在流内插旗：前序任务完成后事件成立；wait 让另一条流阻塞到事件成立后才继续——跨流单向依赖的最小工具。
8. Host↔Device 用 synchronize（粗、代价大）；流间用 Event（精确）；设备间用 Notify 等信号；组内全员用 Barrier/集合通信的隐式栅栏（等最慢者）。
9. 漏同步 → 数据竞争、结果偶发错误（跑十次对九次）；滥同步 → 异步流水被反复掐断，时间线大片空隙，性能跌回串行。
10. 反向从后往前逐层产梯度（有时序错位可利用）、梯度按 Bucket 装箱、通信走独立流异步执行、且在 optimizer.step() 前用同步保证全部聚合完成。

:::

## 11. 本课小结

- Host 是写作业单的，Device 是按单干活的——调用返回不代表执行完成；
- Stream 单行道：同流按序、跨流并行；默认流是隐式刹车；
- 同步四层次，粒度由粗到细：synchronize → Event → Notify → Barrier，能细则细；
- 漏同步是偶发 bug 之源，滥同步是隐形减速之源；
- 三种重叠模式（备料/对账/接力）是并发设计的标准件；
- 分析武器是时间线：看空隙找同步病，看咬死找假并行。

## 参考资料

- [PyTorch：CUDA Semantics（异步与流）](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch NPU：Stream 与 Event](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0002.html)
- [《昇腾与 HCCL》的 01-2：Runtime 的 Stream/Event/Notify 管理](../../ascend/01-ascend-cann/02-cann-stack-execution-path.md)
- [HCCL 源码专题 H01-4：通信原语与同步机制](../../ascend/hccl-source/04-primitives-and-sync.md)
- [CANN 常用命令速查：异步排障与 msprof](../../ascend/cann常用命令.md)

下一课补齐本专栏最后一块拼图：**数据类型、布局与算子融合**——BF16/FP16/FP8、Layout 和 Fusion 怎样影响精度与性能。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
