# 单元 04-1｜Kernel 的解剖：入口、索引与启动开销

> 所属章节：[第 4 章｜Kernel、Tiling 与流水线](../04-kernel-tiling-pipeline.md)

::: info 本单元目标
围绕 **Kernel 的结构与启动** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

第 1 章说过"算子是语义，Kernel 是实现"。本单元打开 Kernel，看它由哪几部分组成、怎样被启动。

## 2. Kernel 的两半：Host 侧与 Device 侧

一段典型的 Kernel 程序天然分成两半：

```text
Host 侧（CPU）                          Device 侧（每个执行单元）
────────────────────                    ─────────────────────
解析输入 shape/dtype                    入口拿到：全局形状 + 切分参数 + 自己的索引
计算切分参数（block 数、tile 形状）  →    按索引算出"我负责哪一段"
准备输入输出内存地址                     for (我的 tile) {
下发 Kernel（Launch）                       搬入 → 计算 → 搬出
等待/异步获取结果                        }
```

Device 侧的关键机制是**索引认领**：每个执行单元在入口处拿到自己的编号（GPU 的 `blockIdx/threadIdx`，NPU 的 Block Index），用同一份代码、不同的编号，各自负责总任务的一小块。**Kernel 代码只写"一份"，执行却发生在所有单元上**——这就是 SPMD（Single Program, Multiple Data）。

### 2.1 启动开销：被忽视的固定成本

从 Host 调用到 Device 真正开始执行，存在微秒级的固定开销（Launch Overhead）：参数传递、任务入队、调度分发。

- 一个 Kernel 本身耗时 5 ms，8 µs 的启动开销可以忽略；
- 一个小算子只有 10 µs 计算量，启动开销可能与计算同量级——**大量小 Kernel 串联时，Host 下发跟不上，Device 出现空隙**。

这解释了两个常见优化：**算子融合**（多个小算子合成一个 Kernel）与**图模式下沉**（整图一次性下发，见 《昇腾与 HCCL》的 01-2 的 GE 执行下沉）。也解释了 Profiling 时间线上"Kernel 之间的空洞"从哪来。

### 2.2 切分参数：Host 与 Device 的合同

Host 算好的切分参数（block 数量、每 block 的 tile 形状、循环次数）随启动一起传给 Device。这份"合同"就是下一单元的 Tiling 策略——**合同签得好，所有单元满负荷、数据高复用；签得差，一部分单元闲置或反复搬运**。

---

[继续单元 04-2 →](02-tiling-strategy.md)
