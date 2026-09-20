# 单元 I02-2｜GPU 内部：SM、Warp 与 SIMT

> 所属章节：[第 2 章｜GPU/NPU 执行模型与计算单元](../02-execution-model.md)

::: info 本单元目标
围绕 **SM、Warp 与 SIMT** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

理解 GPU 的钥匙只有三把：SM 是硬件组织，Warp 是调度单位，SIMT 是执行方式。

### 4.1 SM：GPU 的基本作战单元

一块 GPU 由几十上百个 **SM（Streaming Multiprocessor）** 组成，每个 SM 内部包含：

- 数十个 CUDA Core（标量运算通道）；
- 若干 Tensor Core（矩阵运算单元）；
- 一份寄存器堆（Register File）和共享内存（Shared Memory）；
- Warp 调度器与指令发射单元。

```text
GPU 设备
 ├── SM 0    ├── SM 1    ├── SM 2    ────►
      │           │           │
   CUDA Core × N  Tensor Core  RegFile  SharedMem  Warp 调度器
```

软件侧的 Grid/Block/Thread 会按层级映射到设备/SM/执行通道上。**一个 Kernel 能否吃满 GPU，首先取决于是否有足够多的 Block 分布到所有 SM。**

### 4.2 Warp：32 个线程锁步走

线程不是逐个调度的，而是每 **32 个线程组成一个 Warp**，作为最小调度单位。同一个 Warp 里的线程在同一时刻执行同一条指令，只是操作不同的数据。

Warp 调度器的工作方式很朴素：

```text
Warp 0：发射指令 ──► 等待数据（访存延迟）……
Warp 1：        ──► 就绪，立即发射
Warp 2：        ──► 就绪，立即发射
……
Warp 0：数据到达，重新排队发射
```

这就是上一单元说的"用并行度隐藏延迟"的具体机制：**只要就绪的 Warp 够多，计算单元就总有活干。**

### 5. SIMT 的代价与收益

### 5.1 分支分歧（Divergence）

SIMT（Single Instruction, Multiple Threads）要求一个 Warp 同一时刻只走一条指令路径。如果 32 个线程中有 16 个走 `if`、16 个走 `else`，硬件只能**先串行执行完一个分支，再执行另一个**——两个分支的时间都要付。

对大模型负载影响不大（张量运算里几乎没有数据相关的分支），但动态控制流密集的算子会因此明显变慢。

### 5.2 Tensor Core：从标量到矩阵块

传统 CUDA Core 一次做一次标量乘加（FMA）；**Tensor Core 一次完成一个小矩阵块的乘加**（如 16×8×8 的分块），输入输出直接放在寄存器里。

大模型绝大多数 FLOPs 由 Tensor Core 贡献——GPU 的"AI 算力"实际指的就是它。这也带来约束：要吃满 Tensor Core，任务必须能切成形状友好的矩阵块，否则就退化到 CUDA Core 甚至更低效的路径。

### 5.3 执行模型对 Kernel 的三要求

把上面三把钥匙合起来，一个高效 GPU Kernel 需要：

1. **并行度足够**：Block/Warp 数量覆盖所有 SM，且有冗余以隐藏延迟；
2. **访存友好**：合并访问（Coalesced Access），让一个 Warp 的 32 次读尽可能合成少量大事务；
3. **形状规则**：矩阵尺寸对齐 Tensor Core 的分块要求，减少尾块浪费。

第 4 课的 Tiling 与流水线，就是把这三条要求工程化的方法。

---

[继续单元 I02-3 →](./03-npu-davinci-core.md)
