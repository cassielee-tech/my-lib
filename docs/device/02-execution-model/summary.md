# 第 2 章总结｜GPU/NPU 执行模型与计算单元

> 返回：[第 2 章首页](../02-execution-model.md)

## 把本章串成一条主线

请先合上各单元正文，沿着"晶体管预算 → 执行哲学 → 内部组织 → 利用率损失"复述整章；遇到断点时，再回到对应单元查阅。

一条主线收束本章：

```text
晶体管花在哪？
  CPU：控制 + 缓存 → 低延迟          ┐
  GPU：海量 ALU → 吞吐 + 线程隐藏延迟  ├ 三种执行哲学（02-1）
  NPU：专用通路 → 矩阵能效            ┘
        ↓
内部组织：SM / Warp / SIMT（02-2）  ⇄  Cube / Vector / Scalar + 显式搬运（02-3）
        ↓
利用率 = 并行度 × 数据供给 × 形状效率 × …（02-4）
        ↓
优化方向 → Tiling、流水线、融合、并行切分（后续课程）
```

## 9. 动手练习

### 练习 1：给三种处理器分工

一个训练任务包含：数据预处理（字符串解析 + 截断补齐）、Embedding 查表、40 层 Transformer 前向反向、梯度 AllReduce。把每项工作分给 CPU / GPU / NPU / 网络链路，并用本章概念说明理由。

### 练习 2：算一次尾块损失

某算子切出 130 个 Tile，设备有 32 个 AI Core 可用，每个 Tile 耗时相同。计算：总共需要几轮？最后一轮利用率是多少？整体利用率是多少？若把 Tile 数调整为 128，整体利用率变成多少？

### 练习 3：解释一条 Profiling 现象

某 GEMM 实测只有理论算力的 45%。Profiling 显示：Tensor Pipe 利用率 52%，HBM 带宽 88%，Kernel 前半段满载、后四分之一时间利用率骤降。用本章的五类损失解释这 45%，并提出两个优化假设。

## 10. 自测题

1. CPU 和 GPU 处理访存延迟的策略有什么本质区别？
2. Warp 是什么？为什么它是 GPU 的最小调度单位？
3. SIMT 的分支分歧为什么会带来接近两倍的时间损失？
4. Tensor Core 与 CUDA Core 的区别是什么？
5. 达芬奇架构中 Cube、Vector、Scalar 分别承担什么？
6. NPU 与 GPU 在"延迟处理"上的策略差异是什么？
7. 为什么说"一次 MatMul 不只是 Cube 在工作"？
8. 列出至少四类"理论算力跑不满"的结构性原因。
9. 为什么利用率是乘法而不是加法？这对优化策略意味着什么？
10. "Device 利用率 100% 但任务不快"，给出至少两种解释。

::: details 自测答案

1. CPU 尽量缩短延迟（乱序执行、多级缓存、分支预测）；GPU 不缩短延迟，而是用大量可运行线程切换来隐藏延迟。
2. Warp 是 32 个线程组成的锁步执行组；硬件按 Warp 而不是按线程发射指令和调度，因此它是调度的最小单位。
3. Warp 同一时刻只能走一条指令路径，两个分支只能串行执行，总时间约等于两条路径之和。
4. CUDA Core 一次执行一次标量乘加；Tensor Core 一次完成一个小矩阵块的乘加，是大模型算力的主要来源。
5. Cube 做矩阵乘加；Vector 做逐元素与向量运算；Scalar 做标量计算、地址与循环控制、指令发射。
6. GPU 靠线程切换隐藏延迟；NPU 靠软件显式组织数据搬运与流水，让等待尽量不发生。
7. Scalar 准备控制流，DMA/MTE 负责搬运，Cube 做乘加，FixPipe 等负责写出——MatMul 是一条多单元流水线，任何一环慢都会拖慢整体。
8. 并行度不足、尾块效应、数据供给不足、执行方式惩罚（分歧/不对齐/布局转换）、频率与功耗墙（含慢卡）。
9. 各环节的效率彼此独立地压缩最终吞吐，损失是相乘的；因此优化应优先作用于乘积中最小的那一项，收益最大。
10. 例如：单元在小分块/低效路径上忙碌（形状不对齐）；Warp 分歧串行执行；反复做数据搬运；忙的不是关键路径上的有效工作。

:::

## 11. 本课小结

- 晶体管预算决定执行哲学：CPU 换延迟、GPU 换吞吐、NPU 换矩阵能效；
- GPU 的钥匙是 SM/Warp/SIMT：并行度是生命线，分支分歧与不对齐是结构性惩罚；
- NPU 的钥匙是功能分化 + 显式搬运：Cube/Vector/Scalar 流水协作，软件组织数据流；
- 两者的心智模型差异：GPU 让所有线程有活干，NPU 把搬运和计算排成流水；
- 利用率是乘法：并行度 × 数据供给 × 形状效率 × 频率，先找最小的那一项；
- 执行模型为第 1 章的三类瓶颈提供了结构性解释，也为 Tiling、流水线与并行课程铺路。

## 参考资料

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CANN Learning Hub：什么是 NPU](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/02_what_is_npu.ipynb)
- [昇腾官方文档：AI Core 硬件架构](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [2026 主流 AI 加速卡全景](../accelerator-cards-2026.md)

下一课将进入硬件基础的第二站：**存储层次与数据搬运**——HBM、Cache、片上 Buffer 和寄存器有什么区别，数据为什么经常比计算更贵。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
