# 第 1 章总结｜AI Infra 全景与性能分析

> 返回：[第 1 章首页](../01-landscape-performance.md)



## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## 13. 动手练习

### 练习 1：定义一个性能问题

从工作中选择一个“慢”的场景，用以下模板重新描述：

```text
工作负载：
硬件与设备数：
输入 Shape / 消息大小：
当前指标：
目标指标：
测量区间：
```

### 练习 2：计算扩展效率

某训练任务单卡吞吐 1200 tokens/s，2 卡为 2250，4 卡为 4200，8 卡为 7200。分别计算 Speedup 和 Scaling Efficiency，并观察哪次扩容损失最大。

### 练习 3：比较理论下界

某操作需要 10 TFLOPs、搬运 40 GB 数据。设备峰值为 200 TFLOPs/s，HBM 有效带宽为 1 TB/s。分别计算计算和访存时间下界，并初步判断瓶颈。

## 14. 自测题

1. AI Infra 与模型算法关注点有什么不同？
2. 算子和 Kernel 有什么区别？
3. Runtime 主要负责哪些工作？
4. Latency、Throughput 和 Scaling Efficiency 分别描述什么？
5. 为什么设备利用率 100% 不代表程序已最优？
6. 什么是关键路径？
7. Compute-bound、Memory-bound、Communication-bound 各自受什么限制？
8. 算术强度的定义是什么？
9. 小消息通信为什么更容易受固定延迟影响？
10. 为什么局部 Kernel 加速后必须重新测量端到端指标？

::: details 自测答案

1. 模型算法关注要计算的函数与学习目标；AI Infra 关注这些计算如何在硬件和集群上高效、稳定、经济地执行。
2. 算子是上层计算语义，Kernel 是它针对特定硬件、Shape、数据类型和布局的底层实现；二者不一定一一对应。
3. 管理设备内存、Stream、Event、任务下发、同步以及 Host 与 Device 之间的协作。
4. Latency 是完成一次任务的时间；Throughput 是单位时间完成的工作量；Scaling Efficiency 衡量多设备吞吐相对理想线性增长的比例。
5. 它可能只表示设备一直有任务执行，但这些任务可能 Shape 很小、计算效率低或主要在搬运数据。
6. 决定任务最早完成时间的最长依赖链；只有缩短关键路径，端到端时间才一定下降。
7. 分别受计算单元吞吐、存储系统带宽和设备间通信/同步能力限制。
8. $AI=FLOPs/Bytes\ moved$，表示每搬运一个 Byte 完成多少次计算。
9. 总时间中的启动、同步等固定成本不会随消息字节数同比下降，小消息的数据传输时间很短，固定成本占比因而更大。
10. 局部优化可能不在关键路径上，也可能引入额外转换、同步或资源竞争，只有端到端测量才能确认真实收益。

:::

## 15. 本课小结

- AI Infra 连接模型、框架、编译、Runtime、算子、通信库和硬件；
- 一行模型代码要经过多层软件，最终才成为设备上的计算与搬运任务；
- 性能目标必须明确是延迟、吞吐、扩展效率、成本还是稳定性；
- 性能问题可初步分为计算、访存和通信受限，但结论必须针对具体阶段；
- FLOPs、搬运字节数和通信轮次可以帮助建立理论下界；
- 时间线和关键路径比孤立的算子耗时更能解释端到端性能；
- 正确流程是定义指标、建立基线、观察全局、提出假设、对照验证、确认端到端收益。

## 参考资料

- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [昇腾社区文档](https://www.hiascend.com/document)

下一课将进入硬件基础：比较 CPU、GPU 和 NPU 的执行方式，并理解昇腾 AI Core 中矩阵与向量计算单元各自承担什么工作。

---

<!-- chapter-navigation -->
[返回 AI Infra 目录 →](../index.md)
