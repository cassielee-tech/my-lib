# 第 1 章｜AI Infra 全景与性能分析方法

> 本课目标：理解 AI Infra 各层分别负责什么；能够把一次模型执行拆成框架、编译、Runtime、Kernel 和硬件阶段；建立计算、访存、通信三类瓶颈的基本判断方法；学会用“现象—指标—证据—结论”分析性能问题。

## 本章导学

::: tip 本课只记住 3 件事
1. 模型代码要经过框架、编译、Runtime 和 Kernel，最终才在硬件执行。
2. 性能问题先区分计算、访存、通信与等待，并寻找真正的关键路径。
3. 好的性能结论来自可复现基线和对照实验，不来自“我感觉是这里慢”。
:::

**学习节奏：** 本章拆为 **5 个学习单元，每个约 15 分钟**。先判断数据与任务在哪里，再进入性能公式。

**先看这几张核心图：** Model to Device、Timeline Overlap、Performance Funnel。

- [ ] 我能画出模型到设备的软件栈
- [ ] 我能区分三类核心瓶颈
- [ ] 我能写出一个可验证的性能假设

## 本章单元

- **I01-1（约 15 分钟）**：[AI Infra 分层与执行对象](./01-landscape-performance/01-stack-execution.md)
- **I01-2（约 15 分钟）**：[性能指标与时间线](./01-landscape-performance/02-metrics-timeline.md)
- **I01-3（约 15 分钟）**：[计算、访存与通信瓶颈](./01-landscape-performance/03-performance-bottlenecks.md)
- **I01-4（约 15 分钟）**：[数量级、算术强度与峰值](./01-landscape-performance/04-arithmetic-intensity.md)
- **I01-5（约 15 分钟）**：[性能分析流程与常见误区](./01-landscape-performance/05-analysis-workflow.md)

- **本章总结**：[串联知识、练习与检查](./01-landscape-performance/summary.md)
