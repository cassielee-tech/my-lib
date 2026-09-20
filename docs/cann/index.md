# CANN

::: tip 专栏定位
这一栏把 AI Infra 的通用原理映射到昇腾平台：先看懂 **CANN 软件栈与 NPU 执行过程**，再进入 **Ascend C 算子开发、性能调优和 HCCL 源码**。
:::

CANN（Compute Architecture for Neural Networks）是昇腾 AI 处理器的软件栈。学习它不是背一组 API，而是回答一个核心问题：

> 一段模型代码，怎样经过框架适配、图与算子、Runtime，最终在 NPU 上完成计算和通信？

## 先看全景图

![模型计算与分布式通信在 CANN 中的概念路径](/images/cann/cann-learning-map.svg)

这张图要读出两条路径：

- **计算路径**：模型中的算子经过框架适配和 CANN 执行栈，最终落到 AI Core；
- **通信路径**：分布式并行产生 Collective，HCCL 负责组织设备间的数据交换。

二者会在真实训练任务中交错执行，因此后面还要学习 **Stream、Event、内存、同步和通信计算重叠**。

## 这一栏与 AI Infra 的区别

| AI Infra 讲通用原理 | CANN 讲昇腾上的具体落点 |
| --- | --- |
| Kernel、Tiling、Runtime | Ascend C、Tiling Host、Kernel 侧编程 |
| 计算单元与存储层次 | Cube/Vector 单元、Global/Local Memory |
| Stream、Event、异步执行 | CANN Runtime 中的任务下发与同步 |
| Profiling 与 Roofline | 昇腾工具链中的算子分析与调优 |
| Collective 与通信算法 | HCCL 调用链、拓扑、Executor 与 Transport |

## 为你规划的学习顺序

参考 [CANN Learning Hub](https://gitcode.com/cann/cann-learning-hub) 的公共基础、算子开发和大模型推理路线，并结合集合通信算子岗位，专栏规划为 **6 章**。每章再拆成若干篇约 **15 分钟**的学习单元，最后单独做章总结。

### 第 1 章：认识昇腾与 CANN

- **章节导学**：[认识昇腾与 CANN](./01-ascend-cann.md)
- **C01-1**：[NPU、达芬奇架构与 AI Core](./01-ascend-cann/01-npu-davinci-ai-core.md)
- **C01-2**：[CANN 软件栈与模型执行路径](./01-ascend-cann/02-cann-stack-execution-path.md)
- **C01-3**：[版本、驱动、固件与开发环境](./01-ascend-cann/03-version-driver-firmware-env.md)
- **本章总结**：[画出一段模型代码到 NPU 的完整路径](./01-ascend-cann/summary.md)

### 第 2 章：Runtime 与任务执行

- **C02-1**：Host、Device 与异构计算
- **C02-2**：设备内存、数据搬运与生命周期
- **C02-3**：Stream、Event、Task 与异步执行
- **C02-4**：一次算子调用怎样被下发和完成
- **本章总结**：用时间线解释计算、搬运与同步

### 第 3 章：Ascend C 算子基础

- **C03-1**：算子工程的 Host 侧与 Kernel 侧
- **C03-2**：GlobalTensor、LocalTensor 与存储层次
- **C03-3**：TPipe、TQue 与流水线编程
- **C03-4**：Vector、Cube 与基础 API
- **本章总结**：读懂一个最小 Add 算子的结构

### 第 4 章：完成一个自定义算子

- **C04-1**：输入输出、Shape、数据类型与算子语义
- **C04-2**：Tiling 策略与多核切分
- **C04-3**：Kernel 实现、编译与运行
- **C04-4**：正确性验证与框架接入
- **本章总结**：独立实现并验证一个完整算子

### 第 5 章：性能分析与优化

- **C05-1**：先测量——耗时、带宽与利用率
- **C05-2**：减少搬运与提高数据复用
- **C05-3**：流水线、Double Buffer 与并行度
- **C05-4**：Tiling、多核负载均衡与尾块
- **C05-5**：数值精度、边界条件与性能回归
- **本章总结**：形成“现象 → 假设 → 证据 → 修改 → 复测”的闭环

### 第 6 章：从 PyTorch 走向 HCCL

- **C06-1**：`torch_npu` 在框架与 CANN 之间的位置
- **C06-2**：模型算子如何进入昇腾执行栈
- **C06-3**：Rank、通信域、Stream 与 Notify
- **C06-4**：`torch.distributed` 到 HCCL 的概念调用链
- **C06-5**：阅读 AllReduce 源码前要认识的模块
- **本章总结**：追踪一次计算与集合通信的交汇点

### 深入专题：HCCL 源码学习

第 6 章回答"框架通信怎样进入 HCCL"；想继续往下钻源码，进入独立专题 **[HCCL 源码学习](./hccl-source.md)**。专题基于 [HCCL 开源仓库](https://gitcode.com/cann/hccl) 的官方文档（`docs/zh`）整理，共 7 个单元 + 1 篇总结：

- **H01-1～H01-2**：HCCL 全景、仓库地图与 HCCL/HCOMM 分层架构；
- **H01-3～H01-5**：通信域与 RankGraph、通信原语与同步、通信引擎；
- **H01-6～H01-7**：集合通信算法与代价模型、AllReduce 调用链源码走读。

## 推荐实践路线

不要同时铺开所有方向。按下面的顺序做，知识会更容易连起来：

1. 用 Learning Hub 的 **公共基础**认识 NPU、CANN 和张量运算；
2. 沿 **算子开发路线**完成最小算子，建立 Host/Device、Tiling 和流水线概念；
3. 从 **大模型推理路线**挑选 Profiling、算子融合等内容，练习性能定位；
4. 回到博客的集合通信课程，补齐 Collective、拓扑和代价模型；
5. 最后带着明确问题进入 [HCCL 源码学习专题](./hccl-source.md)，沿调用链读源码。

::: warning 版本提醒
CANN 的组件、接口和源码结构会随版本变化。学习概念时可以使用最新资料；运行代码和读源码时，必须以工作环境中的 **驱动、固件、CANN 与框架插件版本**为准。
:::

## 工具与速查

- [CANN 常用命令速查](./cann常用命令.md)：环境、版本、设备监控、PyTorch、日志、Profiling 与分布式排障。
- [CANN Learning Hub 笔记](./cann-learning-hub.md)：CANN 组件与 CUDA 生态对照。

## 参考资料

- [CANN Learning Hub](https://gitcode.com/cann/cann-learning-hub)：教程、案例、Notebook 与分路线学习地图；
- [Ascend C 算子开发指南](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0018.html)：编程模型、API 与开发流程；
- [Ascend C 高性能编程最佳实践](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0001.html)：数据搬运、流水线、Tiling 和性能优化方法。

下一步从 **第 1 章第 1 单元：NPU、达芬奇架构与 AI Core** 开始。
