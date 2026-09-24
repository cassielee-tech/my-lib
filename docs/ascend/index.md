# 昇腾与 HCCL

> 返回：[课程总览](../roadmap.md)

**专栏目标**：把前面所有通用知识落到昇腾平台与 HCCL 源码上——平台机制、调用链追踪、算法实现三步走。

![模型计算与分布式通信在 CANN 中的概念路径](/images/cann/cann-learning-map.svg)

## 专栏分类

本专栏按主题分为四类，**每类编号独立从 0 开始**：昇腾（平台机制）、Ascend C（算子开发，待写）、HCCL（框架调用链 + 源码课程）、HCOMM（底座源码 + 自定义算子开发）。

### 一、昇腾

平台地基——从芯片架构到软件栈再到 Runtime 执行模型：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 0 | [认识昇腾与 CANN](00-ascend-cann.md) | NPU/达芬奇架构、CANN 软件栈、版本与环境链？ | ✅ |
| 1 | [Runtime 与任务执行](01-runtime-task-execution.md) | Host/Device 分工、内存搬运、Stream/Event、aclnn 两段式与双时间线？ | ✅ |

工具与参考（同属昇腾类）：

- [CANN 常用命令速查](cann常用命令.md)：环境、版本、设备监控、日志、Profiling 与分布式排障；
- [CANN Learning Hub 笔记](cann-learning-hub.md)：CANN 组件与 CUDA 生态对照表。

### 二、Ascend C

算子开发线（待写）——为 AI Core 编写算子的完整路径：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 0-2 | Ascend C 算子开发 | 算子工程的 Host 侧与 Kernel 侧？TPipe/TQue 流水线编程？从语义到 Tiling 到验证的完整流程？性能优化闭环？ | 🚧 二梯队 |

### 三、HCCL

框架调用链收官 + 源码深潜（概念地图与逐段精读双轨融合）：

| 章/课程 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 0 | [从 PyTorch 走向 HCCL](00-pytorch-to-hccl.md) | torch_npu 三件注册、ProcessGroup 门面、Stream 缝合与 AllReduce 六站旅程？ | ✅ 主线收官 |
| 课程 | [HCCL 源码学习](hccl-source.md) | 10 个单元（0-9）+ 总结：三层架构、RankGraph、原语、引擎、算法与 selector、AllReduce 调用链、executor/template、资源与 dlsym、MC2 扩展 | ✅ |

### 四、HCOMM

底座深潜与自定义通信算子开发：

| 课程 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 课程 | [HCOMM 源码学习](hcomm-source.md) | 7 个单元（0-6）+ 总结：仓库地图、编程模型、控制面/数据面走读、AICPU 七步、三引擎对比、实战起步 | ✅ |

两个源码课程是姊妹篇：**[HCCL 源码学习](hccl-source.md)**（基于 [HCCL 开源仓库](https://gitcode.com/cann/hccl)）读"内置算子怎么跑"，**[HCOMM 源码学习](hcomm-source.md)**（基于 [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)）练"自己写扩展算子"。

::: warning 版本提醒
CANN 的组件、接口和源码结构会随版本变化。学概念可用最新资料；运行代码和读源码，必须以工作环境中的 **驱动、固件、CANN 与框架插件版本** 为准。
:::

## 验收清单

- [ ] 不看笔记复述一次 AllReduce 从 `torch.distributed` 到引擎执行的完整调用链；
- [ ] 对着 HCCL 仓库说出 selector / executor / template / Transport 分别在哪个目录层；
- [ ] 用双时间线解释一次 Profiling 采集到的计算与通信交错。

---

[返回课程总览 →](../roadmap.md)
