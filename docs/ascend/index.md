# 昇腾与 HCCL

> 返回：[课程总览](../roadmap.md)

**专栏目标**：把前面所有通用知识落到昇腾平台与 HCCL 源码上——平台机制、调用链追踪、算法实现三步走。

![模型计算与分布式通信在 CANN 中的概念路径](/images/cann/cann-learning-map.svg)

## 章节列表

平台入门（第 1-3 章，已备）：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [认识昇腾与 CANN](./01-ascend-cann.md) | NPU/达芬奇架构、CANN 软件栈、版本与环境链？ | ✅ |
| 2 | [Runtime 与任务执行](./02-runtime-task-execution.md) | Host/Device 分工、内存搬运、Stream/Event、aclnn 两段式与双时间线？ | ✅ |
| 3 | [从 PyTorch 走向 HCCL](./03-pytorch-to-hccl.md) | torch_npu 三件注册、ProcessGroup 门面、Stream 缝合与 AllReduce 六站旅程？ | ✅ 主线收官 |

待写章节：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 4 | Ascend C 算子基础 | 算子工程的 Host 侧与 Kernel 侧？TPipe/TQue 流水线编程？ | 🚧 二梯队 |
| 5 | 完成一个自定义算子 | 从语义到 Tiling 到验证的完整流程？ | 🚧 二梯队 |
| 6 | 性能分析与优化 | 现象 → 假设 → 证据 → 修改 → 复测的闭环？ | 🚧 二梯队 |

## 深入专题：HCCL 源码学习

想直接下钻源码，进入独立专题 **[HCCL 源码学习](./hccl-source.md)**（基于 [HCCL 开源仓库](https://gitcode.com/cann/hccl) 官方文档整理，7 单元 + 总结）：

- **H01-1～H01-2**：HCCL 全景、仓库地图与 HCCL/HCOMM 分层架构；
- **H01-3～H01-5**：通信域与 RankGraph、通信原语与同步、通信引擎；
- **H01-6～H01-7**：集合通信算法与代价模型（《集合通信》的实现对照）、AllReduce 调用链走读。

## 工具与参考

- [CANN 常用命令速查](./cann常用命令.md)：环境、版本、设备监控、日志、Profiling 与分布式排障；
- [CANN Learning Hub 笔记](./cann-learning-hub.md)：CANN 组件与 CUDA 生态对照表。

::: warning 版本提醒
CANN 的组件、接口和源码结构会随版本变化。学概念可用最新资料；运行代码和读源码，必须以工作环境中的 **驱动、固件、CANN 与框架插件版本** 为准。
:::

## 验收清单

- [ ] 不看笔记复述一次 AllReduce 从 `torch.distributed` 到引擎执行的完整调用链；
- [ ] 对着 HCCL 仓库说出 selector / executor / template / Transport 分别在哪个目录层；
- [ ] 用双时间线解释一次 Profiling 采集到的计算与通信交错。

---

[返回课程总览 →](../roadmap.md)
