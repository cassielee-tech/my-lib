# 通信与昇腾

> 返回：[首页](../index.md)

**专栏目标**：上篇（集合通信，岗位核心）从 Collective 语义进入算法、代价模型与工程优化——手推算法、算清账本、看懂选型，是读 HCCL 源码前的全部理论准备；下篇（昇腾与 HCCL）把通用知识落到昇腾平台与 HCCL 源码上——平台机制、调用链追踪、算法实现三步走。

**时间紧？** 集合通信有 [⚡ 速通版](00-速通.md)：5 章各压成约 5 分钟的主干。

![模型计算与分布式通信在 CANN 中的概念路径](/images/cann/cann-learning-map.svg)

## 上篇｜集合通信（岗位核心，第 1-5 章）

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [Collective 语义与代价模型](01-collective-semantics-cost.md) | 六大原语的输入输出是什么？α-β-γ 账本怎么算？ | ✅ |
| 2 | [Ring AllReduce 完整推导](02-ring-allreduce.md) | Ring 为什么等于 RS 加 AG？每轮发送哪一块数据？ | ✅ |
| 3 | [Tree、Recursive Doubling 与算法选择](03-tree-algorithms-selection.md) | 小消息和大消息为什么选不同算法？分界线在哪？ | ✅ |
| 4 | [AllGather、ReduceScatter 与 AlltoAll](04-gather-scatter-alltoall.md) | 数据布局怎样变化？MoE AlltoAll 为什么容易失衡？ | ✅ |
| 5 | [拓扑、分层通信与通信计算重叠](05-topology-hierarchical-overlap.md) | 机内和机间链路怎样组合？Chunk、Channel、重叠各解决什么？ | ✅ |

- 学习顺序即章号顺序：语义账本（1）→ 算法（2/3）→ 原语场景（4）→ 拓扑与兑现（5）；
- **第 2 章务必手推**：4 rank 的 RS+AG 两阶段模拟是本篇的验收硬指标；
- 每读一章，对照 [HCCL 源码 5](hccl-hcomm/05-coll-algorithms.md) 的实现视角看一遍——理论/实现双轨并进。

**验收**：手工模拟 4~8 个 rank 的 Ring AllReduce（每轮谁发哪块）；用 α-β-γ 模型按消息大小与 rank 数比较算法，算出分界点；解释机内/机间分层通信为什么快；说出 chunk/channel/重叠各自解决什么、代价是什么。

## 下篇｜昇腾与 HCCL

### 昇腾（平台机制）

平台地基——从芯片架构到软件栈、Runtime 执行模型，再到框架调用链：

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 0 | [认识昇腾与 CANN](00-ascend-cann.md) | NPU/达芬奇架构、CANN 软件栈、版本与环境链？ | ✅ |
| 1 | [Runtime 与任务执行](01-runtime-task-execution.md) | Host/Device 分工、内存搬运、Stream/Event、aclnn 两段式与双时间线？ | ✅ |
| 2 | [从 PyTorch 走向 HCCL](02-pytorch-to-hccl.md) | torch_npu 三件注册、ProcessGroup 门面、Stream 缝合与 AllReduce 六站旅程？ | ✅ 主线收官 |

工具与参考（同属昇腾类）：

- [CANN 常用命令速查](cann常用命令.md)：环境、版本、设备监控、日志、Profiling 与分布式排障；
- [CANN Learning Hub 笔记](cann-learning-hub.md)：CANN 组件与 CUDA 生态对照表。

### HCCL 与 HCOMM（双仓源码课程）

| 课程 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 课程 | [HCCL 与 HCOMM 源码学习](hccl-hcomm.md) | 12 个单元（0-11）+ 总结，分四幕：双仓全景与架构；底座（控制面 / 数据面 / 引擎）；算子层（算法与 selector、AllReduce 调用链、executor/template、资源与 dlsym）；扩展（七步流程、MC2、实战） | ✅ |

一门课程读透两个仓：基于 [HCCL 开源仓库](https://gitcode.com/cann/hccl) 读"内置算子怎么跑"，基于 [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm) 练"自己写扩展算子"——概念与源码配对成单元，一条主线从 `HcclAllReduce` 走到第一个自定义算子。

### 实战复盘（已独立成专栏）

真实需求的完整生命周期复盘——Issue → 定位 → 方案 → 验证 → 检视 → 合入，见[实战复盘专栏](/practice/)。

::: warning 版本提醒
CANN 的组件、接口和源码结构会随版本变化。学概念可用最新资料；运行代码和读源码，必须以工作环境中的 **驱动、固件、CANN 与框架插件版本** 为准。
:::

## 验收清单

- [ ] 不看笔记复述一次 AllReduce 从 `torch.distributed` 到引擎执行的完整调用链；
- [ ] 对着 HCCL 仓库说出 selector / executor / template / Transport 分别在哪个目录层；
- [ ] 复述自定义通信算子的七步流程，并说出它与样例五步法的对应关系；
- [ ] 用双时间线解释一次 Profiling 采集到的计算与通信交错。

---

[返回首页 →](../index.md) · [进入《实战复盘》 →](/practice/)
