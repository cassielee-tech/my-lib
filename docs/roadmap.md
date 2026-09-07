# 大模型与 AI Infra 学习路线

## 使用方式：一次完成一个 15 分钟闭环

::: tip 课程的最小单位是一个问题，而不是一篇长文章
每个学习单元控制在约 **15 分钟**：先看问题和核心图，再读关键机制，最后用一句话复述结论。长文章只是同一主题的资料集合，不要求一次读完。
:::

| 时间 | 动作 | 产出 |
| ---: | --- | --- |
| 2 分钟 | 阅读“本单元要回答的问题” | 知道本次学习边界 |
| 8 分钟 | 看核心图与关键机制 | 理解一条因果链 |
| 3 分钟 | 回答 1～3 个检查问题 | 发现理解缺口 |
| 2 分钟 | 写一句自己的解释 | 留下可复习的结论 |

一次可以只学 **1 个单元**；有整块时间时，再连续完成 2～4 个单元。实验、公式推导和源码阅读作为独立实践单元安排，不挤进概念学习的 15 分钟。

## 目标与定位

这条路线面向有传统后端开发经验、正在从事昇腾集合通信算子开发的工程师。目标不是泛泛了解大模型应用，而是建立一条能够指导工程实践和源码阅读的主线：

> 模型结构 → 计算图 → 训练与推理 → 并行策略 → 集合通信 → 通信库 → HCCL 源码

完成后应当能够：

- 解释 Transformer 一次前向与反向传播发生了什么；
- 估算模型参数量、显存占用、计算量和通信量；
- 解释 DP、TP、PP、ZeRO/FSDP 的切分方式和通信需求；
- 推导 AllReduce、AllGather、ReduceScatter、AlltoAll 等集合通信原语；
- 从模型并行策略一路追踪到 PyTorch ProcessGroup、HCCL API 和集合通信算子；
- 带着“数据在哪里、谁在等待、瓶颈是什么”阅读 HCCL 源码。

## 整体安排

路线仍可在约 **16 周**完成，但进度改用“学习单元”而不是小时数衡量。建议工作日每天完成 1～2 个 15 分钟单元，周末安排一次实验或博客整理。

| 阶段 | 周数 | 核心问题 | 阶段产出 |
| --- | ---: | --- | --- |
| 0. 全景与工具 | 1 周 | 大模型系统由哪些层组成？ | 一张全景图、术语表 |
| 1. 模型计算基础 | 3 周 | Transformer 到底算了什么？ | 手写小型 Transformer、显存估算表 |
| 2. 训练与推理系统 | 2 周 | 模型怎样训练和生成？ | 训练循环分析、KV Cache 笔记 |
| 3. AI Infra 基础 | 2 周 | 算子怎样在加速器上高效运行？ | Roofline/性能分析笔记 |
| 4. 分布式训练 | 3 周 | 模型如何切到多设备？ | 并行策略对比、DDP 实验 |
| 5. 集合通信 | 3 周 | 通信原语如何实现与选型？ | Ring/Tree 推导、通信量计算 |
| 6. 昇腾与 HCCL | 2 周起 | HCCL 如何承接框架通信？ | 调用链、模块图、源码阅读记录 |

## 阶段 0：建立全景（第 1 周）

### 学习内容

- 大模型的生命周期：数据、预训练、后训练、评测、部署和服务；
- 软件栈：模型框架、编译器、算子库、通信库、运行时、驱动和硬件；
- 训练系统的三类核心资源：计算、显存/内存、网络；
- 从 `loss.backward()` 到一次 AllReduce 的调用链概念。

### 验收标准

- 能在 10 分钟内画出大模型训练软件栈；
- 能解释集合通信为什么不是独立知识，而是并行训练的必要机制；
- 建立不少于 30 个概念的术语表。

## 阶段 1：大模型计算基础（第 2～4 周）

### 第 2 周：必要的数学与 PyTorch

- 张量形状、矩阵乘法、广播与批处理；
- 导数、链式法则、梯度下降、反向传播；
- PyTorch 的 Tensor、Autograd、Module、Optimizer；
- 不追求完整数学证明，重点是能沿计算图推导 shape、FLOPs 和梯度。

### 第 3 周：Transformer

- Tokenization、Embedding、位置编码；
- Q/K/V、Scaled Dot-Product Attention、Multi-Head Attention；
- MLP、残差连接、LayerNorm/RMSNorm；
- Encoder-only、Encoder-Decoder、Decoder-only 的差异。

### 第 4 周：手写与量化分析

- 用 PyTorch 实现一个最小 Decoder-only Transformer；
- 跟踪每层张量 shape；
- 估算参数量、激活值、优化器状态与 FLOPs；
- 理解为什么 Attention 对序列长度呈平方复杂度。

### 推荐资料

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)：先学习第 1～4 章；
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：配合实现阅读，不要求第一次读懂全部细节；
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)：补齐张量、Autograd 和训练循环。

## 阶段 2：训练与推理（第 5～6 周）

### 训练

- 预训练目标、数据批次、损失函数与优化器；
- 混合精度、梯度累积、梯度裁剪、Checkpoint；
- SFT、LoRA、偏好对齐只建立概念与工程位置，不先深挖算法。

### 推理

- Prefill 与 Decode；
- KV Cache、Batching、吞吐与时延；
- Quantization、Speculative Decoding、Continuous Batching；
- 训练通信与推理通信的区别。

### 阶段产出

- 一篇“一个 Token 是怎样生成的”笔记；
- 一篇“训练显存都花在哪里”笔记；
- 对同一模型分别估算训练和推理的资源消耗。

## 阶段 3：AI Infra 基础（第 7～8 周）

### 学习内容

- CPU、GPU/NPU 的执行模型差异；
- HBM、Cache、片上存储、PCIe、互联网络；
- FLOPs、带宽、算术强度与 Roofline Model；
- Kernel、算子融合、数据布局、流水与异步执行；
- Profiling：识别 compute-bound、memory-bound、communication-bound。

### 学习原则

所有性能结论都回答三个问题：

1. 数据从哪里来到哪里去？
2. 搬运多少字节，执行多少计算？
3. 计算、访存和通信能否重叠？

## 阶段 4：分布式训练（第 9～11 周）

### 第 9 周：数据并行

- Rank、World Size、Process Group；
- DDP 的模型复制、梯度 Bucket 与 AllReduce；
- 为什么 DDP 通常采用一设备一进程；
- 通信计算重叠与反向传播 Hook。

### 第 10 周：模型与流水线并行

- Tensor Parallel：按行/列切分矩阵乘法；
- Pipeline Parallel：Stage、Micro-batch 与 Bubble；
- Sequence/Context Parallel；
- Expert Parallel 与 MoE AlltoAll。

### 第 11 周：分片与混合并行

- ZeRO 1/2/3 与 FSDP；
- 参数、梯度、优化器状态分别如何分片；
- DP、TP、PP、CP、EP 的组合与 Device Mesh；
- 为不同并行策略计算通信量。

### 推荐资料

- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)；
- [PyTorch Distributed Tutorials](https://docs.pytorch.org/tutorials/distributed.html)；
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)；
- [Megatron-LM 论文](https://arxiv.org/abs/2104.04473)与[官方代码](https://github.com/NVIDIA/Megatron-LM)。

## 阶段 5：集合通信（第 12～14 周）

### 第 12 周：语义与代价模型

- P2P 与 Collective 的关系；
- Broadcast、Reduce、AllReduce、AllGather、ReduceScatter、AlltoAll；
- Latency-Bandwidth 模型：`T ≈ α × 通信轮次 + β × 数据量`；
- 进程、设备、Rank、Communicator 与 Stream。

### 第 13 周：算法

- Ring AllReduce = ReduceScatter + AllGather；
- Tree、Recursive Doubling、Halving-Doubling；
- 不同消息大小、Rank 数与拓扑下的算法选择；
- 分层集合通信：机内互联与机间网络。

### 第 14 周：工程实现

- 拓扑发现、建链、分片、调度和同步；
- Chunk/Channel/Pipeline；
- 通信计算重叠、并发与流语义；
- 超时、错误传播、容错和性能诊断；
- 对照 NCCL 理解成熟通信库的 API 与执行模型。

### 推荐资料

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)：重点阅读 Collective Operations、Group Calls、Stream Semantics；
- 自己实现 CPU 多进程版 Ring AllReduce，用日志观察每一轮数据流；
- 使用 PyTorch `torch.distributed` 实验 AllReduce、AllGather、ReduceScatter。

## 阶段 6：昇腾与 HCCL（第 15～16 周及以后）

### 阅读顺序

1. 昇腾硬件、CANN、Runtime、Stream/Event、内存模型；
2. Ascend Extension for PyTorch 如何接入 `torch.distributed`；
3. HCCL 对外 API、通信域创建和集合通信调用；
4. 从一个 AllReduce 用例追踪完整调用链；
5. 拆分初始化、拓扑、建链、算法选择、Executor/Transport 等模块；
6. 再进入具体集合通信算子和性能优化。

### 推荐资料

- [HCCL 官方 API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)；
- [Ascend C 中的 HCCL 接口说明](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0869.html)；
- 工作环境所对应 CANN 版本的《集合通信用户指南》和实际源码。

HCCL 文档与代码必须以工作环境的 CANN 版本为准，避免把不同版本的模块和 API 混在一起。

## 每周学习闭环

每周围绕一个主题完成若干 15 分钟单元：

1. **概念单元**：回答“它是什么、解决什么问题”；
2. **机制单元**：画出输入、输出和数据流；
3. **代价单元**：估算 Shape、显存、计算量或通信量；
4. **实践单元**：完成最小实验或手工推导；
5. **输出单元**：整理博客，用自己的话写出结论；
6. **复习单元**：隔天不看资料复述核心流程。

推荐每周完成 **8～12 个概念单元 + 1 个实践单元**。没有完成时直接从上次停下的位置继续，不需要重排整条路线。

## 博客笔记模板

每篇文章尽量保持同一结构：

```md
# 主题

## 一句话解释
## 它解决什么问题
## 核心概念与数据流
## Shape / 显存 / 计算量 / 通信量
## 最小示例
## 与 AI Infra、集合通信的关系
## 容易混淆的点
## 自测问题
## 参考资料
```

重点不是复述资料，而是留下未来阅读 HCCL 源码时能够复用的模型、数据流和代价分析。
