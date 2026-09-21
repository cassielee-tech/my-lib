# HCCL 开发主线｜课程总览

> 目标岗位：**CANN 的 HCCL 集合通信算子开发**。这门课把大模型、AI Infra 与昇腾平台三块知识，按岗位需要的顺序组织成六大专栏——从"模型在算什么"一路走到"集合通信源码怎么实现"。

::: tip 怎么用这门课
- 每章拆为若干 **15 分钟单元**，一次只解决一个问题；
- 每章配 **⚡ 速通页**（约 5 分钟）：一个生活类比讲完整章 + 即点即答小测，跟不动完整版时先读它；
- 标注 **【选修】** 的章不影响主线推进，按需取用。
:::

## 专栏总览

| 专栏 | 主题 | 章节 | 状态 | 直达第一课 |
| --- | --- | --- | --- | --- |
| [模型全景](./model/index.md) | 模型在算什么？通信从哪来？ | 第 1-6 章 | ✅ 已备 | [第 1 章｜从模型到集合通信](./model/01-landscape.md) |
| [训练与推理系统](./systems/index.md) | 训练怎样运转？显存花在哪？ | 第 1-6 章 | ✅ 主线已备 | [第 1 章｜训练循环](./systems/01-training-loop.md) |
| [单卡执行系统](./device/index.md) | 算子怎样在单卡上跑得快？ | 第 1-8 章 + 专题 | ✅ 已备 | [第 1 章｜全景与性能分析](./device/01-landscape-performance.md) |
| [并行策略](./parallel/index.md) | 并行为什么产生通信？ | 第 1-9 章 | ✅ 已备 | [第 4 章｜分布式基础](./parallel/04-distributed-basics.md) |
| [集合通信](./collective/index.md) | 语义、算法、代价模型与工程（岗位核心） | 第 1-5 章 | ✅ 已备 | [第 1 章｜Collective 语义与代价模型](./collective/01-collective-semantics-cost.md) |
| [昇腾与 HCCL](./ascend/index.md) | 平台机制、调用链与源码落地 | 第 1-6 章 + 源码专题 | 🚧 1-3 已备，4-6 二梯队 | [第 1 章｜认识昇腾与 CANN](./ascend/01-ascend-cann.md) |

## 学习路线

```text
模型全景            模型在算什么？通信从哪来？
   ↓
训练与推理系统      训练怎样运转？显存花在哪？
   ↓
单卡执行系统        算子怎样在单卡上跑得快？
   ↓
并行策略            并行为什么产生通信？多少字节？
   ↓
集合通信            语义、算法、代价模型与工程（岗位核心）
   ↓
昇腾与 HCCL         平台机制、调用链与源码落地
```

三条问题贯穿全部课程：

1. **数据在哪里？** 当前张量位于 Host、HBM、片上存储还是其他 Rank？
2. **数据有多少？** Shape、数据类型和切分方式决定多少字节需要计算或搬运。
3. **谁在等待？** 计算、访存和通信之间是否存在严格依赖，能否流水或重叠？

## 主线与选修

按岗位价值排序的阅读路线：

- **主线**：模型全景 1-5 → 训练与推理系统 1-2 → 单卡执行系统 1-8 → 并行策略 1-9 → 集合通信 1-5 → 昇腾与 HCCL 1-3 → [HCCL 源码专题](./ascend/hccl-source.md)；
- **选修**：模型全景第 6 章（RoPE/长上下文）、训练与推理系统第 3-6 章（推理与后训练）、并行策略第 2 章（效率设计）；
- **二梯队**：昇腾与 HCCL 第 4-6 章（Ascend C 算子开发）——进入 HCCL 引擎 template（AICPU/AIV/CCU）开发时再深入。

## 写作进度与下一步

- **P0**：✅ 主线材料全部备齐（六专栏主线章节 + HCCL 源码专题）——下一步是读与练，不是写；
- **P1**：《训练与推理系统》第 5-6 章（推理系统两讲，选修）；
- **P2**：《昇腾与 HCCL》第 4-6 章（Ascend C 算子开发，二梯队）。

## 笔记习惯

每篇笔记保持同一结构，重点是留下**读源码时能复用的模型、数据流和代价分析**：

```md
# 主题
一句话解释 / 解决什么问题 / 核心概念与数据流
Shape · 显存 · 计算量 · 通信量 / 最小示例
与集合通信的关系 / 容易混淆的点 / 自测问题 / 参考资料
```

## 参考资料

- [HCCL 开源仓库](https://gitcode.com/cann/hccl) 与 [HCOMM 通信基础库](https://gitcode.com/cann/hcomm)
- [HCCL 官方 API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)
- [CANN Learning Hub](https://gitcode.com/cann/cann-learning-hub)：教程、案例与分路线学习地图
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)（对照理解成熟通信库）
