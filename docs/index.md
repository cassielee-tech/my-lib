---
layout: home

hero:
  name: "Cassie’s Stack"
  tagline: "AI Infra·昇腾·集合通信领域的系统学习文档"
  actions:
    - theme: brand
      text: 开始第一课
      link: /model/01-landscape
    - theme: alt
      text: 直达源码课程
      link: /ascend/hccl-hcomm

features:
  - icon: 🧠
    title: 模型全景
    details: 模型在算什么？通信从哪来？—— Transformer 骨架、张量流动与并行的必然性（第 1-6 章）。
    link: /model/
    linkText: 进入专栏
  - icon: 🏋️
    title: 训练与推理系统
    details: 训练怎样运转？显存花在哪？—— 训练循环、显存账本与梯度通信；推理与后训练为选修（第 1-6 章）。
    link: /systems/
    linkText: 进入专栏
  - icon: 🔧
    title: 单卡执行系统
    details: 算子怎样在单卡上跑得快？—— 执行模型、存储层次、Roofline、Stream/Event 与图编译（第 1-8 章）。
    link: /device/
    linkText: 进入专栏
  - icon: 🧩
    title: 并行策略
    details: 并行为什么产生通信？—— DP/TP/PP/CP/EP 与 ZeRO/FSDP 的张量切分与通信量推导（第 1-9 章）。
    link: /parallel/
    linkText: 进入专栏
  - icon: 📡
    title: 集合通信（岗位核心）
    details: 语义、算法、代价模型与工程 —— 六大原语、Ring/Tree 推导、拓扑分层与重叠（第 1-5 章）。
    link: /collective/
    linkText: 进入专栏
  - icon: 🔷
    title: 昇腾与 HCCL
    details: 平台机制、调用链与源码落地 —— 三类：昇腾 / Ascend C / HCCL 与 HCOMM，含双仓源码课程。
    link: /ascend/
    linkText: 进入专栏
  - icon: 🤖
    title: AI Agent
    details: 主线之外的工程应用视角：Agent 产品地图、工程化知识树与 AI Coding 实践。
    link: /ai-agent/
    linkText: 查看目录
  - icon: 🧱
    title: 后端知识
    details: 编程语言、数据库、分布式系统与后端工程实践。
    link: /backend/
    linkText: 查看目录
  - icon: 🛠️
    title: 工具与效率
    details: 开发工具、系统快捷键与日常效率工作流。
    link: /tools-productivity/
    linkText: 查看目录
  - icon: ✍️
    title: 个人杂谈
    details: 学习复盘、生活观察，以及一些不设边界的思考。
    link: /essays/
    linkText: 查看目录
---


## 专栏总览

> 目标岗位：**CANN 的 HCCL 集合通信算子开发**。这门课把大模型、AI Infra 与昇腾平台三块知识，按岗位需要的顺序组织成六大专栏——从"模型在算什么"一路走到"集合通信源码怎么实现"。

| 专栏 | 主题 | 章节 | 直达第一课 |
| --- | --- | --- |  --- |
| [模型全景](model/index.md) | 模型在算什么？通信从哪来？ | 第 1-6 章 | [第 1 章｜从模型到集合通信](model/01-landscape.md) |
| [训练与推理系统](systems/index.md) | 训练怎样运转？显存花在哪？ | 第 1-6 章 | [第 1 章｜训练循环](systems/01-training-loop.md) |
| [单卡执行系统](device/index.md) | 算子怎样在单卡上跑得快？ | 第 1-8 章 + 专题 |[第 1 章｜全景与性能分析](device/01-landscape-performance.md) |
| [并行策略](parallel/index.md) | 并行为什么产生通信？ | 第 1-9 章 |  [第 4 章｜分布式基础](parallel/04-distributed-basics.md) |
| [集合通信](collective/index.md) | 语义、算法、代价模型与工程（岗位核心） | 第 1-5 章 |  [第 1 章｜Collective 语义与代价模型](collective/01-collective-semantics-cost.md) |
| [昇腾与 HCCL](ascend/index.md) | 平台机制、调用链与源码落地 | 三类：昇腾 / Ascend C / HCCL 与 HCOMM |  [第 0 章｜认识昇腾与 CANN](ascend/00-ascend-cann.md) |

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

- **主线**：模型全景 1-5 → 训练与推理系统 1-2 → 单卡执行系统 1-8 → 并行策略 1-9 → 集合通信 1-5 → 昇腾与 HCCL（昇腾 0-2）→ [HCCL 与 HCOMM 源码学习](ascend/hccl-hcomm.md)；
- **选修**：模型全景第 6 章（RoPE/长上下文）、训练与推理系统第 3-6 章（推理与后训练）、并行策略第 2 章（效率设计）；
- **二梯队**：昇腾与 HCCL 专栏的 Ascend C 分类（第 0-2 章，算子开发）——进入 HCCL 引擎 template（AICPU/AIV/CCU）开发时再深入。


## 笔记习惯

每篇笔记保持同一结构，重点是留下**读源码时能复用的模型、数据流和代价分析**：

```md
# 主题
一句话解释 / 解决什么问题 / 核心概念与数据流
Shape · 显存 · 计算量 · 通信量 / 最小示例
与集合通信的关系 / 容易混淆的点 / 自测问题 / 参考资料
```