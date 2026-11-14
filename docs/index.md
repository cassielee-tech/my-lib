---
layout: home

hero:
  name: "Cassie’s Stack"
  tagline: "AI Infra·昇腾·集合通信领域的系统学习文档"
  actions:
    - theme: brand
      text: 开始第一课
      link: /model-systems/01-landscape
    - theme: alt
      text: 直达源码课程
      link: /comm-ascend/hccl-hcomm
    - theme: alt
      text: Now · 近期动态
      link: /now

features:
  - icon: 🧠
    title: 模型与系统
    details: 上篇·模型全景：模型在算什么？通信从哪来？（第 1-6 章）；下篇·训练与推理：训练循环、显存账本与梯度通信（第 1-6 章，推理为选修）。
    link: /model-systems/
    linkText: 进入专栏
  - icon: 🔧
    title: 执行与并行
    details: 上篇·单卡执行系统：执行模型、存储层次、Roofline、Stream/Event 与图编译（第 1-8 章）；下篇·并行策略：DP/TP/PP/CP/EP 与 ZeRO/FSDP 的通信量推导（第 1-9 章）。
    link: /execution-parallel/
    linkText: 进入专栏
  - icon: 📡
    title: 通信与昇腾
    details: 上篇·集合通信（岗位核心）：六大原语、Ring/Tree 推导、拓扑分层与重叠（第 1-5 章）；下篇·昇腾与 HCCL：平台机制与双仓源码课程。
    link: /comm-ascend/
    linkText: 进入专栏
  - icon: 🧪
    title: 实战复盘
    details: 真实需求从 Issue 到合入的完整复盘 —— 问题定位、方案演进、分层验证与检视闭环，沉淀可平移的工程方法论。
    link: /practice/
    linkText: 进入专栏
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
    details: 学习复盘、生活观察、AI Agent 观察与实践，以及一些不设边界的思考。
    link: /essays/
    linkText: 查看目录
---


## 专栏总览

> 目标岗位：**CANN 的 HCCL 集合通信算子开发**。这门课把大模型、AI Infra 与昇腾平台三块知识，按岗位需要的顺序组织成三大专栏（各含上下两篇，共原六站课程）——从"模型在算什么"一路走到"集合通信源码怎么实现"，实战复盘独立成篇。

| 专栏 | 主题 | 结构 | 直达第一课 |
| --- | --- | --- | --- |
| [模型与系统](model-systems/index.md) | 模型在算什么？训练怎样运转？ | 上篇·模型全景（1-6 章）＋ 下篇·训练与推理系统（1-6 章） | [第 1 章｜从模型到集合通信](model-systems/01-landscape.md) |
| [执行与并行](execution-parallel/index.md) | 算子怎样在单卡上跑得快？并行为什么产生通信？ | 上篇·单卡执行系统（1-8 章＋专题）＋ 下篇·并行策略（1-9 章） | [第 1 章｜全景与性能分析](execution-parallel/01-landscape-performance.md) |
| [通信与昇腾](comm-ascend/index.md) | 通信算法与代价模型；昇腾平台与源码落地 | 上篇·集合通信（1-5 章，岗位核心）＋ 下篇·昇腾与 HCCL（含双仓源码课程） | [第 1 章｜Collective 语义与代价模型](comm-ascend/01-collective-semantics-cost.md) |
| [实战复盘](practice/index.md) | 真实需求从 Issue 到合入 | 2 篇实战 ＋ 写作模板，持续更新 | [第 1 篇｜UBOE/UB_RTP 建链端口注入](practice/01-uboe-ubrtp-port-injection.md) |

## 学习路线

```text
模型与系统            模型在算什么？训练怎样运转？
   ↓
执行与并行            单卡怎样跑得快？并行为什么产生通信？
   ↓
通信与昇腾            通信算法与代价模型（岗位核心）→ 昇腾平台与源码落地
   ↓
实战复盘              真实需求从 Issue 到合入
```

三条问题贯穿全部课程：

1. **数据在哪里？** 当前张量位于 Host、HBM、片上存储还是其他 Rank？
2. **数据有多少？** Shape、数据类型和切分方式决定多少字节需要计算或搬运。
3. **谁在等待？** 计算、访存和通信之间是否存在严格依赖，能否流水或重叠？

## 主线与选修

按岗位价值排序的阅读路线：

- **主线**：模型与系统（上篇 1-5、下篇 1-2）→ 执行与并行（上篇 1-8、下篇 1-9）→ 通信与昇腾（上篇 1-5、下篇 0-2）→ [HCCL 与 HCOMM 源码学习](comm-ascend/hccl-hcomm.md) → [实战复盘](practice/index.md)；
- **选修**：模型与系统上篇第 6 章（RoPE/长上下文）、下篇第 3-6 章（推理与后训练）、执行与并行下篇第 2 章（效率设计）。


## 笔记习惯

每篇笔记保持同一结构，重点是留下**读源码时能复用的模型、数据流和代价分析**：

```md
# 主题
一句话解释 / 解决什么问题 / 核心概念与数据流
Shape · 显存 · 计算量 · 通信量 / 最小示例
与集合通信的关系 / 容易混淆的点 / 自测问题 / 参考资料
```
