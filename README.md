# my-cann

个人 CANN/HCCL 方向的双轨材料库（Obsidian vault，GitCode 同步）：`study/` 岗位系统学习课程 + `design/` 需求设计交付物。

## 仓库结构

```text
my-cann/
├── study/      # 六大专栏课程（模型 → 系统 → 单卡 → 并行 → 集合通信 → 昇腾HCCL）
├── design/     # 需求交付物（SRS/SD、STC、评审记录、预调研、方法论、模板库）
│   ├── 模板库/
│   └── 26RP4/  # 按迭代组织的需求目录
└── README.md
```

`.obsidian/`、`.claude/` 为本地工具配置，不入库。

## study｜学习课程（入口：[study/index.md](study/index.md)）

| 专栏 | 主题 | 章节 |
| ---- | ---- | ---- |
| [模型全景](study/模型全景/index.md) | 模型在算什么？通信从哪来？ | 6 章 |
| [训练与推理系统](study/训练与推理系统/index.md) | 训练怎样运转？显存花在哪？ | 6 章 |
| [单卡执行系统](study/单卡执行系统/index.md) | 算子怎样在单卡上跑得快？ | 8 章 + 专题 |
| [并行策略](study/并行策略/index.md) | 并行为什么产生通信？多少字节？ | 9 章 |
| [集合通信](study/集合通信/index.md) | 语义、算法、代价模型与工程（岗位核心） | 5 章 |
| [昇腾与 HCCL](study/昇腾与HCCL/index.md) | 平台机制、调用链与源码落地 | 昇腾 / Ascend C / HCCL 三类 |

其中 [HCCL 与 HCOMM 源码学习](study/昇腾与HCCL/HCCL与HCOMM源码学习.md) 为 13 单元源码精读课程（架构分层 → 控制面/数据面 → 引擎算法调用链 → executor/template → 资源与 dlsym → 自定义算子），带 [总结地图](study/昇腾与HCCL/HCCL与HCOMM源码/总结-源码阅读地图.md)。

## design｜需求材料

### 26RP4｜Reduce 64bit+特殊规约分级算法（[目录](design/26RP4/Reduce%2064bit+特殊规约分级算法/)）

需求 AR20260930049502 全链路交付物（hccl 仓 `feature/reduce-64bit-prod-2level-hier`，PR #3755 已合入）：

| 材料 | 说明 |
| ---- | ---- |
| [srs&sd.md](design/26RP4/Reduce%2064bit+特殊规约分级算法/srs&sd.md) | SRS/SD 全文：决策记录、风险表、死锁排查扩展节、cost 三轮迭代与实测数据 |
| [设计文档.md](design/26RP4/Reduce%2064bit+特殊规约分级算法/设计文档.md) | SE 串讲版设计 |
| [前置知识.md](design/26RP4/Reduce%2064bit+特殊规约分级算法/前置知识.md) | 小白版理论向导 |
| [经验总结.md](design/26RP4/Reduce%2064bit+特殊规约分级算法/经验总结.md) | 同行向知识分享（分级原理/架构注册/选路cost/两则实战案例） |
| [Reduce_64bit特殊规约两级分级算法_STC.xlsx](design/26RP4/Reduce%2064bit+特殊规约分级算法/Reduce_64bit特殊规约两级分级算法_STC.xlsx) | 用例台账 58 条（ST 15 / UT 6 / Checker 25 / NPU 12），执行情况已反填 |
| [SRS_SD_review.xlsx](design/26RP4/Reduce%2064bit+特殊规约分级算法/SRS_SD_review.xlsx) | 设计评审记录 9 条（7 闭环 2 开放） |
| [Code_review_3755.xlsx](design/26RP4/Reduce%2064bit+特殊规约分级算法/Code_review_3755.xlsx) | PR #3755 代码评审记录 20 条（人工 6 + AI 去重 14，处置按 14061870 核实） |
| [DPU-Reduce写非root用户输出buffer问题对齐.md](design/26RP4/Reduce%2064bit+特殊规约分级算法/DPU-Reduce写非root用户输出buffer问题对齐.md) | ST 断言暴露的存量缺陷对齐材料 |
| [scripts/](design/26RP4/Reduce%2064bit+特殊规约分级算法/scripts/README.md) | 一键执行六件套（NPU 真机 / ST / Checker / UT / 单用例 / 临界点实验） |

开放项追踪：芯片支持口径待 SE 确认（srs&sd 1.5）；NPU_RD_TH_* 阈值实验真机结果待回填（srs&sd 2.9）；STC TOPO/CHK 三条 GTEST_SKIP 待真机补验。

### 26RP4｜支持 AllGatherV、Reduce 算子的零拷贝模式（预调研）

[预调研-AllGatherV与Reduce零拷贝模式.md](design/26RP4/支持allgatherV、reduce算子的零拷贝模式/预调研-AllGatherV与Reduce零拷贝模式.md)——零拷贝机制现状摸底，`file:line` 事实与【待确认】判断分离，供 SE 需求串讲前对齐。

### 通用材料

- [需求实验方法论参考.md](design/需求实验方法论参考.md)——远程仿真实验全链路经验（环境/工具/构建/观测/实验设计/踩坑/交付，8 章）
- [模板库](design/模板库/)——SRS-SD 文档模板、SRS_SD_review / STC_review / code_review 评审记录模板、测试 STC 用例模板

## 关联仓库

上游 [cann/hccl](https://gitcode.com/cann/hccl)（fork：cassielee-tech/hccl）· [cann/hcomm](https://gitcode.com/cann/hcomm)

## 更易读的博客版本

[Cassie’s Stack
](
https://cassielee-tech.github.io/my-lib/)