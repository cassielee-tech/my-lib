# CassieLee 的知识库

一个基于 VitePress 搭建的个人知识博客。核心内容是 **HCCL 开发主线**——面向 CANN 集合通信算子开发岗位的六站课程，以及日常需求中的交付产物。

**在线阅读**：[cassielee-tech.github.io/my-lib](https://cassielee-tech.github.io/my-lib/) 

## 仓库结构

```text
my-lib/
├── docs/
│   ├── index.md / now.md            # 首页导览 / Now 页（当前在做什么）
│   ├── model-systems/               # 课程①：模型与系统
│   ├── execution-parallel/          # 课程②：执行与并行
│   ├── comm-ascend/                 # 课程③：通信与昇腾
│   │   ├── hccl-hcomm/              #   ├ HCCL/HCOMM 双仓源码课程 
│   │   └── reference/               #   └ HCCL 参考手册 
│   ├── practice/                    # 实战复盘（复盘 + 需求交付物 + 研讨 + 模板库）
│   │   ├── 26RP4/                   #   ├ 需求交付物（按迭代组织）
│   │   └── 研讨/                     #   └ 研讨笔记
│   ├── backend/                     # 后端知识
│   ├── tools-productivity/          # 工具与效率
│   ├── essays/                      # 个人杂谈
│   └── public/                      # 站点静态资源（图片等）
├── .github/workflows/               # GitHub Pages 自动部署
├── docs/.vitepress/config.ts        # VitePress 配置（导航与侧栏）
└── package.json                     # VitePress 站点工程
```

## 内容导览

博客分四类内容：**课程**（按顺序学）、**参考手册**（按需查）、**实战复盘**（真实案例）、**杂项**。

| 内容类型 | 回答的问题 | 入口 |
| --- | --- | --- |
| **课程** | 从零建立心智模型 | [模型与系统](docs/model-systems/index.md) · [执行与并行](docs/execution-parallel/index.md) · [通信与昇腾](docs/comm-ascend/index.md) |
| **参考手册** | 这个算子/错误码/组网怎么查 | [HCCL 参考手册](docs/comm-ascend/reference/index.md)（80 篇） |
| **实战复盘** | 真实需求怎么从 Issue 走到合入 | [实战复盘](docs/practice/index.md)（含需求交付物与研讨） |
| **杂项** | 后端、工具速查、个人思考 | [后端知识](docs/backend/index.md) · [工具与效率](docs/tools-productivity/index.md) · [个人杂谈](docs/essays/index.md) |

---

## 课程｜三大专栏 

### [模型与系统](docs/model-systems/index.md)｜模型在算什么？训练怎样运转？

- **上篇·模型全景（6 章）**：1 [从模型到集合通信](docs/model-systems/01-landscape.md) · 2 [张量、计算图与反向传播](docs/model-systems/02-tensor-autograd.md) · 3 [Tokenization 与 Embedding](docs/model-systems/03-tokenization-embedding.md) · 4 [Self-Attention](docs/model-systems/04-self-attention.md) · 5 [完整 Decoder Block](docs/model-systems/05-decoder-block.md) · 6 [RoPE 与长上下文（选修）](docs/model-systems/06-rope-long-context.md) · ⚡ [速通版](docs/model-systems/00-速通-上篇.md)
- **下篇·训练与推理系统（6 章）**：1 [训练循环](docs/model-systems/01-training-loop.md) · 2 [训练显存与计算优化](docs/model-systems/02-training-memory-compute.md) · 3 [推理与 KV Cache（选修）](docs/model-systems/03-inference-kv-cache.md) · 4 [后训练与对齐（选修）](docs/model-systems/04-post-training-alignment.md) · 5 [推理引擎与 KV Cache（选修）](docs/model-systems/05-inference-engine.md) · 6 [PD 调度与推理指标（选修）](docs/model-systems/06-pd-scheduling.md) · ⚡ [速通版](docs/model-systems/00-速通-下篇.md)

### [执行与并行](docs/execution-parallel/index.md)｜单卡怎样跑得快？并行为什么产生通信？

- **上篇·单卡执行系统（8 章 + 专题）**：1 [全景与性能分析](docs/execution-parallel/01-landscape-performance.md) · 2 [GPU/NPU 执行模型](docs/execution-parallel/02-execution-model.md) · 3 [存储层次与数据搬运](docs/execution-parallel/03-memory-hierarchy.md) · 4 [Kernel、Tiling 与流水线](docs/execution-parallel/04-kernel-tiling-pipeline.md) · 5 [FLOPs、带宽与 Roofline](docs/execution-parallel/05-flops-bandwidth-roofline.md) · 6 [Eager、计算图与图编译](docs/execution-parallel/06-eager-graph-compilation.md) · 7 [Stream、Event 与异步执行](docs/execution-parallel/07-stream-event-async.md) · 8 [数据类型、布局与融合](docs/execution-parallel/08-dtype-layout-fusion.md) · 专题 [2026 主流 AI 加速卡全景](docs/execution-parallel/accelerator-cards-2026.md) · ⚡ [速通版](docs/execution-parallel/00-速通-上篇.md)
- **下篇·并行策略（9 章）**：1 [MoE 与 Expert Parallel](docs/execution-parallel/01-moe-expert-parallel.md) · 2 [效率设计（选修）](docs/execution-parallel/02-efficient-llm-design.md) · 3 [系统分析](docs/execution-parallel/03-llm-systems-analysis.md) · 4 [分布式基础](docs/execution-parallel/04-distributed-basics.md) · 5 [数据并行与 DDP](docs/execution-parallel/05-ddp.md) · 6 [Tensor Parallel](docs/execution-parallel/06-tensor-parallel.md) · 7 [Pipeline Parallel](docs/execution-parallel/07-pipeline-parallel.md) · 8 [Context Parallel](docs/execution-parallel/08-context-parallel.md) · 9 [ZeRO、FSDP 与混合并行](docs/execution-parallel/09-zero-fsdp.md) · ⚡ [速通版](docs/execution-parallel/00-速通-下篇.md)

### [通信与昇腾](docs/comm-ascend/index.md)｜通信算法与代价模型；昇腾平台与源码落地（岗位核心）

- **上篇·集合通信（5 章）**：1 [Collective 语义与代价模型](docs/comm-ascend/01-collective-semantics-cost.md) · 2 [Ring AllReduce 完整推导](docs/comm-ascend/02-ring-allreduce.md) · 3 [Tree、RD 与算法选择](docs/comm-ascend/03-tree-algorithms-selection.md) · 4 [AG、RS 与 AlltoAll](docs/comm-ascend/04-gather-scatter-alltoall.md) · 5 [拓扑、分层与重叠](docs/comm-ascend/05-topology-hierarchical-overlap.md) · ⚡ [速通版](docs/comm-ascend/00-速通.md)
- **下篇·昇腾平台**：0 [认识昇腾与 CANN](docs/comm-ascend/00-ascend-cann.md) · 1 [Runtime 与任务执行](docs/comm-ascend/01-runtime-task-execution.md) · 2 [从 PyTorch 走向 HCCL](docs/comm-ascend/02-pytorch-to-hccl.md) · [CANN 常用命令速查](docs/comm-ascend/cann常用命令.md) · [CANN Learning Hub](docs/comm-ascend/cann-learning-hub.md)
- **源码课程｜[HCCL 与 HCOMM 导学](docs/comm-ascend/hccl-hcomm.md)（13 单元）**：0 [双仓全景与调用链](docs/comm-ascend/hccl-hcomm/00-repo-map.md) · 1 [软件架构与六族 API](docs/comm-ascend/hccl-hcomm/01-architecture-layering.md) · 2 [控制面：通信域的一生](docs/comm-ascend/hccl-hcomm/02-control-plane.md) · 3 [数据面：原语与资源](docs/comm-ascend/hccl-hcomm/03-data-plane.md) · 4 [通信引擎与三引擎编排](docs/comm-ascend/hccl-hcomm/04-comm-engines.md) · 5 [算法与 selector](docs/comm-ascend/hccl-hcomm/05-coll-algorithms.md) · 6 [AllReduce 调用链走读](docs/comm-ascend/hccl-hcomm/06-allreduce-call-chain.md) · 7 [executor 与 template](docs/comm-ascend/hccl-hcomm/07-executor-template.md) · 8 [资源地基与 dlsym](docs/comm-ascend/hccl-hcomm/08-resources-dlsym.md) · 9 [自定义算子七步流程](docs/comm-ascend/hccl-hcomm/09-custom-op-dev.md) · 10 [MC2 框架与官方样例](docs/comm-ascend/hccl-hcomm/10-mc2-custom-ops.md) · 11 [实战与第一个算子](docs/comm-ascend/hccl-hcomm/11-examples-first-op.md) · [总结｜源码阅读地图](docs/comm-ascend/hccl-hcomm/summary.md)

## 参考手册｜HCCL 知识库（80 篇）

[总导读](docs/comm-ascend/reference/index.md)，12 大类按需检索：

[总览](docs/comm-ascend/reference/overview/hccl-overview.md) · [算子参考](docs/comm-ascend/reference/operators/index.md) · [算法参考](docs/comm-ascend/reference/algorithms/index.md) · [组网形态](docs/comm-ascend/reference/networking/index.md) · [协议与传输](docs/comm-ascend/reference/comm/index.md) · [通信域](docs/comm-ascend/reference/communicator/index.md) · [通信引擎](docs/comm-ascend/reference/engine/index.md) · [执行模式](docs/comm-ascend/reference/execution/index.md) · [排障手册](docs/comm-ascend/reference/debug/index.md) · [性能分析](docs/comm-ascend/reference/profiling/index.md) · [运行与约束](docs/comm-ascend/reference/runtime/index.md) · [扩展机制](docs/comm-ascend/reference/extension/index.md)

## 实战复盘｜复盘文章 + 需求交付物 + 研讨

### 复盘文章（[专栏目录](docs/practice/index.md)）

| 篇 | 主题 |
| ---: | --- |
| 1 | [UBOE/UB_RTP 建链端口注入：从 Issue 到合入](docs/practice/01-uboe-ubrtp-port-injection.md) |
| 2 | [Reduce 64bit/特殊规约分级算法：集合通信与 HCCL 基础理论](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/index.md) |
| 3 | [AllGatherV/Reduce 零拷贝（对称内存）：从门禁到数据面](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/index.md) |

### 26RP4/01｜Reduce-64bit 与特殊规约分级算法（[复盘文](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/index.md)）

需求 AR20260930049502 全链路交付物（hccl 仓 `feature/reduce-64bit-prod-2level-hier` 分支，PR #3755 已合入）：

| 材料 | 说明 |
| --- | --- |
| [02-前置知识](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/02-前置知识.md) | 小白版理论向导 |
| [03-SE设计](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/03-SE设计.md) | SE 串讲版设计 |
| [04-SRS_SD](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/04-SRS_SD.md) | SRS/SD 全文：决策记录、风险表、死锁排查扩展节、cost 三轮迭代与实测数据 |
| [05-STC.xlsx](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/05-STC.xlsx) | 用例台账 58 条（ST 15 / UT 6 / Checker 25 / NPU 12），执行情况已反填 |
| [06-SRS_SD评审.xlsx](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/06-SRS_SD评审.xlsx) | 设计评审记录 9 条（7 闭环 2 开放） |
| [07-代码评审-PR3755.xlsx](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/07-代码评审-PR3755.xlsx) | PR #3755 代码评审记录 20 条（人工 6 + AI 去重 14） |
| [08-问题对齐](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/08-问题对齐-DPU-Reduce写非root用户输出buffer.md) | DPU-Reduce 写非 root 输出 buffer 存量缺陷对齐材料 |
| [09-经验总结](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/09-经验总结.md) | 组内知识分享串讲文档 |
| [scripts/](docs/practice/26RP4/01-Reduce-64bit与特殊规约分级算法/scripts) | NPU / ST / Checker 一键执行脚本 |

### 26RP4/02｜AllGatherV 与 Reduce 零拷贝模式（[复盘文](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/index.md)）

需求 AR20261029516478 全链路交付物（hccl 仓 `feature/zero-copy-agv-reduce` 分支）：

| 材料 | 说明 |
| --- | --- |
| [01-预调研](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/01-预调研.md) | 零拷贝机制现状摸底（file:line 事实与待确认项分离表述） |
| [02-前置知识](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/02-前置知识.md) | 对称内存小白版向导 |
| [03-SE设计](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/03-SE设计.md) | 四算子方案全文 |
| [04-SRS_SD](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/04-SRS_SD.md) | 需求与设计 |
| [05-STC.xlsx](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/05-STC.xlsx) | 用例与执行台账 |
| [06-SRS_SD评审.xlsx](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/06-SRS_SD评审.xlsx) | 设计评审记录 |
| [07-代码评审-PR3842.xlsx](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/07-代码评审-PR3842.xlsx) | PR #3842 代码评审记录 |
| [09-经验总结](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/09-经验总结.md) | 经验总结（含 Reduce 对称内存 in-place 边界 WAR 图解） |
| [scripts/](docs/practice/26RP4/02-AllGatherV与Reduce零拷贝模式/scripts) | 真机一键测试脚本 |

### 26RP4/03｜HCCL aclgraph 模式零拷贝方案（进行中）

| 材料 | 说明 |
| --- | --- |
| [03-SE设计](docs/practice/26RP4/03-HCCL-aclgraph模式零拷贝方案/03-SE设计.md) | SE 设计（进行中，尚无复盘文） |

### 研讨（[目录](docs/practice/研讨/allgather-selector-研讨/index.md)）

| 材料 | 说明 |
| --- | --- |
| [AllGather selector 代价模型研讨](docs/practice/研讨/allgather-selector-研讨/index.md) | 选路代价模型可视化：13 图 + corner case 扫描（CSV）+ 图表生成脚本 |
| [决策树与 Corner Case](docs/practice/研讨/allgather-selector-研讨/allgather-selector-decision-tree.md) | 完整决策树、老路径分支表、阈值表、代码行号索引 |
| [选择策略研究](docs/practice/研讨/allgather-selector-研讨/allgather-selector-costmodel-analysis.md) | 引入 cost model 后的算法选择策略研究 |

### 模板与方法论

| 材料 | 说明 |
| --- | --- |
| [SRS/SD 文档模板](docs/practice/00-模板库/01-SRS_SD文档模板.md) | 文档模板（评审 / STC / 代码评审 / 用例模板 xlsx 同目录） |
| [需求实验方法论参考](docs/practice/00-需求实验方法论参考.md) | 远程仿真实验全链路经验（环境/工具/观测/实验设计/踩坑） |
| [实验用例设计与脚本写法经验参考](docs/practice/00-实验用例设计与脚本写法经验参考.md) | UT / ST / 真机三层用例设计与脚本写法经验 |
| [复盘写作模板](docs/practice/template.md) | 复盘手记骨架与完稿自检清单 |

## 杂项

- **[工具与效率](docs/tools-productivity/index.md)**：[VS Code + SSH + Jupyter](docs/tools-productivity/vscode-remote-ssh-jupyter.md) · [Linux 服务器常用命令速查](docs/tools-productivity/linux-server-commands.md) · [Git 实用速查](docs/tools-productivity/git-cheatsheet.md) · [VS Code 快捷键速查](docs/tools-productivity/vscode-shortcuts.md) · [opencode 使用教程](docs/tools-productivity/opencode-tutorial.md) · [Ghostty 快捷键速查](docs/tools-productivity/ghostty-shortcuts.md) · [macOS 快捷键速查](docs/tools-productivity/macos-shortcuts.md) · [Vim 常用按键与技巧](docs/tools-productivity/vim-essentials.md)
- **[个人杂谈](docs/essays/index.md)**：读书笔记（[《金刚经》研读笔记](docs/essays/《金刚经》研读笔记.md) · [《金字塔原理》读书笔记](docs/essays/《金字塔原理》读书笔记.md) · [《认知觉醒》读书笔记](docs/essays/《认知觉醒》读书笔记.md) · [《当下的力量》读书笔记](docs/essays/《当下的力量》读书笔记.md)）｜AI 观察（[2026 热门 AI Agent 产品地图](docs/essays/01-popular-agents.md) · [AI Agent 工程化知识树](docs/essays/agent-engineering-knowledge-tree.md) · [AI Coding：从代码补全到智能体协作](docs/essays/ai-coding-codearts.md) · [实用提示词收集](docs/essays/实用提示词收集.md)）｜生活健康（[《健身解剖完全手册》学习笔记](docs/essays/《健身解剖完全手册》学习笔记.md) · [ADHD 日常管理指南](docs/essays/ADHD日常管理指南.md)）
- **[后端知识](docs/backend/index.md)**：目录

## 交付物命名约定

- **需求目录**：`NN-需求简称`，`NN` 按立项顺序编号；外层按迭代组织（当前 `26RP4/`）。
- **需求文档**：类型编号固定——`01-预调研`、`02-前置知识`、`03-SE设计`、`04-SRS_SD`、`05-STC`、`06-SRS_SD评审`、`07-代码评审`、`08-问题对齐`、`09-经验总结`；某需求没有的类型直接缺号，不为连续而重排。
- **模板库**：`NN-模板名` 只表示模板清单顺序，与需求文档的类型号无关。

> 历史：六专栏（model / systems / device / parallel / collective / ascend）已合并为三大专栏，旧路径由主题脚本自动重定向。
> 2026-11：原 GitCode my-cann 仓（design/note）已融合入 `docs/practice`；其 study 学习底稿即本站三大课程，以本站为准，不再单独维护。本仓同时推送 [GitHub](https://github.com/cassielee-tech/my-lib) 与 [GitCode](https://gitcode.com/cassielee-tech/my-lib)。

## 本地运行

```bash
npm install
npm run docs:dev
```

构建静态站点：

```bash
npm run docs:build
```

## License

MIT
