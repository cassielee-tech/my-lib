# CassieLee 的知识库

一个基于 VitePress 搭建的个人知识博客。核心内容是 **HCCL 开发主线**——面向 CANN 集合通信算子开发岗位的六站课程，以及日常需求中的交付产物。

**在线阅读**：[cassielee-tech.github.io/my-lib](https://cassielee-tech.github.io/my-lib/) 

## 仓库结构

```text
my-lib/
├── docs/
│   ├── index.md / now.md            # 首页导览 / Now 页（当前在做什么）
│   ├── 模型与系统/               # 课程①：模型与系统
│   ├── 执行与并行/          # 课程②：执行与并行
│   ├── 通信与昇腾/                 # 课程③：通信与昇腾
│   │   ├── 源码课程/              #   ├ HCCL/HCOMM 双仓源码课程 
│   │   └── 参考手册/               #   └ HCCL 参考手册 
│   ├── 实战复盘/                    # 实战复盘（四个子专栏）
│   │   ├── 00-模板与方法论/          #   ├ 模板与方法论
│   │   ├── 01-需求迭代/26RP4/        #   ├ 需求迭代（需求交付物，按迭代组织）
│   │   ├── 02-问题修复/              #   ├ 问题修复（Issue → 合入）
│   │   └── 03-研讨笔记/              #   └ 研讨笔记（专题研究）
│   ├── 后端知识/                     # 后端知识
│   ├── 工具与效率/          # 工具与效率
│   ├── 个人杂谈/                      # 个人杂谈
│   └── public/                      # 站点静态资源（图片等）
├── .github/workflows/               # GitHub Pages 自动部署
├── docs/.vitepress/config.ts        # VitePress 配置（导航与侧栏）
└── package.json                     # VitePress 站点工程
```

## 内容导览

博客分四类内容：**课程**（按顺序学）、**参考手册**（按需查）、**实战复盘**（真实案例）、**杂项**。

| 内容类型 | 回答的问题 | 入口 |
| --- | --- | --- |
| **课程** | 从零建立心智模型 | [模型与系统](docs/模型与系统/index.md) · [执行与并行](docs/执行与并行/index.md) · [通信与昇腾](docs/通信与昇腾/index.md) |
| **参考手册** | 这个算子/错误码/组网怎么查 | [HCCL 参考手册](docs/通信与昇腾/参考手册/index.md)（80 篇） |
| **实战复盘** | 真实需求怎么从 Issue 走到合入 | [实战复盘](docs/实战复盘/index.md)（含需求交付物与研讨） |
| **杂项** | 后端、工具速查、个人思考 | [后端知识](docs/后端知识/index.md) · [工具与效率](docs/工具与效率/index.md) · [个人杂谈](docs/个人杂谈/index.md) |

---

## 课程｜三大专栏 

### [模型与系统](docs/模型与系统/index.md)｜模型在算什么？训练怎样运转？

- **上篇·模型全景（6 章）**：1 [从模型到集合通信](docs/模型与系统/01-从模型到集合通信.md) · 2 [张量、计算图与反向传播](docs/模型与系统/02-张量、计算图与反向传播.md) · 3 [Tokenization 与 Embedding](docs/模型与系统/03-模型输入.md) · 4 [Self-Attention](docs/模型与系统/04-self-attention.md) · 5 [完整 Decoder Block](docs/模型与系统/05-完整Decoder-Block.md) · 6 [RoPE 与长上下文（选修）](docs/模型与系统/06-RoPE与长上下文.md) · ⚡ [速通版](docs/模型与系统/00-速通-上篇.md)
- **下篇·训练与推理系统（6 章）**：1 [训练循环](docs/模型与系统/01-训练循环.md) · 2 [训练显存与计算优化](docs/模型与系统/02-训练显存与计算优化.md) · 3 [推理与 KV Cache（选修）](docs/模型与系统/03-推理与KV-Cache.md) · 4 [后训练与对齐（选修）](docs/模型与系统/04-后训练与对齐.md) · 5 [推理引擎与 KV Cache（选修）](docs/模型与系统/05-推理引擎与KV-Cache.md) · 6 [PD 调度与推理指标（选修）](docs/模型与系统/06-PD调度与推理指标.md) · ⚡ [速通版](docs/模型与系统/00-速通-下篇.md)

### [执行与并行](docs/执行与并行/index.md)｜单卡怎样跑得快？并行为什么产生通信？

- **上篇·单卡执行系统（8 章 + 专题）**：1 [全景与性能分析](docs/执行与并行/01-全景与性能分析.md) · 2 [GPU/NPU 执行模型](docs/执行与并行/02-GPU与NPU执行模型.md) · 3 [存储层次与数据搬运](docs/执行与并行/03-存储层次与数据搬运.md) · 4 [Kernel、Tiling 与流水线](docs/执行与并行/04-Kernel、Tiling与流水线.md) · 5 [FLOPs、带宽与 Roofline](docs/执行与并行/05-FLOPs、带宽与Roofline.md) · 6 [Eager、计算图与图编译](docs/执行与并行/06-Eager、计算图与图编译.md) · 7 [Stream、Event 与异步执行](docs/执行与并行/07-Stream、Event与异步执行.md) · 8 [数据类型、布局与融合](docs/执行与并行/08-数据类型、布局与融合.md) · 专题 [2026 主流 AI 加速卡全景](docs/执行与并行/主流AI加速卡全景.md) · ⚡ [速通版](docs/执行与并行/00-速通-上篇.md)
- **下篇·并行策略（9 章）**：1 [MoE 与 Expert Parallel](docs/执行与并行/01-moe-expert-parallel.md) · 2 [效率设计（选修）](docs/执行与并行/02-效率设计.md) · 3 [系统分析](docs/执行与并行/03-大模型系统拆解.md) · 4 [分布式基础](docs/执行与并行/04-分布式基础.md) · 5 [数据并行与 DDP](docs/执行与并行/05-数据并行与DDP.md) · 6 [Tensor Parallel](docs/执行与并行/06-tensor-parallel.md) · 7 [Pipeline Parallel](docs/执行与并行/07-pipeline-parallel.md) · 8 [Context Parallel](docs/执行与并行/08-context-parallel.md) · 9 [ZeRO、FSDP 与混合并行](docs/执行与并行/09-ZeRO、FSDP与混合并行.md) · ⚡ [速通版](docs/执行与并行/00-速通-下篇.md)

### [通信与昇腾](docs/通信与昇腾/index.md)｜通信算法与代价模型；昇腾平台与源码落地（岗位核心）

- **上篇·集合通信（5 章）**：1 [Collective 语义与代价模型](docs/通信与昇腾/01-Collective语义与代价模型.md) · 2 [Ring AllReduce 完整推导](docs/通信与昇腾/02-Ring-AllReduce推导.md) · 3 [Tree、RD 与算法选择](docs/通信与昇腾/03-Tree、RD与算法选择.md) · 4 [AG、RS 与 AlltoAll](docs/通信与昇腾/04-gather-scatter-alltoall.md) · 5 [拓扑、分层与重叠](docs/通信与昇腾/05-拓扑、分层与重叠.md) · ⚡ [速通版](docs/通信与昇腾/00-速通.md)
- **下篇·昇腾平台**：0 [认识昇腾与 CANN](docs/通信与昇腾/00-认识昇腾与CANN.md) · 1 [Runtime 与任务执行](docs/通信与昇腾/01-Runtime与任务执行.md) · 2 [从 PyTorch 走向 HCCL](docs/通信与昇腾/02-从PyTorch走向HCCL.md) · [CANN 常用命令速查](docs/通信与昇腾/CANN常用命令.md) · [CANN Learning Hub](docs/通信与昇腾/cann-learning-hub.md)
- **源码课程｜[HCCL 与 HCOMM 导学](docs/通信与昇腾/源码课程/index.md)（13 单元）**：0 [双仓全景与调用链](docs/通信与昇腾/源码课程/00-双仓全景与调用链旅程.md) · 1 [软件架构与六族 API](docs/通信与昇腾/源码课程/01-软件架构与六族API.md) · 2 [控制面：通信域的一生](docs/通信与昇腾/源码课程/02-控制面-通信域的一生.md) · 3 [数据面：原语与资源](docs/通信与昇腾/源码课程/03-数据面-原语与资源.md) · 4 [通信引擎与三引擎编排](docs/通信与昇腾/源码课程/04-通信引擎与三引擎编排.md) · 5 [算法与 selector](docs/通信与昇腾/源码课程/05-算法与selector.md) · 6 [AllReduce 调用链走读](docs/通信与昇腾/源码课程/06-AllReduce调用链走读.md) · 7 [executor 与 template](docs/通信与昇腾/源码课程/07-executor-template.md) · 8 [资源地基与 dlsym](docs/通信与昇腾/源码课程/08-资源地基与dlsym.md) · 9 [自定义算子七步流程](docs/通信与昇腾/源码课程/09-自定义算子七步流程.md) · 10 [MC2 框架与官方样例](docs/通信与昇腾/源码课程/10-MC2框架与官方样例.md) · 11 [实战与第一个算子](docs/通信与昇腾/源码课程/11-实战与第一个算子.md) · [总结｜源码阅读地图](docs/通信与昇腾/源码课程/总结-源码阅读地图.md)

## 参考手册｜HCCL 知识库（80 篇）

[总导读](docs/通信与昇腾/参考手册/index.md)，12 大类按需检索：

[总览](docs/通信与昇腾/参考手册/总览/HCCL概述.md) · [算子参考](docs/通信与昇腾/参考手册/算子参考/index.md) · [算法参考](docs/通信与昇腾/参考手册/算法参考/index.md) · [组网形态](docs/通信与昇腾/参考手册/组网形态/index.md) · [协议与传输](docs/通信与昇腾/参考手册/协议与传输/index.md) · [通信域](docs/通信与昇腾/参考手册/通信域/index.md) · [通信引擎](docs/通信与昇腾/参考手册/通信引擎/index.md) · [执行模式](docs/通信与昇腾/参考手册/执行模式/index.md) · [排障手册](docs/通信与昇腾/参考手册/排障手册/index.md) · [性能分析](docs/通信与昇腾/参考手册/性能分析/index.md) · [运行与约束](docs/通信与昇腾/参考手册/运行与约束/index.md) · [扩展机制](docs/通信与昇腾/参考手册/扩展机制/index.md)

## 实战复盘｜四个子专栏

[专栏目录](docs/实战复盘/index.md)——**模板与方法论**（怎么写、怎么做实验）、**需求迭代**（立项需求全链路交付物）、**问题修复**（Issue 驱动的修复实战）、**研讨笔记**（专题研讨过程记录）。

### 模板与方法论（`docs/实战复盘/00-模板与方法论/`）

| 材料 | 说明 |
| --- | --- |
| [复盘写作模板](docs/实战复盘/00-模板与方法论/复盘写作模板.md) | 复盘手记骨架与完稿自检清单 |
| [SRS/SD 文档模板](docs/实战复盘/00-模板与方法论/01-SRS_SD文档模板.md) | 文档模板（评审 / STC / 代码评审 / 用例模板 xlsx 同目录） |
| [需求实验方法论参考](docs/实战复盘/00-模板与方法论/需求实验方法论参考.md) | 远程仿真实验全链路经验（环境/工具/观测/实验设计/踩坑） |
| [实验用例设计与脚本写法经验参考](docs/实战复盘/00-模板与方法论/实验用例设计与脚本写法经验参考.md) | UT / ST / 真机三层用例设计与脚本写法经验 |

### 需求迭代（`docs/实战复盘/01-需求迭代/`）

#### 26RP4/01｜Reduce-64bit 与特殊规约分级算法（[复盘文](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/index.md)）

需求 AR20260930049502 全链路交付物（hccl 仓 `feature/reduce-64bit-prod-2level-hier` 分支，PR #3755 已合入）：

| 材料 | 说明 |
| --- | --- |
| [02-前置知识](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/02-前置知识.md) | 小白版理论向导 |
| [03-SE设计](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/03-SE设计.md) | SE 串讲版设计 |
| [04-SRS_SD](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/04-SRS_SD.md) | SRS/SD 全文：决策记录、风险表、死锁排查扩展节、cost 三轮迭代与实测数据 |
| [05-STC.xlsx](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/05-STC.xlsx) | 用例台账 58 条（ST 15 / UT 6 / Checker 25 / NPU 12），执行情况已反填 |
| [06-SRS_SD评审.xlsx](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/06-SRS_SD评审.xlsx) | 设计评审记录 9 条（7 闭环 2 开放） |
| [07-代码评审-PR3755.xlsx](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/07-代码评审-PR3755.xlsx) | PR #3755 代码评审记录 20 条（人工 6 + AI 去重 14） |
| [08-问题对齐](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/08-问题对齐-DPU-Reduce写非root用户输出buffer.md) | DPU-Reduce 写非 root 输出 buffer 存量缺陷对齐材料 |
| [09-经验总结](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/09-经验总结.md) | 组内知识分享串讲文档 |
| [scripts/](docs/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/scripts) | NPU / ST / Checker 一键执行脚本 |

#### 26RP4/02｜AllGatherV 与 Reduce 零拷贝模式（[复盘文](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/index.md)）

需求 AR20261029516478 全链路交付物（hccl 仓 `feature/zero-copy-agv-reduce` 分支）：

| 材料 | 说明 |
| --- | --- |
| [01-预调研](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/01-预调研.md) | 零拷贝机制现状摸底（file:line 事实与待确认项分离表述） |
| [02-前置知识](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/02-前置知识.md) | 对称内存小白版向导 |
| [03-SE设计](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/03-SE设计.md) | 四算子方案全文 |
| [04-SRS_SD](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/04-SRS_SD.md) | 需求与设计 |
| [05-STC.xlsx](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/05-STC.xlsx) | 用例与执行台账 |
| [06-SRS_SD评审.xlsx](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/06-SRS_SD评审.xlsx) | 设计评审记录 |
| [07-代码评审-PR3842.xlsx](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/07-代码评审-PR3842.xlsx) | PR #3842 代码评审记录 |
| [09-经验总结](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/09-经验总结.md) | 经验总结（含 Reduce 对称内存 in-place 边界 WAR 图解） |
| [scripts/](docs/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/scripts) | 真机一键测试脚本 |

### 问题修复（`docs/实战复盘/02-问题修复/`）

| 篇 | 主题 |
| ---: | --- |
| 1 | [UBOE/UB_RTP 建链端口注入：从 Issue 到合入](docs/实战复盘/02-问题修复/01-UBOE与UB_RTP建链端口注入实战.md) |
| 2 | [HCCL aclgraph 模式零拷贝方案](docs/实战复盘/02-问题修复/02-HCCL-aclgraph模式零拷贝方案/03-SE设计.md)（进行中，SE 设计） |

### 研讨笔记｜AllGather Selector 算法选择策略研究（`docs/实战复盘/03-研讨笔记/`）

| 材料 | 说明 |
| --- | --- |
| [AllGather 选算法：从阈值树到代价模型](docs/实战复盘/03-研讨笔记/allgather-selector-算法选择策略研究/index.md) | 合集版：决策树、代价模型、util 阶梯、7 组 corner case、10 分钟分享稿、16 张图、交互式计算器与 PDF |

## 杂项

- **[工具与效率](docs/工具与效率/index.md)**：[VS Code + SSH + Jupyter](docs/工具与效率/vscode-remote-ssh-jupyter.md) · [Linux 服务器常用命令速查](docs/工具与效率/Linux服务器常用命令速查.md) · [Git 实用速查](docs/工具与效率/Git实用速查.md) · [VS Code 快捷键速查](docs/工具与效率/VSCode快捷键速查.md) · [opencode 使用教程](docs/工具与效率/opencode使用教程.md) · [Ghostty 快捷键速查](docs/工具与效率/Ghostty快捷键速查.md) · [macOS 快捷键速查](docs/工具与效率/macOS快捷键速查.md) · [Vim 常用按键与技巧](docs/工具与效率/Vim常用按键与技巧.md)
- **[个人杂谈](docs/个人杂谈/index.md)**：读书笔记（[《金刚经》研读笔记](docs/个人杂谈/《金刚经》研读笔记.md) · [《金字塔原理》读书笔记](docs/个人杂谈/《金字塔原理》读书笔记.md) · [《认知觉醒》读书笔记](docs/个人杂谈/《认知觉醒》读书笔记.md) · [《当下的力量》读书笔记](docs/个人杂谈/《当下的力量》读书笔记.md)）｜AI 观察（[2026 热门 AI Agent 产品地图](docs/个人杂谈/01-2026热门AIAgent产品地图.md) · [AI Agent 工程化知识树](docs/个人杂谈/AIAgent工程化知识树.md) · [AI Coding：从代码补全到智能体协作](docs/个人杂谈/AICoding-从代码补全到智能体协作.md) · [实用提示词收集](docs/个人杂谈/实用提示词收集.md)）｜生活健康（[《健身解剖完全手册》学习笔记](docs/个人杂谈/《健身解剖完全手册》学习笔记.md) · [ADHD 日常管理指南](docs/个人杂谈/ADHD日常管理指南.md)）
- **[后端知识](docs/后端知识/index.md)**：目录

## 交付物命名约定

- **需求迭代**：外层按迭代组织（当前 `26RP4/`）；需求目录 `NN-需求简称`，`NN` 按立项顺序编号。
- **需求文档**：类型编号固定——`01-预调研`、`02-前置知识`、`03-SE设计`、`04-SRS_SD`、`05-STC`、`06-SRS_SD评审`、`07-代码评审`、`08-问题对齐`、`09-经验总结`；某需求没有的类型直接缺号，不为连续而重排。
- **问题修复**：`NN-主题`，按修复顺序编号。
- **模板**：`NN-模板名` 只表示模板清单顺序，与需求文档的类型号无关。

> 历史：六专栏（model / systems / device / parallel / collective / ascend）已合并为三大专栏，旧路径由主题脚本自动重定向。
> 2026-11：原 GitCode my-cann 仓（design/note）已融合入 `docs/实战复盘`；其 study 学习底稿即本站三大课程，以本站为准，不再单独维护。本仓同时推送 [GitHub](https://github.com/cassielee-tech/my-lib) 与 [GitCode](https://gitcode.com/cassielee-tech/my-lib)。

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
