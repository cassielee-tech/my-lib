# 前置知识科普：Reduce 算子 64bit+特殊规约分级算法（小白版）

> **面向读者**：新入职开发——懂 C++ 和基本并行概念，不要求有集合通信或昇腾硬件背景。
> **阅读约定**：术语首次出现时给大白话解释；标〔代码事实〕的内容可在仓内按文件:行号核对；标〔打比方〕的内容仅为帮助理解，不是严谨定义。
> **配套文档**：同目录《reverse-walkthrough-64bit分级算子.md》（反串讲，进阶版）。

---

## 0. 一张图看懂这个需求

**需求一句话**：Reduce 算子遇到 64bit 数据类型（INT64/UINT64/FP64）或 PROD 规约、且是"跨机两级组网"时，目前只能用一种很慢的扁平算法；本需求给它换上"框内+框间"两级编排的分级算法，只求正确、不求快。

```
                 现状（慢，但正确）                本需求（新，仍以正确为先）
  ┌────────────────────────────┐    ┌────────────────────────────────────┐
  │  整个通信域扁平地跑一遍 NHR   │    │  框内先归约(RS-L0)                  │
  │  (不分框内/框间，所有卡全量   │ →  │  框间再归约(RS-L1)                  │
  │   gather 到本卡再逐元素算)   │    │  框间分发(AG-L1) → 框内分发(AG-L0)   │
  └────────────────────────────┘    └────────────────────────────────────┘
```

全文地图：第 1 章补集合通信概念 → 第 2 章补硬件组网与"分级" → 第 3 章看懂 HCCL 代码结构 → 第 4 章理解为什么 64bit/PROD 特殊 → 第 5 章回到需求本身。

---

## 1. 集合通信速成

### 1.1 rank 与通信域

- **通信域（HcclComm）**：一组一起干活的 NPU 卡。〔打比方〕一个讨论小组。
- **rank**：卡在通信域内的编号（0, 1, 2...），`rankSize` 是组的卡数。〔打比方〕组员的学号。
- **root**：Reduce 算子里被指定接收最终结果的那张卡。〔打比方〕课代表。

### 1.2 Reduce 算子做什么

把所有 rank 上的同形状数据**逐元素归约**，结果只落在 root 上。对比几个常被混淆的兄弟算子：

```
Reduce(4 rank, root=0)：        AllReduce：                     ReduceScatter(RS)：
 rank0: [1] ┐                   每人都得到完整归约结果             每人只得到 1/N 段归约结果
 rank1: [2] ├→ rank0 得 [10]     rank0~3 各得 [10][10][10][10]    rank0 得[10] r1 得[20]
 rank2: [3] ┤   (1+2+3+4)                                       rank2 得[30] r3 得[40]
 rank3: [4] ┘
```

**关键认知**：AllReduce、Reduce 在 HCCL 内部常被拆成 **ReduceScatter（先归约并散开）+ AllGather（再拼回全量）** 两段组合实现；本需求的"分级算法"正是在这两段上做文章。

### 1.3 归约操作（reduce op）

对数据的每个元素分别做四则/比较运算：SUM（求和）、PROD（求积）、MAX（取最大）、MIN（取最小）。例：`[1,5] × [2,3]` 做 PROD → `[2,15]`。

### 1.4 "64bit 类型"指什么

INT64 / UINT64 / FP64——每个元素 8 字节（对比 INT8/FP16/FP32 等"窄类型"）。数据类型不只是占多大内存的问题：**底层搬运和计算的硬件单元按"类型×操作"逐组合支持**，8 字节类型和乘法（PROD）恰好落在支持薄弱的组合上（第 4 章展开）。〔代码事实〕PROD 的类型白名单见 `op_common.cc:3422`（int8/int32/int64/uint64/fp16/fp32/fp64）。

---

## 2. 昇腾组网与"分级"

### 2.1 框内与框间：两级拓扑

多卡训练的典型组网是"两级"的：

```
   ┌─────────── 框（服务器）A ───────────┐      ┌─────────── 框 B ───────────┐
   │   NPU0 ↔ NPU1 ↔ NPU2 ↔ NPU3        │ RoCE │  NPU4 ↔ NPU5 ↔ NPU6 ↔ NPU7 │
   │        （HCCS 高速互联，             │ 网络 │        （同样 HCCS）         │
   │          框内带宽很高）              │(较慢) │                            │
   └────────────────────────────────────┘      └────────────────────────────┘
        level0（框内）                              level1（框间）
```

〔代码事实〕代码里用 `topoLevelNums` 表示层数：1=单机、2=跨机两级、3=超大规模三级。本需求只做两级。

### 2.2 分级算法：按层拆开执行

因为框内链路快、框间慢，聪明的做法是**按层拆开编排**，而不是把 8 张卡扁平看待。HCCL 的做法是把 Reduce 拆成 4 步〔代码事实，`reduce_sequence_executor_aicpu_3level.cc` 的编排顺序〕：

```
 RS-L0：框内先各自归约（每框先合出一版"框内小结"）
   ↓
 RS-L1：框间归约（各框的代表之间交换归约）
   ↓
 AG-L1：框间分发（把归约好的各段拼齐）
   ↓
 AG-L0：框内分发（拼回完整结果；Reduce 场景仅 root 写回用户输出内存）
```

〔打比方〕收作业：先在小组内互相对答案（框内），再由各小组课代表碰头合并（框间），最后把合并结果发回去（分发）。分级的好处是"长途运输"（框间）只运合并后的干货，量小得多。
官方原理说明见 `docs/zh/user_guide/coll_algo_intro/hierarchical_comm_principle.md`（延伸阅读）。

### 2.3 NHR 是什么

〔代码事实，官方定义见 `docs/zh/user_guide/coll_algo_intro/NHR.md`〕NHR（Nonuniform Hierarchical Ring，非均衡的层次环）：对 N 个节点构建生成树的最优通信关系，通信步数只有 ⌈log₂N⌉（比 Ring 的线性步数少得多），且无论节点数是否为 2 的幂都能吃满链路。常用于**框间/跨机**链路。
一句话记住：**NHR 是 HCCL 跨机链路上的主力搬运算法之一，本需求的新算法每个层级都用它搬数据**。Mesh（框内全互联直连）、Ring（环形传递）等家族成员见附录 B 的算法总览。

---

## 3. HCCL 软件结构（本需求视角）

### 3.1 一次 HcclReduce 调用的旅程

〔代码事实〕入口 `HcclReduce`（`include/hccl.h:250`）之后的分层：

```
 HcclReduce(...)                          对外 API
   └→ selector（选算法）                    "这次该用哪个算法？"——按拓扑/数据类型/规约操作点名
       └→ executor（编排）                  "按什么顺序跑哪些步骤？"——串行/并发/分层编排
           └→ template（模板，真正干活）      "具体怎么搬数据、怎么做规约"
               └→ wrapper（动作原语）        LocalCopy / LocalReduce / SendRecvWrite...
                   └→ HCOMM 基础通信库       独立仓 cann/hcomm，经 dlsym 动态加载
                       （真正的硬件搬运/规约指令在这里下发）
```

分层是架构硬约束：HCCL 不编译期依赖 HCOMM，跨仓调用全部走 `src/common/hcomm_dlsym/` 的符号表 + dlsym。

### 3.2 算法命名怎么读

算法名是有语法 的：`<引擎><算子><编排模式><框内算法><框间算法>`。拆一个真实例子〔代码事实，`ins_v2_all_reduce_sequence_executor_aicpu.cc:893`〕：

```
 Aicpu  +  AllReduce  +  Sequence  +  Mesh              +  NHRAicpuReduce
 └引擎      └算子         └串行按层编排   └框内用 Mesh 直连搬运   └框间用 NHR 搬 + aicpu 核规约
```

本需求新算法（SE 命名）：`AicpuReduceSequenceNHRAicpuReduceNHR` —— aicpu 引擎 + Reduce 算子 + 两级串行编排 + 框内/框间均用"NHR 搬运 + aicpu 规约"的模板。（注：命名尾段与组合的一致性在反串讲中列为待确认问题 3。）

常见编排模式速记：**Sole**=不分级、整域一段跑完；**Sequence**=按层串行；**Parallel**=多线程并发跑层；**Concur/PipeLine**=并发/流水变体。

### 3.3 注册机制与能力声明（attrs）

executor + 模板组合好后，用 `REGISTER_*` 宏登记进注册表，算法名即检索键；同时用 `REGISTER_ALG_ATTRS` 声明"能力边界"（attrs），例如：
- `topo.minTopoLevelNum / maxTopoLevelNum`：适用几级拓扑；
- `op.unsupportedDataTypes / supportedDataTypes`：支持/排除哪些数据类型；
- `op.isSupportProd`：是否支持 PROD 规约。

〔代码事实〕正是这些 attrs 把 64bit/PROD 挡在了现有分级算法门外（如 `reduce_sequence_executor_aicpu_3level.cc:838`）。

### 3.4 costModel：算法的"报价"机制

selector 之外还有一套按拓扑/数据量估算耗时的排序机制（cost table）。每个模板可以提供一组静态"报价系数"（`CalcCostCoeff`）；**没有报价系数的模板参与的算法不参与耗时排序**，只能由 selector 点名选用。〔打比方〕招标时没交报价单的公司不参与比价，只能被甲方直接指定。
〔代码事实〕本需求新算法的两个模板中，`InsTempReduceScatterAicpuReduceNHRPcie` 无报价系数、但 `InsTempAllGatherNHR` 有——聚合后是否会形成"部分报价单"参与误排序，是反串讲发现 1（高优先级风险），处置方案待评审拍板。

---

## 4. 为什么 64bit / PROD 特殊

### 4.1 三种通信引擎

〔代码事实，`docs/zh/architecture/architecture-brief.md`〕

| 引擎 | 一句话 | 特点 |
| ---- | ---- | ---- |
| AICPU_TS（本需求的"aicpu"） | AICPU 控制核运行通信 Kernel，TS 调度器下发硬件任务 | 不占计算核，适合大数据量 |
| AIV | Vector 向量核直接执行通信算子 | 低延迟，但占 Vector 核 |
| CCU | IO Die 上的硬化集合通信单元，微码执行 | 专用硬件 |

### 4.2 在线归约 vs 先搬后算：问题的根源

常规分级算法的层间模板（`InsTempReduceScatterNHR`）采用**在线归约**：数据在网卡搬运的途中"边收边算"（`HcommWriteReduceOnThread` 等原语，〔代码事实〕`ins_temp_reduce_scatter_nhr.cc:424-434`）。这套硬件在线归约路径**不支持 64bit 类型和 PROD**。

不支持时只剩一条路：**先搬完，再由 AICPU 控制核逐元素标量计算**——〔代码事实〕`alg_data_trans_wrapper.cc:905`，`LocalReduce` 遇到 INT64/UINT64/FP64/PROD 就分流到 `AicpuReduce`（:1319 的逐元素 for 循环）。

```
 在线归约（快，但不支持64bit/PROD）：  先搬后算（本需求路径，支持一切）：
 数据 ──SDMA搬运──┐                    数据 ──SDMA搬运──▶ 缓冲区
                  ├─边搬边算─▶ 结果                        │
                                                  AICPU控制核 for 循环逐元素算
                                                  （算完还要做"断批同步"）
```

### 4.3 断批同步（HcommBatchMode）

控制核做标量计算期间不能与批量任务描述符混排，模板里必须先 `HcommBatchModeEnd/Start` 清空批次、再 `HcommThreadJoin` 等待算完〔代码事实，`ins_temp_reduce_scatter_mesh_1D.cc:139-146`〕。〔打比方〕流水线上插了一段手工工序：得先把传送带上这批货清空，师傅干完活，再重启传送带。这就是"正确性优先、性能无要求"的由来——这套路径天然慢。

### 4.4 现状：回退路径慢在哪

〔代码事实〕两级拓扑 + 64bit/PROD 时，selector 一律回退到 `AicpuReduceSoleNHRAicpuReduce`（注册于 `reduce_sole_executor.cc:357`；路由于 `reduce_auto_selector.cc:281`）：整域扁平跑 NHR、gather 全量数据到本卡再标量算。通信量放大 N 倍、且不区分框内框间。**本需求就是把它升级为第 2.2 节的两级编排**。

---

## 5. 本需求方案一页纸

**改动极小**（SE 设计："只加注册、不改框架"）：

| # | 文件 | 改动 |
| - | ---- | ---- |
| 1 | `reduce_sequence_executor_aicpu_3level.cc` | 文件尾新增一个 `REGISTER_EXEC_V2_MULTI`（6 个模板位：RS-L0/RS-L1/RS-L2 占位/AG-L2 占位/AG-L1/AG-L0，两级注册、L2 会被 skipLevel2 跳过）+ attrs + 补一个 include |
| 2 | `reduce_auto_selector.cc` | :281 的回退分支改为"两级→新算法、三级→维持原样" |

**为什么能这么省**：编排框架（3level executor 支持两级注册）和干活模板（分级版 `InsTempReduceScatterAicpuReduceNHRPcie`、`InsTempAllGatherNHR`）都已存在并被其它算法使用，本次只是新增一种"模板组合"的注册。

**风险的大白话版**（详见反串讲第 5 节）：
1. cost 误选——"没交报价单的算法"可能形成部分报价被误上架比价（需补短路或确认不覆盖）；
2. 命名尾段与组合不一致，影响 HCCL_ALGO 环境变量按名指定；
3. 分轮数暴增可能触发超时误报，需 GB 级数据量用例覆盖；
4. 分级版模板首次作框内（L0）模板跑 HCCS 链路，零历史验证，双机 ST 重点覆盖。

---

## 附录 A. 术语速查表

| 术语 | 一句话 |
| ---- | ---- |
| rank / rankSize / root | 组员编号 / 组员总数 / 接收最终结果的卡 |
| 通信域（HcclComm） | 一起干活的一组 NPU 卡 |
| Reduce / AllReduce / RS / AG | 归约到 root / 归约到全员 / 归约并散开 / 收集拼全量 |
| SUM / PROD / MAX / MIN | 逐元素加 / 乘 / 取大 / 取小 |
| 64bit 类型 | INT64、UINT64、FP64（8 字节元素） |
| level0 / level1 / topoLevelNums | 框内 / 框间 / 拓扑层数（1 单机、2 跨机、3 超大规模） |
| HCCS / RoCE | 框内高速互联 / 框间以太网络 |
| NHR | 非均衡层次环，跨机主力搬运算法，步数 log₂N |
| selector / executor / template | 选算法 / 编排步骤 / 具体搬算 |
| attrs | 算法的能力自述（支持哪些拓扑/类型/操作） |
| Sole / Sequence / Parallel | 扁平一段 / 按层串行 / 多线程并发 |
| costModel / cost table | 按"报价系数"给算法排序的机制 |
| AICPU / AIV / CCU | 控制核引擎（本需求）/ 向量核引擎 / 硬化通信单元 |
| 在线归约 / AicpuReduce | 边搬边算（不支持 64bit/PROD）/ 控制核先搬后逐元素算 |
| HcommBatchMode / ThreadJoin | 断批清空 / 等待线程干完活（标量规约的配套同步） |
| wrapper | LocalCopy、LocalReduce、SendRecvWrite 等动作原语的封装层 |
| HCOMM / dlsym | 基础通信库（独立仓）/ 运行时动态加载符号的机制 |
| ccl buffer / scratch | 通信中转缓冲区 / 算法工作缓冲区（按 multiplier 申请） |
| slice / tailSize | 数据切片 / 最后一 rank 的尾块大小（除不尽时） |
| loop（分轮） | 数据超过单轮容量时分多轮执行 |
| inplace | 收发使用同一块用户内存 |
| 保序（DETERMINISTIC） | 相同输入必得完全相同的输出（浮点累加顺序固定） |
| 对称内存 | 各卡可互访的同一地址空间内存（可选加速特性） |
| task cache | 算子任务展开缓存（64bit/PROD 下当前禁用） |

## 附录 B. 延伸阅读（均在仓内）

| 文档 | 内容 |
| ---- | ---- |
| `docs/zh/user_guide/coll_algo_intro/algo_intro.md` | 集合通信算法总览（Mesh/Ring/RHD/NHR/Pipeline...） |
| `docs/zh/user_guide/coll_algo_intro/NHR.md` | NHR 官方定义、通信过程图与耗时公式 |
| `docs/zh/user_guide/coll_algo_intro/hierarchical_comm_principle.md` | 分级通信原理（官方图解） |
| `docs/zh/architecture/architecture-brief.md` | HCCL 软件分层、通信引擎、架构约束 |
| `docs/zh/user_guide/hccl_env/HCCL_ALGO.md` | HCCL_ALGO 环境变量：手动指定算法的方法 |
| 同目录《reverse-walkthrough-64bit分级算子.md》 | 本需求反串讲（论断核实与风险清单） |
