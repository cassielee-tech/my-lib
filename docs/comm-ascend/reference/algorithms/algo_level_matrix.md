# 各算子分层算法选择矩阵（L0 / L1 / L2）

> **深度参考** · 前置阅读：[《集合通信》第 2 章（Ring）与第 3 章（Tree/选型）](/comm-ascend/02-ring-allreduce.md)。本篇为参考手册体例，检索导向；事实基线见文首引用块。

> HCCL 通信算法解析。整理各通信算子在 Server 内（L0）、Server 间（L1）、超节点间（L2）三个层级的默认算法选择、特点与适用场景。依据：`hccl/docs/zh/user_guide/coll_algo_intro/algo_intro.md`（官方算法简介与分级通信表）、`hccl/src/ops/scatter/scatter.cc`（SetAlgoLevel0/1/2 层级映射源码样本）、`hcomm/legacy/.../alg_configurator.cc`（910 系列）、`hccl/docs/zh/user_guide/hccl_env/HCCL_ALGO.md` 与 inter_server/inter_superpod 支持度列表、各算子 `selector/<op>_auto_selector.cc`。
>
> **L0/L1/L2 的实现原理**（每层的协议、通信语义、引擎、数据面原语、端到端数据流）见 [../03_L0L1L2分级通信实现原理.md](/comm-ascend/reference/overview/l0l1l2-principle.md)；**编排方式的执行语义**见 [combination.md](/comm-ascend/reference/algorithms/combination.md)；**算法名全量与选择/回退机制**见 [algo_selector.md](/comm-ascend/reference/algorithms/algo_selector.md)；**各算法原理与耗时公式**见 [ring](/comm-ascend/reference/algorithms/ring.md) / [fullmesh](/comm-ascend/reference/algorithms/fullmesh.md) / [hdr](/comm-ascend/reference/algorithms/hdr.md) / [star](/comm-ascend/reference/algorithms/star.md) / [nhr](/comm-ascend/reference/algorithms/nhr.md) / [nb](/comm-ascend/reference/algorithms/nb.md) / [ahc](/comm-ascend/reference/algorithms/ahc.md) / [pipeline](/comm-ascend/reference/algorithms/pipeline.md) / [pairwise](/comm-ascend/reference/algorithms/pairwise.md)。

## 1. 分层选择框架

| 层级 | 选择方式 | 可否配置 | 决策依据 |
| --- | --- | --- | --- |
| **L0（Server 内）** | 按硬件拓扑**自动选择** | 不可配（HCCL_ALGO 的 level0 仅支持 NA） | 每机卡数、链路形态（HCCS/SIO/PCIe） |
| **L1（Server 间）** | **自适应**选择 | 可配（HCCL_ALGO level1，配置后自适应失效） | 产品形态、Server 数、数据量 |
| **L2（超节点间）** | 自适应选择（仅 Atlas A3，且仅 AI_CPU 展开模式） | 可配（HCCL_ALGO level2） | 超节点个数、数据量 |

流程：`GetAlgType`（解析环境变量 → 未配置则走各层默认函数）→ `AlgType{L0,L1,L2}` → `SelectAlg` 输出组合算法名 → executor 执行。组合维度详见 [combination.md](/comm-ascend/reference/algorithms/combination.md)。

## 2. L0：Server 内算法

官方口径：Server 内支持 **Mesh、Ring、Double-Ring、Star**，按硬件拓扑自动选择（algo_intro.md）。

### 2.1 通用 Module 默认规则（`GetDefaultAlgoLevel0Module`）

| 每聚合域卡数 | 默认算法 | 说明 |
| --- | --- | --- |
| 8 卡 | `ALG_LEVEL0_8P_RING` | 8 卡 HCCS 环 |
| 4 卡 | `ALG_LEVEL0_4P_MESH` | 4 卡 Mesh（3 固定 stream） |
| 2 卡 / 1 卡 | `ALG_LEVEL0_NP_SINGLE_RING` | 单环 |
| 其他 | `ALG_LEVEL0_WHOLE_RING` | 与 L1/L2 共同组成大环 |
| 910B（HCCS 链路使能、非多 Module 异卡数） | `ALG_LEVEL0_NP_MESH` | 服务器内 3~8p 组 Mesh |
| 910_93 | HCCS_SW 数 = 2×SIO 数 → `NP_DOUBLE_RING`，否则 `NP_SINGLE_RING` | 双环吃满链路 |

### 2.2 标卡分支（见 [board_form_factor.md](/comm-ascend/reference/networking/board_form_factor.md)）

2 卡：910B 用 `NP_SINGLE_RING`、其余 `2P_MESH`；1 卡或 3~8 卡：`NP_SINGLE_RING`（`TOPO_TYPE_NP_SINGLE_RING` 仅用于标卡）。

### 2.3 950 系列（950PR/950DT）：Mesh 族细选

selector 按拓扑与数据量输出 Mesh 系算法名（可配 `HCCL_ALGO` 的 level0）：

| 算法名 | 选择条件（all_reduce_auto_selector.cc 示例） |
| --- | --- |
| `SoleMeshOneShot` | 小数据量一步直达 |
| `SoleMeshTwoShot` | 中数据量两步收发 |
| `SoleMesh` / `SoleMesh2Die` | 常规；跨 Die 场景 |
| `SequenceMesh2Die` | 跨 Die 顺序编排 |
| `SoleMeshConcur` / `ConcurMeshNHRMultiLink` | 多算法/多链路并发 |
| `ParallelMeshNHR` | 矩形拓扑大数据量（层内层间并行） |
| `PipeLineMeshNHR` | 流水编排 |

## 3. L1：Server 间算法

### 3.1 默认选择规则（`GetDefaultAlgoLevel1V1`）

| Server 数 | 默认算法 |
| --- | --- |
| ≥ 8（`HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM=8`） | **H-D_R** |
| 2 的整数次幂（非 1） | **H-D_R** |
| 非 2 的整数次幂（含 1） | **Ring** |
| 910_93 的 H-D_R 结果 | 替换为 **NHR** |

（HCCL_ALGO.md 口径：Atlas 训练系列非 2 幂默认 ring、其他 H-D_R，与代码一致。）

### 3.2 各算法特点与适用场景（algo_intro.md 权威描述）

| 算法 | 步数/复杂度 | 特点 | 适用场景 |
| --- | --- | --- | --- |
| **Ring** | 线性 | 通信关系简单，受网络拥塞影响小，时延相对高 | Server 少、数据量小、网络明显拥塞且 Pipeline 不适用 |
| **H-D_R**（RHD） | 对数 | 非整 2 幂节点规模引入额外通信量，时延相对低 | Server 数为 2 的整数次幂；或非 2 幂但数据量小 |
| **NHR** | 对数 | 非均衡层次环，自研 | Server 数较多且 Pipeline 不适用 |
| **NB** | 对数 | 非均匀数据块（Bruck 变体） | Server 数较多且 Pipeline 不适用 |
| **Pipeline** | 流水 | **并发使用 Server 内与 Server 间链路** | 数据量大、每机多卡；A2 下不支持确定性计算 |
| **Pairwise** | 线性 | 规避一打多（单端口发多 rank） | **仅 AlltoAll/AlltoAllV/AlltoAllVC**，数据量大需防一打多 |
| **AHC** | 层次 | 层次化拼接，层间带宽收敛时收益更好 | **仅 ReduceScatter/AllGather/AllReduce**；多层对称或非对称 NPU 分布 |

### 3.3 算子支持度（按产品，详见官方 inter_server_algo_support.md）

| 算法 | 950PR&950DT | Atlas A3 | Atlas A2 | Atlas 训练系列 |
| --- | --- | --- | --- | --- |
| **NHR** | RS/AG/AR/Bcast/Reduce/Scatter（唯一自适应算法） | RS/AG/AR/Bcast/RSV/Scatter/AGV | RS/AG/AR/Bcast/RSV/Scatter/AGV | RS/AG/AR/Bcast |
| **Ring** | —（NHR 自动接管） | RS/AG/AR/Reduce/RSV/Scatter/AGV | RS/AG/AR/Reduce/RSV/Scatter/AGV | RS/AG/AR/Reduce |
| **H-D_R** | — | —（NHR/ring 接管） | RS/AG/AR/Bcast/Reduce | RS/AG/AR/Bcast/Reduce |
| **NB** | — | RS/AG/AR/Bcast/RSV/Scatter/AGV | RS/AG/AR/Bcast/RSV/Scatter/AGV | RS/AG/AR/Bcast/RSV/Scatter/AGV |
| **NHR_V1** | — | — | RS/AG/AR/Bcast | RS/AG/AR/Bcast（逐步停用，建议 NHR） |
| **AHC** | — | RS/AG/AR | — | — |
| **Pipeline** | — | — | AR/AG/AGV/RS/AlltoAll 族 | — |
| **Pairwise** | — | AlltoAll 族 | AlltoAll 族 | — |
| **CP** | — | — | AlltoAllV（单算子） | — |

**Scatter 特殊约束**：Server 间仅支持 Ring/NHR/NB，其余算法重定向为 Ring（scatter.cc:652）；910_93 仅 Server 间 Ring。

## 4. L2：超节点间算法（仅 Atlas A3）

- 可配算法：**Ring、H-D_R、NHR、NB**（+AG/RS 的 Pipeline，仅单算子模式且开启零拷贝）；支持 AI_CPU 展开模式（`HCCL_OP_EXPANSION_MODE`）。
- 默认：代码 default 分支选 **NHR**（scatter.cc `SetAlgoLevel2` 注释"Level2默认选择NHR算法"）；HCCL_ALGO.md 口径：超节点个数 <8 且非 2 幂 → Ring，其他 → H-D_R。
- **AHC 联动**：L1 配置 AHC 时 L2 自动采用 AHC，显式配置不生效。
- 全部算法支持确定性计算（inter_superpod_algo_support.md）。

## 5. 各算子分级通信编排（两/三级拓扑的阶段表）

Atlas A2 单算子模式（两级拓扑，algo_intro.md 权威表）：

| 算子 | 阶段一 | 阶段二 | 阶段三 |
| --- | --- | --- | --- |
| **ReduceScatter** | Server 间 RS | Server 内 RS | — |
| **ReduceScatterV** | Server 间 RSV | Server 内 RSV | — |
| **AllGather** | Server 内 AG | Server 间 AG | — |
| **AllGatherV** | Server 内 AGV | Server 间 AGV | — |
| **AllReduce** | Server 内 RS | Server 间 AR | Server 内 AG |
| **Scatter** | Server 间 Scatter | Server 内 Scatter | — |
| **Broadcast** | Server 内 Scatter | Server 间 Broadcast | Server 内 AG |
| **Reduce** | Server 内 RS | Server 间 Reduce | Server 内 Gather（结果仅发 root） |
| **AlltoAll / V / VC** | Server 内 A2A | Server 间 A2A | — |

编排要点：归约/散布类先层间后层内（保证 Server 间数据块连续），聚合类先层内后层间；AllReduce 把大数据量阶段放高带宽的 Server 内。950 Sequence 执行器即此编排的四级实例：框内 RS → 框间 RS → 框间 AG → 框内 AG（见 [combination.md 2.1.2](/comm-ascend/reference/algorithms/combination.md)）。三阶段顺序表详见 [algo_selector.md](/comm-ascend/reference/algorithms/algo_selector.md)。

## 6. 按组网形态汇总

| 组网形态 | 生效层级 | 各层典型算法 | 见组网篇 |
| --- | --- | --- | --- |
| 单服务器（8 卡 HCCS） | 仅 L0 | 8P_RING / 910B NP_MESH / 950 Mesh 族 | [server_internal.md](/comm-ascend/reference/networking/server_internal.md) |
| 服务器间 RoCE 集群 | L0 + L1 | L0 按 2.1/2.3；L1 默认 H-D_R/Ring，可配 NHR/NB/Pipeline/Pairwise/AHC | [roce_cluster.md](/comm-ascend/reference/networking/roce_cluster.md) |
| UB 超节点（A3） | L0 + L1 + L2 | L2 Ring/H-D_R/NHR/NB（AI_CPU 模式），NHR-HCF 处理非对称超节点 | [ub_superpod.md](/comm-ascend/reference/networking/ub_superpod.md) |
| 950 UB 直连 | L0 + L1（Mesh×NHR 组合） | `ParallelMeshNHR` 等，CCU 引擎 | [ub_superpod.md](/comm-ascend/reference/networking/ub_superpod.md) |
| 标卡（PCIe） | 仅 L0 | NP_SINGLE_RING / 2P_MESH | [board_form_factor.md](/comm-ascend/reference/networking/board_form_factor.md) |

## 7. 特殊算法

- **分组 Full Mesh**：仅 A3 的 AlltoAll/AlltoAllV/AlltoAllVC，集群规模较大时按并发度分组完成——超节点内并发度高时延小、超节点间并发度低（防一打多）。不可通过 HCCL_ALGO 配置。
- **NHR-HCF**（NHR Highest Common Factor，最大公约数）：仅 A3，超节点间 **Server 数不一致但 Server 内卡数一致**时默认生效，不可配置。按超节点间 Server 数的最大公约数切分为多个对称逻辑超节点，再按新拓扑选算法；"最大公约数 > 1"场景收益更好。
- **CP 算法**：仅 A2 的 AlltoAllV（单算子模式），不支持时回退 Pairwise。

## 8. 参考资料

- `hccl/docs/zh/user_guide/coll_algo_intro/algo_intro.md`：算法简介、特点、α–β 耗时模型与分级通信权威表
- `hccl/docs/zh/user_guide/coll_algo_intro/hierarchical_comm_principle.md`：RS/AG/AR 分级流程图解
- `hccl/docs/zh/user_guide/hccl_env/HCCL_ALGO.md`、`inter_server_algo_support.md`、`inter_superpod_algo_support.md`：配置语法与按产品支持度
- `hccl/src/ops/scatter/scatter.cc`：SetAlgoLevel0/1/2 与 GetDefaultAlgoLevel0Module/Level1V1 层级映射源码
- `hcomm/src/legacy/ascend910/algorithm/impl/alg_configurator.cc`：910 系列层级算法配置
- 各算子 selector：`hccl/src/ops/*/selector/`（950 算法名细选）

