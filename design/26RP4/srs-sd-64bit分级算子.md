# 需求分析与软件设计（SRS/SD）：Reduce 算子 64bit/特殊规约两级分级算法

| 项目 | 内容 |
| ---- | ---- |
| 需求编号 | AR20260930049502 |
| 涉及算子 | Reduce（HcclReduce 单算子 / 图模式入口） |
| 引擎 / 拓扑 | aicpu（AICPU_TS）/ 两级拓扑（topoLevelNums == 2） |
| 关联文档 | 同目录《reverse-walkthrough-64bit分级算子.md》（反串讲）、《background-64bit分级算子-前置知识.md》（前置知识）；SE 串讲文档《plan-64bit分级算子.md》 |
| 事实基准 | hccl 仓内代码（关键论断均带 文件:行号，经反串讲核实） |

---

## 1 简介

### 1.1 目的

本文档是 AR20260930049502 的需求分析（SRS）与软件设计（SD），用于需求反串讲评审，并作为后续开发、测试与代码检视的基线依据。内容基于 SE 串讲文档与开发侧代码核实结论（见反串讲文档第 4 节核实表），凡引用代码处均标注 `文件:行号`。

### 1.2 范围

- **算子范围**：仅 Reduce。AllReduce / ReduceScatter 为同型兄弟需求，由他人负责；三算子 selector / executor 分文件、无共同改动点，本需求与其无代码耦合，仅交付节奏与验收口径需对齐（见 4.1.9）。
- **能力范围**：两级拓扑 + 64bit 数据类型（INT64/UINT64/FP64）或 PROD 规约（任意支持的数据类型）场景，算法路由从扁平 Sole 回退算法升级为两级分级 sequence 算法（nhr+nhr）。
- **明确不做**：三级拓扑、单机路由变化、aiv/ccu 引擎、新增 template、修改 executor/template 主体逻辑、修改 costModel 公式、HCOMM 跨仓变更。

## 2 总体概述

### 2.1 软件概述

#### 2.1.1 项目介绍

HCCL（Huawei Collective Communication Library）是 CANN 的核心集合通信库，对上提供 AllReduce/Broadcast/AllGather/ReduceScatter/Reduce 等集合通信原语，对下经 dlsym 动态加载独立仓 HCOMM 使能昇腾 NPU（架构约束见 `docs/zh/architecture/architecture-brief.md`）。

Reduce 算子（`HcclReduce`，`include/hccl.h:250`）将通信域内所有 rank 的同形状数据逐元素归约，结果仅落在 root rank。其内部普遍拆解为 ReduceScatter + AllGather 两段组合实现。

**本需求**：两级拓扑下 64bit/PROD 场景当前回退到扁平算法 `AicpuReduceSoleNHRAicpuReduce`（注册于 `reduce_sole_executor.cc:357`）——整域扁平跑 NHR、gather 全量数据到本卡后由 AICPU 控制核标量规约，通信量放大 rankSize 倍且不分框内/框间。本需求新增两级分级算法 `AicpuReduceSequenceNHRAicpuReduceNHR`（SE 命名，尾段待确认，见 3.1.5），按"框内→框间→框间分发→框内分发"编排，完全复用既有 executor/template，**只加注册、不改框架**。

一次 `HcclReduce` 调用经过 7 个阶段（详见前置知识文档 3.1），本需求改动位于阶段 3 与注册表：

```
 阶段 0  API 入口与分流          HcclReduce（reduce.cc:23）
 阶段 1  入参检查                ReduceInitAndCheck（reduce.cc:129）
 阶段 2  模式分流与参数构造      ReduceOutPlaceCommon（reduce.cc:219）
 阶段 3  算法选择 ★改动点1       Selector（op_common.cc:160）→ ReduceAutoSelector（:281 特判分支）
 阶段 4  资源计算                HcclExecOp（op_common.cc:812）→ HcclGetAlgRes（:1387）
 阶段 5  执行编排 ★改动点2 受益   executor->Orchestrate（RS-L0→RS-L1→AG-L1→AG-L0）
 阶段 6  模板与硬件执行          template KernelRun → wrapper → HCOMM（dlsym）
```

### 2.2 设计约束

| 编号 | 约束 | 说明 |
| ---- | ---- | ---- |
| C1 | 架构硬约束 | 分层依赖方向不可违反；HCCL 不得编译期依赖 `cann/hcomm`，跨仓调用仅走 `src/common/hcomm_dlsym/` 符号表 + dlsym。本需求无跨仓变更 |
| C2 | 需求自身约束 | 不新增 template、不修改 executor/template 主体逻辑、不改 costModel 公式（SE 设计"变更面最小"原则）；无性能要求，正确性优先 |
| C3 | 兼容性约束 | 三级拓扑维持 Sole 回退；单机各分支路由不变；非 64bit 的 min/max 走常规算法路径；保序（DETERMINISTIC_STRICT）优先拦截不受影响；`HCCL_ALGO` 指定机制对新算法名可用 |
| C4 | 版本约束 | 新注册置于 `#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)` 区（与同文件现有 aicpu sequence 注册一致）；无新增文件，不改 CMakeLists.txt |
| C5 | 编码与合规 | C++17；遵循根目录 `.clang-format`（120 列/4 空格）；编译无告警；通过 CANN 静态检查（CI codecheck）与 pre-commit（clang-format 18.1.8 + OAT）；本需求无新增源文件，不涉及许可头新增 |
| C6 | 结构约束 | 改动位于 `src/ops/reduce/` 标准算子结构（selector + algorithm/{executor,template}）之内，属既有结构扩展，非新算子落盘 |

## 3 需求分析—SRS

### 3.1 AR20260930049502 【HCCL】Reduce算子支持64bit/特殊规约两级分级算法

#### 3.1.1 介绍

**背景与现状**：aicpu 分级算法（Parallel/Sequence 家族）因层间模板采用在线归约原语（`HcommWriteReduceOnThread` 等，`ins_temp_reduce_scatter_nhr.cc:424-434`）而被 attrs 整体排除 64bit 类型与 PROD（`reduce_sequence_executor_aicpu_3level.cc:838` 等）；根因是 `LocalReduce` 对 64bit/PROD 走 AICPU 控制核标量规约（`alg_data_trans_wrapper.cc:905-908`），需 `HcommBatchModeEnd/Start + HcommThreadJoin` 断批同步，仅 aicpu_reduce 系列模板兼容该路径。

**问题**：两级及以上拓扑 + 64bit/PROD 场景只能回退 `AicpuReduceSoleNHRAicpuReduce`（selector 路由于 `reduce_auto_selector.cc:281`）：整域扁平、gather 全量数据（通信量放大 rankSize 倍）、不区分框内/框间链路。

**目标**：为两级拓扑新增"nhr+nhr"两级 sequence 算法，框内/框间均复用 aicpu_reduce 系列分级模板，正确承接 64bit（INT64/UINT64/FP64，SUM/PROD/MAX/MIN）与 PROD（任意支持 dtype）全组合；正确性优先，不设性能指标。

#### 3.1.2 输入

**API 入参**（与现状一致，无新增）：

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| sendBuf / recvBuf | void* | 发送/接收缓冲区 |
| count | uint64_t | 元素个数 |
| dataType | HcclDataType | 本需求关注 INT64/UINT64/FP64 |
| op | HcclReduceOp | SUM/PROD/MAX/MIN |
| root | uint32_t | 结果落地 rank |
| comm / stream | HcclComm / aclrtStream | 通信域与流 |

**新算法路由进入条件**（本次需求定义的核心输入条件）：

```
topoInfo->topoLevelNums == TOPO_LEVEL_NUM_2
且（ Is64BitDataType(dataType)            // INT64/UINT64/FP64，四类规约操作均承接
   ∥ reduceType == HCCL_REDUCE_PROD ）    // 任意支持 dtype（白名单 op_common.cc:3422）
```

非 64bit 的 min/max **不进入**新算法（常规算法已支持且性能更优）——SE 列为设计确认点，默认不改。环境输入：`HCCL_ALGO` 白名单可按算法名指定新算法。

#### 3.1.3 处理

| 编号 | 功能需求 | 说明 |
| ---- | ---- | ---- |
| FR1 | 新算法注册 | `reduce_sequence_executor_aicpu_3level.cc` 文件尾新增 `REGISTER_EXEC_V2_MULTI`（6 模板位）+ `REGISTER_ALG_ATTRS`（min/maxTopoLevelNum=2、isSupportInplace=false）+ 补 include |
| FR2 | selector 路由改造 | `reduce_auto_selector.cc:281` 特判分支：两级拓扑 → 新算法名；三级 → 维持 `AicpuReduceSoleNHRAicpuReduce`；单机分支全部不变 |
| FR3 | cost 隔离 | 新算法不得被 cost table 在常规场景误选。反串讲发现 1：Reduce 3level executor `CalcCostCoeff` 为拼接聚合、无"任一模板空则退出"短路（`reduce_sequence_executor_aicpu_3level.cc:78-110`），且 `InsTempAllGatherNHR` 有静态系数（`ins_temp_all_gather_nhr.cc:18`）→ 新算法将形成非空部分系数。处置三选一（推荐 executor 补短路，对齐 RS executor 模式 `ins_v2_reduce_scatter_sequence_executor_aicpu.cc:133-136`），**待评审拍板** |
| FR4 | 兼容性 | 保序模式优先拦截不变；`HCCL_ALGO` 可指定新算法名（命名需通过 `alg_parse.cc` 字符串解析，见 3.1.5）；对称内存行为随现有 `AicpuReduceSequenceMeshConcurNHR`（UT 验证项）；task cache 对 64bit/PROD 维持禁用（`aicpu_task_cache_policy.cc:198-209`） |
| FR5 | 回退与健壮性 | 资源不足触发 `FallbackOp` 回退链（`op_common.cc` HcclExecOp 内）；attrs 层 min/maxTopoLevelNum=2 过滤三级/单机误入；`TopoMatchTwoLevel` 匹配失败报错兜底 |

#### 3.1.4 输出

- **功能输出**：root rank 的 recvBuf 得到全量归约结果，数值与参考实现逐位一致；非 root rank 不写用户输出（AG-L0 仅 root 落 user output）。
- **日志输出**：selector 选中日志（"Algo match [算法名]"）、`HcclGetAlgRes` 资源分配日志、`OrchestrateLoop` 编排起止日志（均为既有格式，无新增日志项）。
- **测试输出**：UT（路由矩阵/注册完整性）与 ST（双机正确性矩阵）通过记录。

#### 3.1.5 约束分析

| 约束 | 分析与结论 |
| ---- | ---- |
| 无性能要求（C2） | 接受标量规约 + 断批同步的固有开销；但需验证 `CUSTOM_TIMEOUT`（`aicpu_timeout.h`）不因 loop 数增多误报超时（反串讲风险 3） |
| 不新增模板（C2） | 两级 RS 复用分级版 `InsTempReduceScatterAicpuReduceNHRPcie`（具备 GetAlgRank/尾块/stride/repeatNum 全部两级语义）；该模板此前仅作框间模板用于单机 pcie-mix 场景，**首次作 L0 模板跑框内 HCCS 链路，零历史验证**（反串讲风险 4），ST 双机重点覆盖 |
| 三级维持 Sole（C3） | attrs min/maxTopoLevelNum=2 在通信域初始化阶段过滤 + selector 三级分支维持原样，双重保证 |
| min/max 语义确认点 | 需求字面含 min/max；默认仅伴随 64bit 进入。若需求方要求非 64bit min/max 也切换，selector 判定改一行即可，不影响其它设计 |
| scratch 相乘 | NHRPcie `CalcScratchMultiple = rankSize*2`（`ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.cc:282`），两级 multiplier 相乘后单轮可处理数据量骤减、loop 增多；大数据量（GB 级）分轮正确性为 ST 必测项 |
| 命名待确认 | 按 `<引擎><算子>Sequence<框内RS><框间RS>` 惯例，两级均 NHRPcie 的新算法尾段应为 `NHRAicpuReduce`×2，SE 命名尾段为 `NHR`（反串讲问题 3）；需同步验证 `HCCL_ALGO` 解析 |

## 4 软件设计-SD

### 4.1 AR20260930049502 【HCCL】Reduce算子支持64bit/特殊规约两级分级算法

#### 4.1.1 功能描述

新增 aicpu 引擎、两级拓扑的 sequence 分级算法（复用 `ReduceSequenceExecutorAicpu3Level`，两级注册 + `skipLevel2_` 跳层，先例 `AicpuReduceSequenceMeshConcurNHR`，`reduce_sequence_executor_aicpu_3level.cc:841`）：

```
用户输入 ─▶ RS-L0 框内归约：InsTempReduceScatterAicpuReduceNHRPcie
              （NHR 交换 + AICPU LocalReduce + 断批同步）
        ─▶ RS-L1 框间归约：InsTempReduceScatterAicpuReduceNHRPcie
        ─▶ AG-L1 框间分发：InsTempAllGatherNHR
        ─▶ AG-L0 框内分发：InsTempAllGatherNHR（仅 root 写回用户 output）
数据超单轮容量时按 maxCountPerLoop 分 loop 循环执行上述 4 步
```

注册的 6 个模板位（L2 两位因 skipLevel2 不会被拉起，按现有惯例填同款占位）：

| 模板位 | 模板 | 说明 |
| ---- | ---- | ---- |
| T0（RS-L0） | InsTempReduceScatterAicpuReduceNHRPcie | 框内，首次作 L0 使用 |
| T1（RS-L1） | InsTempReduceScatterAicpuReduceNHRPcie | 框间 |
| T2（RS-L2） | InsTempReduceScatterAicpuReduceNHRPcie | 占位（skipLevel2） |
| T3（AG-L2） | InsTempAllGatherNHR | 占位（skipLevel2） |
| T4（AG-L1） | InsTempAllGatherNHR | 框间分发 |
| T5（AG-L0） | InsTempAllGatherNHR | 框内分发，root 输出 |

**能力矩阵**：两级拓扑 × {INT64/UINT64/FP64} × {SUM/PROD/MAX/MIN} 全承接；PROD@任意白名单 dtype；非 64bit 的 SUM/MIN/MAX 维持常规算法。

#### 4.1.2 流程描述

调用旅程与改动位置（阶段定义见 2.1.1）：

```
阶段 3（算法选择）★改动点 1 —— reduce_auto_selector.cc SelectAicpuAlgo（:281 附近）
    if (Is64BitDataType(dataType) || opParam.reduceType == HCCL_REDUCE_PROD) {
        if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_2) {
            selectAlgName = "AicpuReduceSequenceNHRAicpuReduceNHR";  // 新
        } else {
            selectAlgName = "AicpuReduceSoleNHRAicpuReduce";         // 三级维持
        }
    }
    // 其余分支（单机 Mesh/CLOS、保序拦截、非 64bit 常规分级）一律不动

注册表（编译期登记）★改动点 2 —— reduce_sequence_executor_aicpu_3level.cc 文件尾
    REGISTER_EXEC_V2_MULTI(HCCL_CMD_REDUCE, <新算法名>, ReduceSequenceExecutorAicpu3Level,
        TopoMatchTwoLevel, T0..T5 见 4.1.1)
    REGISTER_ALG_ATTRS(<新算法名>, topo.minTopoLevelNum = TOPO_LEVEL_NUM_2;
        topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2; op.isSupportInplace = false;)
    + #include "ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.h"
```

修改文件清单：

| # | 文件 | 改动 |
| - | ---- | ---- |
| 1 | `src/ops/reduce/algorithm/executor/reduce_sequence_executor_aicpu_3level.cc` | 新增注册宏 ×2 + include（置于 9.0.0 版本宏区） |
| 2 | `src/ops/reduce/selector/reduce_auto_selector.cc` | :281 特判分支两级/三级分流 |

（若 FR3 处置选 executor 补短路，则追加 `reduce_sequence_executor_aicpu_3level.cc` 的 `CalcCostCoeff` 短路逻辑，约 10 行。）

运行时关键路径：Selector 选中新算法名 → `HcclGetAlgRes` 经 `CalcAlgHierarchyInfoV2` 计算分级通信域（框内/框间分组）→ `CalcRes` 按模板 `CalcChannelRequestNhr` 分别建框内/框间链路 → `Orchestrate` 按 4.1.1 顺序逐层调用模板 `KernelRun`。

#### 4.1.3 数据描述

**OpParam 关键字段**（`ReduceConstructOpParam`，`reduce.cc:179` 组装）：

| 字段 | 取值 |
| ---- | ---- |
| DataDes.dataType / count | 用户入参（INT64/UINT64/FP64 等） |
| reduceType | SUM/PROD/MAX/MIN |
| root / userRank | 用户指定 / 本卡 rank |
| opType | HCCL_CMD_REDUCE |

**TemplateDataParams**：sliceSize / tailSize（尾块）/ outputSliceStride / repeatNum / buffInfo——与分级版模板语义完全对齐（`ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.cc` 全量支持），executor 编排代码无需修改。

**资源（AlgResourceRequest）**：NHRPcie 模板单线程（slaveThreadNum=0），sequence executor 对 thread/notify 取 max 合并；链路按框内/框间 subCommRanks 分建。

**scratch**：NHRPcie `CalcScratchMultiple = rankSize*2`，两级相乘（≈ 4×L0×L1）后单轮可处理数据量 `maxCountPerLoop` 相应变小，大数据量分多 loop；每 loop 的 RS 阶段伴随一次 `HcommBatchModeEnd/Start + HcommThreadJoin` 断批同步。

**输出数据流**：通信全程经 ccl buffer 中转；AG-L0 阶段仅 root rank 将结果写入用户 recvBuf。

#### 4.1.4 依赖性描述

| 依赖项 | 类型 | 是否修改 |
| ---- | ---- | ---- |
| `ReduceSequenceExecutorAicpu3Level`（编排/资源/层级计算） | 仓内 executor | 否，仅新增注册 |
| `InsTempReduceScatterAicpuReduceNHRPcie`（两级 RS 模板） | 仓内 template | 否 |
| `InsTempAllGatherNHR`（两级 AG 模板） | 仓内 template | 否 |
| HCOMM dlsym 符号：`HcommLocalReduceOnThread`、`HcommBatchModeEnd/Start`、`HcommThreadJoin`、SendRecv 读写系列 | 跨仓（cann/hcomm） | 否，全部为既有符号，无新增跨仓依赖 |
| `REGISTER_EXEC_V2_MULTI` / `REGISTER_ALG_ATTRS` 注册框架 | 仓内 op_common | 否 |
| AllReduce / ReduceScatter 同型需求 | 兄弟特性 | 无代码耦合（分文件）；FR3 若补短路需与其负责人对齐同一方案 |

#### 4.1.5 接口描述

- **对外接口**：零变化。`HcclReduce` / `HcclReduceGraphMode` 签名、参数语义、错误码（含 `HCCL_E_NOT_SUPPORT` 触发条件）均不变。
- **内部接口**：无新增函数/类。新增内容为两条注册（算法名进入 `CollAlgExecRegistryV2` 注册表），新算法名同时成为 `HCCL_ALGO` 环境变量可指定的字符串接口（需通过 `alg_parse.cc` 算法名分段解析，命名确认后验证）。

#### 4.1.6 使用限制

| 限制 | 机制 |
| ---- | ---- |
| 仅两级拓扑 | attrs min/maxTopoLevelNum=2（通信域初始化阶段过滤）+ selector 三级分支维持 Sole |
| 单机场景不进入 | selector 单机分支不变 |
| 非 64bit 的 min/max 不进入 | selector 判定条件（设计确认点，默认不改） |
| 不支持 inplace | `op.isSupportInplace = false`（与现有 Reduce 分级算法对齐） |
| task cache 不适用 | 64bit/PROD 维持既有禁用策略（`aicpu_task_cache_policy.cc`） |
| 保序场景不进入 | DETERMINISTIC_STRICT 优先拦截走 StrictOrdered 路径 |

#### 4.1.7 DFX 设计

- **可观测（日志）**：沿用既有日志体系——selector "Algo match"（HCCL_INFO）、`HcclGetAlgRes` 资源分配日志（线程/每层 channel 数）、`OrchestrateLoop` 起止、wrapper 层 TraceDataSlice（元信息）。可通过日志确认"走了新算法 + 两级编排"，无需新增日志项。
- **可测试**：`HCCL_ALGO` 按名强制指定新算法即可单算法验证；UT 入口 `build.sh -u`，ST 入口 `build.sh -s`（用例目录 `test/st/algorithm/testcase/reduce_*.cc`）。
- **可维护**：注册式扩展，不动框架；算法名遵循既有命名语法（尾段问题见 3.1.5）。
- **异常与回退**：三级/单机误入由 attrs 过滤；算法名查不到由注册表报错；执行期资源不足由 `FallbackOp` 回退链兜底（与既有算法一致）。

#### 4.1.8 资料描述

- 对外 API 资料（`docs/zh/api_ref/comm_op_interface/HcclReduce.md` 及英文版）已按产品粒度声明 sum/prod/max/min 与数据类型支持，本需求不改变对外能力口径，**无需更新**。
- 待确认项：内部实现粒度资料（`docs/zh/user_guide/comm_ops_support_list/`、算法介绍目录）是否需要同步——默认不需要（内部算法路径变化不改变用户可见行为）。

#### 4.1.9 性能&&质量

**性能**：不设指标（需求明确 aicpu 规约 + Sync 操作正确性优先）。可选保留：与回退算法 `AicpuReduceSoleNHRAicpuReduce` 的耗时对照数据，供后续性能需求参考。

**正确性验收矩阵（ST，双机两级环境）**：

| 维度 | 取值 |
| ---- | ---- |
| 数据类型 × 规约 | {INT64, UINT64, FP64} × {SUM, PROD, MIN, MAX}（12 组合）+ PROD@{int8,int32,fp16,fp32} |
| 数据形状 | 整除切分 / 尾块（非整除）/ GB 级大数据分 loop |
| 场景开关 | 对称内存开/关；inplace（预期不支持，校验拒绝路径）；root 非 0 rank |
| 回归 | 非 64bit 常规场景路由不变；三级/单机场景路由不变 |

**质量门禁**：`bash build.sh --pkg` 编译无告警；`build.sh -u` UT 通过（新增路由矩阵用例）；`build.sh -s` ST 通过；pre-commit（clang-format + OAT）与 CI codecheck 通过；PR 关联 Issue、按模板填写描述。

**风险与对策**（承接反串讲第 5 节）：

| 风险 | 等级 | 对策 |
| ---- | ---- | ---- |
| cost 误选（3level executor 无空短路 + AG 模板有系数） | 高 | FR3 处置方案评审拍板（推荐补短路）；UT 增加常规场景不选中新算法的断言 |
| NHRPcie 首次作 L0 模板跑框内 HCCS 链路 | 中 | ST 双机全矩阵覆盖；关注非 pcie 的 SendRecvWrite 路径 |
| loop 暴增误报 CUSTOM_TIMEOUT | 中 | GB 级数据量用例；必要时调整超时语义（另行评审） |
| 命名尾段与 HCCL_ALGO 解析 | 中 | 评审定稿命名后补解析验证用例 |
| Reduce 对称内存路径未特判 | 低 | 与现有两级算法行为一致，ST 开关覆盖；异常则 attrs opCustomCheck 排除并回退 Sole |

## 5 网络安全设计

| 维度 | 分析与结论 |
| ---- | ---- |
| 攻击面 | 无新增外部输入、网络端口、协议、落盘文件；输入仍为用户缓冲区指针与既有 API 参数，阶段 1 入参校验（指针/count/类型/PROD 白名单/root 合法性，`reduce.cc:129-155`）全部沿用 |
| 数据安全 | 通信数据仅在设备内存与 ccl buffer 中转，不落盘、不上报；日志仅输出元信息（指针地址/大小/类型/算法名/rank），不输出数据内容，符合日志规范敏感信息要求 |
| 凭据与遥测 | 无密钥/token/凭据引入；无新增遥测、分析上报或额外网络调用 |
| 第三方依赖 | 无新增 |
| 合规 | 通过仓内 OAT.xml、pre-commit 与 CI codecheck 校验即为合规结论 |
| 结论 | 本需求不引入新的网络安全暴露面 |
