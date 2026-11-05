# 1.AR20261029516478 AllGatherV（SRS）

## 1.1 介绍

`HcclAllGatherV` 将各 rank 长度可不同的数据汇聚到所有 rank。当前 AICPU 非对称路径需要先把跨卡数据写入对端 CCL，再由对端 `PostLocalCopy` 到 userOut，存在额外本地拷贝；大数据还会受 CCL scratch 与 256MB 单轮上限约束。

本需求在 OPBASE 下打通 AllGatherV 的对称内存路径：各 rank 通过对称窗口直接读写对端 userOut，仅保留一次自贡献本地拷贝，从而消除接收侧 CCL→userOut 拷贝，并去掉 256MB/scratch 的单轮限制。

## 1.2 输入

**API 入参**（与现状一致，无新增）：

| 参数 | 类型 | 说明 |
|------|------|------|
| sendBuf | void* | 本端发送缓冲区 |
| sendCount | uint64_t | 本端发送元素数 |
| recvBuf | void* | 接收缓冲区 |
| recvCounts | const void* | 各 rank 接收元素数 |
| recvDispls | const void* | 各 rank 在 recvBuf 中的偏移 |
| dataType | HcclDataType | 数据类型 |
| comm / stream | HcclComm / aclrtStream | 通信域与流 |

**进入零拷贝条件**：

- HCOMM ≥ 9.1.0；
- `opMode == OPBASE`；
- `engine == COMM_ENGINE_AICPU_TS`；
- `topoInfo->level0Topo == MESH_1D`；
- `algName == AicpuAllGatherVSoleMesh`；
- input 与 output 均注册对称内存（双侧探测成功）。

## 1.3 处理

1. 修正 `outputSize` 口径为 `max(recvDispls[i] + recvCounts[i]) * typeSize`，并校验 displs 单调、非空区间不重叠和溢出。
2. `Selector` 选出 `AicpuAllGatherVSoleMesh` 后，由 `HcclExecOp → RefreshSymmetricMemory` 完成单点门禁与双侧窗口探测。
3. 对称生效时，执行器设置 `loopTimes=1`、`scratchMultiple=0`，并把 `maxCountPerLoop` 放大到本次最大 `recvCounts`。
4. `InsTempAllGatherVMesh1D` 保留一次自贡献本地拷贝，N-1 个对端线程并发直写对端 `userOut[displs[myRank]]`，跳过对端 `PostLocalCopy`。

## 1.4 输出

- 各 rank `recvBuf` 按 `recvDispls` 布局，内容与非对称路径逐 bit 一致。
- 不写 `recvBuf` 落点之外的区域。
- 对称路径不污染 `sendBuf`。

## 1.5 约束分析

| 支持的算子名称 | `HcclAllGatherV` |
| --- | --- |
| 支持的算法名称 | `AicpuAllGatherVSoleMesh` |
| 支持的芯片类型 | Ascend 950PR/950DT、Atlas A3 系列（以产品支持表为准） |
| 支持的展开模式 | AI CPU（`COMM_ENGINE_AICPU_TS`） |
| 支持的拓扑形态 | 单级 `MESH_1D`；A3 仅单 Server，其他拓扑回退 |
| 支持的调用类型 | 单算子 OPBASE；图模式不在本次范围（OFFLOAD 仅回归） |
| 支持的数据类型 | 全部 HCCL 数据类型（纯搬运） |
| 支持的数据量 | 无 256MB/scratch 单轮限制，`loopTimes=1` |
| 是否支持绕路 | 否 |
| 是否支持确定性计算 | 是（纯搬运，逐 bit 一致） |

其他约束：

- `sendBuf == recvBuf`（in-place）返回 `HCCL_E_PARA`。
- `recvCounts/recvDispls` 全 rank 一致；displs 单调不减、非空区间不得重叠、末段不超过 `outputSize`。
- 注册契约为全 rank 双侧全部注册或全不注册；部分注册属于契约外行为。
- 框架暂无跨 rank 对称标志协商，探测失败时本 rank 回退 CCL。

---

# 2.AR20261029516478 AllGatherV（SD）

## 2.1 功能描述

激活 `InsTempAllGatherVMesh1D` 的 `enableRemoteMemAccess` 分支：跨卡写落点为对端 `remoteOutputGraphMode.addr`，写偏移为 `displs[myAlgRank]`；本地保留一次 `LocalDataCopy`（input → 自身 output 的 displs 区间）作为发送源，接收侧跳过 `PostLocalCopy`。

## 2.2 流程描述

一次 `HcclAllGatherV` 调用经过 5 个阶段，本需求改动位于阶段 3（门禁/窗口探测）和阶段 5（模板直写）。

```mermaid
sequenceDiagram
    autonumber
    participant U as 调用方
    participant API as HcclAllGatherV
    participant SEL as Selector
    participant EXE as Executor
    participant TPL as InsTempAllGatherVMesh1D
    participant HC as HCOMM
    U->>API: HcclAllGatherV(...)
    API->>API: 输入检查 + outputSize=max(displs+counts)
    rect rgb(255,243,205)
    API->>SEL: OpParam + TopoInfo
    SEL-->>API: AicpuAllGatherVSoleMesh
    end
    API->>EXE: HcclExecOp → RefreshSymmetricMemory
    rect rgb(255,243,205)
    EXE->>HC: HcclCommSymWinGet(input/output)
    HC-->>EXE: 窗口 + offset
    EXE->>EXE: supportSymmetricMemory=true
    end
    EXE->>TPL: loopTimes=1，scratch=0
    rect rgb(232,245,233)
    TPL->>HC: GetSymWinRemoteMem(peer)
    TPL->>TPL: 自贡献 LocalCopy + N-1 线程直写对端 userOut
    end
```

> 黄色为门禁/窗口探测改动，绿色为数据面直写改动。

**改动点 1——对称门禁与窗口探测**

**表 2-1：AllGatherV 对称门禁矩阵**

| 阶段 | 检查项 | AGV 条件 | 不满足 |
| --- | --- | --- | --- |
| 1 基础门禁 | 版本 / 模式 / 引擎 / 一级拓扑 | hcomm≥9.1.0、OPBASE、AICPU_TS、level0 MESH_1D | CCL 回退 |
| 2 算法门禁 | 选路 | `AicpuAllGatherVSoleMesh` | CCL 回退 |
| 3 窗口门禁 | 双侧注册 | input + output 窗口均有效 | CCL 回退 |
| 4 执行 | 全部通过 | `supportSymmetricMemory=true`；`loopTimes=1`、scratch=0；直写对端 userOut | CCL 回退 |

> 门禁顺序：基础 → 算法 → 窗口；任一层失败即本 rank 回退 CCL，不做跨 rank 协商。

> 文件：`src/ops/op_common/op_common.cc:4784`

```cpp
case HcclCmdType::HCCL_CMD_ALLGATHER_V:
    symMemAllowed = std::string(param.algName) == "AicpuAllGatherVSoleMesh";
    break;
```


**改动点 2——数据面直写**

**图 2-2：AllGatherV 普通路径与对称路径数据流**

```mermaid
flowchart TD
    subgraph NORMAL["普通 CCL 路径"]
        N0["rank0 sendBuf"] -->|LocalCopy 自贡献| N1["rank0 userOut[displs0]"]
        N0 -->|N-1 线程并发写| N2["对端 CCL 槽[0]"]
        N2 -->|对端 PostLocalCopy| N3["对端 userOut[displs0]"]
    end
    subgraph SYM["对称路径"]
        S0["rank0 sendBuf"] -->|LocalCopy 自贡献| S1["rank0 userOut[displs0]"]
        S0 -->|N-1 线程并发直写| S2["对端 userOut[displs0]"]
    end
    N3 -.->|消除接收侧 CCL→userOut 拷贝| S2
    classDef existing fill:#f5f5f5,stroke:#999,color:#333;
    classDef direct fill:#e8f5e9,stroke:#2e7d32,color:#000;
    class N0,N1,N2,N3 existing;
    class S0,S1,S2 direct;
```

> 文件：`src/ops/all_gather_v/algorithm/executor/ins_v2_all_gather_v_sole_executor.cc:147`

```cpp
if (param.supportSymmetricMemory && maxSendDataCount > 0) {
    // 对称路径一次传完，不受 cclBuffer 与 UB_MAX_DATA_SIZE 限制
    maxCountPerLoop = maxSendDataCount;
}
u64 loopTimes = (maxSendDataCount == 0) ? 0 : (1 + ((maxSendDataCount - 1) / maxCountPerLoop));
if (param.supportSymmetricMemory) {
    loopTimes = (maxSendDataCount == 0) ? 0 : 1;
}
```

> 文件：`src/ops/all_gather_v/algorithm/template/aicpu/ins_temp_all_gather_v_mesh_1D.cc:190`

```cpp
void* txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : remoteOutBuffAddr;
void* rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : remoteOutBuffAddr;
void* rxDstPtr = (!enableRemoteMemAccess_)
                     ? tempAlgParams_.buffInfo.hcclBuff.addr
                     : tempAlgParams_.buffInfo.outputPtr;
```

普通路径先写对端 CCL，再由对端 `PostLocalCopy` 到 userOut；对称路径保留一次自贡献本地拷贝，N-1 个对端线程并发直写对端 `userOut`，不再执行接收侧 `PostLocalCopy`。

## 2.3 数据描述

| 数据 | 说明 |
|------|------|
| `param.vDataDes.counts` / `param.varData` | recvCounts 与 recvDispls 合并存放 |
| `param.outputSize` | 修正为 `max(recvDispls[i] + recvCounts[i]) * typeSize` |
| `TemplateDataParams.allRankDispls` | 各 rank 在 output 中的偏移 |
| `TemplateDataParams.allRankSliceSize` | 各 rank 本轮处理的字节数 |
| `ChannelInfo.remoteOutputGraphMode.addr` | 对端 userOut 地址，由 `FillChannelSymWinPeerAddrs` 填充 |
| `TemplateDataParams.buffInfo.outputSize` | 本端窗口大小，用于对称落点越界校验 |

> 文件：`src/ops/all_gather_v/all_gather_v.cc:216`

```cpp
AllGatherVDisplsError displsErr{};
HcclResult calcRet = CalcAndCheckAllGatherVOutputSize(
    recvCounts, recvDispls, userRankSize, perDataSize, outputSize, displsErr);
if (calcRet != HCCL_SUCCESS) {
    HCCL_ERROR("[AllGatherV] invalid recvCounts/recvDispls, rank[%u], reason[%s], ret[%d]",
               displsErr.rank, displsErr.reason, calcRet);
    return calcRet;
}
```


**图 2-3：AllGatherV 的 recvBuf / displs 布局**

```mermaid
flowchart LR
    A["rank0 区间<br/>displs[0] + counts[0]"] --- G["空洞"] --- B["rank1 区间<br/>displs[1] + counts[1]"] --- C["rank2 区间<br/>displs[2] + counts[2]"]
    O["outputSize = max(displs[i] + counts[i])"] --- A
    N["rank i 写各 rank 的 displs[i] 区间<br/>非空区间不得重叠"] --- B
    T["所有 rank 使用同一 recvCounts/recvDispls 布局"] --- C
    classDef changed fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#000;
    classDef existing fill:#f5f5f5,stroke:#999,color:#333;
    class O,N changed;
    class A,G,B,C,T existing;
```

**图 2-4：对称窗口注册与地址换算**

```mermaid
sequenceDiagram
    participant R0 as rank0
    participant R1 as rank1
    participant H as HCCL
    participant HCOMM as HCOMM
    R0->>H: HcclCommSymWinRegister(comm, ptr, size, flag=1)
    R1->>H: HcclCommSymWinRegister(comm, ptr, size, flag=1)
    H->>HCOMM: 记录窗口与 offset
    Note over R0,R1: 全 rank 全部注册或全不注册
    rect rgb(255, 243, 205)
    R0->>H: HcclCommSymWinGet(comm, ptr, size)
    H-->>R0: winHandle + offset
    R0->>H: GetSymWinRemoteMem(winHandle, offset, peer=1)
    H-->>R0: peer1 的 user buffer 地址
    end
    rect rgb(232, 245, 233)
    R0->>R1: 直接写/读 peer1 的 userOut
    end
    Note over R0,R1: 任一 rank 查询失败 → 本 rank 回退 CCL
```

## 2.4 依赖性描述

- 公共层：`RefreshSymmetricMemory`、`CheckAndSetSymmetricMemory`、`IsSymMemWindowValid`、`FillChannelSymWinPeerAddrs`。
- HCOMM：`HcclCommSymWinRegister`、`HcclCommSymWinGet`、`GetSymWinRemoteMem`，按 9.1/9.2 版本能力动态加载。
- 执行器/模板：`InsV2AllGatherVSoleExecutor`、`InsTempAllGatherVMesh1D`、`executor_v2_base` 的对称信息传递。
- 不依赖 Scatter/ReduceScatterV 的实现。

## 2.5 接口描述

| 函数原型 | `HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)` |
| --- | --- |
| 函数功能 | 将各 rank 的输入按 `recvCounts/recvDispls` 汇聚到所有 rank 的 `recvBuf`。 |
| 输入说明 | `sendBuf/sendCount` 为本端输入；`recvCounts/recvDispls` 为全 rank 一致的接收布局；`dataType` 为元素类型；`comm/stream` 为通信域与流。 |
| 输出说明 | 各 rank `recvBuf` 按 `recvDispls` 布局；对称路径下跨卡数据直接写入对端 `userOut`。 |
| 返回值说明 | `HCCL_SUCCESS`；参数为空/非法返回 `HCCL_E_PTR`/`HCCL_E_PARA`；内部错误返回 `HCCL_E_INTERNAL`。 |

无 API 变更；注册对称内存且门禁满足时行为变化为跨卡数据不经 CCL，未注册时保持原路径。

## 2.6 约束分析

| 支持的算子名称 | `HcclAllGatherV` |
| --- | --- |
| 支持的算法名称 | `AicpuAllGatherVSoleMesh` |
| 支持的芯片类型 | Ascend 950PR/950DT、Atlas A3 系列（以产品支持表为准） |
| 支持的展开模式 | AI CPU（`COMM_ENGINE_AICPU_TS`） |
| 支持的拓扑形态 | 单级 `MESH_1D`；A3 仅单 Server，其他拓扑回退 |
| 支持的调用类型 | 单算子 OPBASE；图模式不在本次范围（OFFLOAD 仅回归） |
| 支持的数据类型 | 全部 HCCL 数据类型（纯搬运） |
| 支持的数据量 | 无 256MB/scratch 单轮限制，`loopTimes=1` |
| 是否支持绕路 | 否 |
| 是否支持确定性计算 | 是（纯搬运，逐 bit 一致） |

补充约束：

- `remoteOutputGraphMode` 只回填地址、不保证 size，越界校验使用本地 `outputSize`。
- 全 rank 双侧全部注册或全不注册；部分注册属于契约外行为。
- 探测失败或门禁不满足时，本 rank 回退 CCL。

## 2.7 DFX设计

| 校验内容 | 级别 | 搜索内容 |
| --- | --- | --- |
| 对称窗口探测 | INFO | `supportSymmetricMemory`、`symMem`、窗口查询结果 |
| 门禁与算法 | INFO | `RefreshSymmetricMemory`、`algName`、`engine`、`level0Topo` |
| 对端地址/越界 | ERROR | `remote output addr is null`、`output offset out of range` |
| 参数校验 | ERROR | displs 单调/重叠/溢出、in-place |
| 回退 | INFO/WARNING | `supportSymmetricMemory=false`、CCL fallback |

## 2.8 资料描述

- `docs/zh/api_ref/comm_op_interface/HcclAllGatherV.md`：补充对称内存说明、注册契约与 displs 约束。
- `docs/zh/user_guide/comm_ops_support_list/comm_ops_support_a3.md`：按产品支持表更新单算子零拷贝状态。

## 2.9 性能&&质量

- 正确性：ST 12/12 默认路径通过；对称路径与默认路径逐 bit 一致，非落点区域不被踩踏。
- 真机：8p 单 Server、512MB FP32，base 11816.32us / 45.43GB/s，sym 6089.40us / 88.16GB/s，约 1.94x；64MB 约 5.5% 提升。
- 对称 ST 当前未纳入默认 CI，仅保留手工/临时验证；后续按专项 issue 接入。
- 质量约束：大数据 `loopTimes=1`，无 256MB 截断；对端地址空值和越界有防御。

---

# 3.AR20261029517134 Reduce（SRS）

## 3.1 介绍

`HcclReduce` 将所有 rank 的同形状数据逐元素归约，结果仅落在 root。Reduce 是“N 份全量 D → root 一份 D”的向心归约：确定性 + fullmesh 并发要求 N×D 的中间落点，而合法用户介质只有 root 的 1×D，因此只能做 root 侧部分零拷贝。

本需求对 `AicpuReduceSoleNHR`、`AicpuReduceParallelMeshNHR`、`AicpuReduceParallelNHRNHR` 开启 root 侧零拷贝：root 工作区/终局汇集改 userOut；非 root 与交换槽保留 CCL。

**单级 MESH_1D 无零拷贝收益**：单级选到 `SoleMesh` / `SoleMeshTwoShot`，需要 N×D 或 D 槽位；root 只有 1×D 的 userOut，非 root output 语义无效、input 只读，没有合法 user buffer 可替换。即使强制单级 `SoleNHR`，也会被 `topoLevelNums == 1` 门禁拒绝。

## 3.2 输入

**API 入参**（与现状一致，无新增）：

| 参数 | 类型 | 说明 |
|------|------|------|
| sendBuf | void* | 各 rank 输入 |
| recvBuf | void* | root 输出；非 root 仅作窗口锚点 |
| count | uint64_t | 元素个数 |
| dataType | HcclDataType | 数据类型 |
| op | HcclReduceOp | SUM/PROD/MAX/MIN |
| root | uint32_t | 结果落地 rank |
| comm / stream | HcclComm / aclrtStream | 通信域与流 |

**进入零拷贝条件**：

- HCOMM ≥ 9.1.0；
- `opMode == OPBASE`；
- `engine == COMM_ENGINE_AICPU_TS`；
- `topoInfo->level0Topo == MESH_1D`；
- `topoInfo->topoLevelNums > 1`（单级不开启）；
- `algName ∈ {AicpuReduceSoleNHR, AicpuReduceParallelMeshNHR, AicpuReduceParallelNHRNHR}`；
- `param.inputPtr != param.outputPtr`（禁 in-place）；
- output 侧注册对称内存（单边探测成功）。

## 3.3 处理

1. `Selector` 选出 Reduce 白名单算法；`HcclExecOp → RefreshSymmetricMemory` 完成版本/引擎/拓扑/算法/类型/in-place 门禁与 output 单边探测。
2. 对称生效时，执行器把 `supportSymmetricMemory` 与窗口信息传给模板。
3. `SoleNHR`：root 的 PreCopy 目的、工作区和终局落点改 userOut，root 跳过 PostCopy；非 root 工作区/交换槽保持 CCL。
4. `ParallelMeshNHR / ParallelNHRNHR`：RS 阶段保持 CCL（B 守卫），最终 AG 阶段 root 落点改 userOut，并移除 root 终局 LocalCopy。

## 3.4 输出

- root 的 `recvBuf` 内容与非对称路径逐 bit 一致。
- 非 root `recvBuf` 内容语义无效，对称路径不写入非 root output。
- 输入 `sendBuf` 不被污染。

## 3.5 约束分析

| 支持的算子名称 | `HcclReduce` |
| --- | --- |
| 支持的算法名称 | `AicpuReduceSoleNHR`、`AicpuReduceParallelMeshNHR`、`AicpuReduceParallelNHRNHR` |
| 支持的芯片类型 | Ascend 950PR/950DT、Atlas A3 系列（以产品支持表为准） |
| 支持的展开模式 | AI CPU（`COMM_ENGINE_AICPU_TS`） |
| 支持的拓扑形态 | 2/3 级 `MESH_1D`；单级 MESH_1D 不开启；CLOS/MESH_1D_CLOS 走原路径 |
| 支持的调用类型 | 单算子 OPBASE；图模式不在本次范围 |
| 支持的数据类型 | 排除 INT64/UINT64/FP64 与 PROD；其余按算法 attrs |
| 支持的数据量 | 支持多轮 loop；非 root 交换槽仍受 CCL 槽位约束 |
| 是否支持绕路 | 否 |
| 是否支持确定性计算 | 是（固定成对序，逐 bit 一致） |

其他约束：

- 仅 3 个白名单算法；`MultiJetty`、`SoleMesh`、`TwoShot`、`3level`、`AicpuReduceNHR` 不开启。
- 全 rank `recvBuf` 全部注册或全不注册；非 root `recvBuf` 必须非空并注册（窗口锚点）。
- `inputPtr == outputPtr`、INT64/UINT64/FP64、PROD 均回退。
- 框架暂无跨 rank 对称标志协商，依赖注册契约。

---

# 4.AR20261029517134 Reduce（SD）

## 4.1 功能描述

- **SoleNHR**：root 的本地工作区从 CCL 改为自身 userOut；PreCopy 目的、halving/doubling 的原位归约介质、终局落点都指向 userOut；root 跳过 PostCopy。归约顺序与交换模式不变。
- **ParallelMeshNHR / ParallelNHRNHR**：最终 AG 阶段 root 侧工作区/最终落点改 userOut，移除 root 终局 LocalCopy。
- **B 方案守卫**：`InsTempReduceScatterMesh1D` / `InsTempReduceScatterNHR` 的 `supportSymmetricMemAccess_` 改为 `param.supportSymmetricMemory && param.inputSymWindow != nullptr`。Reduce 仅注册 output 时，RS 阶段自动回退 CCL，避免空地址与输入污染。

关键决策：

- 新增 Reduce 专用 `IsReduceAlgSymMemDeterministic`，不改动 AR/RS 既有函数；AR/RS 多级仍保持 false。
- Reduce 白名单仅 `SoleNHR`、`ParallelMeshNHR`、`ParallelNHRNHR`，不含 MultiJetty。
- 单级 MESH_1D 显式无收益；依赖多级拓扑。
- 无跨 rank 对称标志协商，采用“全 rank 注册或全不注册”契约。

## 4.2 流程描述

一次 `HcclReduce` 调用经过 6 个阶段，本需求改动位于阶段 3（门禁/窗口探测）、阶段 5（SoleNHR root 工作区）和阶段 6（Parallel 最终 AG）。

```mermaid
sequenceDiagram
    autonumber
    participant U as 调用方
    participant API as HcclReduce
    participant SEL as Selector
    participant EXE as Executor
    participant TPL as ReduceNHR / Parallel
    participant HC as HCOMM
    U->>API: HcclReduce(...)
    API->>API: 输入检查 + OpParam 构造
    rect rgb(255,243,205)
    API->>SEL: OpParam + TopoInfo
    SEL-->>API: 白名单算法
    end
    API->>EXE: HcclExecOp → RefreshSymmetricMemory
    rect rgb(255,243,205)
    EXE->>HC: HcclCommSymWinGet(output)
    HC-->>EXE: 窗口 + offset
    EXE->>EXE: supportSymmetricMemory=true
    end
    EXE->>TPL: SoleNHR / Parallel 编排
    rect rgb(232,245,233)
    TPL->>TPL: root 工作区/终局落 userOut；非 root 仍 CCL
    end
    TPL-->>U: root recvBuf
```

> 黄色为门禁/窗口探测改动，绿色为 root 侧数据面改动。

**改动点 1——门禁与算法选择**

**表 4-1：Reduce 对称门禁矩阵**

| 阶段 | 检查项 | Reduce 条件 | 不满足 |
| --- | --- | --- | --- |
| 1 基础门禁 | 版本 / 模式 / 引擎 / 一级拓扑 | hcomm≥9.1.0、OPBASE、AICPU_TS、level0 MESH_1D | CCL 回退 |
| 2 拓扑门禁 | 多级 | `topoLevelNums > 1`；单级直接排除 | CCL 回退 |
| 3 类型/调用门禁 | 数据类型 / 规约 / in-place | 非 INT64/UINT64/FP64/PROD；`inputPtr != outputPtr` | CCL 回退 |
| 4 算法门禁 | 白名单 | `SoleNHR` / `ParallelMeshNHR` / `ParallelNHRNHR` | CCL 回退 |
| 5 窗口门禁 | 单边注册 | 仅 output 窗口有效 | CCL 回退 |
| 6 执行 | 全部通过 | `supportSymmetricMemory=true`；root 工作区/终局落 userOut；非 root 仍 CCL | CCL 回退 |

> 门禁顺序：基础 → 拓扑 → 类型/调用 → 算法 → 窗口；任一层失败即本 rank 回退 CCL，不做跨 rank 协商。

> 文件：`src/ops/op_common/op_common.cc:4753`

```cpp
bool IsReduceAlgSymMemDeterministic(
    const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo == nullptr || topoInfo->topoLevelNums == TOPO_LEVEL_NUM_1) {
        return false;
    }
    return algName == "AicpuReduceSoleNHR"
        || algName == "AicpuReduceParallelMeshNHR"
        || algName == "AicpuReduceParallelNHRNHR";
}
```

> 文件：`src/ops/op_common/op_common.cc:4800`

```cpp
case HcclCmdType::HCCL_CMD_REDUCE:
    symMemAllowed = IsReduceOpSymMemAllowed(param) && param.inputPtr != param.outputPtr
                    && IsReduceAlgSymMemDeterministic(param.algName, topoInfo);
    break;
```


**改动点 2——SoleNHR**

**图 4-2：SoleNHR 的 root 工作区与固定成对归约**

```mermaid
flowchart TD
    A["各 rank 输入 D<br/>sendBuf 只读"] --> B{"rank == root?"}
    B -- 是 --> C["PreCopy：input → userOut<br/>输入已在 userOut 时跳过"]
    C --> D["halving/doubling<br/>root 工作区 = userOut<br/>原位累加"]
    D --> E["root 终局结果已在 userOut<br/>跳过 PostCopy"]
    B -- 否 --> F["非 root 工作区 / 交换槽 = CCL"]
    F --> G["非 root 按固定成对顺序参与归约<br/>输出语义无效"]
    D -.->|固定成对 / 固定序| G
    H["归约顺序与交换模式不变"] --- D
    classDef existing fill:#f5f5f5,stroke:#999,color:#333;
    classDef changed fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#000;
    classDef direct fill:#e8f5e9,stroke:#2e7d32,color:#000;
    class A,F,G,H existing;
    class B changed;
    class C,D,E direct;
```

SoleNHR 只把 root 的 PreCopy 目的、工作区和终局落点改到 userOut；非 root 工作区/交换槽仍使用 CCL，halving/doubling 的成对顺序不变。

> 文件：`src/ops/reduce/algorithm/template/aicpu/reduce_nhr.cc:64`、`:236`、`:425`

```cpp
void* ReduceNHR::GetLocalWorkspacePtr() const
{
    if (rootUsesOutputWorkspace_ && myRank_ == root_) {
        return buffInfo_.outputPtr;
    }
    return buffInfo_.hcclBuff.addr;
}

if (rootUsesOutputWorkspace_ && myRank_ == root_) {
    if (buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
        // userIn -> userOut
        void* const src =
            static_cast<void*>(static_cast<u8*>(buffInfo.inputPtr) + buffInfo.inBuffBaseOff);
        void* const dst =
            static_cast<void*>(static_cast<u8*>(buffInfo.outputPtr) + buffInfo.outBuffBaseOff);
        HcommLocalCopyOnThread(thread_, dst, src, sliceSize);
    }
    // 输入已在 userOut 时跳过 PreCopy
}

// PostCopy：root 已在 userOut，直接返回
if (rootUsesOutputWorkspace_ && myRank_ == root_) {
    return HCCL_SUCCESS;
}
```


**改动点 3——Parallel**

**图 4-3：ParallelMeshNHR 的两级流水**

```mermaid
flowchart LR
    A["输入 D"] --> B["RS 阶段<br/>slot = CCL"]
    B --> C["NHR / 中间结果<br/>非 root CCL"]
    C --> D["最终 AG 阶段<br/>root 落点 = userOut"]
    D --> E["rootOutputWritten_<br/>跳过终局 LocalCopy"]
    B -.-> G["B 守卫：inputSymWindow == nullptr → RS 回退 CCL"]
    F["当前实现：enableRemoteMemAccess = false<br/>Parallel 的 AG 仍走 CCL 回退"] -.-> D
    classDef existing fill:#f5f5f5,stroke:#999,color:#333;
    classDef changed fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#000;
    classDef direct fill:#e8f5e9,stroke:#2e7d32,color:#000;
    classDef fallback fill:#fdecea,stroke:#c62828,color:#000;
    class A,B,C existing;
    class G changed;
    class D,E direct;
    class F fallback;
```

RS 阶段保持 CCL，B 守卫在仅 output 注册时回退；最终 AG 阶段的 root 落点设计为 userOut。当前 `enableRemoteMemAccess=false`，Parallel 实际仍走 CCL 回退。

> 文件：`src/ops/reduce/algorithm/executor/reduce_parallel_executor.cc:545`、`:577`、`:813`；`src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc:124`

```cpp
const bool rootOutputStage =
    param_.supportSymmetricMemory && myRank_ == root_ && stageIdx == 1 && stepIdx == 1;
dataParams.buffInfo.outBuffType =
    rootOutputStage ? BufferType::OUTPUT : BufferType::HCCL_BUFFER;
dataParams.buffInfo.outputPtr = rootOutputStage
    ? static_cast<void*>(static_cast<u8*>(param_.outputPtr) + outputLoopBaseOff_)
    : resCtx_.cclMem.addr;
dataParams.buffInfo.outputSize = rootOutputStage
    ? (param_.outputSize - outputLoopBaseOff_) : resCtx_.cclMem.size;

if (dataParams.buffInfo.outBuffType == BufferType::OUTPUT) {
    rootOutputWritten_ = true;
}

// RS 阶段：仅 output 注册时回退 CCL
supportSymmetricMemAccess_ =
    param.supportSymmetricMemory && param.inputSymWindow != nullptr;
```

> 当前 `enableRemoteMemAccess` 未启用，Parallel 的 AG 仍走 CCL 回退；上面是已接线的 root 落点/终局判定与 RS 守卫。


## 4.3 数据描述

| 数据 | 说明 |
|------|------|
| `param.root` | 结果落地 rank；模板 `root_` 由 `InsAlgTemplateBase` 构造时取 `param.root` |
| `param.inputOffset` / `outputOffset` | 仅 output offset 参与 Reduce 对称路径 |
| `ChannelInfo.remoteOutputGraphMode.addr` | 指向对端 userOut；Reduce 只在“对端为 root”时使用 |
| `TemplateDataParams.buffInfo.outBuffType` | root 终局阶段需为 `OUTPUT` |
| `TemplateDataParams.buffInfo.outBuffBaseOff` | 多轮 loop 时叠加到直写地址 |

**图 4-4：Reduce 的空间下界——N×D 输入 vs root 1×D 输出**

```mermaid
flowchart LR
    subgraph IN["N 个 rank 输入（每份 D）"]
        I0["rank0: D"]
        I1["rank1: D"]
        I2["rank2: D"]
    end
    I0 --> C["CCL 每源一槽<br/>N×D"]
    I1 --> C
    I2 --> C
    C --> F["固定序折叠"]
    F --> O["root userOut<br/>1×D"]
    N["非 root userOut<br/>语义无效"] -.->|不能作累加介质| C
    S["userIn 只读"] -.->|不能作累加介质| C
    classDef existing fill:#f5f5f5,stroke:#999,color:#333;
    classDef direct fill:#e8f5e9,stroke:#2e7d32,color:#000;
    classDef fallback fill:#fdecea,stroke:#c62828,color:#000;
    class I0,I1,I2,C,F existing;
    class O direct;
    class N,S fallback;
```

**图 4-5：对称窗口注册与地址换算**

```mermaid
sequenceDiagram
    participant R0 as rank0
    participant R1 as rank1
    participant H as HCCL
    participant HCOMM as HCOMM
    R0->>H: HcclCommSymWinRegister(comm, ptr, size, flag=1)
    R1->>H: HcclCommSymWinRegister(comm, ptr, size, flag=1)
    H->>HCOMM: 记录窗口与 offset
    Note over R0,R1: 全 rank 全部注册或全不注册
    rect rgb(255, 243, 205)
    R0->>H: HcclCommSymWinGet(comm, ptr, size)
    H-->>R0: winHandle + offset
    R0->>H: GetSymWinRemoteMem(winHandle, offset, peer=1)
    H-->>R0: peer1 的 user buffer 地址
    end
    rect rgb(232, 245, 233)
    R0->>R1: 直接写/读 peer1 的 userOut
    end
    Note over R0,R1: 任一 rank 查询失败 → 本 rank 回退 CCL
```

## 4.4 依赖性描述

| 依赖 | 说明 |
|------|------|
| `ReduceNHR` | SoleNHR 模板；PreCopy/RunReduce/RunGather/PostCopy 按 root 选择地址 |
| `ReduceParallelExecutor` | RS Mesh/NHR + AG Mesh/NHR 组合；最终 AG root 落点处理 |
| `InsTempReduceScatterMesh1D/NHR` | B 守卫改动；与 AR/RS 共享，需回归 |
| `InsTempAllGatherMesh1D/NHR` | 使用 `enableRemoteMemAccess` 与 `remoteOutputGraphMode` |
| `FillChannelSymWinPeerAddrs` | 只填地址，不填 size；越界校验使用本地 size |
| 公共层 | `RefreshSymmetricMemory`、`CheckAndSetSymmetricMemory`、`IsSymMemWindowValid`、`IsReduceAlgSymMemDeterministic` |
| HCOMM | `HcclCommSymWinRegister`、`HcclCommSymWinGet`、`GetSymWinRemoteMem` |

## 4.5 接口描述

| 函数原型 | `HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)` |
| --- | --- |
| 函数功能 | 将所有 rank 的 `sendBuf` 逐元素归约，结果写入 root 的 `recvBuf`。 |
| 输入说明 | `sendBuf/count/dataType/op` 为各 rank 输入；`root` 指定结果落地 rank；`comm/stream` 为通信域与流。 |
| 输出说明 | root 的 `recvBuf` 为归约结果；非 root `recvBuf` 内容无效。 |
| 返回值说明 | `HCCL_SUCCESS`；参数为空/非法返回 `HCCL_E_PTR`/`HCCL_E_PARA`；内部错误返回 `HCCL_E_INTERNAL`。 |

无 API 变更；注册对称内存且门禁满足时，root 侧减少一次全量本地拷贝。

## 4.6 约束分析

| 支持的算子名称 | `HcclReduce` |
| --- | --- |
| 支持的算法名称 | `AicpuReduceSoleNHR`、`AicpuReduceParallelMeshNHR`、`AicpuReduceParallelNHRNHR` |
| 支持的芯片类型 | Ascend 950PR/950DT、Atlas A3 系列（以产品支持表为准） |
| 支持的展开模式 | AI CPU（`COMM_ENGINE_AICPU_TS`） |
| 支持的拓扑形态 | 2/3 级 `MESH_1D`；单级 MESH_1D 不开启；CLOS/MESH_1D_CLOS 走原路径 |
| 支持的调用类型 | 单算子 OPBASE；图模式不在本次范围 |
| 支持的数据类型 | 排除 INT64/UINT64/FP64 与 PROD；其余按算法 attrs |
| 支持的数据量 | 支持多轮 loop；非 root 交换槽仍受 CCL 槽位约束 |
| 是否支持绕路 | 否 |
| 是否支持确定性计算 | 是（固定成对序，逐 bit 一致） |

补充约束：

- 单级 MESH_1D 选到 SoleMesh/TwoShot，不在白名单；强制单级 SoleNHR 也被 `topoLevelNums == 1` 排除。
- 全 rank `recvBuf` 全部注册或全不注册；非 root `recvBuf` 必须非空并注册（窗口锚点）。
- RS 阶段仅 output 注册时由 B 守卫回退 CCL，不污染输入。
- 多轮 loop 下直写/原位归约地址必须叠加 `outBuffBaseOff`，notify 握手不得省略。

**图 4-6：Reduce 算法支持范围决策树**

```mermaid
flowchart TD
    A["HcclReduce"] --> B{"topoLevelNums == 1?"}
    B -- 是 --> C["SoleMesh / SoleMeshTwoShot<br/>需要 N×D 或 D 槽位<br/>无合法 userOut → 不开"]
    B -- 否 --> D{"INT64/UINT64/FP64/PROD?<br/>in-place?"}
    D -- 是 --> E["CCL 回退"]
    D -- 否 --> F{"algName 白名单?"}
    F -- 否 --> E
    F -- 是 --> G["SoleNHR / ParallelMeshNHR / ParallelNHRNHR<br/>root 侧部分零拷贝"]
    C --> H["CCL 回退 / 无收益"]
    classDef existing fill:#f5f5f5,stroke:#999,color:#333;
    classDef changed fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#000;
    classDef fallback fill:#fdecea,stroke:#c62828,color:#000;
    classDef direct fill:#e8f5e9,stroke:#2e7d32,color:#000;
    class A existing;
    class B,D,F changed;
    class G direct;
    class C,E,H fallback;
```

## 4.7 DFX设计

| 校验内容 | 级别 | 搜索内容 |
| --- | --- | --- |
| 对称窗口探测 | INFO | `symMem`、`supportSymmetricMemory`、窗口查询结果 |
| 门禁与算法 | INFO | `RefreshSymmetricMemory`、`algName`、`topoLevelNums`、`isCapture` |
| root 工作区/阶段 | INFO | `rootUsesOutputWorkspace_`、PreCopy/PostCopy、最终 AG 阶段 |
| 对端地址/越界 | ERROR | remote root output addr null、output offset out of range |
| 参数/类型/回退 | ERROR/INFO | in-place、64bit/PROD、窗口失败、CCL fallback |

## 4.8 资料描述

- `docs/zh/api_ref/comm_op_interface/HcclReduce.md`：补充对称内存说明（仅 recvBuf 注册、非 root 内容无效、不污染输入、单级无收益、探测失败回退）。
- `docs/zh/user_guide/comm_ops_support_list/comm_ops_support_a3.md`：按白名单算法/拓扑更新支持状态。

## 4.9 性能&&质量

- 正确性：Reduce 默认 ST 22/22 通过；root 输出与非对称路径逐 bit 一致，重复执行逐 bit 一致。
- 单级 MESH_1D 实测：`symMem[0]`，base/sym 512MB 均约 3606us/148.8GB/s，符合“单级无收益”设计。
- 多级 SoleNHR 为 root 侧收益场景，需 2 Server 拓扑验证；Parallel 当前 `enableRemoteMemAccess=false`，仍是 CCL 回退。
- 对称 ST 当前未纳入默认 CI，仅保留手工/临时验证；后续按专项 issue 接入。
- 边界质量：非 root output 不写、输入不污染、多轮偏移与对端空地址有防御。
