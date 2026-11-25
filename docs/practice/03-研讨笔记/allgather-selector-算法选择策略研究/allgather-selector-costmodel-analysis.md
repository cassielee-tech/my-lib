# AllGather Selector 算法选择策略研究：成本模型决策树与 Corner Case

> 仓：`cann/hccl`。本文聚焦 AllGather 算子算法选择器（selector），尤其是引入 cost model 之后形成的"决策树"结构，以及由此暴露的 corner case。

## 1. 两条选路路径的分流点

算法选择的入口在 `src/ops/op_common/op_common.cc:225`：

```cpp
if (!useNdaSelector && ((topoInfo->level0Topo == Level0Shape::MESH_1D) || AutoSelectorBase::IsDevType960())) {
    CHK_RET(SelectorEngine::Global()->Run(comm, param, topoInfo.get(), algName));   // 新 cost model
} else {
    ... ExecuteSelector ...   // 老硬编码决策树
}
```

- **新 cost model 路径**（`SelectorEngine::Run`）：仅当 `level0Topo == MESH_1D` 或 960 机型时走。`SelectorEngine::IsOpSupported`（`selector_engine.cc:42`）白名单含 ALLGATHER。
- **老硬编码路径**（`ExecuteSelector::Run → AutoSelectorBase::Select → AllGatherAutoSelector::Select*`）：CLOS / MESH_1D_CLOS(UBX) / hostDPU 等其它拓扑。

## 2. 老的 `AllGatherAutoSelector` 是一颗字面"决策树"

`src/ops/all_gather/selector/all_gather_auto_selector.cc` 全部是显式 if/else 阈值分支，例如：

- `SMALL_COUNT_512KB`（512KB）
- `AG_CCU_SMALL_DATA_SIZE`（4MB）
- `AG_FLATTEN_MAX_DATA_SIZE`（128KB）
- `AG_2P_DETOUR_DATA_SIZE`（4MB）
- `MAX_RANK_NUM_FOR_CONCURRENT_ALGO`（4）

改动代码前先读架构约束 `docs/zh/architecture/architecture-brief.md` 第 3 节。

## 3. 新 cost model 如何形成"决策树"

`SelectorEngine::Run`（`selector_engine.cc:413`）的 pipeline：

```
InitCostModel → GenerateCostTable → TunerEnrich → SelectMinCost
```

核心数学在 `cost_table.cc:360 CalcAlgCost`：

```
segCost = (A/util + B + Breduce) * dataSize + C
cost    = max( Σ/MAX segCost , totalD )
```

- `A/B/C/D` 由每个算法的 `CalcCostCoeff` 产出（如 `ins_temp_all_gather_mesh_1D.cc` 的 mesh、`ins_temp_all_gather_nhr.cc` 的 NHR）。
- `util(ds)` 是 dataSize 的**分段常数阶跃函数**（来自 `closUbUtilTable_`/`meshUbUtilTable_`，断点 1/2/4/8/16/32/64/128/256MB）。
- `SelectMinCost` 在"最高引擎优先级层"内取 cost 最小者（`selector_engine.cc:597`）。

所以决策树实质是：**每个算法一条分段线性 cost 曲线，曲线随 dataSize 的交叉点构成决策边界**；而显式 if/else 并未消失，它以 `opCustomCheck`/`topoCustomCheck`/`topoPriorityCheck`/`opPriorityCheck` 的形式作为"补丁叶子"挂在树梢：

- `opCustomCheck`（`cost_table.cc:141`）
- `topoCustomCheck`（`cost_model.cc:292`）
- `topoPriorityCheck`（`cost_model.cc:480 ApplyTopoPriority`）
- `opPriorityCheck`（`cost_table.cc:276 ApplyOpPriority`）

## 4. 两级 MESH_1D+CLOS 的候选集（AICPU 引擎）

| 算法 | 结构 | cost 段数 | 参与条件 |
|---|---|---|---|
| `AicpuAllGatherSoleNHR` | 纯 NHR（TopoMatchOneLevel，全 rank 环） | 1 段 | 两级 & level0Topo=MESH_1D；被 `opCustomCheck` 在 16P/2~4MB 区段排除 |
| `AicpuAllGatherParallelMeshNHR` | mesh + NHR 并行（TopoMatchTwoLevel） | 4 段 `[mesh,NHR,mesh,NHR]` | 两级 MESH_1D |
| `AicpuAllGatherSequenceMeshConcurNHR` | 序列两级 | 2 段 | 仅 `totalSize > 4GB` |

核心竞争即 **SoleNHR** vs **ParallelMeshNHR**。

## 5. 闭式代价函数（代入 `CalcAlgCost` 后的决策输入）

单段 `segCost(ds) = (A/util(ds) + B)·ds + C`，整体 `cost = max(Σ/MAX segCost, ΣD)`。`ds` = per-rank 字节数。

**SoleNHR**（`ins_temp_all_gather_nhr.cc:18`，NHR 强制 CLOS）：

```
A = (N-1)/(P·56e9)          B = 1/1400e9
C = 2e-6·max(8log₂N-7, 10)  D = 1.2e-6·9·(max(⌊5log₂N/3⌋,2)-1)
util 查表时 perTransfer = ds·(N/2)          ← NHR 按 rs/2 放大（cost_table.cc:388）
```

**ParallelMeshNHR**（`ins_v2_all_gather_parallel_executor.cc:52`，α=meshFirstRatio）：

```
seg0 Mesh(L0, α):        A=α/56e9,            B=2α/1400e9
seg1 NHR (L1, 1-α):      A=(1-α)(L1-1)/(P·56e9), B=(1-α)/1400e9
seg2 Mesh(L0, (1-α)L1):  A=(1-α)L1/56e9
seg3 NHR (L1, αL0):      A=αL0(L1-1)/(P·56e9)
cost = max(seg0,seg1) + max(seg2,seg3)          ← groupSizes{2,2}+intraMAX
每个 C += 2.3e-5，D += 1.5e-6·(9N+32)/32
```

α 由内置公式给出（`template_utils.cc:733` ALL_GATHER）：`α = L1/(L1 + L0(L1-1)/P)`。

相关成本函数（`cost_model.cc`）：

- `CalcMeshParam`（`cost_model.cc:555`）：MESH 时 `A = n/linkBw`（linkBw = 56e9）
- `CalcNHRParams`（`cost_model.cc:584`）：`A = n·(rankSize-1)/(portNumEff·linkBw)`
- `CalcLocalCopyParams`（`cost_model.cc:608`）：`B = n/localCopyBw`（localCopyBw = 1400e9）
- `CalcLatencyParams`（`cost_model.cc:634`）：AICPU `C = 2e-6·taskNum`
- 带宽常量（`cost_model.cc:114 InitBandwidth`）

## 6. 数值决策表（Mesh/Parallel 与 NHR 的交叉点，per-rank）

假设：L0=8卡/服务器、非 POD、P 为 CLOS 端口数。

| rankSize | L1 | 端口 P | α | 交叉数据量（per-rank） |
|---|---|---|---|---|
| 16 | 2 | 1 | 0.200 | 30.4 KB |
| 16 | 2 | 2 | 0.333 | **60.7KB, 64KB, 108KB, 128KB, 192KB, 256KB, 281KB** |
| 32 | 4 | 1 | 0.143 | **20.7KB, 32KB, 36.9KB** |
| 32 | 4 | 2 | 0.250 | **119KB, 128KB, 170KB** |
| 64 | 8 | 1 | 0.125 | **50.5KB, 64KB, 85.2KB** |
| 128 | 16 | 1 | 0.118 | 72.5 KB |
| 128 | 16 | 4 | 0.348 | 346.7 KB |

规律：小数据 SoleNHR 赢（单 kernel、launch 开销低）；数据一大 Parallel 赢（mesh 带宽 util 0.71 远高于 CLOS 的 0.10~0.56）。交叉点落在几十 KB~几百 KB 量级。

> 注：`portNum`（`GetPhysicalLevelPortNums`）是运行时拓扑输入，无静态值；上表用典型值（mesh=1、CLOS=1/2/4）参数化，具体边界依赖机型。

## 7. Corner Case 清单

### ① 决策边界不是单阈值，而是"多段跳变带"（最重要）

上表中 `20.7KB, 32KB, 36.9KB` 这类**多个交叉点**，说明 `cost_parallel(ds) - cost_nhr(ds)` 在很窄的数据量区间内**多次穿越 0**。根因：`util(ds)` 是分段常数函数（`QueryUbUtil` 的 `lower_bound`），每过一个 util 断点，某一方的斜率跳变、相对顺序翻转。这正是"引入 cost model 之后形成的决策树"最隐蔽的 corner——不再是干净的单阈值二分，而是非单调 flip-flop。

### ② util 断点闭区间语义

`cost_table.cc:486`：

```cpp
auto it = std::lower_bound(table.begin(), table.end(), dataSize, [](const UbUtilEntry& e, u64 ds) {
    return e.upperBound <= ds;
});
```

`dataSize == upperBound` 时跳到下一档 util。即 `perTransfer` 恰为一个端点（1MB/2MB…）时利用率阶跃，交叉点会精确落在这些 2 的幂附近（上表 64KB/128KB/192KB/256KB 的聚集即证据）。

### ③ 已注释的 cost model 误选区段

`sole_executor.cc:409-416`：

```cpp
// 规避: 16P MESH_1D 总数据量 32~64MB(per-rank 2~4MB)区段 costmodel 对 SoleNHR 估值偏低导致误选,
// 先在该区段排除该算法, 待 costmodel 对应区段系数修正后移除
op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
    u64 totalSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType] * topo->userRankSize;
    return !(topo->userRankSize == 16 && totalSize >= 32ULL * 1024 * 1024
             && totalSize <= 64ULL * 1024 * 1024 && topo->level0Topo == Level0Shape::MESH_1D);
};
```

16P 两级 MESH_1D、total 32~64MB（per-rank 2~4MB）时，cost model 对 `AicpuAllGatherSoleNHR` 估值偏低导致误选，靠 `opCustomCheck` 硬排除。与 ① 同类根因：cost model 的分段线性近似在特定 dataSize 带内失真，只能靠硬 gate 打补丁。

### ④ `perTransferSize==0` 防护

`cost_table.cc:392`：NHR `dr×rs/2` 折半后可能归 0，回退到 `dataSize`，属查表路径的边界防护。

```cpp
u64 perTransferSize = static_cast<u64>(static_cast<float>(dataSize) * dr);
if (perTransferSize == 0) {
    perTransferSize = dataSize;
}
```

### ⑤ cost 平票

`selector_engine.cc:693-703`：多个算法 cost 完全相同（浮点相等）时取 `tiedAlgos[0]`（注册序第一个），仅打 WARNING。

### ⑥ AllGather CLOS 小数据 util 钳制

`cost_table.cc:480-485`：AllGather 在 CLOS 且 dataSize < 1MB 时统一用 1MB 的 util，是小数据段的特殊处理。

## 8. 结论

引入 cost model 后，allgather 的"决策树" = **引擎优先级分层 + 每层内 min-cost（分段线性曲线下界）**。真正的分叉点由两类东西决定：

1. `A/B/C/D` 系数（闭式，取决于 rankSize/portNum/isPod）
2. `util(ds)` 分段表（阶跃）

硬编码 if/else 没有消失，而是以 `opCustomCheck`/`topoPriorityCheck` 等形式作为"补丁叶子"挂在树梢——③ 即典型证据，④⑤⑥ 则是边界防护与平票兜底。

## 附：关键文件索引

| 内容 | 位置 |
|---|---|
| 选路分叉 | `src/ops/op_common/op_common.cc:225` |
| 新 cost model 入口 | `src/ops/op_common/selector/selector_engine.cc:413` |
| cost 计算 | `src/ops/op_common/selector/cost_table.cc:360` |
| util 表 + 查表 | `src/ops/op_common/selector/cost_table.cc:437` / `:461` |
| cost model 管理 | `src/ops/op_common/selector/cost_model.cc` |
| 老硬编码决策树 | `src/ops/all_gather/selector/all_gather_auto_selector.cc` |
| base selector | `src/ops/op_common/selector/auto_selector_base.cc` |
| Mesh / NHR 成本 | `src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc` / `ins_temp_all_gather_nhr.cc` |
| Parallel 成本 | `src/ops/all_gather/algorithm/executor/ins_v2_all_gather_parallel_executor.cc:52` |
| 算法注册/attrs | `src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sole_executor.cc` |
| 16P corner case | `src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sole_executor.cc:409` |