# AllGatherV、Reduce 零拷贝模式预调研纪要

| 项 | 内容 |
|----|------|
| 文档状态 | 预调研草稿（供 SE 需求串讲前对齐，非最终设计） |
| 需求名称 | 支持 AllGatherV、Reduce 算子的零拷贝模式 |
| 调研范围 | 本仓 `cann/hccl` 现网实现与文档，不含 `cann/hcomm` 内部实现 |
| 目的 | 摸清"零拷贝"在本仓的真实机制与现状，列出需与 SE 确认的关键设计问题 |

> 说明：本文把"已从代码/文档确认的事实"与"待 SE 确认的判断"分开表述。带 `file:line` 的为事实依据；标注 **【待确认】** 的为需要串讲时对齐的内容。

---

## 1. 背景与目的

新需求为"支持 allgatherV、reduce 算子的零拷贝模式"，目前仅知需求名称。本纪要用于：

1. 说明本仓"零拷贝"当前到底指什么、已经支持哪些算子；
2. 给出 AllGatherV、Reduce 两个算子与现有零拷贝链路的差距；
3. 汇总需与 SE 对齐的定义、范围与关键设计问题，避免后续按错误前提做方案。

---

## 2. 术语与定义（首要待确认项）

### 2.1 本仓"零拷贝"的实际含义

代码中存在两个容易混淆的字段：

- `OpParam::supportSymmetricMemory`（`src/ops/op_common/inc/alg_param.h:688`）：**真正生效的开关**，表示本次调用的用户 buffer 已注册对称内存，可被对端直接读写。
- `OpParam::isZeroCopy`（`src/ops/op_common/inc/alg_param.h:743`、`:773`）：全仓没有任何位置将其置 `true`，仅参与日志打印与 AICPU task cache key。也就是说，`isZeroCopy` 目前不是功能开关。

因此，当前 HCCL 所谓"单算子零拷贝"= **对称内存零拷贝**：用户在通信域注册对称内存后，HCCL 直接对业务传入的 buffer 进行远端读写，省去 ccl buffer 中转。

官方文档定义："为了降低内存拷贝开销，使得 HCCL 可以直接对业务传入的内存进行操作，提升通信性能"（`docs/zh/user_guide/comm_ops_support_list/comm_ops_support_a3.md:5`）。

### 2.2 待 SE 确认清单

| 编号 | 待确认问题 | 影响 |
|------|-----------|------|
| D1 | "零拷贝"是否即指对称内存零拷贝（依赖 `HcclCommSymWinGet` 注册）？是否存在其他零拷贝编程模型？ | 决定整体方案基线 |
| D2 | 目标产品与拓扑范围：是否仅 A3/950、单算子 `OPBASE`、`AICPU_TS`、一级 `MESH_1D`？是否含多级拓扑/超节点间？ | 决定门禁条件与算法覆盖 |
| D3 | 是否只做 AI CPU 展开？CCU、AIV 是否在范围内？ | AIV 当前明确不支持零拷贝（`comm_ops_support_a3.md:1115`） |
| D4 | 变长算子（AllGatherV）的零拷贝是否要求所有 rank 都注册对称内存，还是允许部分 rank 回退？ | 决定探测失败时的行为 |
| D5 | 与确定性计算、重执行的关系是否沿用 AllReduce/ReduceScatter 的口径？ | 影响对外约束与测试 |

---

## 3. 现有零拷贝机制（已支持算子）

### 3.1 端到端链路

| 阶段 | 位置 | 关键动作 |
|------|------|---------|
| 入口探测 | `src/ops/op_common/op_common.cc:4650` `CheckAndSetSymmetricMemory` | 对 input/output 分别调 `HcclCommSymWinGet` 取窗口句柄与 offset；两侧都拿到才置 `supportSymmetricMemory=true`，否则回退普通路径 |
| 入口条件 | 各算子入口，如 `src/ops/all_gather/all_gather.cc:242`、`src/ops/all_reduce/all_reduce.cc:233` | 版本≥9.1.0、`OPBASE`、`AICPU_TS`、特定一级拓扑且满足算子附加条件时才探测 |
| 统一门禁/回退补齐 | `src/ops/op_common/op_common.cc:4609` `RefreshSymmetricMemory`，调用点 `:1046` | 初始选择、资源不足 Fallback、回退缓存命中三条路径统一按最终 `algName/engine` 复核；回退到 `AICPU_TS` 时补做探测 |
| 执行器传参 | `src/ops/op_common/algorithm/executor/executor_v2_base.h:254` `SetSymmetricMemoryInfo` | 把 `inputSymWindow/outputSymWindow/inputOffset/outputOffset` 拷入执行器成员 |
| 对端地址填充 | `executor_v2_base.cc:81` → `template_utils.cc:647` `FillChannelSymWinPeerAddrs` | 用 `GetSymWinRemoteMem` 为每个 channel 填充对端 user buffer 地址 |
| 模板直连 | 例：`src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_omnipipe_mesh_1D.cc:125` | 用对端窗口地址直接读写，跳过 ccl scratch 与本地拷贝 |

对端窗口地址获取封装在 `src/common/hcomm_dlsym/hccl_sym_win_dl.cc:28`，优先 `HcclSymWinGetRemoteAddr`，否则回退 `HcclSymWinGetPeerPointer`，均为 dlsym 动态加载，符合两仓解耦约束。

### 3.2 已支持算子与门禁

`RefreshSymmetricMemory`（`src/ops/op_common/op_common.cc:4609`）中的算子白名单：

| 算子 | 现状 | 附加门禁 |
|------|------|---------|
| AllGather | 支持 | 入口另有拓扑/数据量判断（`all_gather.cc:234-242`） |
| Broadcast | 支持 | 一级 `MESH_1D` |
| AllToAll / AllToAllVC | 支持 | 一级 `MESH_1D` |
| AllReduce | 支持 | `IsReduceOpSymMemAllowed` + `IsReduceOpSymMemDeterministic`（`op_common.cc:4581`、`:4592`） |
| ReduceScatter | 支持 | 同上，且要求 `inputPtr != outputPtr`（`op_common.cc:4627`） |
| **AllGatherV** | **不支持** | 入口未接入探测 |
| **Reduce** | **不支持** | 入口未接入探测 |

规约类算子的两个门禁：

- `IsReduceOpSymMemAllowed`：排除 `int64/uint64/fp64` 与 `HCCL_REDUCE_PROD`（`op_common.cc:4581`）。
- `IsReduceOpSymMemDeterministic`：仅放行由确定顺序的 PipeLineMeshNHR 模板完成归约的算法（`AicpuReduceScatterPipeLineMeshNHR`、`AicpuAllReducePipeLineMeshNHR`），多级拓扑禁止（`op_common.cc:4592`）。

### 3.3 已对外声明的约束

- 零拷贝会就地修改用户输入：AllReduce/ReduceScatter 打开零拷贝后 `sendBuf` 被当作对称窗口参与就地规约，输入会被污染，要求用户自行备份；且这两类算子无法支持重执行（`docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md:59`、`docs/zh/api_ref/comm_op_interface/HcclAllReduce.md:115`）。
- AIV 模式不支持单算子零拷贝（`comm_ops_support_a3.md:1115`）。
- 零拷贝不支持 AICPU task cache 与 warmup（`src/ops/op_common/algorithm/template/aicpu/task_cache/aicpu_task_cache_policy.cc:62`、`src/ops/op_common/op_common.cc:881`）。

---

## 4. AllGatherV 现状与差距

### 4.1 现状

- 入口：`HcclAllGatherV` → `AllGatherVOutPlace`（`src/ops/all_gather_v/all_gather_v.cc:195`），组装 `OpParam`（`param.opType = HCCL_CMD_ALLGATHER_V`，`:253`）后直接 `Selector` + `HcclExecOp`（`:274`）；**全程未调用 `CheckAndSetSymmetricMemory`**，param 的 sym 字段保持默认。
- 选择器：AICPU 仅 `AicpuAllGatherVSoleMesh`（`src/ops/all_gather_v/selector/all_gather_v_auto_selector.cc:91`）；CCU 为 `CcuSchedAllGatherVSoleMesh`（`:63`）。
- 执行器/模板：`InsV2AllGatherVSoleExecutor` → `InsTempAllGatherVMesh1D`。
- 模板中已有 `enableRemoteMemAccess_` 分支（`ins_temp_all_gather_v_mesh_1D.cc:154-162`），但由 `param.opMode == OpMode::OFFLOAD` 驱动（`ins_v2_all_gather_v_sole_executor.cc:104`），属于**图模式远程内存**，并非对称窗口；且仍执行 `LocalDataCopy`/`PostLocalCopy`（`:92`、`:110`）。

### 4.2 差距与风险点

1. **入口未探测**：需在 `AllGatherVOutPlace` 增加 `CheckAndSetSymmetricMemory` 调用，并在 `RefreshSymmetricMemory` 白名单加入 `HCCL_CMD_ALLGATHER_V`。
2. **模板未用对称窗口**：模板只区分"scratch / `remoteOutputGraphMode`"，需改为在 sym 生效时用 `GetSymWinRemoteMem(outputSymWindow, outputOffset, peerRank, ...)` 取对端 output 地址。
3. **变长 semantics**：
   - `recvCounts/recvDispls` 存在 `param.varData`，各 rank 的接收量不同，远端写入偏移需按 `displs[algRank]` 计算。
   - `outputSize` 口径不一致：`AllGatherVOutPlace` 用 `sum(recvCounts)`（`all_gather_v.cc:207`），`AllGatherVOutPlaceGraphMode` 用 `max(displs+count)`（`all_gather_v.cc:312`）。窗口注册与 offset 计算需统一到用户实际 buffer 口径，否则可能漏注册或越界。**【待确认】**
4. **in-place 已禁止**：`sendBuf == recvBuf` 被显式拒绝（`all_gather_v.cc:166`），对零拷贝是有利条件（无读写重叠），但也意味着不需要处理 in-place 场景。
5. **算法覆盖窄**：当前 AICPU 只有单层 `SoleMesh`，多级/超节点算法未接入，需确认需求是否只要求单层。

---

## 5. Reduce 现状与差距

### 5.1 现状

- 入口：`HcclReduce` → `ReduceOutPlaceCommon`（`src/ops/reduce/reduce.cc:219`），组装 `OpParam`（`param.opType = HCCL_CMD_REDUCE`，`:208`）后 `Selector` + `HcclExecOp`（`:261`）；**未接入对称内存探测**。
- 语义：root 语义。以 `ReduceMesh1D` 为例（`src/ops/reduce/algorithm/template/aicpu/reduce_mesh_1D.cc:132` `RunReduce`）：
  - 非 root：`SendData` 把本端 input 写入 root 的 ccl buffer（`:163`、`:168`）；
  - root：`GatherData`（`:144`）+ `ReduceData`（`:160`）在 root 本地完成归约。
- AICPU 可选算法：`AicpuReduceSoleMesh/TwoShot`、`AicpuReduceSoleNHR`、`AicpuReduceParallelMeshNHR/MultiJetty`、`AicpuReduceSequenceMeshConcurNHRNHR` 等（`reduce_auto_selector.cc:282-366`），**没有与 AllReduce/ReduceScatter 对应的 `PipeLineMeshNHR` 确定性算法**。

### 5.2 差距与风险点

1. **入口未探测**：需在 `ReduceOutPlaceCommon` 增加探测，并在 `RefreshSymmetricMemory` 白名单加入 `HCCL_CMD_REDUCE`。
2. **数据流需重设计**：现有是"非 root 写 root ccl buffer → root 本地归约"。零拷贝下有两条可选路径，需与 SE 拍板：
   - 路径 A：root 用对端 input 窗口直接"读 + 归约"到本端 output；
   - 路径 B：各 rank 直接写入 root 的 output 窗口，再由 root 归约。
   两条路径对"哪些 rank 需要 input/output 窗口"要求不同。**【待确认】**
3. **确定性无现成算法**：`IsReduceOpSymMemDeterministic` 只认 PipeLine 类算法，Reduce 的 AICPU 算法集合里没有对应项。要么新增/改造确定性算法，要么把 Reduce 零拷贝限定在特定算法并补门禁。**【待确认】**
4. **类型/规约门禁**：是否沿用 `IsReduceOpSymMemAllowed`（排除 `int64/uint64/fp64` 与 `PROD`）。注意 Reduce 对 64bit/PROD 已有专门的 AICPU 规约分支（`reduce_mesh_1D.cc:150`）。**【待确认】**
5. **in-place**：Reduce 的 `sendBuf == recvBuf` 是否允许、零拷贝下 root 侧是否会被污染，需明确。**【待确认】**

---

## 6. 待与 SE 对齐的关键设计问题（汇总）

**A. 定义与范围**
- A1 零拷贝定义（D1）与产品/拓扑/引擎范围（D2、D3）。
- A2 是否与现有 AllReduce/ReduceScatter 完全同构，还是允许新的窗口/语义约定。

**B. AllGatherV 专项**
- B1 首批支持算法（是否仅 `AicpuAllGatherVSoleMesh`）；是否要求多级。
- B2 `outputSize`/offset 口径统一；`displs` 非连续场景如何处理。
- B3 是否复用模板已有 `enableRemoteMemAccess_` 分支，还是新增 sym 专用分支。

**C. Reduce 专项**
- C1 root 侧数据流选型（路径 A 还是 B）。
- C2 各 rank 窗口需求（input-only / output-only / both）。
- C3 确定性算法方案与门禁口径。
- C4 类型/规约/in-place 约束。

**D. 横切约束**
- D1 是否禁用 AIV、是否禁 task cache/warmup。
- D2 重执行口径（沿用"零拷贝不可重执行"）。
- D3 用户契约：是否需在 API 文档声明输入污染。
- D4 支持表更新：`comm_ops_support_a3.md` 中 AllGatherV（`:129`）、Reduce（`:176`）由 `×` 改 `√` 的条件与时机。

**E. 测试与验收**
- E1 UT/ST 覆盖范围（现有 sym 相关测试较少，可参考 `test/st/algorithm/testcase/reduce_scatter_dpu_testcase.cc` 等）。
- E2 与普通路径的性能对比口径、回退正确性验证。

---

## 7. 后续实施建议（待 SE 对齐后细化）

建议分两阶段，先易后难：

1. **第一阶段：AllGatherV**
   - 入口接入探测；`RefreshSymmetricMemory` 白名单新增 `HCCL_CMD_ALLGATHER_V`；
   - 执行器透传 sym 信息，模板新增/复用对称窗口分支；
   - 明确变长 offset 口径并补测试。
2. **第二阶段：Reduce**
   - 先定 root 数据流与确定性算法方案；
   - 再接入探测、执行器与模板；
   - 补类型/规约/in-place 门禁与污染声明。

每个算子的最小改造面（供后续排期参考）：

| 层次 | 改造点 |
|------|--------|
| 入口 | `<op>.cc` 增加 `CheckAndSetSymmetricMemory`，补算子附加条件 |
| 公共门禁 | `op_common.cc` `RefreshSymmetricMemory` 白名单与门禁函数 |
| 执行器 | `supportSymmetricMemory_ = param.supportSymmetricMemory` + `SetSymmetricMemoryInfo` |
| 模板 | 用 `GetSymWinRemoteMem` 取对端地址，跳过 ccl scratch/本地拷贝 |
| 文档 | 支持表、API 约束（输入污染、重执行） |
| 测试 | UT/ST 覆盖 sym 命中与回退 |

> 本阶段仅做调研，未改动任何源码。

---

## 8. 证据索引

| 主题 | 位置 |
|------|------|
| `supportSymmetricMemory` / `isZeroCopy` 定义 | `src/ops/op_common/inc/alg_param.h:686-688`、`:743`、`:773` |
| 对称内存探测 | `src/ops/op_common/op_common.cc:4650` |
| 统一门禁/回退 | `src/ops/op_common/op_common.cc:4609`、调用点 `:1046` |
| 规约类门禁 | `src/ops/op_common/op_common.cc:4581`、`:4592` |
| AllGather 入口探测 | `src/ops/all_gather/all_gather.cc:242` |
| AllReduce 入口探测/门禁 | `src/ops/all_reduce/all_reduce.cc:233-262` |
| 执行器传参 | `src/ops/op_common/algorithm/executor/executor_v2_base.h:254` |
| 对端地址填充 | `src/ops/op_common/algorithm/executor/executor_v2_base.cc:81`、`src/ops/op_common/algorithm/template/template_utils.cc:647` |
| 远端窗口地址 API | `src/common/hcomm_dlsym/hccl_sym_win_dl.cc:28` |
| 模板直连示例 | `src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_omnipipe_mesh_1D.cc:125` |
| AllGatherV 入口 | `src/ops/all_gather_v/all_gather_v.cc:195`、`:207`、`:253`、`:274` |
| AllGatherV 变长/图模式 | `src/ops/all_gather_v/all_gather_v.cc:166`、`:312` |
| AllGatherV 模板 remote 分支 | `src/ops/all_gather_v/algorithm/template/aicpu/ins_temp_all_gather_v_mesh_1D.cc:82-111`、`:154-162` |
| AllGatherV 选择器 | `src/ops/all_gather_v/selector/all_gather_v_auto_selector.cc:63`、`:91` |
| Reduce 入口 | `src/ops/reduce/reduce.cc:219`、`:208`、`:261` |
| Reduce root 数据流 | `src/ops/reduce/algorithm/template/aicpu/reduce_mesh_1D.cc:132-168` |
| Reduce AICPU 算法 | `src/ops/reduce/selector/reduce_auto_selector.cc:282-366` |
| 支持表现状 | `docs/zh/user_guide/comm_ops_support_list/comm_ops_support_a3.md:129`、`:176`、`:1115` |
| 重执行约束 | `docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md:59` |
| 输入污染声明 | `docs/zh/api_ref/comm_op_interface/HcclAllReduce.md:115` |
| task cache / warmup 禁用 | `src/ops/op_common/algorithm/template/aicpu/task_cache/aicpu_task_cache_policy.cc:62`、`src/ops/op_common/op_common.cc:881` |
