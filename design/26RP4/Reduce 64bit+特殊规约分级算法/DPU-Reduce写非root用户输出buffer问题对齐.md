# DPU Reduce 写非 root 用户输出 buffer 问题（存量缺陷）对齐材料

> **结论先行**：hostDpu 模式下，Reduce 算法 `DpuReduceSequenceMeshNHR` 会把规约中间数据
> 写进**非 root rank 的用户输出 buffer**，违反 HcclReduce 对外语义（仅 root 输出生效）。
> 该缺陷为存量问题（2026-09-02 引入），与 feature 分支"64bit/PROD 两级分级算法"无关，
> 系 ST 校验器新增断言后首次暴露。当前分支先撤断言恢复 CI，产品修复建议单独排期。

## 1. 发现过程时间线

| 时间 | 事件 |
|---|---|
| 2026-09-02 | 违规代码随 `06a6848a` 入库（`InsTempGatherMesh1dIntra` 模板） |
| 10:36 CI run | 新 ST 二进制段错误（另一独立问题，已由 `1b90618e` ST 桩修复）；当时 st_reduce_dpu_test 通过 |
| 12:09 | `d5479596` 给 ST 语义校验器补"非 root 用户输出 buffer 不被写"断言（+14 行） |
| 14:20 CI run | 断言首次生效，`st_reduce_dpu_test` 暴露本缺陷 |

## 2. 失败现象与用例逐字段吻合

失败用例：`ST_REDUCE_DPU_TEST.host_dpu_opbase_reduce_1_fp32_sum`（9 rank=3srv×3rank、
count=1×FP32、SUM、root=1）。校验器报错 7 个字段全部对上：

| 报错字段 | 用例参数 |
|---|---|
| `[rankId:0]`（首个被写的非 root） | root=1，rank0 迭代序最先 |
| `size is 4` | 1 个 FP32 元素 |
| `HcclReduceOp is 0` | SUM（继承自规约后源数据的语义） |
| srcBufs ranks 0-8 | 9 rank 全量血缘 |

## 3. 根因定位（代码证据链）

**选路**：`ENABLE_HOSTDPU_FOR_LLT=1`（reduce_testcase_common.h:140）→ hostDpuOnly
（auto_selector_base.cc:24-28）→ 硬编码选路 `DpuReduceSequenceMeshNHR`
（reduce_auto_selector.cc:470-473）。

**违规写**（缺陷本体）：
- DPU 执行器仅最后一段绑用户输出：stages 1-3 的 output 全绑 cclBuff 的 cclOut 区
  （CCL 类型，合法）；仅 stage4 绑 `param.outputPtr` + `BufferType::OUTPUT`
  （ins_v2_reduce_sequence_executor.cc:232-235，root 已传入 :237）
- stage4 `InsTempGatherMesh1dIntra::LocalDataCopy` **无 root 门控**（仅 in-place 跳过），
  所有 rank 把自己 slice 拷进自己的用户 outputPtr（ins_temp_gather_mesh_1D_intra.cc:207-249）
- **门控缺失的铁证**：同文件 `PostLocalCopy` 有 `if (myRank_ != tempAlgParams_.root)
  continue;`（:266），唯独 LocalDataCopy 没有
- 正确姿势参照：`ReduceSequenceExecutorAicpu3Level::GenBaseTempAlgParams` 仅
  `isRootRank_` 时给最后一段绑 OUTPUT（reduce_sequence_executor_aicpu_3level.cc:398-406）

**报错字段全解释**：rank 0 的 output[0..4B] 语义条目由校验器的语义继承机制生成——
OVERRIDE 拷贝会原样继承源语义块的 isReduce/reduceType/srcBufs
（task_check_op_semantics.cc:231），源的"已规约、9-rank 血缘"来自 stages 1-3 的 cclOut 区，
故报错显示 `HcclReduceOp is 0`、srcBufs 0-8、size 4，全部对上。

## 4. 为什么判定为存量缺陷（非本分支引入）

1. 违规模板/执行器 2026-09-02 入库，本分支功能 2026-10-27；此前仅日志文案类改动
2. 本分支 3 个提交（`5bbdb792`/`d5479596`/`1b90618e`）零 DPU 文件
3. 10:36 run 同一份 DPU 代码通过（当时无断言）——行为未变，只是以前不查
4. 新算法被 64bit/PROD 软门禁挡在失败场景之外（FP32+SUM 进不了其候选池）

## 5. 影响评估

- **触发条件**：hostDpuOnly 部署形态 + Reduce + topoLevelNums>1 + level0 为 MESH_1D（或
  deviceNumPerModule==1）——即选中 `DpuReduceSequenceMeshNHR` 的所有场景；**任意 root**
  （root=0 同样违规，非 root 全体输出被写）
- **真机表现**：非 root rank 用户传入的 recvBuf 被规约中间结果覆盖（静默数据破坏）
- **影响面**：`InsTempGatherMesh1dIntra` 唯一使用方即 `DpuReduceSequenceMeshNHR`；
  建议排查时顺带确认 `DpuReducePipeLineMeshNHRNHR` 等其它 DPU reduce 算法
- 严重级别建议：**中高**（违反对外 API 契约，但依赖 hostDpu 部署形态）

## 6. 修复建议（请决策）

- **方案 A（推荐）**：`InsTempGatherMesh1dIntra` 的 LocalDataCopy/交换路径改走 cclBuff，
  用户输出写统一 root 门控（参照 isRootRank_ 绑定法）；需 ST + hostDPU 真机用例背书，
  注意 hostDPU 热路径性能回归
- **方案 B**：若 DPU 场景存在"允许写非 root 输出"的历史契约，则记录后放开 ST 断言
- 无论 A/B，建议建 GitCode issue 跟踪，与 feature 分支解耦

## 7. 当前分支处置

- 段错误已修复（`1b90618e`，ST 桩 LocalReduce 符号介入，22 条新用例已过）
- 本分支已 revert 断言提交恢复 CI 绿；DPU 修复独立 PR
