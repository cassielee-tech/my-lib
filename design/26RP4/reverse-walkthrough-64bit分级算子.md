# 需求反串讲：Reduce 算子 64bit+特殊规约两级分级算法

| 项目 | 内容 |
| ---- | ---- |
| 反串讲人 | 开发（本人），**仅负责 Reduce 算子** |
| 依据文档 | SE 串讲文档《plan-64bit分级算子.md》（覆盖 AllReduce/ReduceScatter/Reduce 三算子）+ 仓内代码核实（hccl，2026-10-13） |
| 责任边界 | AllReduce/ReduceScatter 为同型兄弟改造，由他人负责；三算子代码各自独立（selector/executor 分文件、无共同改动点），共用模板 `InsTempReduceScatterAicpuReduceNHRPcie` / `InsTempAllGatherNHR` 但**均不修改**，无代码耦合，仅交付节奏与验收口径需对齐 |
| 核实方式 | SE 文档 Reduce 相关论断已逐条对照 `src/ops/` 代码验证，结论见第 4 节 |

## 1. 需求理解（我的复述）

**一句话**：Reduce 算子在 64bit 数据类型（INT64/UINT64/FP64）与特殊规约（PROD，及伴随 64bit 的 MIN/MAX）场景下，跨框两级拓扑不再退化为扁平 Sole NHR，改为"框内+框间"两级 sequence 编排（nhr+nhr），复用现有 aicpu_reduce 系列模板；**正确性优先，无性能要求**。

**范围矩阵**：

| 维度 | 做 | 不做 |
| ---- | ---- | ---- |
| 算子 | **Reduce**（本人） | AllReduce/ReduceScatter（他人，同型参照） |
| 拓扑 | 两级（topoLevelNums==2） | 三级（维持 Sole）、单机（路由不变） |
| 进入条件 | Is64BitDataType ∥ PROD（任意 dtype） | 非 64bit 的 min/max（走常规模板，**设计确认点**） |
| 引擎 | aicpu | aiv/ccu/dpu |
| 交付物 | 1 处 executor 注册 + 1 处 selector 分支 + include + UT/ST | 不新增模板、不改 executor/template 主体、不改 costModel、无 HCOMM 跨仓依赖 |

## 2. 现状理解（已核实，Reduce 部分）

1. **template 层仅有 Sole 实现**：`src/ops/reduce/algorithm/template/aicpu/reduce_aicpu_reduce_nhr.cc`（NHR gather + 本地标量 reduce，扁平单级）。
2. **executor 层仅有 Sole 注册**：`AicpuReduceSoleNHRAicpuReduce`（`reduce_sole_executor.cc:357`，`TopoMatchOneLevel`）——整域扁平跑 NHR，跨框时不分框内/框间链路。
3. **selector 硬编码回退**：`reduce_auto_selector.cc:281`——`topoLevelNums > 1` 且 64bit/PROD 时一律路由 Sole。
4. **根因链**（背景理解）：分级算法（Parallel/Sequence 常规组合）被排除是因为其层间模板 `InsTempReduceScatterNHR` 用在线归约原语（`HcommWriteReduceOnThread` 等，`alg_data_trans_wrapper.cc:202-241` 直通 HCOMM，无 64bit/PROD 兜底）；而 `LocalReduce` 对 64bit/PROD 走 `AicpuReduce` 控制核标量规约（`alg_data_trans_wrapper.cc:905-908`），需 `HcommBatchModeEnd/Start + HcommThreadJoin` 同步——aicpu_reduce 系列模板天然兼容此路径。这就是新方案弃用 `InsTempReduceScatterNHR`、两级均改用 aicpu_reduce 系列模板的原因。

## 3. 方案理解（Reduce）

**编排**：复用 `ReduceSequenceExecutorAicpu3Level` 做两级注册（`TopoMatchTwoLevel` + `skipLevel2_`，现有 `AicpuReduceSequenceMeshConcurNHR` 即此模式，`reduce_sequence_executor_aicpu_3level.cc:841`）：

```
RS-L0 框内: InsTempReduceScatterAicpuReduceNHRPcie（NHR 交换 + aicpu LocalReduce）
RS-L1 框间: InsTempReduceScatterAicpuReduceNHRPcie
AG-L1 框间: InsTempAllGatherNHR
AG-L0 框内: InsTempAllGatherNHR（仅 root 写 user output，isRootRank_ 控制）
L2 两个模板位因 skipLevel2 不会被拉起，按现有惯例填同款占位
```

**注册内容**（`reduce_sequence_executor_aicpu_3level.cc` 文件尾）：

```
REGISTER_EXEC_V2_MULTI(HCCL_CMD_REDUCE, AicpuReduceSequenceNHRAicpuReduceNHR, ReduceSequenceExecutorAicpu3Level,
    TopoMatchTwoLevel,
    InsTempReduceScatterAicpuReduceNHRPcie,   // RS-L0
    InsTempReduceScatterAicpuReduceNHRPcie,   // RS-L1
    InsTempReduceScatterAicpuReduceNHRPcie,   // RS-L2 占位（skipLevel2）
    InsTempAllGatherNHR,                      // AG-L2 占位（skipLevel2）
    InsTempAllGatherNHR,                      // AG-L1
    InsTempAllGatherNHR);                     // AG-L0（root 输出）
+ REGISTER_ALG_ATTRS（min/maxTopoLevelNum=2；op.isSupportInplace=false 对齐现有 Reduce 分级算法）
+ 补 include ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.h
```

**关键选型**（理解的理由）：两级 RS 复用 `InsTempReduceScatterAicpuReduceNHRPcie`（分级版）而非 Sole 版，因为分级版具备 `GetAlgRank` 映射（userRank≠算法内索引）、`tailSize` 尾块、`outputSliceStride`、`repeatNum`、in/out 同 buffer 处理——Sole 版直接用于两级编排会错乱；AG 为纯搬运无规约，64bit 无障碍。已核实该模板具备上述全部语义（`ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.cc:65,74,78,120,164,241,282`），且 `IsPcieProtocol` 自动判链路（:71），非 pcie 链路走 SendRecv 通用路径（跨算子参照：AllReduce/RS 的 Sequence 注册已将其用作框间模板）。

**selector 路由**（`reduce_auto_selector.cc` `SelectAicpuAlgo` :281 分支改造）：

```cpp
if (Is64BitDataType(dataType) || opParam.reduceType == HCCL_REDUCE_PROD) {
    if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_2) {
        selectAlgName = "AicpuReduceSequenceNHRAicpuReduceNHR";  // 新：两级分级
    } else {
        selectAlgName = "AicpuReduceSoleNHRAicpuReduce";         // 三级维持 Sole
    }
}
```

单机各分支、保序（DETERMINISTIC_STRICT）拦截均不变。

**costModel**：新算法靠模板无静态 `CalcCostCoeff` 不参与 cost 排序，仅 selector 特判选中（**此条有出入，见 4-5**）。

## 4. SE 文档论断核实结果（Reduce 相关）

| # | SE 论断 | 核实 | 代码依据 |
| - | ---- | ---- | ---- |
| 1 | Reduce 3level executor 支持两级注册 + skipLevel2 + root 语义（AGL0 仅 root 落 user output） | ✅ | `AicpuReduceSequenceMeshConcurNHR`（`reduce_sequence_executor_aicpu_3level.cc:841`，TopoMatchTwoLevel）；root 语义见 `GenTempAlgParamsAGL0` |
| 2 | NHRPcie 具备 algIdx 映射/尾块/stride/repeatNum 两级编排语义，可作框内/框间模板 | ✅ | `ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.cc:65,74,78,120,164,241,282`；`IsPcieProtocol` 链路自适应（:71） |
| 3 | `InsTempAllGatherNHR` 具备 GetAlgRank 分级映射 | ✅ | `ins_temp_all_gather_nhr.cc:424,460,521` |
| 4 | `InsAlgTemplateBase::CalcCostCoeff` 默认返回空 | ✅ | `alg_v2_template_base.h:28` |
| 5 | **Reduce 3level executor"聚合后同样得到空结果、不参与 cost 排序"** | ⚠️ **不符** | `reduce_sequence_executor_aicpu_3level.cc:78-110` 的 `CalcCostCoeff` 为**直接拼接 p0..p5、无"任一模板空则整体退出"短路**（该短路仅 RS executor 有：`ins_v2_reduce_scatter_sequence_executor_aicpu.cc:133-136`）；且 `InsTempAllGatherNHR` **有**静态 `CalcCostCoeff`（`ins_temp_all_gather_nhr.cc:18`）→ 新算法（NHRPcie 空×3 + AG NHR 有×3）将得到**非空的部分系数** |
| 6 | NHRPcie 已被 sequence 注册用作 RS 模板（跨算子参照） | ✅ | `AicpuAllReduceSequenceMeshNHRAicpuReduce`（`ins_v2_all_reduce_sequence_executor_aicpu.cc:893`）、`AicpuReduceScatterSequenceMeshNHRAicpuReduce`（`ins_v2_reduce_scatter_sequence_executor_aicpu.cc:591`）——但均为单机 pcie-mix 场景、且仅作框间模板（见风险 1） |
| 7 | 注册宏 `REGISTER_EXEC_V2_MULTI` 可用、Reduce Sole 现状描述 | ✅ | `reduce_sequence_executor_aicpu_3level.cc:833` 同型使用；`reduce_sole_executor.cc:357` |

## 5. 风险与补充（反串讲重点）

**SE 已列风险**（框内复用未验证 / scratch 相乘 loop 增多 / 三级误入 / 非 64bit min-max 语义 / Reduce 对称内存）均已知悉，按 Reduce 视角补充 3 条：

1. **【高】cost 排序误选风险（对应 4-5 出入）**：新算法不设 op 侧白名单 + 非空部分系数（仅 AG 部分、缺 RS 项 → cost 低估），若 cost table 参与选择，常规非 64bit 场景可能被误选。注意**不能照抄**现有 `AicpuAllReduceSequenceMeshNHRAicpuReduce` 的白名单（`supportedDataTypes={64bit}` + `isSupportProd=true`）——本需求进入条件是"64bit **或** PROD@任意dtype"，白名单会挡掉 PROD@fp16/fp32。建议三选一：
   - ① Reduce 3level executor `CalcCostCoeff` 补"任一模板空则退出"短路（对齐 RS executor 模式，改动最小且根治，**推荐**；需与 AllReduce 负责人协同，其 executor 同样无短路）；
   - ② `opCustomCheck` 精确表达进入条件（`Is64BitDataType(dt) || op == PROD`）；
   - ③ 确认 cost table 流不覆盖 Reduce selector 特判路径并留证据。
2. **【中】算法命名与组合不一致**：按 `<框内RS><框间RS>` 惯例（对照 `...SequenceMeshNHRAicpuReduce` = Mesh1D+NHRPcie），两级均 NHRPcie 的新算法尾部应为 `NHRAicpuReduce` ×2，SE 命名 `AicpuReduceSequenceNHRAicpuReduceNHR` 尾段为 `NHR`。需确认是刻意简化还是笔误；且 `HCCL_ALGO` 字符串解析（`alg_parse.cc` 按 `nhraicpureduce` 等分段）需验证新名可被正确指定/检索。
3. **【中】loop 数暴增与 `CUSTOM_TIMEOUT`**：scratch multiplier 相乘致 `maxCountPerLoop` 骤减，大数据量下 loop 数与 `HcommBatchModeEnd/Start + ThreadJoin` 次数线性放大——虽无性能要求，但需确认 notify/执行超时（`aicpu_timeout.h` 的 CUSTOM_TIMEOUT）不会因 loop 过多误报超时；建议 ST 覆盖 GB 级数据量。
4. **【SE 风险 1 的 Reduce 侧强化】NHRPcie 框内（L0）复用零验证**：跨算子参照也仅将其用作框间模板、且均在单机 pcie-mix 场景；本需求中它将首次作为 L0 模板运行于**框内 HCCS 链路**（非 pcie → `SendRecvWrite` 路径），需 ST 双机重点覆盖。

## 6. 待确认问题（请 SE 评审时拍板）

1. 发现 1 的处置方案（Reduce 3level executor 补短路 / opCustomCheck / 提供不覆盖证据，推荐 ①，需与 AllReduce 负责人对齐同一方案）；
2. 非 64bit 的 min/max 是否进入新路径（SE 已列为设计确认点，默认不改）；
3. 新算法命名尾段 `NHR` 是否修正为 `NHRAicpuReduce`（含 `HCCL_ALGO` 解析验证）；
4. 三级拓扑维持 Sole 是否为本迭代硬边界（影响命名/attrs 预留）；
5. 三算子虽代码解耦，是否同一 Issue/同一验收口径交付？Reduce 侧是否可独立提 PR、独立验收；
6. （转交）SE 文档中 RS Sole 注册行号 :404 与实测 :605 附近不符，请 AllReduce/RS 负责人复核其文档基线。
