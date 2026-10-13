# 需求反串讲：64bit+特殊规约两级分级算法（AllReduce / ReduceScatter / Reduce）

| 项目 | 内容 |
| ---- | ---- |
| 反串讲人 | 开发（本人） |
| 依据文档 | SE 串讲文档《plan-64bit分级算子.md》+ 仓内代码核实（hccl，2026-10-13） |
| 核实方式 | SE 文档全部关键论断已逐条对照 `src/ops/` 代码验证，结论见第 4 节 |

## 1. 需求理解（我的复述）

**一句话**：三个规约类算子在 64bit 数据类型（INT64/UINT64/FP64）与特殊规约（PROD，及伴随 64bit 的 MIN/MAX）场景下，跨框两级拓扑不再退化为扁平 Sole NHR，改为"框内+框间"两级 sequence 编排（nhr+nhr），复用现有 aicpu_reduce 系列模板；**正确性优先，无性能要求**。

**范围矩阵**：

| 维度 | 做 | 不做 |
| ---- | ---- | ---- |
| 算子 | AllReduce、ReduceScatter、Reduce | 其它算子 |
| 拓扑 | 两级（topoLevelNums==2） | 三级（维持 Sole）、单机（路由不变） |
| 进入条件 | Is64BitDataType ∥ PROD（任意 dtype） | 非 64bit 的 min/max（走常规模板，**设计确认点**） |
| 引擎 | aicpu | aiv/ccu/dpu |
| 交付物 | 3 处 executor 注册 + 3 处 selector 分支 + UT/ST | 不新增模板、不改 executor/template 主体、不改 costModel、无 HCOMM 跨仓依赖 |

## 2. 现状理解（已核实）

1. **template 层仅有 Sole 实现**：三算子的 aicpu_reduce NHR 模板均为扁平单级（如 `reduce_aicpu_reduce_nhr.cc`，`ReduceSoleExecutor` 使用）。
2. **executor 层仅有 Sole 注册**（均 `TopoMatchOneLevel`，整域扁平跑 NHR，不分框内/框间）：
   - `AicpuAllReduceSoleNHRAicpuReduce`（`ins_v2_all_reduce_sole_executor.cc:432`）
   - `AicpuReduceScatterSoleNHRAicpuReduce`（`ins_v2_reduce_scatter_sole_executor.cc` 注册宏在 :605 一带，SE 文档写 :404，行号待对齐）
   - `AicpuReduceSoleNHRAicpuReduce`（`reduce_sole_executor.cc:357`）
3. **selector 硬编码回退**：`topoLevelNums > 1` 且 64bit/PROD 时一律路由 Sole（`all_reduce_auto_selector.cc:439`、`reduce_scatter_auto_selector.cc:416`、`reduce_auto_selector.cc:281`）。
4. **根因链**（补充的背景理解）：常规模板被排除是因为 `LocalReduce` 对 64bit/PROD 走 `AicpuReduce` 控制核标量规约（`alg_data_trans_wrapper.cc:905-908`），需 `HcommBatchModeEnd/Start + HcommThreadJoin` 同步——aicpu_reduce 系列模板天然兼容此路径。

## 3. 方案理解

**编排**（sequence executor 串行按层，均已存在、仅加注册）：

```
AllReduce:       RS-L0(NHRPcie) → RS-L1(NHRPcie) → AG-L1(AG NHR) → AG-L0(AG NHR)
ReduceScatter:   RS-L0(NHRPcie) → RS-L1(NHRPcie)
Reduce:          同 AllReduce，但 AGL0 仅 root 写 user output（3level executor 两级注册 + skipLevel2，L2 模板位占位）
```

**关键选型**（理解的理由）：两级 RS 复用 `InsTempReduceScatterAicpuReduceNHRPcie` 而非 Sole 版，因为分级版具备 `GetAlgRank` 映射（userRank≠算法内索引）、`tailSize` 尾块、`outputSliceStride`、`repeatNum`、in/out 同 buffer 处理——Sole 版直接用于两级编排会错乱；AG 为纯搬运无规约，64bit 无障碍。已核实该模板确实具备上述全部语义（`ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.cc:65,74,78,120,164,241,282`），且 `IsPcieProtocol` 自动判链路（:71），非 pcie 走 SendRecv 通用路径。

**selector 路由**：三算子 `SelectAicpuAlgo` 的特殊分支改为"两级→新算法、三级→维持 Sole"；单机全部不变；保序（DETERMINISTIC_STRICT）仍优先拦截。

**costModel**：新算法靠模板无静态 `CalcCostCoeff` 不参与 cost 排序，仅 selector 特判选中（**此条有出入，见 4-6**）。

## 4. SE 文档论断核实结果

| # | SE 论断 | 核实 | 代码依据 |
| - | ---- | ---- | ---- |
| 1 | 现有 `AicpuAllReduceSequenceMeshNHRAicpuReduce`（Mesh框内+NHRPcie框间）存在且已验证 NHRPcie 作 sequence RS 模板 | ✅ | `ins_v2_all_reduce_sequence_executor_aicpu.cc:893` |
| 2 | RS 侧 `AicpuReduceScatterSequenceMeshNHRAicpuReduce` 同上 | ✅ | `ins_v2_reduce_scatter_sequence_executor_aicpu.cc:591` |
| 3 | NHRPcie 具备 algIdx 映射/尾块/stride/repeatNum 两级语义 | ✅ | 见第 3 节行号 |
| 4 | `InsAlgTemplateBase::CalcCostCoeff` 默认返回空 | ✅ | `alg_v2_template_base.h:28` |
| 5 | RS sequence executor"任一模板空则整体退出 cost"短路存在 | ✅ | `ins_v2_reduce_scatter_sequence_executor_aicpu.cc:133-136` |
| 6 | **AllReduce（及 Reduce 3level）sequence executor"聚合后同样为空"** | ⚠️ **不符** | `ins_v2_all_reduce_sequence_executor_aicpu.cc:84-105`、`reduce_sequence_executor_aicpu_3level.cc:78-110` 均为**直接拼接、无短路**；且 `InsTempAllGatherNHR` **有**静态 `CalcCostCoeff`（`ins_temp_all_gather_nhr.cc:18`）→ 新算法（NHRPcie 空×2 + AG NHR 有×2）将得到**非空的部分系数** |
| 7 | Reduce 3level executor 支持两级注册 + skipLevel2 + root 语义 | ✅ | `AicpuReduceSequenceMeshConcurNHR`（`reduce_sequence_executor_aicpu_3level.cc:841`，TopoMatchTwoLevel） |
| 8 | `InsTempAllGatherNHR` 具备 GetAlgRank 分级映射 | ✅ | `ins_temp_all_gather_nhr.cc:424,460,521` |
| 9 | 注册宏（BY_FOUR_TEMPS/BY_TWO_TEMPS/V2_MULTI）均存在可用 | ✅ | 三文件内已有同型使用 |

## 5. 风险与补充（反串讲重点）

**SE 已列风险**（框内复用未验证 / scratch 相乘 loop 增多 / 三级误入 / 非 64bit min-max 语义 / Reduce 对称内存）均已知悉，补充 3 条：

1. **【高】cost 排序误选风险（对应 4-6 出入）**：新算法不设 op 侧白名单 + 非空部分系数（仅 AG 部分、缺 RS 项 → cost 低估），若 cost table 参与选择，常规非 64bit 场景可能被误选。注意**不能照抄**现有 `AicpuAllReduceSequenceMeshNHRAicpuReduce` 的白名单（`supportedDataTypes={64bit}` + `isSupportProd=true`，`ins_v2_all_reduce_sequence_executor_aicpu.cc:901-903`）——本需求进入条件是"64bit **或** PROD@任意dtype"，白名单会挡掉 PROD@fp16/fp32。建议三选一：
   - ① executor 侧补"任一模板空则退出"短路（对齐 RS executor，改动最小且根治，**推荐**）；
   - ② `opCustomCheck` 精确表达进入条件；
   - ③ 确认 cost table 流不覆盖这三个 selector 特判路径并留证据。
2. **【中】算法命名与组合不一致**：按 `<框内RS><框间RS>` 惯例（对照 `...SequenceMeshNHRAicpuReduce` = Mesh1D+NHRPcie），两级均 NHRPcie 的新算法尾部应为 `NHRAicpuReduce` ×2，SE 命名尾段为 `NHR`。需确认：是刻意简化还是笔误；且 `HCCL_ALGO` 字符串解析（`alg_parse.cc` 按 `nhraicpureduce` 等分段）需验证新名可被正确指定/检索。
3. **【中】loop 数暴增与 `CUSTOM_TIMEOUT`**：scratch multiplier 相乘（RS executor ≈ L0×2 × L1×2）致 `maxCountPerLoop` 骤减，大数据量下 loop 数与 `HcommBatchModeEnd/Start + ThreadJoin` 次数线性放大——虽无性能要求，但需确认 notify/执行超时（`aicpu_timeout.h` 的 CUSTOM_TIMEOUT）不会因 loop 过多误报超时；建议 ST 覆盖 GB 级数据量。

## 6. 工作量修正

（对比此前基于需求标题的 10–26 人日预估；SE 方案确认"无性能要求 + 全复用模板 + 无 HCOMM 依赖"后显著下调）

| 项 | 内容 | 人日 |
| - | ---- | ---- |
| 代码 | 3 注册 + include + 3 selector 分支（~100 行）+ 第 5 节问题处置 | 1–2 |
| UT | 路由矩阵/注册完整性 | 1–2 |
| ST | 三算子 × {INT64,UINT64,FP64} × {SUM,PROD,MIN,MAX} × {整除/尾块/大数据/inplace/对称内存} 双机正确性 + 常规场景回归 | 2–3 |
| 联调排障 | 双机两级环境（预留发现 1/3 的修复迭代） | 2–4 |
| 检视/RFC/合规 | pre-commit、静态检查、PR | 1–2 |
| **合计** | | **7–13 人日（约 2 周）** |

## 7. 待确认问题（请 SE 评审时拍板）

1. 发现 1 的处置方案（短路 / opCustomCheck / 提供不覆盖证据，推荐 ①）；
2. 非 64bit 的 min/max 是否进入新路径（SE 已列为设计确认点，默认不改）；
3. 新算法命名尾段 `NHR` 是否修正为 `NHRAicpuReduce`（含 `HCCL_ALGO` 解析验证）；
4. RS Sole 注册行号勘误（:404 → 实测 :605 附近），请 SE 确认文档版本基线；
5. 三级拓扑维持 Sole 是否为本迭代硬边界（后续迭代是否已有规划，影响命名/attrs 预留）。
