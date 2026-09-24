# 单元 5｜集合通信算法与 selector 选择器

> 所属课程：[HCCL 源码学习](../hccl-source.md) · 第 5 单元（共 10 单元）
> 精读对象：`src/ops/op_common/selector/`（registry + base + execute）与 `src/ops/all_reduce/selector/all_reduce_auto_selector.cc`（728 行）

::: info 本单元目标
读完后，你能够用 **α-β-γ 代价模型**比较 **Ring / RHD / NHR / Pairwise** 等算法，解释**分级通信**为什么"先机内后机间"；再进入源码，看懂 selector 的**注册表、引擎瀑布与决策树**——算法文档里的选型表是怎么变成代码的。
:::

## 先记住 6 个结论

1. **算法差异的本质是通信步数与每步数据量的不同取舍**：Ring 步数 O(n−1) 但每步只传 1/n；RHD/NHR 步数 O(log n) 但每步数据量更大——前者吃带宽、后者吃延迟。
2. **分层拓扑必然导向分级通信**：把大数据量放在高带宽的 Server 内（Layer0），把跨 Server（Layer1）的传输压缩到最少。
3. **selector 四层结构**：`Selector()`（op_common.cc:84，总入口：算 topo → 选算法 → 设引擎/超时）→ `SelectorRegistry`（注册表：按 opType + priority 存选择器）→ `ExecuteSelector::Run`（按 priority 升序遍历，首个 MATCH 胜出）→ `AllReduceAutoSelector`（真正的决策树）。
4. **MATCH/NOT_MATCH 责任链**：`NOT_MATCH` 不是错误，是"我处理不了，请降级"。引擎瀑布：CCU_MS → CCU_SCHED → AIV → AICPU，逐级降级（`AutoSelectorBase::Select`，auto_selector_base.cc:17）。
5. **算法选择结果是一个字符串**（如 `"AicpuAllReduceSoleMeshTwoShot"`），executor 侧再按名字注册/查找——selector 与 executor 通过字符串名解耦。
6. **决策输入五元组**：数据量（dataSize）× 拓扑（level0Topo / 层数 / 2Die）× 卡数（userRankSize）× 类型约束（int64/fp64/prod/int8）× 特殊模式（inplace / 保序 / 环境变量覆盖）。

## 第一部分｜概念地图：算法与代价模型

### 1. 代价模型：读算法文档的钥匙

官方算法文档的耗时公式全部基于 **α-β-γ 模型**：

| 符号 | 含义 | 对应什么 |
| --- | --- | --- |
| **α** | 每次通信的启动延迟 | 下发一次任务的固定开销 |
| **β** | 单位数据的传输时间（1/带宽） | 链路带宽 |
| **γ** | 单位数据的本地计算时间 | 本地 Reduce（如加法）开销 |

一个算法的总耗时 ≈ 步数 × α + 总数据量 × β +（规约类）总数据量 × γ。**比较算法就是比较这三个系数的加权**：小消息场景 α 主导（要少步数），大消息场景 β 主导（要总搬运量小、带宽利用高）。带着这把钥匙，下面的算法对比会非常自然。

### 2. Ring：环形接力

![Ring 拓扑：每张卡与左手卡、右手卡相连（图源：HCCL 官方文档）](/images/cann/hccl/official/ring-topology.png)

所有 NPU 连成环，每张卡有左手卡与右手卡，一个负责接收、一个负责发送：

![Ring 算法 AllReduce 流程：ReduceScatter 转一圈 + AllGather 再转一圈（图源：HCCL 官方文档）](/images/cann/hccl/official/ring-principle.png)

- AllReduce 被拆成 **ReduceScatter（转一圈）+ AllGather（再转一圈）**；
- p 个节点需 **p−1 步**，每步交换 **1/p** 的数据；
- 时间复杂度 **O(n−1)**，适用于"星型"或"胖树"拓扑。

AllReduce 耗时：$2(p-1)\alpha + 2\frac{p-1}{p}n\beta + \frac{p-1}{p}n\gamma$

**读法**：步数 2(p−1) 很多——小消息时 α 被放大 p 倍，不划算；但每步只传 n/p，总搬运量 2n(p−1)/p ≈ 2n，且**任何时刻每张卡都在收发**，带宽利用充分——大数据量时接近链路极限。

### 3. RHD：递归折半-倍增

当规模增大（如 4K rank），Ring 的问题暴露：环太长、转太多次。RHD（Recursive Halving-Doubling）通过**递归折半及倍增**交换数据，以 5 个 rank（2²+1）为例的官方流程图：

![RHD 算法流程：先合并到 2 的整数次幂，再两两对半交换求和、两两拼接，最后还原（图源：HCCL 官方文档）](/images/cann/hccl/official/rhd.png)

- 通信对象每步翻倍/折半，步数 **⌈log₂N⌉**；
- 非 2 的整数次幂时先"合并"到最近 2 幂（例如 5 rank：先把 rank1 的数据并入 rank0 变成 4 rank，最后再还原回去），因此会引入额外通信步数；
- 适用于"星型"或"胖树"拓扑。

2 的整数次幂时 AllReduce 耗时：$2\log(p)\alpha + 2\frac{p-1}{p}n\beta + \frac{p-1}{p}n\gamma$

**与 Ring 对比**：α 项从 2(p−1) 降到 2log(p)，小消息大幅占优；代价是**每个通信阶段的对象在变化，链路也随之变化**——大流量场景可能引起交换机流量冲突，导致带宽下降。

### 4. NHR：非均衡层次环

RHD 的两个残余问题：非 2 幂规模时额外开销（出现"N−1 规模比 N 规模还慢"的怪现象）、链路变化引起冲突。NHR（Nonuniform Hierarchical Ring）针对这两点：

- 对 N 个节点构建 **N 棵生成树**，通过生成树构建最优通信关系；树深（步数）为 **⌈log₂N⌉**；
- 通过**重排数据片编号聚合发送**，保证地址连续的数据切片连续发送（2 幂时每步收发均为 1 份）；
- 最大通信流量集中在**物理位置相近**的节点间，减少流量冲突；
- 无论规模是否 2 幂都能充分利用链路；**小数据包场景进一步退化为只建 1 棵树**，减少网络数据包数量与芯片并发任务数。

rank size 为 4（2 的整数次幂）与 5（非 2 幂）时的通信过程对比：

![NHR 算法：rank size 为 4 时的通信过程，每步收发数据份数均为 1（图源：HCCL 官方文档）](/images/cann/hccl/official/nhr-4rank-flow.png)

![NHR 算法：rank size 为 5 时大部分数据切片可连续收发，仅少部分离散（图源：HCCL 官方文档）](/images/cann/hccl/official/nhr-5rank-flow.png)

**一句话记忆**：Ring 吃满带宽、RHD 压缩步数、NHR 兼顾步数与链路稳定性，且对非 2 幂友好。

### 5. 其他家族成员（混个脸熟）

| 算法 | 关键词 | 适用 |
| --- | --- | --- |
| **Mesh** | 全互联，一步完成（如 4 卡 HCCS 全互联） | 小规模全互联组网 |
| **NB**（Nonblocking） | 非阻塞分段 | 与流水/并行执行配合 |
| **Pairwise** | 两两成对交换，单卡一进一出 | **数据并行梯度同步**（等量交换场景最优） |
| **Pipeline** | 数据切块流水推进 | 与计算重叠 |
| **Star** | 中心节点聚合分发 | 特定拓扑/小规模 |
| **AHC** | 自研算法族 | 见官方算法文档 |

其中 Mesh 与 Pairwise 的拓扑最直观：

![Mesh 算法：所有 NPU 全互联，可一步完成数据交换（图源：HCCL 官方文档）](/images/cann/hccl/official/mesh.png)

![Pairwise 算法：两两成对交换，每张卡一进一出（图源：HCCL 官方文档）](/images/cann/hccl/official/pairwise.png)

不需要一次记住全部——读 selector 代码或排查性能时再回来查表。

### 6. 分级通信：分层拓扑的正确用法

单元 2 说"通信质量随层级递减"，算法层面的直接产物就是**分级通信**（hierarchical communication）。以 8 机 × 8 卡集群的 AllReduce 为例：

![AllReduce 分级通信流程：机内 ReduceScatter → 机间 AllReduce → 机内 AllGather（图源：HCCL 官方文档）](/images/cann/hccl/official/allreduce-hierarchical.png)

1. **Server 内执行 ReduceScatter**：先在 HCCS 高带宽域内归约切分；
2. **Server 间执行 AllReduce**：跨机只剩"每机一份"的小数据量，昂贵的 Layer1 链路被压到最少；
3. **Server 内执行 AllGather**：结果在机内高带宽扩散。

官方文档特别指出一个精妙处：AllReduce 的输出是完整归约结果，**不要求严格遵循 ReduceScatter + AllGather 的语义顺序**，因此可以把大数据量的过程放在带宽更高的 Server 内——语义允许的灵活性与拓扑约束在此互相成就。

| 算子 | 分级顺序 | 原因 |
| --- | --- | --- |
| ReduceScatter | 先机间、后机内 | 保证 Server 间通信数据块的连续性 |
| AllGather | 先机内、后机间 | 同上 |
| AllReduce | 机内 RS → 机间 AR → 机内 AG | 大数据量留在机内高带宽域 |

ReduceScatter 与 AllGather 的分级流程图（注意两者顺序恰好相反）：

![ReduceScatter 分级通信流程：先 Server 间、后 Server 内（图源：HCCL 官方文档）](/images/cann/hccl/official/reduce-scatter-hierarchical.png)

![AllGather 分级通信流程：先 Server 内、后 Server 间（图源：HCCL 官方文档）](/images/cann/hccl/official/allgather-hierarchical.png)

### 7. 算法选择的三层入口

算法的"选择权"在三层可见：

1. **自动选择**：`src/ops/<op>/selector/`（如 `all_reduce_auto_selector.cc`）依据数据量、rank 数、拓扑层级、芯片能力综合决策——下面第二部分逐段精读；
2. **环境变量干预**：`HCCL_ALG` 指定算法，`HCCL_ALGO_MULTIPLE_DIMENSION_SPLIT_RATIO` 等调节切分比例——调优时先读 `user_guide/hccl_env/`；
3. **耗时公式验证**：用第一部分的 α-β-γ 公式手工估算，与 Profiling 数据对照（`perf_analysis/typical_op_behavior.md`）。

::: warning 注意
算法可用性与具体产品型号相关（如 CCU 引擎仅 950 系），跨平台结论必须回到当前环境的支持清单（`comm_ops_support_list/`）核对。
:::

## 第二部分｜源码精读：selector 怎么把选型表变成代码

### 8. 注册机制：宏 + 静态初始化（selector_registry）

```c
// all_reduce_auto_selector.cc:727 —— 文件末尾一行完成注册
REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLREDUCE, 18, AllReduceAutoSelector);
```

宏展开（selector_registry.h:45）：

```c
#define REGISTER_SELECTOR_BY_OPTYPE_HELPER(ctr, optype, priority, name, selector) \
    static HcclResult g_func_##priority##_##name##_##ctr                          \
        = SelectorRegistry::Global()->RegisterByOpType(optype, priority, new selector())
```

- **静态对象初始化时机注册**：动态库被 dlopen/load 时，全局变量的构造函数执行，选择器自动进注册表——新增算子目录只需在自己的 .cc 末尾加一行，**无需修改任何中心化 if/else**。
- `__COUNTER__` 保证多次注册生成不同变量名。
- 注册表数据结构：`map<HcclCMDType, map<u32 priority, AutoSelectorBase*>>`（selector_registry.cc:38），priority 决定遍历顺序（`std::map` 升序），重复 priority 报错——每个 opType 内 priority 唯一。
- 顺带的小设计：选择器对象 `new` 出来后**永不析构**（进程生命周期单例），换来的是无锁读取。

### 9. 遍历与命中：`ExecuteSelector::Run`（execute_selector.cc:20）

```c
selectors = SelectorRegistry::Global()->GetSelectorsByOpType(opParam.opType);
for (auto iter : selectors) {                    // 按 priority 升序
    if (iter.second->Select(opParam, topoInfo, selectAlgName) == SelectorStatus::MATCH) {
        return HCCL_SUCCESS;                     // 首个 MATCH 胜出
    }
}
return HCCL_E_NOT_SUPPORT;                       // 全部 NOT_MATCH 才是错误
```

- MC2 特例（L25~39）：`opParam.isMc2` 时**硬编码取 priority=18 的选择器**（即 AllReduce 系的 CCU 选择器）——MC2 自定义算子复用 AllReduce 的选算法逻辑（单元 9 展开）。
- 由此读码技巧：**想知道某 opType 有哪些选择器，grep `REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::XXX`**。

### 10. 引擎瀑布：`AutoSelectorBase::Select`（auto_selector_base.cc:17~68）★ 核心机制

```text
hostDPUOnly? ──► engine=CPU, SelectDPUAlgo（DPU 专用，直接返回）
opExecuteConfig == CCU_MS  ──► SelectCcuMsAlgo    NOT_MATCH ⇒ 降级为 CCU_SCHED
opExecuteConfig == CCU_SCHED ─► SelectCcuScheduleAlgo  NOT_MATCH ⇒ 降级为 CCU_FAIL
AIV / AIV_ONLY ────────────► ProcessAivConfig → SelectAivAlgo
                               NOT_MATCH 且是 AIV_ONLY ⇒ 直接失败（禁止回退）
                               NOT_MATCH 且是 AIV     ⇒ 降级为 CCU_FAIL
AICPU_TS / HOSTCPU_TS / CCU_FAIL（IsStarsState）
    ├─ IsRollBackAiv 命中? ⇒ 强制改回 AIV_ONLY 重选（反常规的"升级"路径）
    └─ SelectAicpuAlgo ⇒ MATCH 则 opExecuteConfig = AICPU_TS
```

要点拆解：

- **降级链的方向**：专用引擎（CCU/AIV，走片上通信单元/向量核）→ 通用引擎（AICPU，走控制核）。专用引擎快但约束多，约束不满足就逐级让位（单元 4 的物理视角）。
- **`IsRollBackAiv`（L70）是反方向的补丁**：`level0PcieMix && level0BigClosRange && (AlltoAll 系 || INT64 归约)` 时，AICPU 反而不如 AIV，强制改回 AIV——**瀑布不是单向的，性能实测数据可以打补丁**。
- **AIV_ONLY 语义**（L360）：用户显式指定 AIV（环境变量），选中失败直接报错不回退——"我要求的就是 AIV，别给我换"。
- `configAlgMap = GetExternalInputHcclAlgoConfigAllType()`（L21）：**环境变量 `HCCL_ALGO` 的解析结果**在此时进入决策——用户配置可以左右算法选择（主要在 DPU 分支消费）。

### 11. 决策树的输入：topoInfo 关键字段速查

| 字段 | 含义 | 典型用法 |
| --- | --- | --- |
| `topoLevelNums` | 组网层数（1/2/3 级） | 3 级直接排除 AIV/CCU_SCHED |
| `level0Topo` | 单机形态：`CLOS` / `MESH_1D` / `MESH_1D_CLOS` | 决定走 NHR 还是 Mesh 族 |
| `level0MeshType` | `TWO_DIE_REGULAR` 等（双 Die 芯片形态） | 2Die 专用算法 |
| `userRankSize` | 通信域总卡数 | 阈值缩放、4P/8P/64P 分档 |
| `Level1Nhr` | level1 是否 NHR 互联（GCD==1 时置位） | 跨机直接选 NHR |
| `level0PcieMix` | 是否 PCIE 混连定制机型 | 排除 CCU、走 Pcie 算法 |
| `level2Ubg/Uboe` | 三级组网 UB 网关形态 | PipeLine/RSAG 算法 |

配套谓词（auto_selector_base）：

- `IsSmallData = dataSize < 512KB`（UB 协议单次传输上限）；`IsLargeData = dataSize >= 1MB`
- `IsInputOutputOverlap`（L311）：**inplace 判定的实现**——地址区间求交：`inputStart <= outputEnd && outputStart <= inputEnd`。单元 6 思考题的一半答案：CCU 引擎不支持 inplace，overlap 即 NOT_MATCH。
- `Is64BitDataType`：int64/uint64/fp64——数值宽度影响硬件 reduce 指令可用性。
- `CalcFrameNum`（L106）：用 `instSizeListOfLayer[0]` 的 **GCD** 算框数，注释明说"避免非对称场景各 rank 局部值不同"——非对称组网下用全局量而不是局部量。

（这些字段怎么算出来的？`topoInfo` 的计算流水线在单元 8 精读。）

### 12. AllReduce 决策树实例精读

#### 12.1 CCU_MS 分支（all_reduce_auto_selector.cc:40）

先过五道排除闸：保序模式 / 多级组网 / int8 / PROD / 64 位类型 → 任一命中即 NOT_MATCH。然后 `SelectMeshAlgo`（L119）：

```c
if (level0Topo == MESH_1D) {
    if (IsInputOutputOverlap(opParam)) return NOT_MATCH;       // 不支持 inplace
    if (level0MeshType == TWO_DIE_REGULAR) {
        selectAlgName = IsSmallData(dataSize) ? "CcuMSAllReduceSoleMesh2Die"     // 小数据
                                              : "CcuMSAllReduceSequenceMesh2Die"; // 大数据流水
    } else if (IsSmallData(dataSize)) {
        selectAlgName = "CcuMSAllReduceSoleMeshOneShot";        // 小数据一发完
    } else {
        selectAlgName = IsDevType960() && dataSize > 16M && IsTwoLevelNetLayer(...)
                        ? "CcuAllReduceSoleMeshMsConcur" : "CcuMSAllReduceSoleMesh";
    }
} else if (level0Topo == MESH_1D_CLOS) { ... SelectMeshUBXAlgo ... }
```

**OneShot vs TwoShot**（集合通信经典取舍）：OneShot 每卡直接读所有卡的输入（p−1 次跨卡读，时延最优）；TwoShot 先 ReduceScatter 后 AllGather（数据量减半再扩散，带宽最优）。**小数据走时延、大数据走带宽**——512KB/1MB 阈值就是两种模式的 crossover。

#### 12.2 AICPU 分支的单机路径：`SelectMeshAlgoAicpu`（L513）

```c
double ratio = DEFAULT_RANK_SIZE / userRankSize / userRankSize;   // = 8/p²
...
if (isDataTypeOrReduceTypeSpecial) {          // int64/uint64/fp64/PROD
    selectAlgName = dataSize <= 8M ? "AicpuAllReduceSoleMeshOneShot" : "AicpuAllReduceSoleMeshTwoShot";
} else if (dataSize <= 8M) {
    selectAlgName = "AicpuAllReduceSoleMeshOneShot";
} else if (dataSize * ratio > 32M) {          // ★ 阈值随卡数平方缩放
    selectAlgName = "AicpuAllReduceSoleMeshChunkTwoShot";         // 分块两步
} else {
    selectAlgName = "AicpuAllReduceSoleMeshTwoShot";
}
```

**最精妙的一行**：`dataSize * ratio > 32M` 等价于 `dataSize > 32M × p²/8`。卡数越多 → Mesh 算法每步要经过的跳数越多 → **切阈值越早切换到 Chunk（分块流水）算法**，让分块流水掩盖逐跳时延。这是第一部分 α-β 代价模型在代码里的直接落点。

#### 12.3 AIV 分支（L587）

排除条件比 CCU 更多（level2Ubg / 3 级组网 / 保序 / PROD / uint64+fp64 / 超 MAX_RANK_SIZE），再加两条**资源硬约束**：

```c
HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize);  // 通信缓冲区大小
if (dataSize >= AIV_MAX_PER_RANK_DATA_SIZE * userRankSize) return NOT_MATCH; // 8M/卡 上限
if (dataSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) return NOT_MATCH;        // 缓冲区循环上限
```

AIV 是向量核执行、数据要过 cclBuffer 中转（单元 4 的约束表），所以**缓冲区物理上限**进入选择条件。然后按拓扑选 OneShot/TwoShot，板内 8P 有独立拐点 128KB（`AR_AIV_SMALL_DATA_SIZE_IN_BOARD`）。

#### 12.4 `HCCL_AIV_NOT_MATCH_LOG` 宏（auto_selector_base.h:130）

AIV_ONLY 模式下 NOT_MATCH 时额外打 ERROR——普通降级只有 DEBUG 日志（默认不可见），用户显式要求 AIV 却没选上必须显式报错。**降级静默、显式要求响亮**。

### 13. 算法命名的语法（读名知算法）

以 `AicpuAllReduceSoleMeshChunkTwoShot` 为例，名字是结构化句子：

```text
[引擎]   [算子]      [执行模式]                  [拓扑]  [变体]
Aicpu    AllReduce   Sole(单阶段)/Sequence(流水)  Mesh    Chunk(分块)+TwoShot(两步)
CcuMS/   AllGather/  Parallel(并行)/PipeLine     NHR/    MultiLink/UBX/Pcie/
CcuSched Reduce...   Concur(并发)                2Die/   OrderPreserved/...
Aiv/Dpu/                                         Clos/
Ins(实例化框架)
```

这张"词汇表"由 `OP_TYPE_TO_AICPU_SOLE_ALG_MAP`（auto_selector_base.h:37）等映射表正式化——每个 opType 一个默认算法名的表，executor 注册时用同名 key（单元 7 接上）。

### 14. 入口 `Selector()` 的收尾动作（op_common.cc:84~134）

选中算法名 ≠ 结束，还有几件事：

```c
CHK_RET(HcclCalcTopoInfo(comm, param, topoInfo));       // ① topoInfo 是在这里算出来的
CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));  // ② 选算法
CHK_RET(SetCommEngine(param));                          // ③ 算法名 → 引擎类型回填
if (engine == AICPU_TS/CPU)  CHK_RET(LoadAICPUKernel());    // ④ 降级到 AICPU 需补加载内核
if (engine == AIV)            CHK_RET(RegisterKernel());    //    降级到 AIV 需补注册内核
CHK_RET(SetOpParamAlgTag(param, algName));              // ⑤ 资源缓存 key = 算子+算法
CHK_RET(SetExecTimeout(param));                         // ⑥ 执行超时
CHK_RET(SetMultipleDimensionSplitRatio(comm, param));   // ⑦ 多维切分比例
```

单元 6 的 tag（`AllReduce_<commName>`）+ 这里的 algTag 共同构成**资源复用的 key**——"同 comm 同算法"的重复调用可复用已申请的通道/notify 资源（单元 8 展开）。

### 本部分调用链全景

```text
Selector() (op_common.cc:84)
 ├─ HcclCommGetStatus       ← comm 就绪检查（未就绪返回 HCCL_E_SUSPENDING）
 ├─ HcclCalcTopoInfo        ← topoInfo（level0Topo/层数/卡数/2Die...）
 ├─ ExecuteSelector::Run
 │    └─ SelectorRegistry::GetSelectorsByOpType(ALLREDUCE) → [18: AllReduceAutoSelector]
 │         └─ AllReduceAutoSelector::Select（引擎瀑布）
 │              ├─ SelectCcuMsAlgo ─────► "CcuMSAllReduceSoleMeshOneShot" 等
 │              ├─ SelectCcuScheduleAlgo ► "CcuSchedAllReduceParallelMeshNHR" 等
 │              ├─ SelectAivAlgo ──────► "AivAllReduceSoleMeshTwoShot" 等
 │              └─ SelectAicpuAlgo ─────► "AicpuAllReduceSoleMeshChunkTwoShot" 等
 └─ SetCommEngine / LoadAICPUKernel / SetOpParamAlgTag / SetExecTimeout ...
```

## 15. 自测题

1. α、β、γ 分别建模什么？小消息与大数据量场景分别由谁主导？
2. Ring 的 AllReduce 耗时公式是什么？为什么大数据量场景它占优？
3. RHD 解决了 Ring 的什么问题？又引入了什么新问题？
4. NHR 相比 RHD 的两个改进是什么？
5. 为什么 `NOT_MATCH` 要设计成正常返回值而不是错误码？
6. `dataSize * ratio > 32M` 里 ratio=8/p²，8 是什么？
7. 注册表按 priority 升序遍历，如果两个选择器都能 MATCH 同一场景，怎么保证确定性？
8. AIV 分支要先查 `HcclGetHcclBuffer`——为什么 CCU 分支不用查？

::: details 自测答案

1. α = 每步启动延迟，β = 单位数据传输时间（带宽倒数），γ = 单位数据本地规约时间。小消息由 α（步数）主导；大数据量由 β（总搬运量与带宽利用）主导。
2. $2(p-1)\alpha + 2\frac{p-1}{p}n\beta + \frac{p-1}{p}n\gamma$。步数虽多（2(p−1)），但每步只传 n/p，且每张卡始终在收发，带宽利用充分，总传输量约 2n，适合大数据量。
3. RHD 把步数从 O(n−1) 压到 ⌈log₂N⌉，改善小消息延迟；但非 2 幂规模需先合并/还原引入额外步骤，且每步通信对象与链路变化，大流量时可能引发交换机流量冲突。
4. 一是对非 2 幂规模友好（N 棵生成树，任何规模都充分利用链路，无"N−1 比 N 慢"现象）；二是通过重排数据片编号聚合发送、流量集中在物理位置相近节点，减少链路变化与流量冲突。
5. 责任链 + 降级的语义：`NOT_MATCH` 表示"我处理不了，请降级到下一个选择器"，全部 NOT_MATCH 才是真正的错误——它是引擎瀑布（单元 4）得以成立的控制信号。
6. `DEFAULT_RANK_SIZE`，基准卡数——阈值以 8 卡为标定点，卡数越多阈值越早切换到 Chunk 分块流水。
7. 注册时 priority 唯一性检查（`Register` 里重复 priority 报错），`std::map` 按 priority 升序遍历且首个 MATCH 胜出——顺序确定。
8. CCU 有独立片上通路，AIV 数据面走 cclBuffer 中转（单元 4 约束表），所以 AIV 必须检查缓冲区物理上限。

:::

## 本单元小结

- **代价模型**：α-β-γ 是读一切算法耗时公式的钥匙；
- **三大算法**：Ring（带宽型，O(n−1) 步）、RHD（延迟型，log 步、非 2 幂有额外开销）、NHR（兼顾步数与链路稳定，非 2 幂友好）；
- **分级通信**：机内高带宽域消化大数据量，机间传输压到最少；
- **selector 机制**：宏注册表（priority 升序责任链）+ 引擎瀑布（专用→通用降级）+ 决策树（拓扑×数据量×卡数×类型约束）；
- **字符串解耦**：算法名连接 selector 与 executor，命名语法"引擎+算子+执行模式+拓扑+变体"可读名知算法；
- **知识连接**：这里选出的算法名，正是单元 7 executor 注册表的 key。

## 参考资料

- [集合通信算法介绍：Ring（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/Ring.md)
- [集合通信算法介绍：RHD（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/RHD.md)
- [集合通信算法介绍：NHR（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/NHR.md)
- [分级通信原理（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/coll_algo_intro/hierarchical_comm_principle.md)
- [环境变量参考：HCCL_ALG](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_ALGO.md)

---

下一单元进入 **[6｜AllReduce 调用链走读](06-allreduce-call-chain.md)**。

[返回课程导学 →](../hccl-source.md)
