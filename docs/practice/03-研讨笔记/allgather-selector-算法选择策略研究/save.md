
---

## 讲之前：这套 deck 的主线是什么

一句话：**选路从"按条件查表"变成"给每个候选算一笔账、选最便宜的"；而这笔账里的 `util` 是分段常数阶梯，所以"最便宜"会随数据量跳变，决策边界是一条带，不是一个阈值。**

12 页是这条线的四次展开：**结构**（S2–S3）→ **账本**（S4–S6）→ **怎么找边界**（S7–S10）→ **工程上怎么办**（S11–S12）。

---

## S1｜封面：四个数字就是四个钩子

**这页说什么**：2 条路径、5+3 管线、4 段代价、1 字节翻转。

**四个数字分别指向**：

| 数字 | 指向 | 为什么选它 |
|---|---|---|
| 2 条 | `SelectorEngine` vs `ExecuteSelector` | 全场最大的结构事实：同一个算子有两套选路并存 |
| 5+3 | costModel 五步 + costTable 三阶段 | 新路径的完整流水线，S3 展开 |
| 4 段 | ParallelMeshNHR 的代价分解 | 一个算法内部还能再拆，S5 展开 |
| 1 B | 65535→65536 胜负翻转 | 边界不连续的最强证据，S8 展开 |

**讲的时候要留一句余地**：这个"1 字节"只在 64P、P=1、且只有 SoleNHR 与 ParallelMeshNHR 两个候选的模型里成立。它证明的是**性质**（cost 在 util 断点处不连续），不是**通例**（不是任何 1 字节都会翻转）。这句话先说，后面 S8 就不用再补。

**要记住的一句**：选路已经是可行集上的比价，但边界呈跳变带。

---

## S2｜全景：三条入口、三层抽象

**这页说什么**：入口怎么分流，以及"算法"这个词在代码里其实是三层。

### 真实的调用链（这页最容易讲错的地方）

```
src/ops/all_gather/all_gather.cc:226     CHK_RET(Selector(comm, param, topoInfo, algName));
        ↓
src/ops/op_common/op_common.cc:225       if (!useNdaSelector &&
                                             ((topoInfo->level0Topo == Level0Shape::MESH_1D)
                                              || AutoSelectorBase::IsDevType960())) {
:226                                        SelectorEngine::Global()->Run(...)      // 新路径
:228                                      } else { ExecuteSelector ... }            // 老路径
```

三个必须讲准的点：

1. **AllGather 自己不做分流**。它只是调共享的 `Selector()`，分流发生在 `op_common.cc:225`。所以"入口在 op_common.cc:225"说的是**共享分流点**，不是 AllGather 的入口。
2. **谓词里有一个 `!useNdaSelector` 前置**。也就是：NDA 机型即使拓扑是 MESH_1D，也走老路径。NDA 判定是 `hostDPUOnly && CheckSupportNda`（`op_common.cc:218-220`），两个条件缺一不可。
3. **`IsDevType960()` 是"全拓扑放行"**。非 960 机型只有 `level0Topo == MESH_1D` 才进新路径；960 机型不管 L0 是 CLOS 还是 UBX，全部进 `SelectorEngine`。这一条直接决定 S12 那个开放问题怎么问——"CLOS/UBX 要不要迁 cost model"只对非 960 成立。

### 三层抽象

| 层 | 代码里的形态 | 关键接口 |
|---|---|---|
| Selector | `SelectorEngine` / `ExecuteSelector` / `SelectNdaAlgo` | 选出一个**算法名** |
| Executor | `InsV2AllGather{Sole,Parallel,Sequence,Concurrent,OmniPipe}Executor` | `Orchestrate()`（`executor_v2_base.h:70`，纯虚） |
| Template | `InsTempAllGather*`，按引擎目录 aicpu / aiv / ccu | `KernelRun()` |

**为什么要强调三层**：cost model 的候选不是"算法名"，而是 **Executor × Template 的组合**。同一个 Executor 配 Mesh 模板和配 NHR 模板是两笔不同的账。这就是为什么 AllGather 能注册出 **35 个唯一算法名**（Sole 12、Parallel 5、OmniPipe 4、SequenceAicpu 4、Sequence 3、Sequence3Level 2、Sequence3LevelDpu 2、Concurrent 2、OmniPipe2D 1）。

**两个可以主动交代的小坑，显得你真读过代码**：

- **DPU 不是 `EngineType`**。枚举只有 AICPU / CCU / CCU_CIR_MODE / AIV / CPU（`cost_model.h:125-131`）。`Dpu*` 类算法走 `HOSTCPU/CPU`，模板放在 `aicpu/` 目录下。"Template = AICPU/AIV/CCU/DPU" 是命名上的简写，严格讲是"引擎 + 执行位置"。
- **`ReSelector`（`op_common.cc:957`）少了 `!useNdaSelector`**。重选路径上 NDA 机型可能进 `SelectorEngine`，和首选的二选一不完全对称。

**要记住的一句**：过去是按条件查表，现在是先筛可行集、再比代价。

---

## S3｜管线：过滤 → 定价 → 分层 → 择优

**这页说什么**：新路径的四段流水，以及"硬前提 / 软策略"的分层。

### 四段各自的作用域（这页最有价值、也最容易被略过的一点）

| 阶段 | 位置 | 作用域 |
|---|---|---|
| costModel 五步 | `cost_model.cc:314` | **每个通信域、每个引擎档建一次并复用** |
| costTable 三阶段 | `cost_table.cc:163` | **每个 op 重建** |
| tuner | `selector_engine.cc:438` 起 | 可选，未加载则全程 no-op |
| SelectMinCost | `selector_engine.cc:597` | 每个 op 选一次 |

这个"模型复用、表重建"的差异值得单独说一句：它解释了为什么模型里可以放贵的拓扑标定，而表必须便宜。

**五步**（`cost_model.cc:318-323`，顺序固定）：`InitModel` → `FilterByEngine` → `ApplyHcclAlgoPriority` → `FilterByTopoAndCalibrate` → `ApplyTopoPriority`。

**三阶段**（`cost_table.cc:185-189`）：`OpFilterAndCalcCost` → `ApplyOpPriority` → `ConvergeHcclAlgoTier`。

### 硬前提与软策略：这里幻灯片说得太绝对，要口头修正

幻灯片写的是"`opCustomCheck` / `topoCustomCheck` / priority 让位"。代码里其实是**两种不同的让位方式**：

| 机制 | preferred `HCCL_ALGO` 时 | tuner 加载时 | 代码 |
|---|---|---|---|
| `topoPrecondCheck` | **完全不绕过** | **完全不绕过** | `cost_model.cc:283-285` |
| `opCustomCheck` | 跳过 | 跳过 | `cost_table.cc:226` |
| `topoCustomCheck` | 跳过 | 跳过 | `cost_model.cc:424` |
| `ApplyOpPriority`（Phase 2） | **仍执行**，只是偏好项被豁免于剔除 | **整段跳过** | `cost_table.cc:186`、`:290` |
| `ApplyTopoPriority` | **仍执行**，偏好项豁免 | 该组跳过 | `cost_model.cc:502`、`:512` |

所以准确说法是：**customCheck 类会整体让位；priority 类不让位，只是对偏好项做"排他豁免"**。这个区别很重要——它意味着 `HCCL_ALGO` 并没有把优先级过滤整个关掉，只是保证你点的那个不会被别人挤掉。

### 择优的两条规则

- **按引擎优先级分层**：`GetEnginePriority`（`selector_engine.cc:74`）返回 `(引擎档, 优先级)` 列表；`AIV_URMA` 档给 URMA 优先级 1，**URMA 算法存在则必选**（除非被功能过滤掉）；在 AIV 模式下 AIV_URMA 会被压平到 0，所以这时不强制 URMA。
- **层内取 min**：`SelectMinCost`（`:597`）用**严格小于** `cost < minCost`（`:665`）比较，因此**平票取表里靠前的那个**，而表序就是注册序（`AddAlgToAllAlgos` 的 append 顺序）。无效项（`name == nullptr || cost < 0`）不参与 `activeMaxPrio` 统计（`:629`）。

### tuner 的语义（一句话讲清）

插件契约在 `hccl_tuner_plugin.h:133`：**`cost < 0` = 禁用，`cost == 0` = 偏好，`cost > 0` = 覆盖**。所以负数不是"算错了"，是**主动排除**——`SelectMinCost` 会跳过它，而不是把它钳到 0。

**要记住的一句**：阈值没有消失，它变成了补丁叶子。

---

## S4｜代价函数：α-β 模型的工程化

**这页说什么**：一笔账由四项构成，只有一项会随数据量跳变。

### 公式

```
segCost = (A/util + B + Breduce) × dataSize + C
cost    = max( Σ_group [ Σ 或 max_seg segCost ] , Σ D )
```

代码位置：`cost_table.cc:402`（`abCost`）、`:403`（`segCost`）、`:404`（组内 SUM/MAX）、`:407`（组间恒求和）、`:410`（`cost = std::max(cost, totalD)`）。

### 四项的物理含义（逐项讲）

| 项 | 含义 | 取值 |
|---|---|---|
| **A** | 跨卡传输斜率 | Mesh(1DMESH) = `n/BW`；**Mesh(CLOS) = `n·(rs−1)/(P·BW)`**；NHR = `n·(rs−1)/(P·BW)` |
| **B** | 本地拷贝斜率 | `1/1400GB/s`（`cost_model.cc:118`，`localCopyBw_ = 1400e9`） |
| **Breduce** | 规约附加项 | 仅 `dts==8` 或 `PROD` 时非零：`1.2e-9 × reduceCopies / dts` |
| **C** | 固定时延 | AICPU `2μs × taskNum`、AIV `1μs × taskNum`（`cost_model.cc:638/640`） |
| **D** | 下发/同步 | NHR `1.2μs × taskNum`、Mesh `0.6μs × taskNum`（模板内 `:49`） |
| **BW** | 链路带宽 | `crossChip 56e9`；CPU 引擎用 `roce 25e9` |

**必须修正幻灯片的一处**：上面写 `Mesh: n / BW`，那只对 `COMM_TOPO_1DMESH` 成立。**CLOS 下的 Mesh 是 `n·(rs−1)/(P·BW)`**（`cost_model.cc:562-574`）。讲的时候可以说"Mesh 的斜率取决于是不是一维全连接"，一句话就把两种形态带过了。

**两个容易讲错的点**：

1. **`perTransfer` 只用于查 `util`，`dataSize` 才用于乘 `cost`**。
   - `cost_table.cc:391`：`perTransferSize = dataSize × dr`
   - `:396`：拿 `perTransferSize` 去 `QueryUbUtil`
   - `:402`：乘的是函数入参 `dataSize`
   
   这解释了为什么"数据量翻倍、cost 不一定翻倍"——util 会换档。
2. **`util` 只除 A，不除 B**。`:402` 是 `A/util + B + Breduce`，`:400` 的注释明确写了"不除 util"。物理上合理：`util` 描述的是**链路利用率**，本地拷贝不经过链路。

### `max(…, ΣD)` 这一层的意义

`ΣD` 不是"加一个常数"，是**下界**。小数据时 `A/util + B` 项很小，cost 由 `ΣD` 兜底，所以曲线是平的。S8 里 ParallelMeshNHR 全程 276.0μs 一动不动，就是因为它 **D-bound**——四个 D 项相加，与数据量无关。这一点讲透，S8 的"为什么只有一条线在动"就不用再解释。

**要记住的一句**：`util` 是唯一随 `dataSize` 跳变的项，所以边界的形状由它决定。

---

## S5｜多段算法：Parallel 的 4 段与 α 配平

**这页说什么**：一个算法内部把数据切成四段，比例不是拍的，是配平出来的。

### 段与比例

`ins_v2_all_gather_parallel_executor.cc:159-174`：

```
段序     : [L0-mesh, L1-NHR, L0-mesh, L1-NHR]
dataRatio: [α, 1−α, (1−α)·L1, α·L0]
```

注意：`α` 在代码里叫 `meshFirstRatio`，`1−α` 叫 `closFirstRatio`。先算 clos 再取补，别被变量名绕进去。

### α 的来源

`template_utils.cc:735-741` 的注释直接给了理由：

> 第二阶段为耗时主导项，因此按第二阶段配平。

于是：

```
timeCoeff.mesh = L0 · (L1 − 1) / P
timeCoeff.clos = L1
α_raw          = timeCoeff.clos / (timeCoeff.clos + timeCoeff.mesh)
α              = quantize(α_raw, 1/8)   // 四舍五入后钳到 [1/8, 7/8]
```

**64P（L0=8, L1=8, P=1）的算术**：

```
timeCoeff.mesh = 8 × 7 / 1 = 56
timeCoeff.clos = 8
α_raw = 8 / (8 + 56) = 0.125
量化：round(0.125 / 0.125) = 1 → 钳到 [1,7] → 1 × 1/8 = 0.125
```

四段比例：**0.125 / 0.875 / 7.0 / 1.0**。验算配平：`0.125 × 56 = 7.0 = 0.875 × 8`。

### 聚合

`cost_table.cc:404`：

```cpp
groupCost = (meta.intraGroupMode == CostAggMode::MAX)
              ? std::max(groupCost, segCost)
              : groupCost + segCost;
```

Executor 声明 `intraGroupMode = MAX`、`groupSizes = {2,2}`（`parallel_executor.cc:164-165`），所以是：

```
cost = max( seg0, seg1 ) + max( seg2, seg3 )     // 组内取 max
     → 再与 ΣD 取 max                              // :410
```

**为什么组内取 max 而不是求和**：同一组的两段是**并行**发生的，耗时由慢的那段决定。

### 两个幻灯片里看不到、但 code review 会问的项

- `bConst = 23e-6`（`:115`）：**加到每一段的 C 上**，也就是每段凭空多 23μs。名字叫 bConst，实际是 C 项。
- `dAdd = 1.5e-6 × (9·rank + 32) / 32`（`:119`）：只在 `D > 0` 时加到 D 上（`:120-123`）。

**适用条件**（这句能救场）：`α = 0.125` 只在 `multipleDimensionSplitRatioSource == BUILTIN_FORMULA` 且 `SupportsParallelAdaptiveSplit` 时成立；否则退回配置值，再退就是 **0.5**。所以"看到 Parallel 就以为按 1:7 切"是不对的。

**要记住的一句**：α 不是拍脑袋，是按第二阶段主导项配平后量化到 1/8。

---

## S6｜util：分档查表 + 两个特殊处理

**这页说什么**：账本里唯一会跳的项，长什么样、怎么查、被谁改过。

### 四张表，不是两张

`cost_table.cc:437-457`：

| 表 | 用途 | 2MB 档 |
|---|---|---|
| `closUbUtilTable_` | 基础 CLOS | 0.3 |
| `meshUbUtilTable_` | MESH | 0.7758 |
| `closOneJettyOnePortUbUtilTable_` | CLOS 上的 MESH 类算法 | 0.3453 |
| `closAivUbUtilTable_` | CLOS + AIV 引擎 | 0.622 |

CLOS 基础表的完整档位：

```
128K/256K/512K → 0.10388
1M            → 0.1855
2M            → 0.3
4M            → 0.4288
8M            → 0.5302
16M           → 0.568
32M           → 0.6549
64M           → 0.7184
128M          → 0.7408
256M          → 0.7644
```

### 查表语义：等于边界就跳到下一档

`QueryUbUtil`（`:461-505`）用的是：

```cpp
auto it = std::lower_bound(table.begin(), table.end(), dataSize,
                           [](const UbUtilEntry& e, u64 ds) { return e.upperBound <= ds; });
```

容器是 `lower_bound`，但比较器是 `upperBound <= ds`，**等价于"第一个 `upperBound > ds` 的档位"**。所以 `ds` 恰好等于 2MB 时，会跳过 2MB 行、命中 4MB 行。

这一条是整个 corner case 的**唯一机关**，值得在台上重复一遍。

### 两个特殊处理

**① AG + CLOS 钳制**（`:480-484`）：

```cpp
// AllGather: CLOS 小数据量(< 1MB)统一用 1MB 的 util
if (opType == HCCL_CMD_ALLGATHER && netType == COMM_TOPO_CLOS && dataSize < 1MB) {
    dataSize = 1MB;
}
```

注意三点：条件**没有 engine / algo 限制**（AIV 也钳）；钳的是 **`perTransfer` 实参**，不是原始 `dataSize`；钳到**恰好** 1MB。

于是：钳到 1MB → `lower_bound` 跳过 1MB 行（0.1855）→ 命中 2MB 行（0.3）。

**结果是：`perTransfer < 2MB` 全段 util 恒为 0.3，到 2MB 跳到 0.4288。**

**② 注释与实现不一致**：注释说"用 1MB 的 util"，但实际拿到的是 0.3。而且更要命的是——**这个钳制是把 util 从 0.1855 抬到了 0.3**，不是"限制在 1MB 档"。也就是说，它现在的效果和注释描述的意图相反。这是 S9 的第三处债。

**③ NHR 的放大**（`:387-391`）：

```cpp
if (at == NHR || at == NHR_MULTILINK || at == NHR_NDA) {
    dr *= rankSize / 2.0f;
}
perTransferSize = dataSize * dr;
```

准确式是 `perTransfer = ds × dr × (段 rankSize / 2)`，`dr` 初值是 `dataRatio`。**只有 SoleNHR 的 `dr == 1`** 时才退化成 `ds × rs/2`；Parallel 的 NHR 段 `dr` 是 `closFirstRatio` 之类，不能直接用 `rs/2` 反解断点。

**要记住的一句**：1MB 钳制没有把 util 压到 1MB 档，反而把它抬到了 0.3。

---

## S7｜方法：把边界当成"可枚举的断点"

**这页说什么**：corner case 不是靠直觉找的，是靠断点反解枚举出来的。

### 四步

1. **每段求 scale**：`s = dataRatio`；NHR 段再 `× rankSize/2`。
2. **断点反解**：util 表的断点 `B`（2/4/8/16/…MB）反解成 per-rank `ds = B / s`。
3. **两侧取点**：在 `ds − 1` 和 `ds` 上各算一次 Δcost。
4. **变号即候选**：`Δcost` 在两侧异号 → 这个断点上选路会翻。

### 为什么只扫 ≥ 2MB 的断点

因为 AG-CLOS 钳制把 `perTransfer < 2MB` 全段压成 0.3，那些断点**在 AllGather + CLOS 下不可见**。扫它们只会得到一堆"跨过去也没变化"的假候选。

这是一个很好的方法论细节：**扫描空间要先被语义裁剪，否则枚举出来的是噪声。**

### 扫描空间与结果

`rank ∈ {16, 32, 64, 128, 256, 512} × P ∈ {1, 2, 4, 8}`，命中 **7 组**，全部方向一致（Parallel → SoleNHR）。

这 7 组不是"7 个 bug"，是**同一机制在 7 个参数点上的投影**：SoleNHR 的 `perTransfer = ds × rs/2` 落到 CLOS util 的某一档边界上。

**要记住的一句**：这一步把"找边界"从经验问题变成了可重复的枚举问题。

---

## S8｜Corener case：64P、P=1、64KB

**这页说什么**：一次可以用字节数精确描述的选路翻转。

### 数字（已独立复算，全部确认）

| per-rank ds | SoleNHR | ParallelMeshNHR | 胜者 |
|---|---|---|---|
| 65,535 B | **327.803 μs** | 276.000 μs | Parallel，领先 51.803 μs |
| 65,536 B | **253.987 μs** | 276.000 μs | SoleNHR，领先 22.013 μs |

SoleNHR 在 1 字节内**向下跳 73.816 μs**。

### 机制，三步讲完

1. `perTransfer = 65,536 × 32 = 2,097,152 B`，**恰好等于 2MB**。
2. `lower_bound` 语义是"第一个 `upperBound > ds`"，所以 `ds == 2MB` 命中 **4MB 行** → `util = 0.4288`。
3. 65535 时 `perTransfer = 2,097,120 < 2MB` → 命中 **2MB 行** → `util = 0.3`。

`util` 变大 → `A/util` 变小 → 整段 cost 下降。B 和 C 没动，所以跳变量正好是 `A × ds × (1/0.3 − 1/0.4288)` 这一块。

### 为什么 Parallel 一动不动

它在四个场景里全是 **D-bound**：`cost == ΣD`，而 D 只和 taskNum 有关，与数据量无关。所以图上紫线是一条水平线。**只有橙线在动**——这是这张图最值得指出的一点。

### 台上必须补的一句（否则会被追问）

> 这不是"数据量越大反而选小数据算法"的单调性结论。SoleNHR 在小数据段本来就赢（D 更小），到 51KB 附近第一次被 Parallel 反超，64KB 又翻回来，72KB 之后再被反超。**64KB 只是这条交替带上的一个断点，不是分界线。**

而且：**老路径看不到这个翻转**。老的 `all_gather_auto_selector` 在 `dataSize ≤ 1MB` 时直接给 `AicpuAllGatherSoleNHR`，`> 1MB` 可能给 Sequence。所以这是**新路径（cost model）才暴露出来的边界**——这个限定一说，整页的可信度立刻不一样。

**要记住的一句**：不是 bug，是分段常数 util 的固有边界；模型复刻，非上板实测。

---

## S9｜不是孤例：7 组翻转 + 3 处模型/工程债

**这页说什么**：一个 corner case 是巧合，七个就是规律；规律背后还有账没还。

### 7 组翻转

方向全部是 Parallel → SoleNHR，跳降从 **15.7μs 到 139.7μs**：

| rank | P | per-rank 断点 | 命中的 util 档 | 跳降 |
|---|---|---|---|---|
| 16P | 4 | 512KB | 4MB → 8MB | 15.7 μs |
| 32P | 2 | 128KB | 2MB → 4MB | 36.3 μs |
| 32P | 8 | 1MB | 16MB → 32MB | 17.0 μs |
| 64P | 1 | 64KB | 2MB → 4MB | 73.8 μs |
| 128P | 1 | 64KB | 4MB → 8MB | 66.3 μs |
| 256P | 1 | 64KB | 8MB → 16MB | 37.4 μs |
| 512P | 1 | 64KB | 16MB → 32MB | 139.7 μs |

**两个"脆弱边界"要主动标出来**（这比多讲一组更有说服力）：

- **32P / P=8**：翻转后只领先 **1.26μs**。任何系数扰动（POD、端口组、dtype）都会把它推回去。
- **256P**：翻转前 Parallel 的领先区间只有 **约 1.4KB** 宽。也就是说这条边界"擦着"就过去了。

### 三处债（幻灯片里的表述需要两处修正）

**债 A｜16P 的硬补丁**（`ins_v2_all_gather_sole_executor.cc:409-416`）

```cpp
// 规避: 16P MESH_1D 总数据量 32~64MB(per-rank 2~4MB)区段 costmodel 对 SoleNHR 估值偏低导致误选,
// 先在该区段排除该算法, 待 costmodel 对应区段系数修正后移除
topo->userRankSize == 16 &&
totalSize >= 32MB && totalSize <= 64MB &&
topo->level0Topo == Level0Shape::MESH_1D
```

三个精确点：

1. **区间是闭区间** `[32MB, 64MB]`，不是半开。
2. **还要求 `level0Topo == MESH_1D`**，所以 16P 的 CLOS/UBX 不受影响。
3. **它是"软过滤"，不是"硬排除"**。`opCustomCheck` 在代码里被归类为软策略（`selector_engine.cc:455`），tuner 加载或 `HCCL_ALGO` 优先级 > 0 时会被跳过（`cost_table.cc:225-226`）。准确说法是：**默认自动选路时被排除，preferred/tuner 可绕过**。

**债 B｜阈值比较符不一致**（幻灯片已按复核结果修正）

- 同一 4MB：`>=` 只出现在 **2P detour / CCU_MS** 分支（`all_gather_auto_selector.cc:52/161/362`，都带 `userRankSize == 2`），NDA 分支用的是 `>`（`:529`）。还有第三处：CCU 序列上限用 `<`（`:270`）。
- 同一 16MB：CCU_SCHED 960 用 `>`（`:157`）；**用 `<` 的是 AICPU（`:373/:395`）和 CCU_SCHED UBX（`:132`）**——不是 NDA。NDA 根本没有 16MB 比较。

**债 C｜AG-CLOS 钳制的注释与实现相反**：注释想用 1MB 档（0.1855），实现拿到 2MB 档（0.3），而且效果是**抬高**了 util。

### 一个可以主动加分的点：账其实不止三处

复核时又找到至少四处同类硬编码：

1. `aiv_temp_all_gather_mesh_1D.cc:31-34`：非 POD 机型 `portNum / 2` 的实测口径校准。
2. `ins_v2_all_gather_sequence_executor_aicpu.cc:594-597`：4GB 的 `opCustomCheck` 硬上限，与 selector 的 `AG_AICPU_SEQUENCE_DATA_SIZE` 重复。
3. `ins_v2_all_gather_sole_executor.cc:509-516`：AIV SoleMesh 的 buffer 容量门槛。
4. `ins_v2_all_gather_omnipipe_executor.cc:162-165`：OmniPipe 第三轴的"实测校准"常数。

所以在台上说"我列了三处，实际上同类至少七处"，比说"只有三处"更稳。

**要记住的一句**：这些不是实现 bug，是"模型近似 + 工程补丁"的真实边界。

---

## S10｜规律：边界不是阈值，是"跳变带"

**这页说什么**：把 Δcost 画出来，你会看到它多次穿越零线。

### 定义与读法

```
Δcost = cost(SoleNHR) − cost(ParallelMeshNHR)
Δcost < 0  → SoleNHR 赢
Δcost > 0  → Parallel 赢
```

### 关键事实

64P / P=1 下，`Δcost` 在 **51KB、64.35KB、72.6KB** 附近各自穿越一次零线：

- **51KB**：区间内的线性增长导致的第一次反超（SoleNHR → Parallel）。
- **64KB**：util 跳档导致的一次断崖（Parallel → SoleNHR）。
- **72.6KB**：跳档后重新爬升，再次被反超（SoleNHR → Parallel）。

**所以"首次翻转点"和"util 断点翻转"是两个概念**，这页的价值就在于把这两个概念分开：

| 概念 | 是什么 | S7/S9 的 7 组用的是哪个 |
|---|---|---|
| 首次穿越点 | Δcost 第一次变号的位置，随 rank/P 漂移 | 否 |
| util 断点翻转 | 在反解出的 `ds` 上，因跳档发生的翻转 | **是** |

S9 表里的 7 组，全部是**第二类**。如果你在台上把两者混着说，很容易被一句"那到底哪个才是边界"问住。

### 为什么会有多次穿越（机制）

在一个 util 档内，`cost(ds)` 是**斜率为 `(A/util + B)` 的直线**；跨档时**向下跳一段**。对手 Parallel 是**水平线**（D-bound）。一条"锯齿上升的斜线"对一条水平线，自然会**反复穿越**。

**要记住的一句**：回归必须覆盖断点 ±1 字节，而不是只测连续区间。

---

## S11｜工程含义：三类动作

**这页说什么**：把"现象"变成"可回归的约束"。

| 类别 | 动作 | 具体做什么 |
|---|---|---|
| **回归** | 把边界变成用例 | 断点 ±1 字节纳入 UT/ST；覆盖 rank × P 组合；固定 dtype / in-place / cclBuffer |
| **模型** | 把系数变成假设 | 复核 util 表标定来源；决定是否强制连续/单调；修正 16P 区段系数 |
| **治理** | 把补丁变成债 | 补丁登记 owner 与移除条件；日志留痕；DFX dump |

### 三个必须讲准的工程事实

**① 日志三件套确实存在**

- `[CalcAlgCost]`（`cost_table.cc:412`）：每个幸存候选一次，先打总 cost、再逐段打 `A/B/Bre/C/D/util/segCost`（`:428`）。
- `[SelectorEngine] SelectMinCost`（`selector_engine.cc:636`）：打 costTable 规模、opType、dataSize，最后打选中的 algName。
- `[DFX_CostTableDump]`（`cost_table.cc:155`，实现在 `DumpCostTable:153`）：无条件 `HCCL_INFO`，不是可开关的 dump 符号。

**② 但 dump 的位置有个坑**：`DumpCostTable` 在 `GenerateCostTable` 末尾（`:191`）调用，**早于 tuner 改写（`selector_engine.cc:467`）和 `SelectMinCost`**。所以 dump 里看不到 tuner 之后的最终 cost，也看不到最终选择。要复现一次选路，得把 `[DFX_CostTableDump]` 和后面的 `[CalcAlgCost]`、`SelectMinCost` 三条日志拼起来看。

**③ UT 现在抓不到这类问题**

`test/ut/selector_engine/stub/selector_deps_stub.cc:193-197` 把每个候选的 cost 都塞成 `1.0`，并且**完全忽略 dataSize**。也就是说，现有 UT 里不存在"cost 随数据量变化"这回事，任何 util 档位翻转都不可能被测出来。ST 虽然有 1MB / 4MB+1 / 16MB 这些尺寸，但断言的是功能和模板，不是选路结果。

**这段是整份 deck 最有说服力的地方**——它证明"缺回归"不是猜测，是有代码证据的。

**要记住的一句**：让选路在边界上"可解释、可回归、可追踪"。

---

## S12｜结论与开放问题

**这页说什么**：三件带走的事，三个还没答案的问题。

### 三件事

1. **选路不是一棵树**，是可行集 + 比价；阈值以补丁叶子的形式存在。
2. **边界会跳变**，因为 util 是分段常数阶梯。
3. **模型仍有失真区**，16P 的补丁说明系数还没标定准。

### 三个开放问题，按"能不能从代码回答"分类

| 问题 | 性质 | 已知 |
|---|---|---|
| AG-CLOS 钳制的预期是 1MB 档 0.1855 还是 2MB 档 0.3？ | **半开放**：行为已确定（0.3），缺的是设计意图 | 代码行为确定，注释相反 |
| 16P 补丁何时移除、谁负责？ | **真开放** | 代码只写了"待系数修正后移除"，没有 owner / issue / 验收用例 |
| CLOS / UBX 是否迁到 cost model？ | **真开放，但要加限定** | 960 机型**已经**走 cost model；问题只对非 960 成立 |

第三个问题一定要加限定，否则台下懂的人会当场指出来。

### 如果被追问"这套模型可信吗"

标准回答：

> 它的定位是**趋势判断和边界定位**，不是性能预测。所有数字都来自按仓内公式的复刻，不是上板实测；真实选路以 `[CalcAlgCost]` / `[SelectMinCost]` 日志为准。它最大的价值是告诉我们**该测哪些点**，而不是直接告诉我们**线上会选谁**。

---