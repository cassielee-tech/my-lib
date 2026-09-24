# 单元 7｜executor 与 template：执行机制

> 所属课程：[HCCL 源码学习](../hccl-source.md) · 第 7 单元（共 10 单元）
> 精读对象：`src/ops/op_common/executor/`（registry + base + HcclExecOp 总控）、`src/ops/all_reduce/executor/`（13 个执行器）、`src/ops/op_common/template/`（基类 + 数据结构 + wrapper）与 `src/ops/all_reduce/template/{aicpu,aiv,ccu}/`

::: info 本单元目标
单元 6 结束在 `HcclExecOp()`——selector 选出的算法名怎么变成一次真实通信？本单元读完，你能够说清 **executor 的组合公式（算法名 = 执行骨架 ×（拓扑匹配器 + N 个算法模板））**、**总控 `HcclExecOp` 的回退记忆与资源复用**，并逐段精读一个具体模板（Mesh RS）看它如何落到 Hcomm 原语。
:::

## 先记住 7 个结论

1. **总控是 `HcclExecOp()`**（op_common.cc:618）：回退记忆 → 注册表查执行器 → 资源申请/复用 → 按引擎分派（AICPU 走 kernel 入口 / AIV 走 kernel 入口+缓存 / CCU 走 Orchestrate）。
2. **核心公式：算法名 = 执行骨架 ×（拓扑匹配器 + N 个算法模板）**。例如 `AicpuAllReduceParallelMeshNHR = InsAllReduceParallelExecutor<TopoMatchMultilevel, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR>`。
3. **AllReduce 的分解**：一次 AllReduce = 框内 ReduceScatter + 框间 ReduceScatter + 框间 AllGather + 框内 AllGather（4 个模板串起来）——单元 5 分级通信的代码化身。
4. **引擎差异在模板不在骨架**：`CcuSchedAllReduceParallelMeshNHR` 与 AICPU 版用**同一个** `InsAllReduceParallelExecutor` 骨架，只是模板从 `InsTemp*` 换成 `CcuTemp*`。
5. **模板生命周期两阶段**：`CalcRes`（声明要多少线程/notify/通道）→ `KernelRun`（真正编排搬运）。executor 是"搭流水线的人"，模板是"流水线上的一个工位"。
6. **模板不直接碰 dlsym**：它调 `wrapper/alg_data_trans_wrapper.cc` 的 `SendRecvWrite/LocalReduce/...`，wrapper 再调 `HcommWriteOnThread` 等原语（那里才走 dlsym → hcomm）。
7. **Mesh RS 的真实语义 = 全互联写入 + 本地归约**：不是"折半递归"，而是每卡把自己数据的 p 个分片并行写给所有对端的 cclBuff（远端只暴露 cclBuff），然后各自归约。

## 第一部分｜executor 执行器

### 1. 执行器注册表：与 selector 同款机制（coll_alg_v2_exec_registry.h）

```c
std::map<HcclCMDType, std::map<std::string, const CollExecCreatorV2>> execCreators_;
// key: opType → 算法名 → creator 工厂函数
```

注册宏一族（L43~127），按模板参数个数选型：

| 宏 | 模板参数 | 例子 |
| --- | --- | --- |
| `REGISTER_EXECUTOR_IMPL` | 无模板 | 简单算子 |
| `REGISTER_EXECUTOR_BY_TWO_TEMPS` | 骨架+2模板 | TwoShot 系 |
| `REGISTER_EXECUTOR_BY_FOUR_TEMPS` | 骨架+4模板 | RS+AG 四步（最常见） |
| `REGISTER_EXEC_V2_MULTI` | 变长模板 | omnipipe 多步 |

与 selector 注册表（单元 5）三点差异：① key 是**算法名字符串**而非 priority；② 存的是 **creator 工厂函数**（`DefaultExecCreatorV2<P>` 每次 `new`），查询时创建新实例；③ 同名重复注册同样报错（宏生成静态变量名含 name 和 `__COUNTER__`）。

### 2. 铁证：注册处的组合公式（all_reduce/executor/*.cc 文件末尾）

```c
// ins_v2_all_reduce_parallel_executor.cc:1296
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceParallelMeshNHR, InsAllReduceParallelExecutor, TopoMatchMultilevel,
    InsTempReduceScatterMesh1D, InsTempReduceScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR);

// 同一个骨架，换模板 = 换算法（Pcie 定制机型）
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceParallelMesh1DNHRPcie, InsAllReduceParallelExecutor, TopoMatchPcieMix,
    InsTempReduceScatterMesh1D, InsTempReduceScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR);

// 同一个骨架，换 Ccu 模板 = CCU 引擎版本（单元 5 选出的算法在这里落地）
#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuSchedAllReduceParallelMeshNHR, InsAllReduceParallelExecutor,
    TopoMatchMultilevel, CcuTempReduceScatterMesh1DMem2Mem, CcuTempReduceScatterNHR1DMem2Mem,
    CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
#endif
```

三个观察：

- **骨架复用**：`InsAllReduceParallelExecutor` 一个类注册了 7+ 个算法名——执行模式（并行编排逻辑）只写一遍；
- **引擎在模板层**：`InsTemp*`（AICPU/AIV 用）vs `CcuTemp*Mem2Mem`（CCU 用）——单元 4"引擎差异在模板层"在这里变成代码；
- **编译隔离**：`#ifndef AICPU_COMPILE` 保证 CCU 注册只进 host 侧产物；`#if CANN_VERSION_NUM >= CANN_VERSION(9,0,0)` 是版本门控。

### 3. 执行器接口：`InsCollAlgBase`（executor_v2_base.h:28）

三个纯虚函数构成生命周期，外加两个可选：

```c
class InsCollAlgBase {
    virtual HcclResult CalcAlgHierarchyInfo(...) = 0;  // ① 分级信息：我属于哪个框/哪个层级子组
    virtual HcclResult CalcRes(...) = 0;               // ② 资源计算：要几个线程/通道/notify
    virtual HcclResult Orchestrate(const OpParam&, const AlgResourceCtxSerializable&) = 0;  // ③ 编排执行
    virtual HcclResult FastLaunch(const OpParam&, const CcuFastLaunchCtx*);               // 快发（CCU）
    virtual HcclResult RestoreChannelMap(...) const;   // 通道恢复（图模式复用）
};
```

三个阶段正好对应"**我是谁 → 我要什么 → 我怎么跑**"。成员变量是执行上下文（myRank_/rankSize_/dataType_...），由 topoInfo + OpParam 在 `InitCommInfo` 填充。

### 4. 总控 `HcclExecOp` 全流程（op_common.cc:618~757）★

```c
// ① 回退记忆：该 comm+算法组合若回退过，engineCtx 里有 fallbackTag 记录 → 直接走缓存的新算法名
if (HcclEngineCtxGet(comm, param.fallbackTag, ...) == HCCL_SUCCESS) {
    param.engine = COMM_ENGINE_AICPU_TS;                       // 回退结果永远定格在 AICPU
    CHK_RET(HcclExecOp(comm, param, topoInfo, newAlgName, resPack));  // 递归一次
    return HCCL_SUCCESS;
}
// ② 算法名与 commModeTag（commName + "_opbase"/"_offload"）落进 param
// ③ 注册表查执行器
std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
// ④ 资源：申请 或 复用（tag 相同 + 资源未失效 → isResourceReused = true，免掉重建通道/线程）
auto resRet = HcclGetAlgRes(...);
if (resRet == HCCL_E_UNAVAIL) {            // 资源不够（如 CCU 通信域超额）
    CHK_RET(FallbackOp(...));              // → ReSelector 强制 AICPU_TS 重选 + 把结果记进 engineCtx
}
// ⑤ 按引擎分派
if (engine == AICPU_TS/CPU)   → HcclAicpuKernelEntranceLaunch(...);      // AICPU 内核入口
else if (engine == AIV)       → HcclAivKernelEntranceLaunch(...)         // AIV 内核入口
                                  + ExecuteAivCacheLogic(...);            // 记录/重放指令流
else if (engine == CCU)       → （复用则先反序列化 resCtx）
                                  executor->Orchestrate(param, *resCtxHost);  // 生成 CCU 指令流
else                          → executor->Orchestrate(param, *resCtxHost);
```

**两个精妙设计**：

- **回退记忆（①）**：一旦"这个算法在这个 comm 上"资源不够回退到 AICPU，**以后每次直接回退**，不再重复尝试失败的路径——失败结果也是可缓存的知识。
- **串行资源复用（④，见 CalcRes）**：sequence 执行器的 4 步是串行的，线程/notify 取各步需求的 `max` 而不是求和（"step1、2、3、4 为串行，因此 slaveThread 和对应 notify 可以复用"，ins_v2_all_reduce_sequence_executor.cc:113）——资源按峰值而非总量申请。

### 5. `OrchestrateLoop` 精读：四步流水（sequence executor）

数据流水线全景（ins_v2_all_reduce_sequence_executor.cc:167~）：

```text
cclBuffer 对半分：[ cclIn（中转/暂存） | cclOut（结果暂存） ]

user输入 ─► ①框内ReduceScatter ─► ②框间ReduceScatter ─► ③框间AllGather ─► ④框内AllGather ─► user输出
               (Mesh1D,L0)          (NHR,L1)               (NHR,L1)          (Mesh1D,L0)
   in→cclOut     cclOut→cclOut       cclOut→cclOut          cclOut→out
```

四步的 buffer 接线（代码原文，L177~215）：

```c
tempAlgParamsStepOne.buffInfo   = { inputPtr: param.inputPtr,  outputPtr: cclOut, hcclBuff: cclIn }; // ①
tempAlgParamsStepTwo.buffInfo   = { inputPtr: cclOut,          outputPtr: cclOut, hcclBuff: cclIn }; // ②
tempAlgParamsStepThree.buffInfo = { inputPtr: cclOut,          outputPtr: cclOut, hcclBuff: cclIn }; // ③
tempAlgParamsStepFour.buffInfo  = { inputPtr: cclOut,          outputPtr: param.outputPtr, ... };    // ④
```

**数据比缓冲区大怎么办？——分圈（loop）**：

```c
u64 maxCountPerLoop = hcclBuff.size / 2 / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
u64 loopTimes = dataCount_ / maxCountPerLoop + (dataCount_ % maxCountPerLoop != 0);
for (u64 loop = 0; loop < loopTimes; loop++) {
    u64 currDataCount = (最后一圈) ? 尾块 : maxCountPerLoop;   // 尾块单独算长度
    ...四步跑一遍，处理这一圈的数据...
}
```

这就是单元 5 AIV 分支 `cclBufferSize × AIV_MAX_CCL_LOOP_NUM` 上限的另一半答案：**大数据 = 多圈流水，每圈一套 RS+RS+AG+AG**。

每个模板拿到的资源包 `TemplateResource`：`channels = remoteRankToChannelInfo_[level]`（本层通道）+ `threads`（共享线程）+ npu2Dpu/dpu2Npu shmem（跨 DPU 共享内存指针）。

### 6. 13 个 executor 的命名规律（all_reduce/executor/）

| 命名 | 执行模式 | 解决什么 |
| --- | --- | --- |
| `*_sole_executor` | 单阶段 | OneShot 类：一步到位（小数据） |
| `*_sequence_executor` | 串行四步 | RS→RS→AG→AG 依次执行（通用） |
| `*_parallel_executor` | 并行切分 | 数据分块，多块流水重叠（大数据） |
| `*_concurrent_executor` | 多算法并发 | Mesh 与 Clos 链路同时跑 |
| `*_omnipipe_executor` / `*_2d` | 全流水/二维 | 2 级流水线深度重叠 |
| `*_two_shot_sole_executor` | 两步 | TwoShot（单元 5 阈值背后的实现） |
| `*_sequence_executor_aicpu(_3level)` | AICPU 专用 | 三级组网变体 |
| `*_order_preserved_executor` | 保序 | DETERMINISTIC_STRICT 模式（单元 5 保序分支对应） |
| `*2die_executor` | 双 Die | TWO_DIE_REGULAR 形态专用 |

**命名 = 执行模式 + 硬件形态**，与算法名后半段（单元 5 的命名语法）一一对应。

### 7. AIV 缓存重放（收掉单元 6 的悬念）

`ExecuteAivCacheLogic`（op_common.cc:718 调用，L545~551 附近实现）：

- **记录**：首次执行时把 AIV 指令流录进 `g_recordingQueue`，同时记录 `g_baseInputAddr/g_baseOutputAddr`；
- **重放**：同 shape（同缓存 key）再调用直接重放指令流；
- **地址变了怎么办**：重放时按新 `inputPtr/outputPtr` 与基址的**偏移补丁**修正——所以 sendBuf 地址变化不影响命中，shape 才是 key。这就是单元 6"快速路径3"能整链省掉的原因。

## 第二部分｜template 模板

### 8. 类体系（op_common/template/）

```text
CommonAlgTemplateBase                      common_alg_template_base.h
 ├─ Describe() = 0                         自描述（日志用）
 ├─ CalcRes(...) = 0                       资源声明
 ├─ GetRes(...) = 0 / GetThreadNum() = 0   资源回填
 ├─ CalcScratchMultiple(...) = 0           中转内存倍率（cclBuff 要多大）
 ├─ KernelRun(...) = 0                     ★ 执行
 └─ FastLaunch(...) = 0                    快发（CCU/缓存路径）
      ↑
InsAlgTemplateBase                         alg_v2_template_base.h:18
 ├─ 新增上下文：myRank_ / subCommRanks_ / buffInfo_ / opMode_
 ├─ notifyIdxMainToSub_ / notifyIdxSubToMain_   主从线程同步的 notify 编号分配
 ├─ enableRemoteMemAccess_ / supportSymmetricMemory_  远端直访/对称内存能力位
 └─ DPUKernelRun(...)                      DPU 变体入口
      ↑
InsTempReduceScatterMesh1DIntra 等          all_reduce/template/aicpu/
```

（注：`alg_template_base.h` 里的 `AlgTemplateBase` 是旧框架 v1 遗留，v2 执行器体系用的是上面这条链。）

### 9. 数据结构三件套（template_utils.h）

```c
struct BuffInfo {            // 三块内存 + 各自的基址偏移
    void* inputPtr;          // userIn
    void* outputPtr;         // userOut
    HcclMem hcclBuff;        // 中转缓冲
    u64 inBuffBaseOff / outBuffBaseOff / hcclBuffBaseOff;   // 分圈(loop)时的圈内偏移
};

struct TemplateDataParams {  // 模板的"任务单"
    BuffInfo buffInfo;
    u64 count / sliceSize / inputSliceStride / outputSliceStride;  // 切片参数
    u64 repeatNum / inputRepeatStride / outputRepeatStride;        // 重复块
    u64 tailSize;                                            // 尾块
    std::vector<u64> allRankSliceSize / allRankDispls;       // 每张卡的块大小/偏移（不等长 RS 用）
    StepSliceInfo stepSliceInfo;                             // omnipipe 分步切片
    ... + Serialize()/DeSerialize()                          // ★ 可序列化
};

struct TemplateResource {    // 模板的"工具箱"
    std::map<u32, std::vector<ChannelInfo>> channels;  // 远端 rank → 通道列表
    std::vector<ThreadHandle> threads;                 // 线程（含主线程）
    std::vector<CcuKernelHandle> ccuKernels;           // CCU kernel 句柄
    std::vector<CcuKernelSubmitInfo> submitInfos;      // CCU 下发参数（含 cachedArgs）
    void* npu2DpuShmemPtr / dpu2NpuShmemPtr;           // 跨 DPU 共享内存
    void* aivCommInfoPtr;
};
```

**`Serialize()` 是钥匙**：`HcclExecOp` 里 CCU 资源复用时 `resCtxHost->DeSerialize(seq)` 反序列化的就是这套结构——**执行计划被序列化存进 engineCtx，复用时还原**，这是图模式和 CCU 快发能成立的数据基础（engineCtx 缓存主干在单元 8 展开）。

### 10. 精读 `InsTempReduceScatterMesh1DIntra`（aicpu 模板）

#### 10.1 `CalcRes`（资源声明，ins_temp_reduce_scatter_mesh_1D_intra.cc:23）

```c
u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
resourceRequest.slaveThreadNum = threadNum - 1;        // 从线程数 = 对端数
resourceRequest.notifyNumPerThread.push_back(1);       // 每从线程 1 个 notify
resourceRequest.notifyNumOnMainThread = threadNum - 1; // 主线程收 p-1 个回执
CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels);
resourceRequest.channels.push_back(level0Channels);
```

**Mesh1D 的资源哲学：一个对端一个线程**。8 卡框内 RS → 7 个从线程，每个线程只跟一个对端说话——线程即并行度。

#### 10.2 `KernelRun`（执行骨架，L49）

```c
PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_);   // ① 主→从：开工
RunReduceScatter(channels, threads, tempAlgParams);                 // ② 算法本体
PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_);  // ③ 从→主：完工
// ④ 特殊类型兜底（L70）：
if (dataType 是 INT64/UINT64/FP64 || reduceOp 是 PROD) {
    HcommBatchModeEnd(algTag); HcommBatchModeStart(algTag);         // 关批处理模式
    for (thread : threads) HcommThreadJoin(thread, CUSTOM_TIMEOUT); // 硬等所有线程
}
PostCopy(tempAlgParams, threads);                                    // ⑤ 收尾归约
```

**④ 就是单元 5 `isDataTypeOrReduceTypeSpecial` 的实锤**：64 位类型和 PROD 的硬件 DMA-Reduce 不可靠/不支持，必须**批模式关闭 + 线程级 join**保证归约顺序——软件排序换正确性。selector 给这类组合单独选算法（`AicpuAllReduceSoleNHRAicpuReduce` 等）也是同一原因。

①③ 的 `PreSyncInterThreads` / `PostSyncInterThreads` 正是单元 3 ThreadNotify 的 Record/Wait 封装。

#### 10.3 `RunReduceScatter`（算法本体，L130）

```c
// ① 本地片：thread 0 直接拷贝 userIn → output（"DMA消减：让thread 0做本地拷贝"）
LocalCopy(threads[0], srcSlice, dstSlice);

// ② 远端片：每个从线程负责一个对端（queIdx=1..p-1，nextRank=(myAlgRank+queIdx)%p）
for (queIdx = 1; queIdx < threadNum_; queIdx++) {
    // write 语义：把"对端的那片"从我的 userIn 写到【对端的 cclBuff】
    txDstSlice = DataSlice(remoteCclBuffAddr, ... + myAlgRank * sendSize, ...);
    // rx 槽位：我的 cclBuff 里给对端留的位置
    rxDstSlice = DataSlice(hcclBuff.addr, ... + nextRank * recvSize, ...);
    SendRecvWrite(sendRecvInfo, threads[queIdx]);    // 双向同时发
}
```

两条源码注释道破天机：

> "由于进程只能访问远端的 HcclBuffer，所以只能通过 **write** 的方式将自己 userIn 上的数据写到远端 HcclBuffer 上"

> "在接收的时候接收源应该是远端地址，但是由于 rs 的 mesh 算法用的是 write，所以 **rx 不用 care**"

**Mesh RS 的真实语义 = 全互联写入 + 本地归约**（结论 7）。`remoteCclBuffAddr = linkSend.remoteCclMem.addr`——**远端缓冲地址在建通道时就已经互相交换好了**（控制面提前做的功课，交换点在单元 8 的 `BuildChannelInfo`）。

#### 10.4 `PostCopy`（收尾，L86）

```c
for (tmpRank = 0..p-1, tmpRank != rankIdx) {
    srcSlice = cclBuff 中 tmpRank 写进来的那片;
    dstSlice = userOut;
    LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_);  // 逐对端累加
}
```

p−1 次 `LocalReduce` 完成最终归约。"数据量为 0 的数据片无需 Reduce"（L105）——不等长切片的边角处理。

### 11. wrapper 层：模板与 dlsym 之间的缓冲垫

`op_common/template/wrapper/alg_data_trans_wrapper.cc`（模板 include 的 `alg_data_trans_wrapper.h`）：

```c
SendRecvWrite(sendRecvInfo, thread)                     // 模板视角：语义化收发
  └─ RunWriteAndNotify(...)                             // wrapper 内部
       ├─ CheckReduceSlicePair / TraceDataSlice         // 切片校验 + 数据面 trace
       ├─ HcommWriteOnThread(thread, channel, dst, src, size)              // 普通写
       ├─ HcommWriteReduceOnThread(thread, channel, dst, src, count, ...)  // 写时归约
       └─ HcommChannelNotifyRecordOnThread(thread, channel, NOTIFY_IDX_DATA_SIGNAL)  // 搬完打点
```

wrapper 的增值：DataSlice 抽象（地址+偏移+大小+元素数）、零长度防护、**最后一个非空 slice 才挂 notify**（一次同步信号覆盖整批写）、trace 日志。`Hcomm*OnThread` 一族就是 L3 原语经 dlsym 进来的入口——**数据面的最底层**（单元 3 概念、单元 8 机制的汇合点）。

### 12. 三套引擎模板对比（同一算法三种写法）

| | aicpu（`InsTemp*`） | ccu（`ccu_temp_*`） | aiv（`aiv_temp_*` + `kernel/`） |
| --- | --- | --- | --- |
| 资源形态 | 每对端 1 线程 + notify | **`slaveThreadNum = 0`**，`ccuKernelNum = [1]` | kernel + aivCommInfoPtr |
| 编排方式 | host 上逐条调 `Hcomm*OnThread` 原语 | 声明 kernel：`kernelFuncName = "CcuKernelAllReduceMesh1DMem2Mem"` + **函数指针** + `CcuKernelArg*` 参数包 + 通道描述 | kernel 源码在 `template/aiv/kernel/`（oneshot/twoshot/**superkernel**） |
| 执行者 | 控制核逐条下发 | CCU 微码执行整条指令流 | 向量核执行 kernel |
| 快发 | — | `FastLaunch`：直接 `HcommCcuKernelLaunch(threads[0], cachedKernelHandle, cachedArgs)` | 指令流缓存重放（本单元第 7 节） |
| CalcRes 摘录 | `threadNum = rankSize` | `strcpy_s(kernelInfo.kernelFuncName, ..., "CcuKernelAllReduceMesh1DMem2Mem"); kernelInfo.kernelFunc = (void*)CcuAllReduceMeshMem2Mem1DKernel;` | — |

CCU 模板的精髓：**资源声明里连"算法的实现函数"都注册好了**（名字 + 函数指针 + 参数包），`FillCachedArgs` 把参数填进 `cachedArgs[]` 数组缓存——单元 6"CCU fast launch"快速路径的原材料。且 CalcRes 里连通道选择都带拓扑意识（`MESH_1D_CLOS` 时只挑 `COMM_PROTOCOL_UBC_CTP` 协议的通道，ccu_temp_all_reduce_mesh_1D_mem2mem.cc:92~99）。

## 13. 本单元在全链路中的位置

```text
模板调用栈（一次框内 ReduceScatter）：
InsTempReduceScatterMesh1DIntra::KernelRun
 ├─ PreSyncInterThreads / PostSyncInterThreads     主从同步（notify 编号）
 ├─ LocalCopy / LocalReduce                        wrapper → Hcomm 原语
 ├─ SendRecvWrite ─► HcommWriteOnThread            wrapper → Hcomm 原语 → dlsym → hcomm
 └─ (64bit/PROD) HcommBatchMode + ThreadJoin       特殊类型软件排序
```

至此主线闭环：**API（单元 1）→ op 入口（单元 6）→ selector 选名（单元 5）→ executor 组装（本单元）→ template 落到 Hcomm 原语（本单元）**。剩下两个单元往支撑系统与扩展走：资源地基与 dlsym 解耦（单元 8）、MC2 扩展框架（单元 9）。

## 14. 思考题

1. 为什么执行器注册表存 creator 工厂函数，而 selector 注册表存单例指针？（提示：executor 每次 `new` 是否携带状态？`unique_ptr` 生命周期在哪结束）
2. 四步串行取 `max` 合并资源，那 parallel executor（数据分块并行）的资源应该怎么合并？为什么？
3. Mesh RS 的 cclBuff 需要 `rankSize` 倍（`CalcScratchMultiple` 返回 `templateRankSize_`）——为什么每卡要留出 p 个槽位？
4. `SendRecvWrite` 挂 notify 只挂"最后一个非空 slice"，如果中途某个 slice 失败了，对端怎么知道？
5. CCU 模板 `slaveThreadNum = 0` 却还有 `threads[0]`，这个线程是什么？
6. 回退记忆按 `commName+算法名` 为 key，如果 CCU 通信域后来腾出资源了，还能恢复用 CCU 吗？

::: details 参考答案要点

1. selector 无状态（纯决策，单例够用、无锁读）；executor 携带执行上下文（rank/线程/通道句柄），每次执行独立实例，用完即毁。
2. parallel 的分块同时活跃，资源需按**各步之和**再与串行部分取 max——并行度抬高峰值；因为并行块真的同时在用通道/线程。
3. 对端们各写各的槽位（p−1 个对端 + 自己那片对齐），槽位数 = 对端数 + 布局对齐。
4. notify 语义是"这一批都完成了"；异常路径靠错误码上抛 + 超时兜底（单元 6 的 `SetExecTimeout`）。
5. 主线程/提交线程——指令流的提交者，不是执行者（CCU 微码才是执行者）。
6. 不能自动恢复——fallbackCtx 不会主动清除（可 grep `fallbackTag` 的写入点验证），回退是"粘性"的。

:::

## 本单元小结

- **组合公式**：算法名 = 执行骨架 ×（拓扑匹配器 + N 个模板）；骨架复用、引擎在模板层切换；
- **总控 `HcclExecOp`**：回退记忆（失败也缓存）+ 资源按峰值申请 + 按引擎分派；
- **四步流水**：框内 RS → 框间 RS → 框间 AG → 框内 AG，大数据分圈（loop）处理；
- **模板精读**：Mesh RS = 全互联 write + 本地归约；wrapper 是模板与 dlsym 之间的缓冲垫；
- **贯穿全课程的四个设计模式**在此全部现身：静态注册+工厂（两个注册表）、责任链降级（回退记忆）、序列化缓存（resCtx）、字符串解耦（算法名/tag）。

## 参考资料

- [HCCL & HCOMM 软件架构简介（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [通信算子执行行为（性能分析）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/perf_analysis/profiling_op_behavior.md)

---

下一单元进入 **[8｜资源地基与 dlsym 解耦](08-resources-dlsym.md)**。

[返回课程导学 →](../hccl-source.md)
