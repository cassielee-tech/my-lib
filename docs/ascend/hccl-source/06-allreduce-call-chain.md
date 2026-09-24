# 单元 6｜AllReduce 调用链走读

> 所属课程：[HCCL 源码学习](../hccl-source.md) · 第 6 单元（共 10 单元）
> 精读对象：`src/ops/all_reduce/all_reduce_op.cc`（281 行）+ `all_reduce_op.h`

::: info 本单元目标
以 `HcclAllReduce` 为线索，从**使用侧剧本**（建域 → 调用 → 销毁）走进**源码路径**（`all_reduce_op.cc` 的三道闸门、四步主干、四条快速路径），把前面单元的概念全部钉到具体文件与调用顺序上。
:::

## 先记住 5 个结论

1. **使用侧永远是三段式**：创建通信域（HCOMM 接口）→ 在 Stream 上调用算子（`hccl.h`）→ 销毁域释放资源。
2. **入口 = 三道分流闸门**：版本兼容闸门（hcomm 版本 < 9.0.0 走老实现 `HcclAllReduceInner`）→ 设备能力闸门（仅 950/960 走新 out-place 流程）→ `count == 0` 快速返回。
3. **主干四步**：`AllReduceInitAndCheck`（环境变量 + 参数校验 + tag）→ `AllReduceEntryLog`（入口日志）→ `AllReduceOutPlace`（执行）→ `LogHcclExit`（出口计时日志）。
4. **`AllReduceOutPlaceCommon` 是心脏**：先过 4 条快速路径（CCU 兼容回落 / CCU fast launch / AIV 缓存重放 / 单卡），都不命中才走 `Selector()` 选算法（单元 5）→ `HcclExecOp()` 执行（单元 7）。
5. **双模式入口**：`HcclAllReduce`（单算子 OPBASE）与 `HcclAllReduceGraphMode`（图模式 OFFLOAD），最终汇入同一个 `AllReduceOutPlaceCommon`。

## 第一部分｜使用侧剧本

### 1. 官方主流程与算子语义

官方给出使用通信库 API 的主流程：**配置集群信息 → 创建通信域 → 执行通信操作（点对点 / 集合）→ 销毁通信域**：

![集合通信操作流程：建域 → 通信操作 → 销毁（图源：HCCL 官方文档）](/images/cann/hccl/official/operation-flow.png)

AllReduce 的算子语义：将通信域内所有 rank 的输入归约（sum/prod/max/min）后，把结果发送到所有 rank 的输出 buffer，且**每个 rank 只能有一个输入**：

![AllReduce 算子图示（图源：HCCL 官方文档）](/images/cann/hccl/official/allreduce-op.png)

官方快速入门的 8 卡 AllReduce 样例，浓缩成剧本（细节见 `user_guide/quick_start.md` 与 hcomm 仓 examples）：

```c
// ① 设备与内存准备（ACL）
aclrtSetDevice(device);
aclrtMalloc(&sendBuf, size, ...);  aclrtMalloc(&recvBuf, size, ...);

// ② 创建通信域（L2-comm，HCOMM 接口）
HcclGetRootInfo(rootInfo);                          // root 生成标识信息并广播
HcclCommInitRootInfo(nRanks, rootInfo, device, &comm);

// ③ 在 Stream 上调用算子（L1，HCCL 接口）
aclrtCreateStream(&stream);
HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32,
              HCCL_REDUCE_SUM, comm, stream);
aclrtSynchronizeStream(stream);                     // 异步！必须同步等完成

// ④ 销毁与释放
HcclCommDestroy(comm);
```

三个读码要点：

- **异步语义**：`HcclAllReduce` 把任务挂到 Stream 就返回，完成靠 `aclrtSynchronizeStream`——这就是"通信与计算可重叠"的机制根源；
- **建域方式有三种**：rank table 文件（`HcclCommInitClusterInfo`）、root 信息（`HcclCommInitRootInfo`）、单机批量（`HcclCommInitAll`），另可用 `HcclCreateSubCommConfig` 切子域；
- **每个 rank 只能有一个输入**（AllReduce 语义约束），reduce op 支持 sum/prod/max/min。

### 2. 官方约束（集成时的高频坑）

- 多个通信域下的所有通信算子在每个 Device 上**必须串行下发**，不允许乱序、多线程并发下发，不支持线程重入；
- 同一 Device 上，同一通信域内所有通信算子的下发线程需使用**相同的 Context**；
- 同一通信域内**不支持图模式与单算子模式混用**；同一 NPU 上需串行创建多个通信域。

## 第二部分｜源码精读：`all_reduce_op.cc`（281 行）

### 3. 文件结构总览

```text
all_reduce_op.cc
├── HcclAllReduce()               L23   单算子模式入口（extern "C"）
├── HcclAllReduceGraphMode()      L54   图模式入口（extern "C"）
└── namespace ops_hccl
    ├── AllReduceInitAndCheck()   L107  初始化 + 参数校验
    ├── CheckAllReduceInputPara() L135  指针非空校验（错误码规范上报）
    ├── FillAllReduceOpParam()    L159  填充 OpParam
    ├── AllReduceOutPlaceCommon() L189  ★ 核心分发逻辑
    ├── AllReduceEntryLog()       L236  入口日志
    ├── AllReduceOutPlaceGraphMode() L259 图模式包装
    └── AllReduceOutPlace()       L270  单算子模式包装
```

### 4. `HcclAllReduce` 主入口（L23~52）—— 三道闸门 + 四步主干

```c
HcclResult HcclAllReduce(void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAllReduce");
    // 闸门1：hcomm 版本 < 9.0.0 → 回落老实现
    if (GetHcommVersion() < CANN_VERSION(9, 0, 0)) { // compat handle
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }

    // 闸门2：设备不支持 out-place → 回落老实现
    bool isOutPlace = false;
    CHK_RET(IsOutPlaceDevice(isOutPlace));
    if (!isOutPlace) {
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }
    // 闸门3：count == 0 → 直接成功
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW();                 // 计时起点（闸门耗时不算）
    OpParam param;                               // 算子上下文（贯穿全程）
    CHK_RET(AllReduceInitAndCheck(...));         // 步骤1：环境变量 + 校验
    CHK_RET(AllReduceEntryLog(...));             // 步骤2：接口交互日志
    CHK_RET_AND_PRINT_IDE(AllReduceOutPlace(...), param.tag);  // 步骤3：执行
    CHK_RET(LogHcclExit("HcclAllReduce", param.tag, startut)); // 步骤4：出口日志
    return HCCL_SUCCESS;
}
```

**闸门逐个拆解**：

- **闸门 1（版本兼容）**：`GetHcommVersion()` 声明于 `src/common/hcomm_dlsym/hcomm_dlsym.h:23`，运行时通过 dlsym 从已安装的 hcomm 库取版本。版本不够 → 调 `HcclAllReduceInner`（老版 HCCL 的实现，**本仓不定义**——它是外部符号，ST 仿真桩 `test/st/.../hccl_stub.cc:196` 里直接返回 NOT_SUPPORT，真实环境由旧版运行库提供）。这就是"两仓独立版本演进"（单元 1）的活样本。
- **闸门 2（设备能力）**：`IsOutPlaceDevice` 定义在 `src/common/hccl_common.h:268`：

  ```c
  inline bool shouldGoOutPlace(HcclDevType deviceType)
  {
      return deviceType == HcclDevType::DEV_TYPE_950 || deviceType == HcclDevType::DEV_TYPE_960;
  }
  ```

  **out-place 指新版执行流程**（"从 HCOMM 中拿出执行权，由本仓新框架执行"），当前只在 950/960 设备启用，其余设备仍走老实现——新框架按设备灰度放量。
- **闸门 3（空操作）**：`count == 0` 是合法输入，语义为"什么都不做即成功"——接口宽容性的体现。

### 5. 错误处理宏族（`src/common/log.h:173~227`）—— "C 语言式异常"

全文件的 `CHK_*` 都是日志宏，语义一张表记住：

| 宏 | 行为 | 典型用途 |
| --- | --- | --- |
| `CHK_PTR_NULL(ptr)` | 空指针 → 打 ERROR → `return HCCL_E_PTR` | 指针校验 |
| `CHK_PRT_RET(cond, log, ret)` | cond 成立 → 执行 log → `return ret` | 自定义返回值（含成功） |
| `CHK_PRT_CONT(cond, log)` | cond 成立 → 只打日志，**不返回** | 记录告警继续跑 |
| `CHK_RET(call)` | `HcclResult != SUCCESS` → 打 trace → 原样 return | **最常用**，错误逐层向上冒泡 |
| `CHK_RET_AND_PRINT_IDE(call, id)` | 同上 + 额外打印通信域标识 | 主干执行点，便于定位是哪个 comm 出错 |

设计动机：C++14 无异常（嵌入式/性能约束），用宏实现"check-and-return"链，且**错误码一路上抛不吞**。`HCCL_E_AGAIN` 单独降级为 WARNING（可重试错误不算事故）。

### 6. `AllReduceInitAndCheck`（L107~133）—— 校验 + 装配上下文

```c
HcclResult AllReduceInitAndCheck(HcclComm comm, ..., OpParam& param)
{
    CHK_RET(InitEnvConfig());                          // ① 每次入口都解析环境变量
    CHK_RET(CheckAllReduceInputPara(comm, sendBuf, recvBuf, stream));  // ② 指针校验
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));         // ③ 查 rank 数
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));           // ④ 查本卡 rank
    CHK_RET(HcclGetCommName(comm, param.commName));    // ⑤ 查通信域名
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllReduce_%s", param.commName);  // ⑥ 生成 tag
    CHK_RET(HcclCheckTag(param.tag));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), param.tag);
    CHK_RET(CheckCount(count));                        // ⑦ 数值域校验
    CHK_RET(CheckDataType(dataType, true));
    CHK_RET(CheckReduceOp(dataType, op));
    return HCCL_SUCCESS;
}
```

- ③④⑤ 的 `HcclGetRankSize/HcclGetRankId/HcclGetCommName` 声明于**已安装的** `hccl/hccl_comm.h`（L2 域管理 API），实现在已安装运行库中，本仓不重复实现——控制面查询越层拿到，正是解耦设计的体现。
- ⑥ **tag 规则**：`"AllReduce_<commName>"`。注释点明"所有相同的算子可以共享"——tag 是后面 topoInfo/资源缓存的 key，同一 comm 上重复调用 AllReduce 可复用拓扑计算结果（单元 8 实锤）。
- ⑦ `CheckCount/CheckDataType/CheckReduceOp` 来自 `src/common/param_check.cc`，全算子共用。

### 7. `CheckAllReduceInputPara`（L135~157）—— 错误码规范上报

```c
RPT_INPUT_ERR(stream == nullptr, "EI0003",
    std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
    std::vector<std::string>({"HcclAllReduce", "nullptr", "stream", "non-null pointer"}));
CHK_PTR_NULL(stream);
// comm / sendBuf / recvBuf 同款四连
```

`RPT_INPUT_ERR` 把错误按 **CANN 统一错误码规范**（EI0003 = 输入参数错误）上报，占位符填充成"算子名 + 实际值 + 参数名 + 期望值"——用户侧看到的是规范报错文案而不是裸的 `HCCL_E_PTR`。**先 RPT 上报、再 CHK 返回**是固定搭配。

### 8. `FillAllReduceOpParam`（L159~187）—— 装配 OpParam

```c
u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];  // dataType → 字节宽度查表
u64 outputSize = count * perDataSize;             // 元素数 × 宽度 = 字节数（在这里换算！）
param.inputPtr/inputSize/outputPtr/outputSize ... // 指针 + 字节数
param.DataDes.count = count;                      // 元素描述保留元素数
param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;   // 算子类型枚举
param.enableDetour = false;                       // 绕行模式开关
param.deviceType = deviceType;
```

单元 1 的悬念在此收口：**API 层传元素个数，执行层要字节数，转换点就在这**。`OpParam`（定义于 `src/ops/op_common/inc/alg_param.h`）是贯穿 selector/executor/template 的通用算子上下文。

### 9. `AllReduceOutPlaceCommon`（L189~234）—— ★ 心脏：四条快速路径 + 正常路径

```c
CHK_RET(FillAllReduceOpParam(...));
CHK_RET(HcclGetOpExpansionMode(comm, param));       // ① 决定 param.engine（AICPU/AIV/CCU）

// 快速路径1：9.0.0 的 CCU 模式未接入新框架 → 回落老实现
if (opMode == OpMode::OPBASE && GetHcommVersion() == CANN_VERSION(9, 0, 0)
    && param.engine == CommEngine::COMM_ENGINE_CCU) {
    return HcclAllReduceInner(...);
}

// 快速路径2：CCU 快速下发（命中则免掉完整的计划构建）
CcuFastLaunchCtx* ccuFastLaunchCtx = nullptr;
if ((opMode == OpMode::OPBASE) && ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
    return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
}

// 快速路径3：AIV 通信计划缓存重放（同一 shape 重复调用直接重放指令流）
if (param.engine == CommEngine::COMM_ENGINE_AIV) {
    bool aivCacheHit = false;
    CHK_RET(HcclAivCacheCheckAndReplay(comm, param, aivCacheHit));
    if (aivCacheHit) { return HCCL_SUCCESS; }
}

// 快速路径4：单卡（rankSize == 1 退化为本地拷贝/空操作）
u32 userRankSize;
CHK_RET(HcclGetRankSize(comm, &userRankSize));
if (userRankSize == 1) {
    CHK_RET(SingleRankProc(comm, param));
    return HCCL_SUCCESS;
}

// 正常路径：选算法 → 执行
std::string algName;
std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
CHK_RET(Selector(comm, param, topoInfo, algName));     // op_common.h:170
CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));  // op_common.h:35
```

**为什么要这么多快速路径？**——集合通信的调用模式是"同 shape 高频重复"（训练每步都 AllReduce 同一大小的梯度）。快速路径的本质是**用缓存换规划开销**：

| 路径 | 命中条件 | 省掉什么 |
| --- | --- | --- |
| CCU 兼容回落 | 9.0.0 + CCU 引擎 | 新框架尚未支持的组合（正确性优先） |
| CCU fast launch | 满足 CCU 快发条件 | 完整 plan 构建 |
| AIV cache replay | AIV 引擎 + 同 shape 缓存命中 | selector + 资源申请 + 指令生成，**整条链** |
| 单卡 | rankSize == 1 | 一切通信开销 |

`Selector` 与 `HcclExecOp` 都声明于 `src/ops/op_common/op_common.h`——策略与机制的分离点，分别在单元 5、7 精读。

### 10. 图模式入口 `HcclAllReduceGraphMode`（L54~104）与双模式汇合

```c
HcclResult HcclAllReduceGraphMode(void* sendBuf, void* recvBuf, uint64_t sendCount, ..., const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize)
{
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));   // 图模式用 group 名换 comm 句柄
    ...
    ResPackGraphMode resPack;                           // 图模式资源包：tag + streams[] + scratchMem
    ...
    CHK_RET(AllReduceOutPlaceGraphMode(...));           // → AllReduceOutPlaceCommon(OpMode::OFFLOAD, resPack, ...)
}
```

两种模式的差异与汇合：

| | 单算子（OPBASE） | 图模式（OFFLOAD） |
| --- | --- | --- |
| comm 来源 | 调用方直接给 `HcclComm` | `group` 字符串 → `HcomGetCommHandleByGroup` 反查 |
| stream | 单 stream | `streams[]` 多流 + `scratchMem` 暂存内存 |
| tag | 自动生成 `AllReduce_<commName>` | 调用方显式指定（图内标识通信节点） |
| 汇合点 | `AllReduceOutPlaceCommon(OpMode::OPBASE, ...)` | 同一函数，`OpMode::OFFLOAD` |

图模式服务于**整图下发**场景：通信算子嵌在 AI 计算图里，由图引擎（GE）统一调度，编译期确定执行序，减少 Host 下发开销——对应 `src/ops/interface_graph_mode/` 目录。框架集成（`user_guide/framework_integration.md`）说明 torch_npu / MindSpore 等如何在这两种模式间选择。

### 11. `AllReduceEntryLog`（L236~257）—— 可控的入口日志

```c
if (forceLog || GetExternalInputHcclEnableEntryLog()) {   // 环境变量开关
    ... aclrtGetDevice(&deviceId); aclrtStreamGetId(stream, &streamId);
    snprintf_s(..., "tag[%s], sendBuf[%p], ..., streamId[%d], deviceId[%d]", ...);
    HCCL_RUN_INFO("Entry-%s:%s", opName.c_str(), ...);
}
```

默认不打（高频路径日志开销不可忽视），由环境变量 `HCCL_ENABLE_ENTRY_LOG` 类开关或图模式 `forceLog` 强制打开——**每次通信的完整入参可追溯**，是现场定位问题的第一手段。

## 第三部分｜把调用链放回全景

### 12. 共享组件：`op_common` 四件套

单看 all_reduce 会以为组件是私有的，其实它们住在 `src/ops/op_common/`：

| 目录 | 职责 | 与前面单元的连接 |
| --- | --- | --- |
| `selector/` | 算法/引擎选择的通用框架 | 单元 5 的决策树 |
| `executor/`（含 `channel/`、`registry/`） | 执行器框架与通道编排 | 单元 3 的原语在这里被调用 |
| `template/`（aicpu/aiv/ccu/dpu/wrapper/registry） | 算法模板框架 | 单元 4 的引擎维度 |
| `topo/` | rankGraph 拓扑信息的获取与转换 | 单元 2 的 RankGraph 在算子侧的适配 |

读码顺序建议：**先读 `all_reduce_op.cc` 看它如何调用 selector，再进 selector 看判定条件（单元 5 已读），最后挑一个最简单的 executor（如 sequence）读它的原语编排（单元 7）**。复杂的 executor（omnipipe 等）留到需要时再啃。

### 13. 带着读码三问走一遍

用课程导学的方法论检查这条链路：

**① 入口在哪？**
`HcclAllReduce` 的声明在 `include/hccl.h`；实现在 `src/ops/all_reduce/all_reduce_op.cc`。图模式入口则在 `src/ops/interface_graph_mode/`（区分单算子/图模式两条路）。

**② 数据在哪？**
用户 buffer（CommMem，单元 3）→ executor 编排的通道搬运 → 链路 → 远端 CommMem。中间可能经过 HCCL buffer 中转（性能分析文档的 usermem ↔ hcclbuffer 通路）。追踪点：executor 里对 Read/Write/LocalReduce 原语的调用序列（单元 7）。

**③ 谁在等待？**
三层同步（单元 3/4）：Thread 内算子顺序依赖；引擎内 Thread 间 ThreadNotify；跨 rank 的 ChannelNotify。宿主侧则是 `aclrtSynchronizeStream` 等 Stream 完成。追踪点：executor 代码中 Notify 的 Record/Wait 配对。

### 14. 自测题

1. 使用侧三段式是什么？为什么 `HcclAllReduce` 返回后还需要同步？
2. 建通信域的三种方式分别适用什么场景？
3. 三道闸门分别拦什么？`HcclAllReduceInner` 为什么是外部符号？
4. `count == 0` 返回 SUCCESS 而不是报参数错误，好还是不好？（对比 MPI 的处理）
5. 四条快速路径各自的命中条件是什么？为什么集合通信特别适合"缓存换规划"？
6. AIV 缓存重放如果遇到 `sendBuf` 地址变了但 shape 没变，还能命中吗？

::: details 自测答案

1. 建域 → Stream 上调用算子 → 销毁域。因为算子是异步语义，调用只是把任务挂到 Stream，必须用 `aclrtSynchronizeStream` 等待完成，这也是通信与计算可重叠的机制基础。
2. rank table 文件（`HcclCommInitClusterInfo`，有完整集群信息）；root 节点信息（`HcclCommInitRootInfo`，无 rank table 时两阶段协商）；`HcclCommInitAll`（单机批量建域）；另有 `HcclCreateSubCommConfig` 从已有域切子域。
3. 闸门 1 拦 hcomm 版本 < 9.0.0（走老实现）；闸门 2 拦非 950/960 设备（新框架灰度放量）；闸门 3 拦 count==0（空操作直接成功）。`HcclAllReduceInner` 由旧版运行库提供、本仓不定义——仿真环境里是 NOT_SUPPORT 桩，真实部署环境才有实现。
4. 见仁见智的接口设计题：MPI 同样允许 count=0；宽容性换来"空输入不算错误"的语义一致性，代价是可能掩盖上层逻辑 bug——HCCL 打了一条 WARNING 作为折中。
5. CCU 兼容回落（9.0.0 + CCU）、CCU fast launch（快发条件）、AIV cache replay（同 shape 缓存命中）、单卡（rankSize==1）。因为训练的调用模式是"同 shape 高频重复"——梯度每步同样大小，缓存命中率天然高。
6. 能。重放时按新 `inputPtr/outputPtr` 与基址的**偏移补丁**修正指令流，shape 才是缓存 key（单元 7 的 AIV 缓存逻辑展开）。

:::

## 本单元小结

- **使用侧**：建域（HCOMM）→ 算子（HCCL L1）→ 销毁；异步 Stream 语义是重叠的根源；
- **源码侧**：`all_reduce_op.cc` 三道闸门 → 四步主干 → 心脏 `AllReduceOutPlaceCommon`（四条快速路径 + `Selector()`/`HcclExecOp()` 正常路径）；
- **双模式**：OPBASE 与 OFFLOAD 汇合于同一心脏，图模式多带 `ResPackGraphMode` 上下文；
- **共享组件**：selector / executor / template / topo 四件套在 `op_common`，被所有算子复用；
- **读码三问**贯穿：入口（op.cc）、数据（executor 原语序列）、等待（Notify 配对 + Stream 同步）。

## 参考资料

- [使用通信库 API 实现通信功能（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/api_comm_impl.md)
- [HcclAllReduce 接口参考](https://gitcode.com/cann/hccl/blob/master/docs/zh/api_ref/comm_op_interface/HcclAllReduce.md)
- [通信域管理接口（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/README.md)
- [快速入门样例](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/quick_start.md)
- [examples：集合通信样例](https://gitcode.com/cann/hccl/tree/master/examples)

---

下一单元进入 **[7｜executor 与 template：执行机制](07-executor-template.md)**。

[返回课程导学 →](../hccl-source.md)
