# 单元 8｜资源地基与 dlsym 解耦

> 所属课程：[HCCL 源码学习](../hccl-source.md) · 第 8 单元（共 10 单元）
> 精读对象：`src/ops/op_common/op_common.cc`（HcclCalcTopoInfo / HcclGetAlgRes / HcclGetThread / HcclGetChannel）、`src/ops/op_common/topo/topo_host.cc`（1132 行）、`src/common/hcomm_dlsym/`（29 个文件）+ `src/common/compat.cc` + `src/hccl.cmake`

::: info 本单元目标
前七个单元讲完"一次调用"的动态执行；本单元往下挖两层地基：**op_common 的资源系统**（engineCtx 缓存主干、topo 计算流水线、通道/线程申请）与 **hcomm_dlsym 解耦机制**（弱符号、三层兼容、版本探测）——回答"拓扑字段谁算的、缓存挂在哪、两仓为什么能独立发版不崩"。
:::

## 先记住 7 个结论

1. **engineCtx 是全系统的缓存主干**：`HcclEngineCtxGet/Create/Copy` 是一个按 `tag + engine` 寻址的 KV 存储——topoInfo 缓存、执行计划缓存（resCtx）、回退记忆（fallbackTag）、增量建链的 hostCache 全挂在它上面。
2. **topo 计算一次、序列化永存**：`HcclCalcTopoInfo` 先查缓存（key = 单元 6 的 `tag = "AllReduce_<commName>"`），未命中才算 `CalcTopoShape` 流水线——"topoInfo 的 tag 所有相同算子可以共享"注释的实锤。
3. **资源复用三条件**：OPBASE 模式（或 CCU 引擎）+ 非 BATCH_SEND_RECV + algTag 命中 engineCtx → `isResourceReused = true`，线程/通道全部免建。
4. **对端 cclBuffer 地址的交换点**：`BuildChannelInfo` 里 `HcclChannelGetHcclBuffer` 拿到对端 cclBuffer 地址存进 `channel.remoteCclMem`（op_common.cc:1821）——单元 7 模板里 `linkSend.remoteCclMem.addr` 的数据来源。
5. **解耦 ≠ 不链接**：`libhccl.so` 链接期就依赖 `libhcomm.so`，解耦指的是**版本可以错配而系统不崩**——新 HCCL 配旧 hcomm（弱符号兜底 + 能力降级），或反之。
6. **三层兼容体系**：编译期（头文件桩 + `CANN_VERSION_NUM` 门控）→ 链接期（弱符号桩打进 `hccl_compat.so`）→ 运行期（`GetHcommVersion` 版本号 + `HcommIsSupport*` 能力探测）。
7. **一个符号的完整生命周期**：`DECL_WEAK_FUNC`（声明）→ `DEFINE_WEAK_FUNC`（弱定义 + 报错桩 + 能力标志）→ `INIT_SUPPORT_FLAG`（dlsym 探测置位）→ 业务代码按 `HcommIsSupport<func>()` 降级。

## 第一部分｜op_common 支撑系统：topo、资源与缓存

### 1. engineCtx：一切缓存的底座

| 调用点 | tag | 存什么 |
| --- | --- | --- |
| `HcclCalcTopoInfo`（op_common.cc:1104） | `param.tag`（`AllReduce_<commName>`） | topoInfo 序列化（CPU_TS 内存） |
| `TryReuseResource`（L1167） | `param.algTag`（算子+算法名） | 执行计划 resCtx |
| `HcclExecOp` 回退记忆（单元 7） | `fallbackTag` | 回退后的新算法名 |
| `GetAlgResAICPU`（L1454） | `algTag + "_hostCache"` | 增量建链的 host 侧通道表 |
| `GetOrCreateAivCacheIndexCtx`（单元 7） | AIV 专用 | 指令流录制队列 |

**设计本质**：把"算过一次就不要再算"的知识（拓扑、计划、失败记忆）统一挂在通信域上，按 tag 寻址、按 engine 分内存类型。engineCtx 本身又是经 dlsym 调 hcomm 的（L2 域管理服务）。

### 2. topo 计算流水线（topo_host.cc）

#### 2.1 入口：缓存优先（op_common.cc:1098）

```c
HcclResult ret = HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CPU_TS, &ctx, &size);
if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
    CHK_RET(InitRankInfo(comm, topoInfo.get()));        // 首次：真算
    std::vector<char> seq = topoInfo->Serialize();
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, ..., size, &ctx));   // 存缓存
    memcpy_s(ctx, size, seq.data(), size);
    return HCCL_SUCCESS;
}
topoInfoTemp.DeSerialize(seq);                            // 命中：反序列化直接用
```

#### 2.2 `CalcTopoShape` 九步流水线（topo_host.cc:771）

```c
ExtractNetLayerDetails   // 网络层数、每层实例规模（instSizeListOfLayer）
→ CalcLevel1Nhr          // ★ GCD 规则
→ ExtractTopoDetails     // 每层各拓扑形态的 rank 数（rankNumForTopoType）
→ CalcLevel0TopoShape    // ★ 单机形态分类
→ Is2DieFullMesh         // 双 Die 全互联判定
→ IsLevel0PcieMix        // PCIE 混连判定
→ CalcLevel0MeshType     // TWO_DIE_REGULAR / NOT_REGULAR
→ CalcLevel2Uboe         // 三级网 UBOE 协议判定
→ CalcLevel2Ubg
```

**两个关键算法**：

`CalcLevel0TopoShape`（L578）——单机形态三分法：

```c
if (topoInstNum == 1 && 只有 1DMESH 记录)        → level0Topo = MESH_1D;
else if (topoInstNum == 1 && 只有 CLOS 记录)      → level0Topo = CLOS;
else if (topoInstNum == MESH_1D_CLOS 实例数)      → level0Topo = MESH_1D_CLOS;
    且 closRankNum > BIG_CLOS_RANGE → level0BigClosRange = true;   // 单元 5 回退AIV的条件之一！
else  → 默认 CLOS（A2 场景兜底）
```

`CalcLevel1Nhr`（L638）——注释直白：

> "当 Level0 GCD 为 1 时，Mesh 无意义，需要退化为单级 NHR"

即 `gcd(instSizeListOfLayer[0]) == 1` → `Level1Nhr = true`（框规模互质时无法按框分组，跨机直接 NHR 一步到位）。单元 5 决策树里 `topoInfo->Level1Nhr` 优先选 `SoleNHR` 的判定源头在此。

`CalcLevel2Uboe`（L662）则展示**跨仓查询的姿势**：`HcclRankGraphGetLinks`（L3 拓扑资源 API，经 dlsym）取链路属性看协议是不是 `COMM_PROTOCOL_UBOE`——HCCL 自己不维护链路数据库，按需查 hcomm 的 rank graph（单元 2 概念的落地）。

### 3. 资源申请全流程 `HcclGetAlgRes`（op_common.cc:1177）

```c
① TryReuseResource(...)                                  // 复用判定
② needInconsistentCheck = NeedInconsistentCheck(...)      // 首次下发检测（context 未创建 ⇒ 首次）
③ executor->CalcAlgHierarchyInfo(...)                     // 匹配器算分级子组
④ executor->CalcRes(...) → resRequest                     // 模板声明的资源清单
⑤ GetAlgResWithEngine(...)                                // 按引擎分派:
     AICPU_TS → GetAlgResAICPU     AIV → GetAlgResAiv
     CCU       → GetAlgResCcu      CPU → GetAlgResDPU
⑥ FillOpExchangeInfo + CompareOpExchangeInfos             // ★ 跨 rank 参数一致性校验
```

#### 3.1 复用判定 `TryReuseResource`（L1144）

三条排除规则 + 一次查询：

```c
BATCH_SEND_RECV + OPBASE → increCreateChannelFlag = true（增量建链模式，不复用）
图模式 且 非CCU引擎       → 不复用（图模式资源随图生命周期）
// 引擎内存类型映射：AIV→CPU_TS（host内存）、CPU(DPU)→AICPU_TS（device内存）
HcclEngineCtxGet(comm, algTag, ctxEngine, ...) 命中 → isResourceReused = true
```

#### 3.2 AICPU 路径 `GetAlgResAICPU`（L1444）三件套

```c
HcclAllocAlgResourceAICPU:
 ├─ HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize)   // ① 中转缓冲
 │    resCtxHost->cclMem = {DEVICE, addr, size};              //    "CCL IN使用所有的CCL Buffer"
 ├─ HcclGetThread(...)                                        // ② 线程
 └─ HcclGetChannel(...)                                       // ③ 通道
之后: resCtxHost->Serialize() → HcclMemcpyCtxHostToDevice     // ④ 计划落进 device 内存
```

**④ 值得注意**：AICPU 引擎的执行计划存在 **device 内存**（`HcclEngineCtxCreate(COMM_ENGINE_AICPU_TS)` + `HcclEngineCtxCopy`），因为真正消费计划的是设备上的 AICPU kernel——host 只生产计划。

#### 3.3 线程申请 `HcclGetThreadWithConfig`（L1518）

```c
threadConfigs[0].notifyNumPerThread = resRequest.notifyNumOnMainThread + 1;  // 主流多1个：host-device同步
threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i-1];    // 从线程按需
HcclThreadAcquireWithConfig(comm, COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS, ...);
// 展开流（unfoldThread）：单独申请 1 个 CPU 引擎线程
//   "展开流需要一个Notify用于 AICPU按序下发"（ORDER_UNFOLD_THREAD_NOTIFY_NUM）
//   已存在则复用（GetUnfoldThreadInfo）
```

主线程多出的 1 个 notify 专用于 host↔device 同步——notify 不只是线程间信号，还是**跨越 host/device 边界的信号**。旧版 hcomm 不支持 per-thread 配置时降级为"全员 maxNotifyNum+1"（L1578~1586）——又一个按能力降级的例子。

#### 3.4 通道申请 `HcclGetChannel`（L1753）与 `BuildChannelInfo`（L1784）★

```c
// 图模式先注册用户 buffer（input/output 两个 tag + memHandles）
if (opMode == OFFLOAD) RegGraphModeBuffers(...);
// 每层通道请求按端点位置分流
locType == ENDPOINT_LOC_TYPE_DEVICE → device 建链（AICPU_TS）
locType == ENDPOINT_LOC_TYPE_HOST   → host 建链（CPU）
```

`BuildChannelInfo` 填出 `ChannelInfo` 完整画像：

```c
channel.remoteRank / protocol / notifyNum / handle;                  // 基本属性
HcclRankGraphGetEndpointInfo(ENDPOINT_ATTR_BW_COEFF) → portGroupSize; // 端口组大小（带宽系数）
HcclRankGraphGetEndpointInfo(ENDPOINT_ATTR_DIE_ID)  → dieId;          // Die 编号（POD收敛调度用）
HcclChannelGetHcclBuffer(handle, &remoteCclBufferAddr, &size);        // ★ 对端 cclBuffer 地址
channel.remoteCclMem = {DEVICE, remoteCclBufferAddr, size};
```

**L1821 就是单元 7 的答案**：模板里 `linkSend.remoteCclMem.addr` 的数据来源——建链时通过 `HcclChannelGetHcclBuffer` 交换到的对端缓冲地址。图模式还额外 `GetGraphModeBuffers` 把用户 input/output buffer 注册进通道（对端才写得了你的 userOut）。

#### 3.5 一致性校验（L1224）——分布式特有的防御

首次下发时，把本次参数打包成 `OpExchangeInfo`（cclBufferSize / root / opType / dataType / count / **aivCoreLimit** / group / tag），跨 rank 交换比对：

- **为什么**：集合通信是"所有 rank 必须跑同一个算法同一套参数"的强一致协议，任何一张卡的 count/dataType 不同都会死锁或数据错乱；
- `aivCoreLimit` 也在校验清单里：各卡向量核配额不同会导致 AIV block 数不一致 → 流水错位；
- 单元 6 提到的 `inconsistent_check.cc`（`src/common/`）就是这个机制的实现文件。

#### 3.6 增量建链（BATCH_SEND_RECV 专用，L1467~1476）

```c
hostCtxObj.DeSerialize(cachedData);                       // 取 hostCache 里的已有通道表
CompReqChannelWithExistChannel(hostCtxObj.channels, resRequest);   // remoteRank 集合做差
if (差集为空) → ReuseCachedDeviceCtx(...)                 // 全有：直接复用
else          → IncrementalCreateChannel(...)             // 只补缺的链路
```

批量收发对端集合动态变化——增量建链 = 已有通道做差集，只补缺不重建。

### 4. 第一部分小结图

```text
Selector() ──► HcclCalcTopoInfo ──► 【缓存命中?】
                    │ miss
                    └─► InitRankInfo → CalcTopoShape 九步流水线 → 序列化入 engineCtx
HcclExecOp ──► HcclGetAlgRes
                    ├─ TryReuseResource（algTag 命中?）
                    ├─ CalcAlgHierarchyInfo（匹配器）+ CalcRes（模板清单）
                    ├─ GetAlgResWithEngine → Thread / Channel / cclBuffer 三件套
                    │                          └─ BuildChannelInfo：remoteCclMem 交换点(L1821)
                    └─ 一致性校验（跨 rank 参数比对）
```

## 第二部分｜hcomm_dlsym：两仓解耦与三层兼容机制

### 5. 初始化链：库加载即探测

```c
// src/common/compat.cc:21
__attribute__((constructor)) void InitCompat()          // libhccl.so 被 dlopen/load 时自动执行
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, CompatSymInit);                  // 进程内只跑一次
}
void CompatSymInit(void) { HcommDlInit(); }              // "增加强制依赖"
```

```c
// hcomm_dlsym.cc:63
void HcommDlInit(void)
{
    if (gLibHandle != nullptr) return;                   // 幂等（aicpu_task_cache_clear.cc 也显式调它）
    gLibHandle = HcclDlopen("libhcomm.so", RTLD_NOW);    // RTLD_NOW: 立即解析全部符号
    ...
    HcclResDlInit(gLibHandle);          // 资源管理域
    HcclRankGraphDlInit(gLibHandle);    // 拓扑/链路查询域
    HcommPrimitivesDlInit(gLibHandle);  // L3 数据面原语域
    HcclInnerDlInit(gLibHandle);        // MC2 inner 接口域
    HcommProfilingDlInit(gLibHandle);   // profiling 域
    HcclCommDlInit(gLibHandle);         // 通信域管理域
    HcclResExptDlInit(gLibHandle);      // 试验资源域
    CcuResDlInit / HcclCcuResDlInit / CcuLaunchDlInit / CcuPrimitivesImplDlInit(...);  // CCU 四件套
}
```

**目录组织 = 按域分文件**：29 个文件里 12 个 `*_dl.cc` 对应 12 个功能域，每个域一个 `XxxDlInit(handle)`。`RTLD_NOW` 保证探测时机前置——不等第一次调用才发现符号缺失。

### 6. 核心三件套宏（dlsym_common.h:152~175）

```c
#define DECL_WEAK_FUNC(type, func_name, ...)
        type func_name(__VA_ARGS__) __attribute__((weak));        // 弱声明：本仓可以没有真实现

#define DEFINE_WEAK_FUNC(type, func_name, ...)                    // 弱定义三合一
    static bool g_##func_name##Supported = false;                 // ① 能力标志
    extern "C" bool HcommIsSupport##func_name(void) { ... }       // ② 能力查询函数
    type func_name(__VA_ARGS__) __attribute__((weak));
    type func_name(__VA_ARGS__)                                    // ③ 兜底桩：报错并返回 -1
    { HCCL_COMPAT_ERROR("[HcclWrapper] %s not supported", __func__); return (type)(-1); }

#define INIT_SUPPORT_FLAG(handle, func_name)                      // 初始化时探测
    do { void* ptr = (void*)HcclDlsym(handle, #func_name);        // dlsym 只为探测存在性
         g_##func_name##Supported = (ptr != nullptr); } while (0);
```

**弱符号的语义**：动态链接时若 `libhcomm.so` 导出了强符号 → 用真的；若安装的是旧版没有这个符号 → `hccl_compat.so` 里的弱桩生效，调用返回 -1 且打 `not supported`——**链接不炸、运行可查、调用方可降级**。

实际用法（hcomm_primitives_dl.cc，L3 数据面原语的完整清单）：

```c
DEFINE_WEAK_FUNC(int32_t, HcommWriteWithNotifyOnThread, ThreadHandle, ChannelHandle, void* dst, ...);
DEFINE_WEAK_FUNC(int32_t, HcommWriteReduceWithNotifyOnThread, ...);
DEFINE_WEAK_FUNC(int32_t, HcommReadNbiOnThread, ...);
DEFINE_WEAK_FUNC(int32_t, HcommChannelNotifyRecordOnThread, ...);   // 单元 7 wrapper 调用的原语家族
DEFINE_WEAK_FUNC(HcclResult, HcommThreadJoin, ThreadHandle, uint32_t timeout);  // 单元 7 特殊类型 barrier 用的
DEFINE_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheStart, ...);         // AICPU task cache
...
void HcommPrimitivesDlInit(void* handle)
{
    INIT_SUPPORT_FLAG(handle, HcommWriteWithNotifyOnThread);        // 逐个探测置位
    INIT_SUPPORT_FLAG(handle, HcommWriteReduceWithNotifyOnThread);
    ...
}
```

单元 7 模板 wrapper 调的 `HcommWriteOnThread` 一族、本单元第一部分的 `HcommIsSupportHcclThreadAcquireWithConfig()` 分支，**源头都在这个文件**。

### 7. 版本探测（hcomm_dlsym.cc:32）

```c
int GetHcommVersion(void)
{
    if (gHcommVersion == 0) {
        char hcommPkgName[] = "hcomm";
        if (aclsysGetVersionNum(hcommPkgName, &gHcommVersion) != ACL_SUCCESS)  // 问 ACL 系统层
            gHcommVersion = 0;
    }
    return gHcommVersion;   // 静态缓存
}
```

版本号编码规则（dlsym_common.h:14）：

```c
CANN_VERSION(M, m, p)    = M*10000000 + m*100000 + p*1000
CANN_VERSION(M, m, p, b) = 正式号 - 200 + b     // beta 子版本插在两个 patch 之间
```

配套查询函数展示"版本 + 能力"双判定：

```c
bool HcommIsExportThreadSupported()
{
    return GetHcommVersion() >= CANN_VERSION(9, 0, 0)          // 版本够
        && HcommIsSupportHcclThreadExportToCommEngine();       // 且符号在（老版本同号库可能被裁剪）
}
```

**单元 6 版本闸门的实现本体**：`GetHcommVersion()` 不走 dlsym，问 ACL 系统层 `aclsysGetVersionNum("hcomm", ...)`，结果缓存静态变量。

### 8. 编译期兼容：头文件桩 + 版本宏

**CMake 注入**（CMakeLists.txt:67）：

```cmake
if(HCCL_CANN_VERSION_NUM GREATER 0 AND HCCL_CANN_VERSION_NUM LESS 90000000)
    set(HCCL_CANN_COMPAT_850 ON)                # 8.5 兼容模式
endif()
function(hccl_apply_cann_compat target)
    target_compile_definitions(${target} PRIVATE CANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM})
endfunction()
```

**类型桩**（dlsym_common.h:38~113，注释即规则）：

```c
/* beta.1 起 hccl_types.h 已提供 HcclCommStatus，仅 < 9.1.0_beta.1 (8.5.0/9.0.0) 需要桩 */
#if CANN_VERSION_NUM < CANN_VERSION(9, 1, 0, 1)
typedef enum { HCCL_COMM_STATUS_READY = 0, ... } HcclCommStatus;
#endif
/* 9.0.0 起 hccl_types.h 已提供 ThreadHandle，仅 < 9.0.0 (8.5.x) 需要桩 */
#if CANN_VERSION_NUM < CANN_VERSION(9, 0, 0)
typedef uint64_t ThreadHandle;
#endif
// 还有 HcclOpDesc / HcclKernelFuncInfo / HcclCommStatePhase 等一大批
```

`hccl_host_comm_dl.h` 同款：`HcclOpExpansionMode` 枚举在 `< 9.1.0_beta.1` 时本地造桩。**效果**：同一份本仓源码，可以在 8.5 / 9.0 / 9.1 三代头文件环境下编译——"新代码编译于旧环境"。

### 9. 链接组织（hccl.cmake + hccl_compat.cmake）

```cmake
# hccl.cmake
target_link_libraries(hccl PRIVATE
    -Wl,--no-as-needed
    hcomm          # 真符号来源（全量 CANN 树内构建时指向 hcomm 目标）
    hccl_compat    # 弱符号桩库
    acl_rt ...)
add_dependencies(hccl hccl_compat)

# hccl_compat.cmake
if(STATIC_MODE)
    target_sources(hccl PRIVATE hccl_dl.cc hcomm_primitives_dl.cc ...)   # 直接编进静态库
else()
    add_library(hccl_compat SHARED)                                       # 独立小动态库
```

另有 `hccl_dl.cc` 的小技巧——dl 包装函数自身用**弱别名**导出：

```c
void* __HcclDlopen(const char* libName, int mode) { return dlopen(libName, mode); }
weak_alias(__HcclDlopen, HcclDlopen);     // 允许测试环境覆盖 dlopen 行为
```

### 10. 三种调用形态并存（以 hcomm_primitives_dl.cc 为例）

| 形态 | 例子 | 机制 | 用途 |
| --- | --- | --- | --- |
| **弱符号直调** | `HcommWriteOnThread(...)` | 链接期符号解析（真符号或弱桩） | 大多数原语 |
| **函数指针** | `g_HcommBatchTransferOnThread`（dlsym 取出后经 wrapper 转发） | dlsym + 指针 | 带复杂结构体参数的接口（`HcclHcommBatchTransferDesc`） |
| **能力查询** | `HcommIsSupport<func>()` | dlsym 探测结果标志 | 分支降级（`ThreadAcquireWithConfig`） |

### 11. 全链路收口：前面单元的调用点在这里对号入座

| 前面单元看到的调用 | 本单元的出处 |
| --- | --- |
| 单元 6 `GetHcommVersion() < CANN_VERSION(9,0,0)` 闸门 | hcomm_dlsym.cc:32，`aclsysGetVersionNum` |
| 单元 6 `HcclGetRankSize/HcclGetRankId/HcclGetCommName` | 声明于**已安装的** `hccl/hccl_comm.h`，符号由 libhcomm.so 提供，链接期直调（不在 dl 桩清单里——它们是老接口，所有版本都有） |
| 单元 5 selector 的 CCU/AIV 能力分支 | `HcommIsSupport*` 探测标志 |
| 单元 7 CCU `HcclEngineCtxGet/Create` | `hccl_res_dl.cc` 域（engineCtx 是 hcomm 提供的服务，本单元缓存主干由此而来） |
| 单元 7 wrapper 层 `HcommWrite/Read/Notify*OnThread` | `hcomm_primitives_dl.cc` 弱符号族 |
| 本单元 `HcommIsSupportHcclThreadAcquireWithConfig()` 降级 | `INIT_SUPPORT_FLAG` 探测 |
| 本单元 `HcclRankGraphGetLinks` | `hccl_rank_graph_dl.cc` 域 |

### 12. 架构收益清单

1. **独立发版**：hccl 与 hcomm 各自升级，`-Wl,--no-as-needed` + 弱桩保证任意版本组合可加载；
2. **灰度放量**：新特性（CCU 四件套、AIV 缓存）按 `HcommIsSupport*` 逐符号灰度，环境不够自动走老路径（单元 6 闸门、单元 7 回退的全链条支撑）；
3. **可测试性**：ST 仿真桩 `hccl_stub.cc` 只需覆盖外部符号（`HcclAllReduceInner`/`HcclGetRankSize` 全是桩），不依赖真实 hcomm；
4. **编译可移植**：一份源码三代头文件环境（8.5/9.0/9.1）都能编。

## 13. 思考题

1. topoInfo 缓存 key 用 `AllReduce_<commName>` 而资源缓存 key 用 `algTag`（含算法名）——为什么粒度不同？
2. 图模式非 CCU 不复用资源，为什么 CCU 例外？
3. 主线程 notify 数 `+1` 用于 host-device 同步——如果去掉这个同步会怎样？
4. `DEFINE_WEAK_FUNC` 的桩返回 `(type)(-1)`——调用方靠什么避免把 -1 当正常错误处理？
5. 为什么 `INIT_SUPPORT_FLAG` 用 dlsym 探测，而调用本身走弱符号？两者信息源一致吗？
6. 若 libhcomm.so 完全不存在，`HcclDlopen` 失败后系统什么行为？

::: details 参考答案要点

1. 拓扑与算法无关、与 comm 强相关（同 comm 上所有算子共享一张图）；资源与算法强相关（不同算法要不同的线程/通道组合）——缓存粒度跟着"变化的原因"走。
2. CCU 的指令流是纯 device 侧对象，图捕获的是下发动作（序列化后可反序列化复用，单元 7 的 `DeSerialize` 路径）；其他引擎的资源与图生命周期绑定更紧。
3. host 下发完毕 ≠ device 执行完毕，stream 语义要求顺序可依赖——去掉会出现 device 侧读到未就绪计划的竞态。
4. 能力查询先行：`HcommIsSupport*` 为 false 就不该走到调用（调用方分支降级兜底）。
5. 一致——同一个 handle 的同一符号表；探测与解耦调用一次完成，直调避免每次函数指针跳转。
6. fprintf stderr 后返回，所有能力标志保持 false → 单元 6 版本闸门 `GetHcommVersion()==0 < 9.0.0` → 全部走 `HcclAllReduceInner` 老实现（真实部署环境由旧版运行库提供该符号；仿真环境是 NOT_SUPPORT 桩）。

:::

## 本单元小结

- **engineCtx 缓存主干**：topoInfo / resCtx / 回退记忆 / hostCache 全挂通信域，按 tag 寻址、按 engine 分内存；
- **topo 九步流水线**：`level0Topo` 三分法 + `Level1Nhr` GCD 规则 + 跨仓查询姿势——单元 5 决策树输入的产地；
- **资源三件套**：cclBuffer / Thread / Channel；`BuildChannelInfo` 是对端缓冲地址交换点；一致性校验是分布式防御；
- **dlsym 三层兼容**：编译桩 / 弱符号 / 运行探测，两仓独立发版不崩；
- **设计模式再现**：序列化缓存（engineCtx）+ 责任链降级（能力探测）——与单元 7 一脉相承。

## 参考资料

- [HCCL & HCOMM 软件架构简介（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [HCCL 性能分析：通信数据通路（usermem/hcclbuffer）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/perf_analysis/perf_data_analysis.md)
- [源码构建（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/build/build.md)

---

下一单元进入 **[9｜MC2 自定义算子框架](09-mc2-custom-ops.md)**（收官）。

[返回课程导学 →](../hccl-source.md)
