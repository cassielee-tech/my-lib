# 单元 9｜MC2 自定义算子框架

> 所属课程：[HCCL 源码学习](../hccl-source.md) · 第 9 单元（共 10 单元，收官）
> 精读对象：`include/hccl_mc2.h`（97 行）+ `src/common/hccl_mc2.cc`（168 行）+ `examples/04_custom_ops_p2p` + `examples/05_custom_ops_allgather`

::: info 本单元目标
读完本单元，你将知道**自定义通信算子的两条开发路径**（声明式 Kfc 参数包 vs 手写资源编排），并通过官方样例验证一个重要判断：**前面八个单元学的资源模型不是内部实现细节，而是层次清晰的公开编程模型**。最后用一张知识地图收束整个课程。
:::

## 先记住 3 个结论

1. **自定义通信算子有两条路径**：路径 A（`hccl_mc2.h` 的 Kfc 参数包 + `HcclCreateOpResCtx`，面向 MC2 通算融合场景的**声明式**接口）与路径 B（`hccl_res_expt.h` 资源 API + L3 原语**手写**算子，官方 examples 走的路）。
2. **examples/04 的 send.cc 就是 op_common 的教学版**：tag → engineCtx 复用判定 → 线程申请/导出 → 建链 → 取本端/远端 cclBuffer → resCtx 拷入 device → LaunchKernel——单元 7/8 学的资源模型原样复现，证明那套模型就是**公开的编程模型**。
3. **MC2 与主框架的连接点**是单元 5 埋的伏笔：`opParam.isMc2` 时 `ExecuteSelector::Run` 硬编码取 priority 18 选择器（AllReduce 系 CCU 选择器）——MC2 自定义算子复用主框架的选算法逻辑。

## 1. 路径 A：`hccl_mc2.h` —— Kfc 参数包 + 资源上下文预创建

### 1.1 API 全景（97 行全是这套）

```c
HcclKfcAllocOpArgs(void** opArgs);                      // 分配参数包
HcclKfcFreeOpArgs(void* opArgs);                        // 释放
HcclKfcOpArgsSetSrcDataType / SetDstDataType            // 数据类型
HcclKfcOpArgsSetReduceType / SetCount                   // 归约类型 / 元素数
HcclKfcOpArgsSetAlgConfig(opArgs, char* algConfig);     // ★ 算法配置字符串
HcclKfcOpArgsSetCommEngine(opArgs, uint8_t engine);     // ★ 引擎选择
HcclCreateOpResCtx(HcclComm comm, uint8_t opType, void* opArgs, void** opResCtx);  // ★ 预创建资源
```

### 1.2 不透明指针的 ABI 哲学（hccl_mc2.cc:21）

```c
struct HcclOpArgs {                  // 定义在 .cc 里，头文件只见 void*
    HcclDataType srcDataType;
    HcclDataType dstDataType;
    HcclReduceOp reduceType;
    uint64_t count;
    char algConfig[ALG_CONFIG_SIZE]; // 128 字节算法配置
    CommEngine commEngine;
    uint64_t reverse;
    void Init() { srcDataType = FP16; dstDataType = FP16; reduceType = SUM; count = 0; }  // 安全默认值
};
```

**头文件只有 `void*`，结构体藏在实现里**：字段怎么加都不破坏 ABI——和单元 1 `extern "C"` 的稳定性哲学一脉相承。setter 逐字段校验（`HcomCheckDataType` / `SYS_MAX_COUNT` 上限），`Init()` 给安全默认值。

### 1.3 两个关键约束与一个闭环

```c
// 约束1：引擎白名单（hccl_mc2.cc:131）
if (commEngine != COMM_ENGINE_AICPU && commEngine != COMM_ENGINE_AIV) return HCCL_E_NOT_SUPPORT;
// （注释：A3只支持AICPU和AIV场景）

// 约束2：opType 必须在 HcclCMDType 枚举范围内（L147）

// 闭环：HcclCreateOpResCtx 的真正实现不在本仓
CHK_RET(HcclCreateOpResCtxInner(comm, opType, ...));   // ← dlsym 到 hcomm（hccl_inner_dl.h）
```

`HcclCreateOpResCtxInner` 正是单元 8 `hccl_inner_dl.h` 里 `DECL_WEAK_FUNC` 声明的那个符号——**MC2 的资源创建落在 hcomm 侧**，本仓只做参数包装与校验。`algConfig` 字符串则由单元 5 selector 的 `configAlgMap` 消费（`GetExternalInputHcclAlgoConfigAllType` 同源机制）。

## 2. 路径 B：官方样例 04（P2P）—— 五步法手写算子

`examples/04_custom_ops_p2p/op_host/send.cc` 的 `HcclSendCustom`，逐段对照前面单元：

```c
// STEP 0：上下文（与单元 6 FillAllReduceOpParam 同构）
sprintf_s(param.tag, ..., "hccl_custom_p2p");           // 自定义算子自己起 tag
HcclGetCommName(comm, param.commName);
param.opType = HcclCMDType::HCCL_CMD_SEND;              // 复用主框架的 opType 枚举

// STEP 1：拓扑信息（单元 8 InitRankInfo 同款）
HcclGetRankId / HcclGetRankSize / GetDeviceType

// STEP 2：资源（与单元 8 TryReuseResource + HcclAllocAlgResourceAICPU 一模一样的模式）
if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == SUCCESS) {
    param.resCtx = ctx;                                  // 复用
} else {
    HcclEngineCtxCreate(...);                            // 新建
    // 2.1 线程：stream→CPU_TS线程→导出为AICPU可用；反向同理
    HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &param.cpuThread);
    HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, COMM_ENGINE_AICPU_TS, &...);
    // 2.2 建链（样例 utils.cc:44，两种姿势）
    // 2.3 本端/远端中转内存（单元 8 L1821 同款）
    HcclGetHcclBuffer(comm, &localBuffer...);
    HcclChannelGetHcclBuffer(comm, channelHandle, &remoteBuffer...);   // ← 远端地址交换
    aclrtMemcpy(resCtx, size, &resCtxHost, size, H2D);    // 计划落 device（单元 8 同款）
}

// STEP 3：LaunchKernel(param, stream)                    // 下发 AICPU kernel
```

**建链的两种姿势**（utils.cc:44 `AcquireChannel`）——单元 8 知识的实战：

```c
// A2/A3：简单声明式（remoteRank + HCCS 协议 + notifyNum=2 直接建）
HcclChannelDescInit(&desc, 1); desc.remoteRank = dstRank; desc.channelProtocol = COMM_PROTOCOL_HCCS;
HcclChannelAcquire(comm, engine, &desc, 1, channel);

// A5(950)：先查 rank graph 再按链路属性建（CalcLevel2Uboe 同款查询姿势）
HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize);
for (link : linkList) 若 linkProtocol == COMM_PROTOCOL_UBC_CTP:
    desc.localEndpoint/remoteEndpoint ← link 的 endpoint 全量信息
    HcclChannelAcquire(...);
```

结论：**样例证明了单元 7/8 不是"内部实现细节"，而是层次清晰的公开编程模型**——HCCL 自己的算子和用户的自定义算子用同一套积木。

## 3. 官方样例 05（AllGather）：双引擎对照实验

```text
examples/05_custom_ops_allgather/
├── aicpu/          # AICPU 引擎版：op_host + op_kernel_aicpu
├── ccu/            # CCU 引擎版：op_host + op_kernel_ccu
└── op_host/all_gather.cc
```

同一算法、两种引擎各写一遍——单元 4/7"引擎差异在模板层"的知识在这里变成可对照运行的代码。样例 README 明说：AIV 版基于"AIV 通信编程接口"，含 `aiv_communication_base_v2.h`（通信基类）与 `sync_interface.h`（同步接口）。

## 4. 全课程知识地图：十个单元一图收束

```text
单元 0  全景与仓库地图      三层架构、目录地图、调用链预览
单元 1  分层架构 + hccl.h   五层 API；14 个 C ABI 原语；comm/stream/dataType 三概念
单元 2  通信域与 RankGraph  七个拓扑概念；Edge→Link→Channel 递进链
单元 3  原语与同步          四原语概念；网络/内存两组动词；ThreadNotify/ChannelNotify
单元 4  通信引擎            AICPU=控制核+TS调度；AIV=向量核换时延；CCU=专用协处理器
单元 5  算法 + selector     注册表(priority) + 引擎瀑布(CCU→AIV→AICPU) + 决策树(拓扑×数据量×卡数)
单元 6  AllReduce 调用链    三道闸门（版本/设备/count==0）→ 校验 → 四条快速路径 → 心脏
单元 7  executor+template   算法名=骨架×(匹配器+模板)；AllReduce=RS+RS+AG+AG；回退记忆；Mesh RS=全互联write+本地归约
单元 8  资源 + dlsym        engineCtx 缓存主干；topo 九步流水线；通道=资源交换点；一致性校验；三层兼容
单元 9  MC2 + examples      Kfc 参数包声明式接口；样例=公开编程模型的活文档
```

贯穿十个单元的四个设计模式：

| 模式 | 出现位置 |
| --- | --- |
| **静态注册 + 工厂** | selector 注册表、executor 注册表（宏 + `__COUNTER__` + 静态初始化） |
| **责任链降级** | 引擎瀑布、回退记忆、dlsym 弱符号桩、版本闸门 |
| **序列化缓存** | topoInfo、执行计划、AIV 指令流、回退结果（全挂 engineCtx） |
| **字符串解耦** | 算法名连接 selector 与 executor；tag 连接一切缓存 |

## 5. 结业思考题

1. 若要给 HCCL 加一个新集合算子（如 AlltoAllW），要动哪几个目录？写一份 checklist。
   （参考答案骨架：`include/hccl.h` 加 API → `src/ops/all_to_all_w/` 四段式 → selector 注册 + executor 注册 → param_check 增类型 → experimental/ 先行灰度 → UT/ST）
2. 用户用路径 B 手写的算子，能享受单元 7 的"回退记忆"和 AIV 缓存重放吗？为什么？
   （提示：那些机制依赖 op_common 的总控流程；路径 B 是旁路——所以样例自己写 engineCtx 复用）
3. 全课程哪个设计让你觉得最值得抄走用到自己的项目里？为什么？

## 6. 下一步建议

- **动手路线**：跑通 `examples/04`（最快见全貌）→ 读 `docs/zh/build/build.md` 上板 → 用 profiling 工具实测一次 AllReduce；
- **深入路线**：见[课程总结](summary.md)的"继续深入五条路线"（写自定义算子 / 调优 / 排障 / 架构 / 回到训练系统）；
- **姊妹课程**：继续进入 **[HCOMM 源码学习](../hcomm-source.md)**——下两层（域管理 + 基础通信）深潜与自定义通信算子开发，与本单元路径 B 无缝衔接；
- **源码对照**：`experimental/ops/` 是社区试验算子，结构同 `src` 但不编入商用——读它没有历史包袱。

## 本单元小结

- **两条开发路径**：Kfc 参数包（声明式、面向 MC2 通算融合）vs 资源 API + 原语（手写、完全掌控）；
- **ABI 哲学**：不透明指针 + setter 校验 + 安全默认值——扩展性不靠改头文件；
- **样例即文档**：examples/04 的五步法与 op_common 同构，公开编程模型实锤；
- **知识地图 + 四个设计模式**：整个课程的收束——带着这张图回到训练系统，每个环节都能落到具体文件。

## 参考资料

- [MC2 自定义算子框架头文件（hccl_mc2.h）](https://gitcode.com/cann/hccl/blob/master/include/hccl_mc2.h)
- [examples：自定义通信算子样例](https://gitcode.com/cann/hccl/tree/master/examples)
- [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)
- [通信算子开发指南（hcomm 仓文档）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/README.md)

---

进入 **[课程总结｜HCCL 源码阅读地图 →](summary.md)**

[返回课程导学 →](../hccl-source.md)
