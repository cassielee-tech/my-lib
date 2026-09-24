# 单元 1｜软件架构：分层与对外 API

> 所属课程：[HCCL 源码学习](../hccl-source.md) · 第 1 单元（共 10 单元）

::: info 本单元目标
读完后，你能够画出 HCCL & HCOMM 的**三层软件结构**与**五层对外 API**，解释三个设计决策（控制面/数据面分离、dlsym 解耦、legacy 冻结）；并逐段精读 `include/hccl.h`（262 行），看清 L1 算子 API 的 C ABI 设计。
:::

## 先记住 5 个结论

1. **软件分三层**：HCCL 集合通信算子（算子入口 → 算法选择 → 算法执行）→ HCOMM 集合通信域管理（通信域 + 拓扑 + 资源）→ HCOMM 基础通信（资源管理 + 通信原语执行）。
2. **控制面管资源，数据面管搬运**：拓扑查询、通道分配属控制面；Write/Read/Reduce 与 Notify 属数据面，两层接口独立演进。
3. **HCCL 与 HCOMM 通过 dlsym 动态加载衔接**，两仓可独立编译、独立发版；`legacy/` 只做历史兼容、不承接新特性。
4. **`hccl.h` = 纯 C ABI + 14 个通信原语**，四类模式：归约类 / 收集类 / 分发类 / 点对点交换类；API 层薄、稳、向后兼容。
5. **全部 API 共享统一签名模式** `数据指针 + count + dataType (+op/+root/+displs) + comm + stream` → 实现侧必有统一的参数校验与分发层（单元 6 验证）。

## 第一部分｜概念地图

### 1. 为什么必须分层

假设不分层：AllReduce 的代码里既要选算法、又要管拓扑发现、还要直接操作网卡队列。后果是——

- 换一颗芯片（如 950 引入 CCU 引擎），所有算子代码都要改；
- 框架想建通信域，必须依赖整个算子库；
- 自定义通信算子的开发者必须读懂全部细节才能写一个新算子。

分层之后，**变化被隔离**：算法与引擎的演进发生在 HCCL 算子层；拓扑与资源管理演进发生在 HCOMM；芯片差异被封装在基础通信层之下。

### 2. 三层结构总览

![HCCL 在 CANN 架构中的位置（图源：HCCL 官方文档）](/images/cann/hccl/official/architecture.png)

先用一张图看三层结构在源码仓中的落点：HCCL 算子层在 hccl 仓，域管理与基础通信在 hcomm 仓的两大模块：

![HCCL & HCOMM 软件分层逻辑视图（图源：HCCL 官方文档）](/images/cann/hccl/official/hccl-hcomm-logical-view.svg)

| 软件层次 | 职责 | 代码位置 |
| --- | --- | --- |
| **HCCL 集合通信算子** | 算子入口 → 算法选择 → 算法执行 | `hccl` 仓 `src/ops` |
| **HCOMM 集合通信域管理（HCCM）** | 通信域 + 拓扑管理（rank_graph）+ 资源管理 | `hcomm` 仓 `coll_communicator_mgr` |
| **HCOMM 基础通信（base_comm）** | 基础通信资源 + 通信原语执行 | `hcomm` 仓 `base_comm` |

对应到 HCOMM 仓库的目标目录（读码时对照）：

```text
hcomm/src
├── base_comm/                  # 基础通信层
│   ├── common/                 #   公共基础功能
│   ├── primitives/             #   通信原语（数据面）
│   └── resource/               #   通信资源（Endpoint/Channel/CommMem）
├── coll_communicator_mgr/      # 集合通信域管理
│   ├── api_c_adpt/             #   C 接口适配
│   ├── communicator/           #   通信域
│   ├── rank_graph/             #   拓扑管理（控制面核心）
│   ├── config_mgr/  resource_mgr/  dfx/  common/
└── legacy/                     # 历史版本兼容（A2&A3、A5 旧流程），不持续演进
```

::: tip 判断代码归属的口诀
**"选算法"在 HCCL，"建连接"在 HCOMM。** 看到一个 `.cc` 文件，先问它在回答哪个问题：是"这次 AllReduce 用 Ring 还是 NHR"（HCCL），还是"这两个 rank 之间能建什么链"（HCOMM）。
:::

### 3. 五层对外 API

分层不是内部洁癖，而是直接决定了对外暴露的接口层次：

![HCCL & HCOMM 对外接口分层：L1 算子 / L2 通信域与拓扑资源 / L3 原语与资源（图源：HCCL 官方文档）](/images/cann/hccl/official/hccl-hcomm-api.svg)

| 层次 | 接口（头文件） | 面向 | 职责 |
| --- | --- | --- | --- |
| **L1** | HCCL 算子（`hccl.h`） | AI 框架适配层 | AllReduce 等标准集合通信算子入口 |
| **L2-comm** | HCOMM 通信域（`hccl_comm.h`） | 框架适配层 | 通信域创建/销毁/子域切分 |
| **L2-res** | HCOMM 拓扑与资源（`hccl_res.h` / `hccl_rank_graph.h`） | 算子开发者 | 拓扑查询，Thread/Channel 等资源获取 |
| **L3-prim** | HCOMM 通信原语（`hcomm_primitives.h`） | 算子/通信库开发者 | Write/Read/Reduce + Notify |
| **L3-res** | HCOMM 基础资源（`hcomm_res.h`） | 通信库开发者 | 通信设备/通道/内存资源的管理 |

两个官方强调的组合，直接对应两类开发者：

- **L2-res（rank_graph）+ L3-prim = 自定义通信算子开发接口**：你可以查拓扑、拿通道，然后用自己的编排调用搬运与同步原语——这就是 `examples/04_custom_ops_p2p`、`05_custom_ops_allgather` 的玩法（单元 9 实战）；
- **L3-res + L3-prim = 通信库开发接口**：面向要自己写一个"HCCL"的场景。

::: warning 位置澄清
L2/L3 的头文件与通信域接口位于 [hcomm 仓库](https://gitcode.com/cann/hcomm)，不要在 hccl 仓里找 `hccl_comm.h`。HCCL 仓的 `include/hccl.h` 只有 L1 算子 API。
:::

### 4. 三条架构约束

`architecture-brief.md` 末尾给出四条约束，其中三条最值得在读码前记住：

**4.1 分层依赖方向**：上层依赖下层，**禁止反向依赖**：

```text
base_comm ✗→ coll_communicator_mgr   （基础层不能回头找管理层）
coll_communicator_mgr / base_comm ✗→ coll_comm_ops（管理层/基础层不能依赖算子层）
```

读码时的用途：在 `base_comm` 里看到引用上层符号，要么是理解错了，要么是历史遗留（去 `legacy/` 找线索）。

**4.2 控制面 / 数据面分离**：

| 平面 | 内容 | 特点 |
| --- | --- | --- |
| **控制面** | 资源管理、拓扑查询 | 建域/建链时执行，频率低 |
| **数据面** | Write / Read / Reduce / Notify | 通信热路径，频率极高 |

分离的价值：数据面接口可以按极致性能优化（单元 3 会看到它只剩几个"动词"），而不用拖着资源管理的复杂度。

**4.3 dlsym 动态加载解耦**：HCCL 算子层**编译期不链接 HCOMM**，运行时通过 `dlsym` 动态加载其接口（对应 `src/common/hcomm_dlsym`，单元 8 精读）。好处：两仓独立编译、独立版本演进，发布节奏互不阻塞；HCOMM 不在时 HCCL 仍可完成编译检查。

第四条约束（legacy 不持续演进）在单元 0 已说明，读码时把 `legacy/` 当"博物馆"即可。

### 5. 用这张分层图重看仓库地图

把单元 0 的地图按本单元视角重新标注：

```text
你调用的 API            HcclAllReduce(...)                 ← L1
                        │
HCCL 算子层             all_reduce_op.cc → selector        ← "用哪个算法/引擎"
（hccl 仓 src/ops）     template → executor                 ← "把算法执行出来"
                        │  dlsym 动态加载
HCOMM 域管理层          communicator / rank_graph           ← 建域、查拓扑（控制面）
（hcomm 仓）            │
HCOMM 基础通信层        primitives: Write/Read/Notify       ← 数据面
                        resource: Endpoint/Channel/Mem      ← 资源
                        │
硬件                    RoCE 网卡 / SDMA / UB / CCU …
```

## 第二部分｜源码精读：`include/hccl.h`（262 行）

> 精读对象：`include/hccl.h`——L1 算子 API 的全部声明。

### 段 1：C ABI 保护（L18~20, L259~261）

```c
#ifdef __cplusplus
extern "C" {
#endif
// ... 全部 API 声明 ...
#ifdef __cplusplus
}
#endif
```

`extern "C"` 保证 C++ 框架（MindSpore/Paddle 等）链接时符号名不被 C++ name mangling 破坏。HCCL 是纯 C ABI 的动态库——**API 层薄、稳、向后兼容**，这是 L1 作为公共契约的第一原则。

### 段 2：头文件依赖（L14~16）

```c
#include <hccl/hccl_types.h>   // 基础类型：HcclResult, HcclDataType, HcclReduceOp
#include <hccl/hccl_comm.h>    // 通信器：HcclComm 及 comm 管理接口
#include <acl/acl.h>           // ACL 运行时：aclrtStream
```

- 这三个头**不在本仓**，来自 CANN 安装目录。本仓的 `hccl.h` 只依赖"类型 + 通信器 + stream"，刻意保持薄。
- 依赖 `acl/acl.h` 只为一个类型：`aclrtStream` —— HCCL 的任务是**异步挂到 ACL stream 上**执行，这是它与普通计算算子的最大区别。

### 段 3：`HcclAllReduce` 签名（L35~37）—— 本文件最重要的一行

```c
extern HcclResult HcclAllReduce(
    void* sendBuf,          // 输入：本卡待归约的数据
    void* recvBuf,          // 输出：归约结果（每卡拿到完整结果）
    uint64_t count,         // 元素个数（不是字节数！）
    HcclDataType dataType,  // int8/.../float16/bfp16，决定单元素宽度与计算指令
    HcclReduceOp op,        // SUM / MIN / MAX / PROD
    HcclComm comm,          // 通信器：标识"哪一组卡、什么拓扑"
    aclrtStream stream);    // 异步执行载体
```

三个必懂语义（与 NCCL 同构）：

1. **归约语义**：数学上 `recvBuf[i] = Σ_{r=0}^{p-1} sendBuf_r[i]`（以 SUM 为例），p 卡参与、结果每卡一份。
2. **异步语义**：调用只做参数检查 + 任务下发，真正通信在 stream 上异步执行；确认完成需 `aclrtSynchronizeStream(stream)`。
3. **In-place 兼容**：传 `sendBuf == recvBuf` 即原地操作，实现侧要能处理。

核心概念三个：**comm（哪些卡）、stream（异步挂载）、dataType/op（怎么算）**。

### 段 4：全 API 家族（L39~257）—— 四类通信模式

| 模式 | API | 行号 | 语义 |
| --- | --- | --- | --- |
| 归约类 | `HcclAllReduce` | L35 | 先算后传，每卡一份结果 |
| 归约类 | `HcclReduce` | L245 | 算完只给 root |
| 归约类 | `HcclReduceScatter` | L67 | 算完按块散开（每卡拿自己那块） |
| 收集类 | `HcclAllGather` | L120 | 每卡出一份拼起来 |
| 收集类 | `HcclAllGatherV` | L138 | 各卡数量不等（counts + displs） |
| 收集类 | `HcclReduceScatterV` | L87 | 归约 + 不等量散开 |
| 分发类 | `HcclBroadcast` | L51 | root 的数据发给大家 |
| 分发类 | `HcclScatter` | L104 | root 数据切块发给各卡 |
| 点对点 | `HcclSend` / `HcclRecv` | L154 / L168 | 点对点收发 |
| 交换类 | `HcclAlltoAll(V/VC)` | L227 / L208 / L185 | 全交换；V = 不等量；VC 用矩阵描述 |
| 批量 | `HcclBatchSendRecv` | L256 | 批量收发（`HcclSendRecvItem` 数组） |

**V 后缀规律**：带 V 的接口多出 `counts[] + displs[]` 两个数组，描述"第 i 个 rank 发/收多少、偏移在哪"——等量版本是它的特例。

### 值得注意的工程细节

- `ReduceScatterV` 注释里数据类型**少了 uint 系列**、op **少了 prod**（L80~82）——实现支持的类型比 AllReduce 窄，**注释即契约**，与实现强一致。
- `AlltoAllVC`（L185）没有 counts/displs，用 `sendCountMatrix` 二维矩阵描述全量收发关系。
- 14 个 API 全部返回 `HcclResult`，全部接收 `HcclComm + aclrtStream` 结尾——这就是"统一签名模式"。

## 6. 自测题

1. 三层软件结构分别叫什么，各自职责一句话？
2. L2-res + L3-prim 组合面向谁？L3-res + L3-prim 又面向谁？
3. 为什么 `base_comm` 不允许依赖 `coll_communicator_mgr`？
4. HCCL 与 HCOMM 是怎样做到独立编译、独立发版的？
5. 为什么 `HcclAllReduce` 的 `count` 用元素个数而不是字节数？
6. 如果框架同时发起两个 AllReduce 到同一个 stream，执行顺序如何保证？

::: details 自测答案

1. HCCL 集合通信算子层（算子入口→算法选择→执行）；HCOMM 集合通信域管理层（通信域、拓扑、资源管理）；HCOMM 基础通信层（基础资源与通信原语执行）。
2. L2-res + L3-prim 是自定义通信算子开发接口（查拓扑拿资源 + 用原语编排数据搬运）；L3-res + L3-prim 是通信库开发接口（面向自研集合通信库）。
3. 保持依赖方向单向。基础层反向依赖管理层会让上层演进被下层锁死，分层失去意义；这类依赖只允许出现在 legacy 兼容代码中。
4. HCCL 通过 dlsym 在运行时动态加载 HCOMM 接口（src/common/hcomm_dlsym），编译期不链接，因此两仓可独立编译与版本演进。
5. dataType 决定单元素宽度，算子语义以元素为单位；换算成字节数是执行层的职责（单元 6 的 `FillAllReduceOpParam` 里 `count × DATATYPE_SIZE_TABLE[dataType]`）。
6. stream 语义 = 顺序执行队列，同 stream 上的任务按提交顺序执行——这是异步语义的正确性基础。

:::

## 本单元小结

- **三层**：算子层（hccl 仓）→ 域管理层 → 基础通信层（hcomm 仓）；
- **五层 API**：L1 算子 / L2 通信域 / L2 拓扑资源 / L3 原语 / L3 基础资源，组合出"自定义算子"与"通信库"两条开发路线；
- **约束**：依赖单向、控制面/数据面分离、dlsym 解耦、legacy 冻结；
- **L1 精读**：`hccl.h` 是纯 C ABI 的 14 个原语，统一签名模式预示实现侧有统一的校验与分发层；
- **读码地图**：从此看任何文件，先定位它在哪一层、属于哪个平面。

## 参考资料

- [HCCL & HCOMM 软件架构简介（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)
- [HCCL 头文件与库文件说明](https://gitcode.com/cann/hccl/blob/master/docs/zh/api_ref/hccl_header_and_lib.md)

---

下一单元进入 **[2｜通信域、Rank 与 RankGraph](02-comm-domain-rank-graph.md)**。

[返回课程导学 →](../hccl-source.md)
