# UB_CTP / UB_RTP / UBoE：UB 网络语义通信

> **深度参考** · 前置阅读：[源码课程单元 3｜数据面：原语与资源](/通信与昇腾/源码课程/03-data-plane.md)。本篇为参考手册体例，检索导向；事实基线见文首引用块。

> **网络语义**通信方式族（`COMM_PROTOCOL_UB_CTP=4`、`UB_RTP=9`、`UBOE=7`、`UBC_TP=5`）。基于**灵衢统一总线（UB，Unified Bus）**——华为新一代互联协议，旨在统一数据中心内部所有互联，提供低时延、高带宽通信。核心对象是 **Jetty**（UB 的 QP 等价物），经 **HCCP** 适配层统一管理，覆盖 **L0/L1/L2** 全层级。
>
> 事实来源（可信度：代码 > 仓内文档 > 社区资料）：
> `hcomm/include/hcomm_res_defs.h:124-132`（UB_CTP/UBC_TP/UBOE/UB_RTP 枚举）、`hcomm/src/legacy/ascend950/unified_platform/external_system/orion_adapter_hccp.h`（Jetty/JFC/TP/PostSend/EID 全量接口）、`hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.{h,cc}`（AICPU 侧 UB 传输）、`hcomm/src/legacy/ascend950/unified_platform/resource/ccu_transport/`（CCU 侧 UB 传输）、`hccl/docs/zh/user_guide/concepts.md`（HCCP 术语）、`shmem/docs/glossary.md`（UB 术语）。

---

## 1. UB 协议族

| 协议 | 值 | 旧名 | 说明 | 语义 | 典型层级 |
| --- | --- | --- | --- | --- | --- |
| `COMM_PROTOCOL_UB_CTP` | 4 | `UBC_CTP` / `UBC_TP` | 灵衢统一总线 CTP | 网络语义 | L0 / L1 |
| `COMM_PROTOCOL_UBC_TP` | 5 | — | 华为统一总线 UBC_TP | 网络语义 | L0 / L1 |
| `COMM_PROTOCOL_UBOE` | 7 | — | UB over Ethernet（UB 经以太） | 网络语义 | L1 / L2 |
| `COMM_PROTOCOL_UB_RTP` | 9 | `UBG` | 灵衢统一总线 RTP | 网络语义 | L1 / L2 |

> **UB（Unified Bus，灵衢总线）**：华为新一代互联协议，统一数据中心内部所有互联，提供低时延、高带宽通信（`shmem/docs/glossary.md`）。UB 超节点组网详见 [../networking/ub_superpod.md](/通信与昇腾/参考手册/组网形态/ub_superpod.md)。

`TpProtocol` 枚举（`orion_adapter_hccp.h:465`）：

```cpp
MAKE_ENUM(TpProtocol, CTP, TP, UBOE, UB_RTP)
```

---

## 2. 网络语义与 Jetty 通道

UB 网络语义与 [RoCE](/通信与昇腾/参考手册/协议与传输/roce_rdma.md) 同属网络语义（Channel + Write/Read/Notify），但**通道实体是 Jetty 而非 QP**：

![网络语义模型](/images/cann/hccl/wiki/architecture/semantic_communication.png)

> 图源：`hcomm/docs/zh/comm_op_dev_guide/prog_models_concepts/figures/semantic_communication.png`

| 概念 | RoCE 对应 | 说明 |
| --- | --- | --- |
| **Jetty** | QP | UB 的通信端点，`JettyHandle`/`TargetJettyHandle` |
| **JFC** | CQ | Jetty Flow Control，`JfcHandle`，完成队列控制 |
| **TP** | — | Transport，`TpProtocol`（CTP/TP/UBOE/UB_RTP） |
| **EID** | IP/GID | Endpoint ID（16 字节），UB 寻址 |
| **Notify** | Notify | 同步信号 |
| **Variable** | —（CCU 独有） | CCU 通用寄存器同步（见 [../engine/ccu.md](/通信与昇腾/参考手册/通信引擎/ccu.md)） |

---

## 3. Jetty 生命周期（HCCP 适配）

`orion_adapter_hccp.h` 提供 UB Jetty 的完整管理接口：

### 3.1 创建与模式

```cpp
MAKE_ENUM(HrtJettyMode, STANDARD, HOST_OFFLOAD, HOST_OPBASE, DEV_USED, CACHE_LOCK_DWQE, CCU_CCUM_CACHE)
HrtRaUbJettyCreatedOutParam HrtRaUbCreateJetty(RdmaHandle handle, const HrtRaUbCreateJettyParam& in);
```

| Jetty 模式 | 说明 |
| --- | --- |
| `STANDARD` | URMA 标准 CreateJetty |
| `HOST_OFFLOAD` | Host 侧展开下沉算子，需指定 sqeBbNum |
| `HOST_OPBASE` | Host 展开单算子，需指定 sqeBbNum |
| `DEV_USED` | 在 Dev 的 AICPU 展开算子，STARS 不能用 UB DirectWQE task，可用 UB DbSend task |
| `CACHE_LOCK_DWQE` | STARS 仅能用 UB DirectWQE task，不能用 DbSend task，需指定 sqeBbNum |
| `CCU_CCUM_CACHE` | CCU 专用（不需要指定 sqeBbNum），见 [../engine/ccu.md §13](/通信与昇腾/参考手册/通信引擎/ccu.md) |

**预留 jettyId**（`orion_adapter_hccp.h:485-487`）：`[1024, 1024+127]` 为 ccuJetty 预留；`[1024+192, 1024+192+4K-1]` 为 starsJetty 预留。

### 3.2 JFC 模式

```cpp
MAKE_ENUM(HrtUbJfcMode, NORMAL, STARS_POLL, CCU_POLL, USER_CTL)
JfcHandle HrtRaUbCreateJfc(RdmaHandle handle, CqCreateInfo& cqInfo, HrtUbJfcMode mode, u32 cqDepth);
```

### 3.3 导入与绑定

| 接口 | 作用 |
| --- | --- |
| `RaUbImportJetty(handle, key, keyLen, tokenValue)` | 导入对端 Jetty |
| `RaUbTpImportJetty(handle, key, keyLen, tokenValue, jettyImportCfg)` | 按 TP 导入（`JettyImportCfg`：localTpHandle/remoteTpHandle/localPsn/remotePsn/protocol） |
| `HrtRaUbJettyBind(jettyHandle, targetJettyHandle)` | 绑定本端与对端 Jetty |
| `HrtRaUbJettyUnbind` / `HrtRaUbUnimportJetty` / `HrtRaUbDestroyJetty` | 解绑 / 解导入 / 销毁 |

```mermaid
flowchart LR
    A["HrtRaUbCtxInit<br/>UB 上下文"] --> B["RaUbAllocTokenIdHandle<br/>分配 token"]
    B --> C["HrtRaUbCreateJfc<br/>创建 JFC"]
    C --> D["HrtRaUbCreateJetty<br/>创建本端 Jetty"]
    D --> E["交换 key<br/>（socket 握手）"]
    E --> F["RaUbImportJetty<br/>导入对端 Jetty"]
    F --> G["HrtRaUbJettyBind<br/>绑定"]
    G --> H["HrtRaUbPostSend<br/>下发 WQE 通信"]
```

---

## 4. 数据下发：PostSend 与 WQE

```cpp
MAKE_ENUM(HrtUbSendWrOpCode, WRITE, WRITE_WITH_NOTIFY, READ, NOP)
HrtRaUbSendWrRespParam HrtRaUbPostSend(JettyHandle jettyHandle, HrtRaUbSendWrReqParam& in);
```

`HrtRaUbSendWrReqParam` 关键字段：

| 字段 | 说明 |
| --- | --- |
| `opcode` | WRITE / WRITE_WITH_NOTIFY / READ / NOP |
| `inlineFlag` / `inlineData` | 内联数据 |
| `inlineReduceFlag` / `dataType` / `reduceOp` | 内联归约（随路 Reduce） |
| `notifyData` / `notifyAddr` / `notifyHandle` | notify 信息 |
| `localAddr` / `remoteAddr` / `size` | 本地/远端地址与大小 |
| `lmemHandle` / `rmemHandle` | 本地/远端内存句柄 |
| `handle`（TargetJettyHandle） | RM 模式下的对端 Jetty |

`HrtRaUbSendWrRespParam` 返回 `jettyId`/`funcId`/`dieId`/`piVal`/`dwqe[128]`/`dwqeSize`——即 **DirectWQE（DWQE）** 内容与 producer index。

> **两种下发方式**：**DirectWQE（DWQE）** 直接写 WQE 内容到网卡；**DbSend** 经 doorbell 触发。`DEV_USED` 模式用 DbSend，`CACHE_LOCK_DWQE` 模式用 DWQE。

---

## 5. EID 寻址

UB 用 **EID（Endpoint ID，16 字节）** 寻址（`orion_adapter_hccp.h:624-637`）：

```cpp
struct HrtDevEidInfo {
    std::string name;
    IpAddress ipAddress;
    uint32_t eidIndex;
    uint32_t type;
    uint32_t dieId;
    uint32_t chipId;
    uint32_t funcId;
    uint32_t devFeature;
};
std::vector<HrtDevEidInfo> HrtRaGetDevEidInfoList(const HRaInfo& raInfo);
HcclResult HrtRaGetEidByIp(RdmaHandle handle, const vector<IpAddress>& ipV4AddrList, vector<IpAddress>& eidAddrList);
```

> rank table 的 `rank_addr_list[].addr_type=EID` 即通过 HCCN TOOL 查询获取（见 [../04_昇腾集群组网总览.md §4.2](/通信与昇腾/参考手册/总览/cluster-topology.md)）。`devFeature` 右移 19 位取 UBoE 标志位（`HrtCheckUboeSupported`）。

---

## 6. AICPU 侧：`UbTransportLiteImpl`

`hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.h`（继承 `BaseTransportLiteImpl`，`DfxLinkType::UB`）：

| 类别 | 接口 |
| --- | --- |
| **EID/Jetty 查询** | `GetLocEid()`/`GetRmtEid()`、`GetJettyHandle()`/`GetJettyId()`/`GetTpn()` |
| **RMA 传输** | `Read`、`Write`、`ReadReduce`、`WriteReduce`、`WriteWithNotify`、`WriteReduceWithNotify` |
| **批量** | `BatchOneSidedRead`/`BatchOneSidedWrite`、`BatchTransfer`、`BatchTransferAll`（aicpu 批量下发） |
| **同步** | `Post`、`Wait`、`WaitWithTimeout`、`Fence`、`Drain` |
| **WQE 缓存** | `PreLaunchWqe`/`PostLaunchWqe`（配合 AICPU Task Cache，`EnableWqeTasks`/`GetWqeTasks`/`DumpWqeContent`） |

> 内部资源：`RmtUbBufLite`（addr/size/tokenId/tokenValue/notifyId）、`LocUbBufLite`、`DrainNotify`、`RmaConnLite`（`UbConnLite`）。WQE 经 `BuildUbDbSendTask` 构造 DbSend task，profiling 经 `DbSqeProfInfo`（含 jettyHandle/jettyId/tpn/locEid/rmtEid）上报。

---

## 7. CCU 侧：`CcuTransport`

`hcomm/src/legacy/ascend950/unified_platform/resource/ccu_transport/ccu_transport.cpp`——CCU 经 UB Jetty 通信，按连接类型分两种（`ccu_transport.cpp:33-41`）：

| 连接类型 | 协议 | 类 |
| --- | --- | --- |
| `CcuTransport::CcuConnectionType::UB_CTP` | UB_CTP | `CcuCtpConnection` |
| 其它（UB_RTP） | UB_RTP | `CcuTpConnection` |

CCU Jetty 模式为 `CCU_CCUM_CACHE`、JFC 模式为 `CCU_POLL`（`ccu_jetty.cpp:32-34`），详见 [../engine/ccu.md §13](/通信与昇腾/参考手册/通信引擎/ccu.md)。

---

## 8. 多平面与多 Jetty

UB 网络语义支持每卡多网口形成多个对等平面（详见 [../networking/multi_plane.md](/通信与昇腾/参考手册/组网形态/multi_plane.md)）：

| 机制 | 说明 |
| --- | --- |
| **多 Jetty** | `MultiJetty` 算法名后缀；20 个 CCU 多 Jetty 模板 |
| **多 Link** | `MultiLink` 算法名后缀、`ConcurMeshNHRMultiLink` |
| **TP 查询** | `RaUbGetTpInfo(handle, {locAddr, rmtAddr, tpProtocol, qos, slLevelCount}, out, num)` 获取 TP 信息（SL→jetty priority 映射，qos 默认 4） |
| **TpAttr** | `HrtRaGetTpAttr`/`HrtRaSetTpAttr`（UBoE 的 sip/dip 用 IPv4） |

---

## 9. 与其他通信方式的关系

| 对比项 | UB 网络语义 | [UB_MEM](/通信与昇腾/参考手册/协议与传输/ub_mem.md) | [URMA](/通信与昇腾/参考手册/协议与传输/urma.md) | [RoCE](/通信与昇腾/参考手册/协议与传输/roce_rdma.md) |
| --- | --- | --- | --- | --- |
| **语义** | 网络语义 | 内存语义 | 内存语义（硬化） | 网络语义 |
| **协议** | UB_CTP/RTP/UBoE | UB_MEM | 基于 UB 总线 | RoCE(v2) |
| **通道实体** | **Jetty** | 映射内存 | WQE + token | **QP** |
| **寻址** | EID | token/memKey | token | IP/GID |
| **搬运硬件** | UB 网卡 | UB 引擎 | UB 引擎 | RDMA 网卡 |
| **主力引擎** | AICPU_TS、CCU | AIV、AICPU_TS | CCU、AIV | AICPU_TS、Host CPU_TS |

---

## 10. 代码落点

| 组件 | 路径 |
| --- | --- |
| **HCCP/Jetty 适配（950）** | `hcomm/src/legacy/ascend950/unified_platform/external_system/orion_adapter_hccp.{h,cc}` |
| **AICPU UB 传输** | `hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.{h,cc}`、`ub_conn_lite.*` |
| **CCU UB 传输** | `hcomm/src/legacy/ascend950/unified_platform/resource/ccu_transport/`：`ccu_transport.cpp`、`ccu_jetty.cpp`、`ccu_connection.cpp`、`ccu_transport_group.cpp` |
| **UB 连接** | `hcomm/src/legacy/ascend950/unified_platform/resource/connection/dev_ub_connection.h`、`host_ub_connection.h` |
| **HCCP 适配（910）** | `hcomm/src/legacy/ascend910/platform/common/adapter/adapter_hccp.cc` |
| **协议枚举** | `hcomm/include/hcomm_res_defs.h:124-132` |

---

## 11. 参考资料

- `hcomm/docs/zh/comm_op_dev_guide/prog_models_concepts/comm_model.md`：网络语义通信模型
- `hcomm/docs/zh/comm_op_dev_guide/prog_models_concepts/CCU_models_concepts.md`：CCU 经 UB Jetty 通信
- `hccl/docs/zh/user_guide/concepts.md`：HCCP/QP 术语
- `hccl/docs/zh/user_guide/cluster_info_config/rank_table_config_a5.md`：`protocols=UB_CTP/UB_RTP/UBOE`、EID 地址类型
- `shmem/docs/glossary.md`：UB 协议术语
- 灵衢 UB 官网（unifiedbus.com）：UB 协议公开资料
- 关联：[README.md](/通信与昇腾/参考手册/协议与传输/)（通信方式总览）、[urma.md](/通信与昇腾/参考手册/协议与传输/urma.md)、[ub_mem.md](/通信与昇腾/参考手册/协议与传输/ub_mem.md)、[../networking/ub_superpod.md](/通信与昇腾/参考手册/组网形态/ub_superpod.md)、[../networking/multi_plane.md](/通信与昇腾/参考手册/组网形态/multi_plane.md)、[../engine/ccu.md](/通信与昇腾/参考手册/通信引擎/ccu.md)

