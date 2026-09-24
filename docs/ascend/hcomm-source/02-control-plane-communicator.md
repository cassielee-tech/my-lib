# 单元 2｜控制面走读：通信域的一生

> 所属课程：[HCOMM 源码学习](../hcomm-source.md) · 第 2 单元（共 7 单元）

::: info 本单元目标
读完后，你能够按源码模块讲出 **一次建域的完整流程**（入口 → 探测 → 建图 → 配资源 → 可用），说出 `coll_communicator_mgr` 各子目录的职责，并列出 L2 控制面接口的四个家族。
:::

## 先记住 3 个结论

1. **建域 = 探测 → 建图 → 配资源**：`api_c_adpt` 是所有 `HcclComm*`/`HcclRankGraph*` 的 C 入口；`rank_info_detect` 先探测成员，`rank_graph` 建拓扑图，`resource_mgr` 按 local/remote 配置资源——全部发生在控制面。
2. **L2 接口四个家族**：建域族（Init 四变体）、查询族（RankId/Size + RankGraph 13 个 Get）、生命周期族（Destroy/Suspend/Resume/状态回调）、内存族（SetMemoryRange/Activate）。
3. **源码里藏着三个 HCCL 课程没展开的宝贝**：`team/`（子域/team 语义）、`dfx/` 五件套（cluster_monitor/ns_recovery/profiling/rtsq_poll/taskException）、以及 `HcclCommSuspend/Resume` 的**故障恢复状态机**。

## 1. 通信域的一生（源码视角）

```text
HcclCommInitRootInfo / HcclCommInitClusterInfo（api_c_adpt 入口）
        ↓
rank_info_detect          # 探测：成员 rank 的设备/网络信息（root info 或 rank table 两种引入方式）
        ↓
rank_graph/               # 建图：phy_topo（物理拓扑）→ rank_graph_builder → topo_info
                          #   产出 [HCCL 源码 2] 的 Node/Edge/Link/层级模型
        ↓
resource_mgr (local/remote)  # 配资源：本端 Thread/内存 + 远端信息交换
        ↓
communicator/             # 域对象成型：通信上下文 + group_schedule_mgr（组调度）
        ↓
此后的每次算子调用只读图、复用资源（控制面低频，数据面高频——HCCL 源码 1 的分离约束）
```

对照 [第 4 章 04-2](../../parallel/04-distributed-basics.md#_04-2-通信域-谁和谁是一伙的)：框架侧 `init_process_group` 的第三步"建立通信域"，落到 HCOMM 源码就是这条流水线。

## 2. L2 控制面接口四家族

以 `include/hccl/hccl_comm.h` 与 `hccl_rank_graph.h` 的真实声明为准：

| 家族 | 代表接口（节选） | 用途 |
| --- | --- | --- |
| **建域** | `HcclCommInitRootInfo` / `HcclCommInitClusterInfo`（+ Config / Scalable 变体）、`HcclCreateSubCommConfig`、`HcclCommInitAll` | 全量建域 / 子域 / 多设备一次建 |
| **查询** | `HcclGetRankId` / `HcclGetRankSize` / `HcclGetCommName`；`HcclRankGraphGetLayers / GetLinks / GetRanksByLayer / GetTopoType...`（13 个） | 算子开发的第一步（单元 4 步骤 2） |
| **生命周期** | `HcclCommDestroy`、`HcclCommSuspend` / `HcclCommResume` / `HcclCommRegCommStateCallback` / `HcclCommGetStatus` | 销毁 + **故障恢复状态机** |
| **内存** | `HcclCommSetMemoryRange` / `UnsetMemoryRange` / `ActivateCommMemory` / `Deactivate` | 域级内存窗口管理 |

::: tip 拓扑查询的 13 个动词
`HcclRankGraphGet*`：Layers（层级数）、TopoType（拓扑类型）、Links（可建链信息）、EndpointDesc/Info/Num、RanksByLayer / RanksByTopoInst、RankSizeByLayer、TopoInstsByLayer、InstSizeListByLayer、TopoTypeByLayer、HeterogMode（异构模式）。**算子的算法选择（Ring？分层？）就靠它们喂数据**——这是单元 4 步骤 2/3 的输入。
:::

## 3. 源码模块速览

| 子目录 | 职责 | 读码提示 |
| --- | --- | --- |
| `api_c_adpt/`（dev/resource） | 全部 L2 C 接口入口 | 读建域流程从这里进 |
| `communicator/`（device、group_schedule_mgr） | 域对象与组调度 | 域状态机的家 |
| `rank_graph/`（phy_topo、builder、topo_info、rank_table_info） | 拓扑建模全链 | 与 HCCL 源码 2 概念一一对应 |
| `rank_info_detect/` | 成员信息探测 | root info 与 rank table 两条入口路 |
| `resource_mgr/`（local/remote） | 资源两端配置 | local 管"我的"，remote 管"交换来的" |
| `team/`（hccl、hcomm 两族） | 子域/team 语义 | `HcclTeamCreate`/`MemberToRank` 接口的家 |
| `dfx/`（cluster_monitor、ns_recovery、profiling、rtsq_poll、taskException） | 维测五件套 | 排障与监控的源码入口 |

## 4. 自测题

1. 一次建域在 HCOMM 源码里的四步流水线是什么？
2. L2 控制面接口分哪四家族？各举一例。
3. `rank_graph` 子目录的建模流水线（子模块顺序）是什么？
4. `resource_mgr` 的 local/remote 分工是什么？
5. `HcclCommSuspend/Resume + RegCommStateCallback` 组合起来支撑什么能力？

::: details 自测答案

1. api_c_adpt 入口 → rank_info_detect 探测 → rank_graph 建图 → resource_mgr 配资源，最终成型于 communicator。
2. 建域（HcclCommInitRootInfo）、查询（HcclRankGraphGetLinks）、生命周期（HcclCommSuspend/Resume）、内存（HcclCommSetMemoryRange）。
3. phy_topo（物理拓扑）→ rank_graph_builder（建图）→ topo_info（查询视图）；rank_table_info 提供 rank table 引入路径。
4. local 管本端资源（Thread/内存的分配），remote 管与对端交换的信息（远端内存/Endpoint 描述）。
5. 通信域的故障恢复：Suspend 挂起域、状态回调通知框架、Resume 恢复——生产级通信库的可靠性能力。

:::

## 本单元小结

- 建域流水线：入口 → 探测 → 建图 → 配资源 → 域成型；此后算子只读图复用；
- L2 四家族接口：建域/查询/生命周期/内存——头文件即清单；
- 三个进阶宝贝：team 子域、dfx 五件套、Suspend/Resume 状态机；
- 控制面的一切都是"名词"——真正搬数据的"动词"在下一单元。

## 参考资料

- [HCOMM 仓库：coll_communicator_mgr 源码目录](https://gitcode.com/cann/hcomm/tree/master/src/coll_communicator_mgr)
- [API 参考：通信域管理（comm_mgr_c）](https://gitcode.com/cann/hcomm/tree/master/docs/zh/api_ref/comm_mgr_c)
- [HCCL 源码 2：通信域、Rank 与 RankGraph](../hccl-source/02-comm-domain-rank-graph.md)

---

下一单元进入 **[3｜数据面走读：原语与资源](03-data-plane-primitives.md)**。

[返回课程导学 →](../hcomm-source.md)
