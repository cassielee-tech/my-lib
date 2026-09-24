# 单元 6｜实战：examples 与第一个自定义算子

> 所属课程：[HCOMM 源码学习](../hcomm-source.md) · 第 6 单元（共 7 单元）

::: info 本单元目标
读完后，你能够说清 **两个仓库的样例分布**（hcomm：建域三式 + ACL Graph；hccl：自定义算子），理解 ACL Graph 的"一次捕获、多次重放"，并按五步路线图跑通自己的第一个扩展通信算子。
:::

## 先记住 3 个结论

1. **样例分两仓**：HCOMM 仓 `examples/` 给**建域**三种姿势（root info / rank table / 每线程一设备）与 **ACL Graph** 演示；**算子开发样例在 HCCL 仓** `examples/04_custom_ops_p2p`（Send/Recv）与 `05_custom_ops_allgather`。
2. **建域三式**对应三种部署形态：root info（小集群免 rank table）、rank table（大规模预分配）、每线程一设备（单进程多卡多线程）。
3. **ACL Graph = 一次捕获、N 次重放**：把整段通信任务的 Host 下发开销摊销到一次 capture——它治的是 [《单卡执行系统》06 章](../../device/06-eager-graph-compilation.md#_06-1-eager-执行-一行代码的即时旅程)讲过的 Eager dispatch 瓶颈。

## 1. HCOMM 仓：建域三式

`examples/01_communicators/` 的三个样例（[官方说明](https://gitcode.com/cann/hcomm/blob/master/examples/README.md)）：

| 样例 | 初始化方式 | 适用 |
| --- | --- | --- |
| `01_one_device_per_process` | **root info**：rank 0 生成 `HcclGetRootInfo`，经 rendezvous 广播，全员 `HcclCommInitRootInfo` | 中小规模，免 rank table 文件 |
| `02_one_device_per_process_rank_table` | **rank table**：预生成的成员表文件，各进程 `HcclCommInitClusterInfo` | 大规模/静态分配集群 |
| `03_one_device_per_pthread` | **每线程一设备**：单进程多线程，各线程独立通信域 | 单机多卡推理/测试 |

三个样例练的是 [单元 2](02-control-plane-communicator.md) 建域家族的三个入口——**读样例 = 把 L2 接口表变成肌肉记忆**。

## 2. ACL Graph：通信任务的"整图重放"

`examples/02_aclgraphs/`（`allreduce_aclgraph_demo.cpp/.py`）演示用 ACL Graph 包住 AllReduce：

```text
Eager 模式：每个算子 = 一次完整 Host dispatch → Device 空闲等下发
ACL Graph：  capture 期记录整段任务 DAG（暂存不执行）
             replay 期 aclmdlRIExecuteAsync 一次性下发整段 → N 次 1-syscall 重放
```

与 [《单卡执行系统》06-3 图编译](../../device/06-eager-graph-compilation.md#_06-3-图编译-从-trace-到优化执行)同一逻辑（"一次下发、多次执行"摊销 Host 开销），但 ACL Graph 是**运行时捕获**而非编译期构图——通信任务每次形状相同、重复亿次，正是捕获重放的理想客户。官方文档还给出多 stream 拓扑与 torch_npu 对接细节（`aclgraph_introduction.md`）。

## 3. HCCL 仓：自定义算子样例

算子开发样例不在 HCOMM 仓：

| 样例 | 内容 | 对应本课程 |
| --- | --- | --- |
| [04_custom_ops_p2p](https://gitcode.com/cann/hccl/tree/master/examples/04_custom_ops_p2p) | Send/Receive（AI CPU 引擎） | [单元 4](04-aicpu-op-dev.md) 七步的最小实现 |
| 05_custom_ops_allgather | AllGather 自定义实现 | 七步 + 算法选择的真实版 |

Send/Recv 样例的接口调用链（单元 4 §5 已列）：`HcclGetRankId/Size → HcclThreadAcquire → HcclChannelAcquire → HcclChannelGetHcclBuffer → HcommLocalCopyOnThread → HcommChannelNotify...`——**建议逐行标注它落在七步的哪一步**，这是最好的自测。HCCL 侧对这两个样例的源码走读见 [HCCL 源码 9：MC2 自定义算子框架](../hccl-source/09-mc2-custom-ops.md)。

## 4. 五步起步路线图

```text
① 跑通建域    hcomm/examples/01_communicators 任选其一（建议 root info 式）
     ↓
② 编译库      按 docs/zh/build/build.md 走 build.sh（版本配套：以 release-management 为准）
     ↓
③ 读样例      hccl/examples/04_custom_ops_p2p：对照单元 4 七步逐行标注
     ↓
④ 改样例      把 Send/Recv 改成"带归约的 Send"（Read 换 ReadReduce）——最小改动练原语
     ↓
⑤ 换算法      参考 05_custom_ops_allgather，把直发改成按拓扑分层的两跳——练"查拓扑+选算法"
```

第 ⑤ 步正是岗位面试的经典题：**"给定 8 机 64 卡，你的自定义 AllGather 怎么利用 rank graph 分层？"**——答案素材在 [单元 2 的拓扑查询 13 动词](02-control-plane-communicator.md)与 [《集合通信》05 章分层算法](../../collective/05-topology-hierarchical-overlap.md#_05-2-分层算法-先内后外)。

## 5. 自测题

1. 两个仓库的样例分别覆盖什么主题？
2. 建域三式的初始化方式与适用场景分别是什么？
3. ACL Graph 解决什么瓶颈？与图编译的异同？
4. Send/Recv 样例的接口调用序列是什么？对应七步的哪几步？
5. 路线图第 ⑤ 步"换算法"要用到哪些拓扑查询接口？

::: details 自测答案

1. hcomm 仓：建域三式 + ACL Graph；hccl 仓：自定义通信算子（04_p2p / 05_allgather）。
2. root info（中小集群免表）/ rank table（大规模静态分配）/ 每线程一设备（单进程多卡）。
3. Eager 模式逐算子 dispatch 的 Host 瓶颈；同图编译一样"一次生成多次执行"，但 ACL Graph 是运行时捕获（capture/replay），不需编译期构图。
4. GetRankId/Size（查拓扑）→ ThreadAcquire/ChannelAcquire/GetHcclBuffer（建资源）→ LocalCopy/ChannelNotify 系（编排）——对应步骤②④⑥。
5. HcclRankGraphGetLayers / GetRanksByLayer / GetLinks / GetTopoType 等——按层级取成员与链路，决定分层两跳的分组。

:::

## 本单元小结

- 样例两仓分工：hcomm 管建域与图捕获，hccl 管算子开发；
- 建域三式覆盖三种部署形态，读样例即背 L2 接口；
- ACL Graph：capture/replay 摊销通信任务的 Host 开销；
- 五步路线图：跑建域 → 编译 → 读样例 → 改样例 → 换算法；
- 换算法 = 拓扑查询 + 分层思想的综合应用（面试高频）。

## 参考资料

- [HCOMM examples 目录](https://gitcode.com/cann/hcomm/tree/master/examples)（建域三式 + ACL Graph）
- [ACL Graph 介绍](https://gitcode.com/cann/hcomm/blob/master/docs/zh/aclgraph/aclgraph_introduction.md)
- [HCCL examples：04_custom_ops_p2p](https://gitcode.com/cann/hccl/tree/master/examples/04_custom_ops_p2p)
- [源码构建指南](https://gitcode.com/cann/hcomm/blob/master/docs/zh/build/build.md)

---

进入 **[课程总结：HCOMM 源码阅读地图 →](summary.md)**

[返回课程导学 →](../hcomm-source.md)
