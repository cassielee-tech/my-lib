# 单元 H01-7｜源码走读：一次 AllReduce 的调用链

> 所属专题：[HCCL 源码学习](../hccl-source.md)

::: info 本单元目标
以 `HcclAllReduce` 为线索，从**使用侧剧本**（建域 → 调用 → 销毁）走到**源码路径**（`all_reduce_op.cc` → selector → template → executor），把前六个单元的概念全部钉到具体文件与调用顺序上。
:::

## 先记住 3 个结论

1. **使用侧永远是三段式**：创建通信域（HCOMM 接口）→ 在 Stream 上调用算子（`hccl.h`）→ 销毁域释放资源。
2. **算子内部是三段式**：`<op>_op.cc` 入口 → selector 选算法/引擎 → template + executor 执行——selector 与 executor 的分离正是"策略与机制分离"。
3. **读码从 `src/ops/all_reduce/` 开始性价比最高**：它是结构最完整的算子目录，且 `op_common/` 的共享组件都在场。

## 1. 使用侧剧本（回顾快速入门样例）

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

### 官方约束（集成时的高频坑）

- 多个通信域下的所有通信算子在每个 Device 上**必须串行下发**，不允许乱序、多线程并发下发，不支持线程重入；
- 同一 Device 上，同一通信域内所有通信算子的下发线程需使用**相同的 Context**；
- 同一通信域内**不支持图模式与单算子模式混用**；同一 NPU 上需串行创建多个通信域。

## 2. 源码路径：从入口到执行

现在进入 `hccl` 仓。一次 `HcclAllReduce` 的概念路径：

```text
include/hccl.h                       ← L1 API 声明
   │
src/ops/all_reduce/all_reduce_op.cc  ← 算子入口：参数检查、预处理、
   │                                   组装 op 上下文
   ▼
src/ops/all_reduce/selector/
   all_reduce_auto_selector.cc       ← 依据数据量/拓扑/规模/芯片
   │                                   选择算法（Ring/NHR/…）与引擎
   ▼                                   （aicpu/aiv/ccu，H01-5）
src/ops/all_reduce/template/
   ├── aicpu/  ├── aiv/  └── ccu/    ← 算法在选定引擎上的模板流程
   │
src/ops/all_reduce/executor/         ← 真正执行：编排数据面原语
   ins_v2_all_reduce_sequence_executor.cc            （串行）
   ins_v2_all_reduce_parallel_executor.cc            （并行）
   ins_v2_all_reduce_omnipipe_executor.cc            （全流水）
   ins_v2_all_reduce_concurrent_executor.cc          （并发）
   ins_v2_all_reduce_two_shot_sole_executor.cc       （两段式）
   …
   │  ── dlsym 加载（H01-2）──
   ▼
HCOMM L3-prim：Write / Read / Reduce + Notify        （H01-4）
   │
通信硬件：RoCE / SDMA / UB / CCU …                   （H01-5）
```

注意 executor 目录的命名已经暴露信息：`sequence`（串行逐步）、`parallel`（并行）、`omnipipe`（全流水）、`two_shot`（两段式，对应分级通信）、`order_preserved`（保序）——**执行器的差异就是算法 + 引擎 + 优化策略的组合**。

## 3. 共享组件：`op_common` 四件套

单看 all_reduce 会以为组件是私有的，其实它们住在 `src/ops/op_common/`：

| 目录 | 职责 | 与前面单元的连接 |
| --- | --- | --- |
| `selector/` | 算法/引擎选择的通用框架 | H01-6 的代价模型是选择依据 |
| `executor/`（含 `channel/`、`registry/`） | 执行器框架与通道编排 | H01-4 的原语在这里被调用 |
| `template/`（aicpu/aiv/ccu/dpu/wrapper/registry） | 算法模板框架 | H01-5 的引擎维度 |
| `topo/` | rankGraph 拓扑信息的获取与转换 | H01-3 的 RankGraph 在算子侧的适配 |

读码顺序建议：**先读 `all_reduce_op.cc` 看它如何调用 selector，再进 selector 看判定条件，最后挑一个最简单的 executor（如 sequence）读它的原语编排**。复杂的 executor（omnipipe 等）留到需要时再啃。

## 4. 带着读码三问走一遍

用专题导学的方法论检查这条链路：

**① 入口在哪？**
`HcclAllReduce` 的声明在 `include/hccl.h`；实现在 `src/ops/all_reduce/all_reduce_op.cc`。图模式入口则在 `src/ops/interface_graph_mode/`（区分单算子/图模式两条路）。

**② 数据在哪？**
用户 buffer（CommMem，H01-4）→ executor 编排的通道搬运 → 链路 → 远端 CommMem。中间可能经过 HCCL buffer 中转（性能分析文档的 usermem ↔ hcclbuffer 通路）。追踪点：executor 里对 Read/Write/LocalReduce 原语的调用序列。

**③ 谁在等待？**
三层同步（H01-4/H01-5）：Thread 内算子顺序依赖；引擎内 Thread 间 ThreadNotify；跨 rank 的 ChannelNotify。宿主侧则是 `aclrtSynchronizeStream` 等 Stream 完成。追踪点：executor 代码中 Notify 的 Record/Wait 配对。

## 5. 图模式与单算子模式

- **单算子模式**：每次 `HcclAllReduce` 独立下发（本单元主路径），适合动态执行；
- **图模式**：通信算子作为节点编入计算图（GE），由 `src/ops/interface_graph_mode/` 承接，编译期确定执行序，减少 Host 下发开销；
- 框架集成（`user_guide/framework_integration.md`）说明 torch_npu / MindSpore 等如何在这两种模式间选择。

## 6. 自测题

1. 使用侧三段式是什么？为什么 `HcclAllReduce` 返回后还需要同步？
2. 建通信域的三种方式分别适用什么场景？
3. `all_reduce_op.cc` 之后依次进入哪两个目录？各回答什么问题？
4. `op_common/topo/` 存在的意义是什么？
5. executor 命名里的 `two_shot`、`order_preserved` 大概率对应什么策略？

::: details 自测答案

1. 建域 → Stream 上调用算子 → 销毁域。因为算子是异步语义，调用只是把任务挂到 Stream，必须用 `aclrtSynchronizeStream` 等待完成，这也是通信与计算可重叠的机制基础。
2. rank table 文件（`HcclCommInitClusterInfo`，有完整集群信息）；root 节点信息（`HcclCommInitRootInfo`，无 rank table 时两阶段协商）；`HcclCommInitAll`（单机批量建域）；另有 `HcclCreateSubCommConfig` 从已有域切子域。
3. 先进 `selector/`（选哪个算法、哪个引擎——策略），再进 `template/` + `executor/`（在选定引擎上执行算法——机制）。
4. 把 HCOMM 的 rankGraph 拓扑描述转换成算子层可消费的形态，是 H01-3 拓扑模型在算子侧的适配层，被所有算子共享。
5. `two_shot` 对应分级/两段式通信（如机内 RS + 机间 AR）；`order_preserved` 对应保序执行（对数据顺序敏感的场景）。（以实际代码注释为准，这里练的是"从命名反推策略"的习惯。）

:::

## 本单元小结

- **使用侧**：建域（HCOMM）→ 算子（HCCL L1）→ 销毁；异步 Stream 语义是重叠的根源；
- **源码侧**：`all_reduce_op.cc` → `selector/`（策略）→ `template/{aicpu,aiv,ccu}` + `executor/`（机制）；
- **共享组件**：selector / executor / template / topo 四件套在 `op_common`，被所有算子复用；
- **读码三问**贯穿：入口（op.cc）、数据（executor 原语序列）、等待（Notify 配对 + Stream 同步）；
- **两种执行模式**：单算子 vs 图模式（`interface_graph_mode`）。

## 参考资料

- [使用通信库 API 实现通信功能（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/api_comm_impl.md)
- [HcclAllReduce 接口参考](https://gitcode.com/cann/hccl/blob/master/docs/zh/api_ref/comm_op_interface/HcclAllReduce.md)
- [通信域管理接口（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/README.md)
- [快速入门样例](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/quick_start.md)
- [examples：集合通信样例](https://gitcode.com/cann/hccl/tree/master/examples)

---

进入最后一篇：[专题总结——HCCL 源码阅读地图 →](summary.md)

[返回专题导学 →](../hccl-source.md)
