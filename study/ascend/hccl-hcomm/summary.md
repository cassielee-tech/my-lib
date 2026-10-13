# 课程总结｜HCCL 与 HCOMM 源码阅读地图

> 所属课程：[HCCL 与 HCOMM 源码学习](../hccl-hcomm.md)

十二个单元走完（概念地图 + 逐段精读双轨），把所有结论压缩成一张可复用的地图，并给出继续深入的路线与岗位能力对照。

## 一张图收束整个课程

```text
使用侧          HcclCommInitRootInfo ──► HcclAllReduce ──► HcclCommDestroy
                          │                    │                  │
L1/L2 API       （hcomm，L2-comm，单元 2）  include/hccl.h（单元 1）│
                          │                    ▼
HCCL 算子层                          all_reduce_op.cc（单元 6）
（hccl 仓）                          三道闸门 + 四条快速路径
                          ▼                    ▼
                   rank_graph 建图      selector ──► 算法：Ring/RHD/NHR/分级（单元 5）
                   （单元 2）           引擎瀑布：CCU→AIV→AICPU（单元 4）
                                              │
                                              ▼
                                      executor：算法名=骨架×(匹配器+模板)（单元 7）
                                      template/{aicpu,aiv,ccu} ──► 引擎（单元 4）
                                              │
                                              ▼
                                      原语编排：Write/Read/Reduce + Notify（单元 7/3）
                                              │
HCOMM 基础层    base_comm：primitives + resource      │  dlsym 解耦（单元 1/8）
                engineCtx 缓存主干 / topo 流水线（单元 8）
                                              ▼
硬件            RoCE / SDMA / UB / CCU / HCCS …（单元 0/2/4）

扩展            自定义算子：七步流程（单元 9）/ MC2 两条路径（单元 10）/ 实战五步（单元 11）
                —— 从 HcclEngineCtxGet 直接进入 HCOMM，绕过算子层（单元 11 回环图）
```

## 十二个单元的精读对象全景

| 单元 | 主题 | 精读对象 |
| ---: | --- | --- |
| 0 | 双仓全景与调用链旅程 | 两仓目录地图、调用链总览 |
| 1 | 软件架构与六族 API | `include/hccl.h`（262 行） |
| 2 | 控制面：通信域的一生 | hcomm 仓建域流水线（api_c_adpt → rank_graph → resource_mgr） |
| 3 | 数据面：原语与资源 | `hcomm_primitives.h`（30 个数据面动词） |
| 4 | 通信引擎与三引擎编排 | `comm_engine_utils.h` + `template/{aicpu,aiv,ccu}` + `ccu/*.hpp` |
| 5 | 集合算法与 selector | `op_common/selector/` + `all_reduce/selector/`（728 行） |
| 6 | AllReduce 调用链走读 | `all_reduce_op.cc`（281 行） |
| 7 | executor 与 template | executor 注册表 + 13 个执行器 + Mesh RS 模板 |
| 8 | 资源地基与 dlsym | `op_common.cc` 资源链 + `topo_host.cc`（1132 行）+ `hcomm_dlsym/`（29 个文件） |
| 9 | 自定义算子七步流程 | 官方七步时序 + EngineCtx/TaskCache 复用机制 |
| 10 | MC2 框架与官方样例 | `hccl_mc2.h` + `examples/04、05`（五步法 ↔ 七步对照） |
| 11 | 实战与第一个算子 | 建域三式 + ACL Graph + 五步起步路线图 |

## 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| 双仓分工 | HCCL 算子层（base_comm 之上的决策层）+ HCOMM 底座（域/拓扑/资源/原语），dlsym 衔接、独立发版 | 单元 0 |
| 三块源码 | base_comm / coll_communicator_mgr / legacy——README 是地图，源码才是地形 | 单元 0 |
| 六族 API | L1 算子 / L2 域 / L2 拓扑资源 / L3 原语 / L3 资源 + CCU 引擎族——前缀即层级身份证 | 单元 1 |
| 控制面/数据面 | 建图管资源（低频）vs Write/Read/Notify 搬数据（高频） | 单元 1、3 |
| RankGraph | Node/Endpoint/Edge/Link/netLayer/Fabric/TopoInstance 建模"谁和谁怎么连" | 单元 2 |
| 递进链 | Edge（谁连）→ Link（怎么建）→ Channel（怎么通信） | 单元 2 |
| 建域流水线 | 入口 → 探测 → 建图 → 配资源 → 域成型；此后算子只读图复用 | 单元 2 |
| 拓扑 13 动词 | HcclRankGraphGet*：算子开发与算法选择的第一输入 | 单元 2 |
| 四原语名词 | Endpoint / Channel / CommMem / CommEngine；Channel = 两端 Endpoint + 协议 + N 个 Notify | 单元 3 |
| 两种语义 | 网络语义（RoCE/UB，走 Channel）vs 内存语义（HCCS/UB_MEM，映射直写） | 单元 3 |
| 两种同步 | ThreadNotify（实体内）/ ChannelNotify（跨实体） | 单元 3 |
| 原语四家族 | 本地 / Thread 同步 / Channel 通信 / 批量缓存；OnThread=绑定执行、Nbi=非阻塞、WithNotify=搬运+通知 | 单元 3 |
| HCCL Buffer | 200MB 锁页中转内存，解决异步悬空指针 | 单元 3 |
| 四种引擎 | AICPU_TS / CPU_TS / AIV / CCU，"Thread + 调度器 + 硬件"统一模型 | 单元 4 |
| 编排时机 | AICPU 动态（Kernel 启动后）/ AIV 静态（随 Kernel 下发）/ CCU 硬件固化（无编排） | 单元 4 |
| CCU 三优势 | 访存降一个数量级 / 归约精度与顺序确定性 / 低时延不占核 | 单元 4 |
| KernelArg/TaskArg | 编排参数走 Kernel 入参；执行参数 LoadArg 动态加载 | 单元 4 |
| 引擎瀑布 | 专用→通用的降级链（CCU_MS→CCU_SCHED→AIV→AICPU），NOT_MATCH 是正规信号 | 单元 4、5 |
| α-β-γ 模型 | 步数 × α + 数据量 × β(+γ)，算法比较的钥匙 | 单元 5 |
| 分级通信 | 大数据量放机内高带宽域，机间传输压到最少 | 单元 5 |
| 算子三段式 | op.cc 入口 → selector 策略 → template/executor 机制 | 单元 6 |
| 快速路径 | 同 shape 高频重复 → 用缓存换规划开销（CCU 快发/AIV 重放/单卡） | 单元 6 |
| 组合公式 | 算法名 = 执行骨架 ×（拓扑匹配器 + N 个算法模板） | 单元 7 |
| Mesh RS 语义 | 全互联 write + 本地归约，远端只暴露 cclBuff | 单元 7 |
| engineCtx | 按 tag+engine 寻址的缓存主干：拓扑/计划/回退记忆全挂这 | 单元 8 |
| 三层兼容 | 编译桩 → 弱符号桩 → 运行探测，版本错配不崩 | 单元 8 |
| 七步流程 | 定义接口→查拓扑→选算法→建资源→下发→编排→完成同步 | 单元 9 |
| MC2 两条路径 | Kfc 参数包声明式 vs 资源 API + 原语手写 | 单元 10 |
| 样例两仓 | hcomm：建域三式 + ACLGraph；hccl：custom_ops（五步法） | 单元 11 |

贯穿全程的四个设计模式（单元 7/10 集中现身）：**静态注册+工厂**（selector/executor 两个注册表）、**责任链降级**（引擎瀑布/回退记忆/弱符号桩/版本闸门）、**序列化缓存**（engineCtx 全家桶：topoInfo/执行计划/AIV 指令流/回退结果）、**字符串解耦**（算法名连接 selector 与 executor；tag 连接一切缓存）。

## 岗位能力对照

学完本课程，对照 HCCL 集合通信算子开发岗位的核心能力：

- [ ] **读懂内置算子**：selector/algorithm/executor 的调用链（单元 5/6/7）；
- [ ] **读懂底座**：建域流水线、拓扑建模、原语与资源（单元 2/3/4）；
- [ ] **写出扩展算子**：七步流程 + 三引擎选型 + 样例起步（单元 9/10/11）；
- [ ] **排障能力**：dfx 五件套入口（单元 2）+ 错误码文档（`docs/zh/error_codes`，EI0001-EI0020）+ [CANN 常用命令速查](../cann常用命令.md)；
- [ ] **跟进社区方向**：仓库 `docs/zh/rfcs/`（0001 topology-based-ccl-monitor、0002 host-nic-plugin）——开源社区的演进路标。

## 继续深入的六条路线

官方 `docs/zh` 还藏着几条现成的进阶路线，按目标选择：

### 1. 写一个自定义通信算子（开发向）

- 读 `api_ref/` 中 L2-res 与 L3-prim 的接口文档；
- 跑通 `examples/04_custom_ops_p2p`、`05_custom_ops_allgather`（单元 10 已走读）；
- 按 [单元 11 五步路线图](11-examples-first-op.md) 逐级进阶：跑建域 → 编译 → 读样例 → 改样例 → 换算法——纸上得来终觉浅。

### 2. 性能分析与调优（调优向）

- `user_guide/perf_analysis/`：数据采集 → 算子行为分析（通信任务/同步任务/数据面任务拆解）→ 慢快卡定位 → 交换网拥塞丢包；
- `user_guide/hccl_env/`：按场景查环境变量（`HCCL_ALG`、`HCCL_BUFFSIZE`、RDMA 相关、重试相关）；
- 方法论闭环：现象 → 用代价模型提假设 → Profiling 拿证据 → 改配置/算法 → 复测。

### 3. 故障诊断（排障向）

- `user_guide/fault_diagnosis/`：按阶段切入——集群信息协商、通信域初始化、参数链路、任务执行，以及 EI0001 等典型错误码；
- 与本课程的知识直接对接：知道建图在 HCOMM（单元 2/8）、执行在引擎（单元 4/7），报错才定位得准。

### 4. 设计与演进（架构向）

- `rfcs/`：读 1～2 篇 RFC（如 0001-add-batch-invariant-reducescatter），看一个特性从动机、方案到约束的完整论证；
- 对照 `git log` 观察架构约束（单元 1 四条）在演进中如何被维护。

### 5. 回到训练系统（应用向）

- 用 `user_guide/framework_integration.md` 补齐 torch_npu 等框架的集成细节；
- 回读博客 [从 PyTorch 调用到 HCCL](../../model/01-landscape.md#_01-3-从-pytorch-调用到-hccl)，此刻每个环节都应该能落到本课程的具体文件。

### 6. 对照 NCCL（横向对照向）

- 把 HCOMM 的 Channel/QP/Notify 与 NCCL 的 Channel/Proxy 对读——两套实现同一套问题（[NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)）；
- 注意命名差异（单元 2 的 warning）：HCCL 的 Edge ≈ NCCL 的 Link，HCCL 的 Link ≈ NCCL 的 Path。

## 源码阅读的长期习惯

1. **每次只带一个问题进源码**：selector 怎么选引擎 / Notify 在哪配对 / 数据何时中转，一次一个问题；
2. **用官方文档当地图，用代码当地形**：文档描述的是目标结构，代码里会有 legacy 与过渡期差异；
3. **改动前先看约束**：单元 1 的四条架构约束 + `CONTRIBUTING.md` 的规范（含 pre-commit 检查）；
4. **结论写回博客**：按课程的"读码三问"格式沉淀，复用价值最高。

## 课程自检清单

- [ ] 我能画出三层软件结构与六族 API，并说明 dlsym 解耦的意义（单元 1/8）
- [ ] 我能用七个概念解释 RankGraph，并复述建域四步流水线（单元 2）
- [ ] 我能把 30 个数据面动词归入四家族，并用原语写出 Send/Recv 三步握手（单元 3）
- [ ] 我能对比四种引擎的 trade-off，用"编排时机"区分三引擎编程模型（单元 4）
- [ ] 我能用 α-β-γ 模型解释 Ring 与 RHD 的适用场景，并读懂 selector 的阈值代码（单元 5）
- [ ] 我能复述 AllReduce 从建域到销毁的完整链路，并定位每段在源码的位置（单元 6）
- [ ] 我能解释算法名的组合公式，说出 Mesh RS 的真实语义（单元 7）
- [ ] 我知道 topoInfo、执行计划、回退记忆分别缓存在哪、key 是什么（单元 8）
- [ ] 我能复述自定义算子七步流程，并说出"下发在前、编排在后"的原因（单元 9）
- [ ] 我能把样例五步法对齐到七步流程，说出 MC2 两条路径的差异（单元 10）
- [ ] 我知道遇到性能/故障问题时，官方文档的哪个目录是我的第一站（总结·路线 2/3）

## 最终参考资料

- [HCCL 开源仓库](https://gitcode.com/cann/hccl) 与 [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)
- [HCCL & HCOMM 软件架构简介](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)（本课程的主轴文档）
- [HCCL 用户指南目录](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/README.md)
- [通信算子开发指南（hcomm 仓）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/README.md)
- [通信算子开发 API 参考（hcomm 仓）](https://gitcode.com/cann/hcomm/tree/master/docs/zh/api_ref/comm_opdev)
- [HCCL API 参考](https://gitcode.com/cann/hccl/blob/master/docs/zh/api_ref/README.md)
- [CANN 官方 HCCL 文档中心](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)

---

[返回课程导学 →](../hccl-hcomm.md) · [返回专栏目录 →](../index.md)
