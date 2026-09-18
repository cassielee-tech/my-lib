# 专题总结｜HCCL 源码阅读地图

> 所属专题：[HCCL 源码学习](../hccl-source.md)

七个单元走完，把所有结论压缩成一张可复用的地图，并给出继续深入的路线。

## 一张图收束整个专题

```text
使用侧          HcclCommInitRootInfo ──► HcclAllReduce ──► HcclCommDestroy
                        │                    │                  │
L1 算子 API     （hcomm，L2-comm）    include/hccl.h            │
                        │                    ▼
HCCL 算子层                          all_reduce_op.cc
（hccl 仓）                                  │
                        ▼                    ▼
                 rank_graph 建图      selector ──► 算法：Ring/RHD/NHR/分级（H01-6）
                 （H01-3）                   │
                                            ▼
                                    template/{aicpu,aiv,ccu} ──► 引擎（H01-5）
                                            │
                                            ▼
                                    executor ──► Write/Read/Reduce + Notify（H01-4）
                                            │
HCOMM 基础层    base_comm：primitives + resource      │  dlsym 解耦（H01-2）
                                            ▼
硬件            RoCE / SDMA / UB / CCU / HCCS …（H01-1/H01-3）
```

## 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| 三层软件结构 | HCCL 算子层 → HCOMM 域管理层 → HCOMM 基础通信层 | H01-2 |
| 五层 API | L1 算子 / L2 域 / L2 拓扑资源 / L3 原语 / L3 资源 | H01-2 |
| 控制面/数据面 | 建图管资源 vs Write/Read/Notify 搬数据 | H01-2、H01-4 |
| RankGraph | Node/Endpoint/Edge/Link/netLayer/Fabric/TopoInstance 建模"谁和谁怎么连" | H01-3 |
| 递进链 | Edge（谁连）→ Link（怎么建）→ Channel（怎么通信） | H01-3 |
| 两种语义 | 网络语义（RoCE/UB）vs 内存语义（HCCS/UB_MEM） | H01-4 |
| 两种同步 | ThreadNotify（实体内）/ ChannelNotify（跨实体） | H01-4 |
| 四种引擎 | AICPU_TS / CPU_TS / AIV / CCU，"Thread + 调度器 + 硬件"统一模型 | H01-5 |
| α-β-γ 模型 | 步数 × α + 数据量 × β(+γ)，算法比较的钥匙 | H01-6 |
| 分级通信 | 大数据量放机内高带宽域，机间传输压到最少 | H01-6 |
| 算子三段式 | op.cc 入口 → selector 策略 → template/executor 机制 | H01-7 |

## 继续深入的五条路线

官方 `docs/zh` 还藏着几条现成的进阶路线，按目标选择：

### 1. 写一个自定义通信算子（开发向）

- 读 `api_ref/` 中 L2-res 与 L3-prim 的接口文档；
- 跑通 `examples/04_custom_ops_p2p`、`05_custom_ops_allgather`；
- 这是 HCOMM 新开放算子编程接口（`hccl_res.h`/`hccl_rank_graph.h` + `hcomm_primitives.h`）的官方落地路径。

### 2. 性能分析与调优（调优向）

- `user_guide/perf_analysis/`：数据采集 → 算子行为分析（通信任务/同步任务/数据面任务拆解）→ 慢快卡定位 → 交换网拥塞丢包；
- `user_guide/hccl_env/`：按场景查环境变量（`HCCL_ALG`、`HCCL_BUFFSIZE`、RDMA 相关、重试相关）；
- 方法论闭环：现象 → 用代价模型提假设 → Profiling 拿证据 → 改配置/算法 → 复测。

### 3. 故障诊断（排障向）

- `user_guide/fault_diagnosis/`：按阶段切入——集群信息协商、通信域初始化、参数链路、任务执行，以及 EI0001 等典型错误码；
- 与本专题的知识直接对接：知道建图在 HCOMM、执行在引擎，报错才定位得准。

### 4. 设计与演进（架构向）

- `rfcs/`：读 1～2 篇 RFC（如 0001-add-batch-invariant-reducescatter），看一个特性从动机、方案到约束的完整论证；
- 对照 `git log` 观察架构约束（H01-2 四条）在演进中如何被维护。

### 5. 回到训练系统（应用向）

- 用 `user_guide/framework_integration.md` 补齐 torch_npu 等框架的集成细节；
- 回读博客 [从 PyTorch 调用到 HCCL](../../llm-foundations/01-landscape/03-pytorch-to-hccl.md)，此刻每个环节都应该能落到本专题的具体文件。

## 源码阅读的长期习惯

1. **每次只带一个问题进源码**：selector 怎么选引擎 / Notify 在哪配对 / 数据何时中转，一次一个问题；
2. **用官方文档当地图，用代码当地形**：文档描述的是目标结构，代码里会有 legacy 与过渡期差异；
3. **改动前先看约束**：H01-2 的四条架构约束 + `CONTRIBUTING.md` 的规范（含 pre-commit 检查）；
4. **结论写回博客**：按本专题的"读码三问"格式沉淀，复用价值最高。

## 专题自检清单

- [ ] 我能画出三层软件结构与五层 API，并说明 dlsym 解耦的意义
- [ ] 我能用七个概念解释 RankGraph，并说出 Edge → Link → Channel 的递进
- [ ] 我能对比四种引擎的 trade-off，并指出源码中对应 template 子目录
- [ ] 我能用 α-β-γ 模型解释 Ring 与 RHD 的适用场景
- [ ] 我能复述 AllReduce 从建域到销毁的完整链路，并定位每段在源码的位置
- [ ] 我知道遇到性能/故障问题时，官方文档的哪个目录是我的第一站

## 最终参考资料

- [HCCL 开源仓库](https://gitcode.com/cann/hccl) 与 [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)
- [HCCL & HCOMM 软件架构简介](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)（本专题的主轴文档）
- [HCCL 用户指南目录](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/README.md)
- [HCCL API 参考](https://gitcode.com/cann/hccl/blob/master/docs/zh/api_ref/README.md)
- [CANN 官方 HCCL 文档中心](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)

---

[返回专题导学 →](../hccl-source.md) · [返回 CANN 目录 →](../index.md)
