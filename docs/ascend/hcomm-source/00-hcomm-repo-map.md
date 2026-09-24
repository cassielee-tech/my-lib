# 单元 0｜HCOMM 全景与仓库地图

> 所属课程：[HCOMM 源码学习](../hcomm-source.md) · 第 0 单元（共 7 单元）

::: info 本单元目标
读完后，你能够说出 **HCOMM 仓库的三大源码块与三族对外头文件**，把任一目录/接口归位到"控制面或数据面的哪一层"，并知道从哪个文件开始读起。
:::

## 先记住 3 个结论

1. **HCOMM 是 HCCL 的底座仓**：HCCL 仓负责"算子入口 + 算法选择 + 执行编排"，HCOMM 仓负责"通信域 + 拓扑 + 资源 + 原语"——[HCCL 源码 1](../hccl-source/01-architecture-layering.md) 三层图中的下两层全部住在这里。
2. **源码三大块**：`src/base_comm`（基础通信层：协议与资源封装）、`src/coll_communicator_mgr`（域管理层：通信域/拓扑/资源管理）、`src/legacy`（历史兼容博物馆，A2&A3 与 A5 旧流程）。
3. **对外头文件三族**：`include/hccl/*`（L2：通信域与拓扑资源）、`include/hcomm_*.h`（L3：原语与基础资源）、`include/ccu/*.hpp`（CCU 引擎的 C++ 编程接口）——**接口前缀就是层级身份证**。

## 1. 为什么值得读这个仓

官方在开发指南《简介》里把动机说得很直白：万卡级集群与通算融合趋势下，**内置通信算法难以在所有场景持续最优**，传统集合通信库的黑盒设计限制了新通信原语的探索。HCOMM 的解法是把底层通信能力开放成轻量接口，让通信算子可以**独立开发、构建、部署**。

对岗位的意义：HCCL 算子开发 = "读懂内置算子（[HCCL 源码学习](../hccl-source.md)）+ 能写扩展算子（本课程）"。后者的一切素材——接口、模型、流程、样例——都从 HCOMM 仓出发。

## 2. 仓库地图

浅 clone 后的第一眼（`src/` 两层目录）：

```text
hcomm/src
├── base_comm/                    # 基础通信层（与业务无关，封装硬件与协议）
│   ├── common/                   #   公共基础功能
│   ├── config_mgr/  dfx/         #   配置与维测
│   ├── primitives/               #   通信原语（数据面动词）
│   │   ├── aicpu/                #     AICPU 侧原语与任务缓存
│   │   └── api_c_adpt/           #     C 接口适配层
│   └── resources/                #   通信资源（控制面名词）
│       ├── endpoints/  endpoint_pairs/
│       ├── reged_mems/           #     注册内存
│       ├── comm_engine_res/      #     引擎资源
│       ├── hccp/                 #     HCCP 自研协议（Huawei Collective Communication Protocol）
│       ├── ccu/  southbound_adpt/  # CCU 资源与南向适配
├── coll_communicator_mgr/        # 集合通信域管理层（控制面核心）
│   ├── api_c_adpt/               #   所有 HcclComm*/HcclRankGraph* 的 C 入口
│   ├── communicator/             #   通信域（含 device、group_schedule_mgr）
│   ├── rank_graph/               #   拓扑建模（phy_topo → builder → topo_info）
│   ├── rank_info_detect/         #   rank 信息探测
│   ├── resource_mgr/             #   资源管理（local/remote）
│   ├── team/                     #   子通信域（hccl/hcomm 两族）
│   └── config_mgr/  dfx/  common/
└── legacy/                       # 历史兼容（ascend910：A2&A3；ascend950：A5 旧流程）
```

与 README 的目录说明相比，实际源码多出三个值得注意的目录：`rank_info_detect`（建域前的 rank 探测）、`team`（子域/team 语义）、`base_comm/resources/hccp`（自研协议实现）——**README 是地图，源码才是地形**。

## 3. 对外头文件三族

`include/` 是全部对外能力的清单，按前缀分三族：

| 头文件族 | 代表文件 | 层级 | 面向 |
| --- | --- | --- | --- |
| `hccl/*.h` | `hccl_comm.h`（建域）、`hccl_rank_graph.h`（拓扑查询）、`hccl_channel.h`/`hccl_res.h`（域内资源）、`hccl_team.h` | L2 | 框架适配 / 算子开发者 |
| `hcomm_*.h` | `hcomm_primitives.h`（数据面原语）、`hcomm_res.h`（Endpoint/Mem/Thread）、`hcomm_channel.h` | L3 | 算子 / 通信库开发者 |
| `ccu/*.hpp` | `ccu_primitives.hpp`、`ccu_buffer.hpp`、`ccu_variable.hpp`、`ccu_loop.hpp`… | 引擎专用 | CCU 算子开发者（C++） |

读码起点建议：**先读头文件、再进源码**——`hccl_comm.h` 里能数出建域接口的全家族，`hcomm_primitives.h` 里躺着数据面的全部"动词"，`ccu/*.hpp` 则展示了与 C 风格完全不同的 C++ 资源抽象。

## 4. 归位练习：随便点开一个文件

拿着这张地图做归位（HCCL 源码 1 的口诀"选算法在 HCCL，建连接在 HCOMM"继续适用）：

| 看到的符号 | 归位 |
| --- | --- |
| `HcclCommInitRootInfo` | L2 控制面：`coll_communicator_mgr/api_c_adpt` 入口（单元 2 展开） |
| `HcclRankGraphGetLinks` | L2 控制面：拓扑查询（单元 1/2） |
| `HcommWriteOnThread` | L3 数据面：Channel 通信原语（单元 3/4） |
| `HcommThreadAlloc` | L3 控制面：基础资源（Thread） |
| `ccu::Variable` | CCU 数据面资源抽象（单元 5） |
| `src/legacy/**` | 博物馆：只看不改 |

## 5. 自测题

1. HCOMM 仓的三大源码块是什么？各自一句话职责？
2. 对外头文件分哪三族？前缀与层级怎么对应？
3. `base_comm` 与 `coll_communicator_mgr` 的分工边界是什么？
4. HCCP 是什么？源码在哪个目录？
5. `HcclCommInitRootInfo` 与 `HcommWriteOnThread` 分别住在哪个层级、哪个平面？

::: details 自测答案

1. `base_comm`（基础通信层：封装硬件与协议，与业务无关）；`coll_communicator_mgr`（域管理层：通信域、拓扑、资源管理）；`legacy`（历史版本兼容，不承接新特性）。
2. `hccl/*`（L2 通信域与拓扑资源）、`hcomm_*.h`（L3 原语与基础资源）、`ccu/*.hpp`（CCU 引擎 C++ 接口）。
3. base_comm 提供资源与原语的"原子能力"；coll_communicator_mgr 在其上组织"域"——建域、建图、按域分配资源。前者不知道 rank 语义，后者不知道协议细节。
4. Huawei Collective Communication Protocol，HCOMM 的自研通信协议模块；源码在 `src/base_comm/resources/hccp/`。
5. 前者是 L2 控制面（建域入口，api_c_adpt）；后者是 L3 数据面（Channel 通信原语）。

:::

## 本单元小结

- HCOMM = HCCL 底座仓：控制面（域/拓扑/资源）+ 数据面（原语）；
- 源码三大块，头文件三族——**前缀即层级身份证**；
- README 之外注意三个"隐藏"目录：rank_info_detect、team、hccp；
- 读码顺序：头文件清单 → api_c_adpt 入口 → 按单元主线深入。

## 参考资料

- [HCOMM 仓库 README（目录结构说明）](https://gitcode.com/cann/hcomm/blob/master/README.md)
- [通信算子开发指南：简介](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/intro.md)
- [HCCL 源码 1：软件架构与对外 API](../hccl-source/01-architecture-layering.md)

---

下一单元进入 **[1｜编程模型：通信、并发与拓扑](01-prog-models.md)**。

[返回课程导学 →](../hcomm-source.md)
