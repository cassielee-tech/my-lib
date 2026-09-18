# 单元 H01-2｜软件架构：HCCL 与 HCOMM 分层

> 所属专题：[HCCL 源码学习](../hccl-source.md)

::: info 本单元目标
读完后，你能够画出 HCCL & HCOMM 的**三层软件结构**与**五层对外 API**，并解释三个设计决策：控制面/数据面分离、dlsym 动态加载解耦、legacy 目录的存在意义。
:::

## 先记住 3 个结论

1. **软件分三层**：HCCL 集合通信算子（算子入口 → 算法选择 → 算法执行）→ HCOMM 集合通信域管理（通信域 + 拓扑 + 资源）→ HCOMM 基础通信（资源管理 + 通信原语执行）。
2. **控制面管资源，数据面管搬运**：拓扑查询、通道分配属控制面；Write/Read/Reduce 与 Notify 属数据面，两层接口独立演进。
3. **HCCL 与 HCOMM 通过 dlsym 动态加载衔接**，两仓可独立编译、独立发版；`legacy/` 只做历史兼容、不承接新特性。

## 1. 为什么必须分层

假设不分层：AllReduce 的代码里既要选算法、又要管拓扑发现、还要直接操作网卡队列。后果是——

- 换一颗芯片（如 950 引入 CCU 引擎），所有算子代码都要改；
- 框架想建通信域，必须依赖整个算子库；
- 自定义通信算子的开发者必须读懂全部细节才能写一个新算子。

分层之后，**变化被隔离**：算法与引擎的演进发生在 HCCL 算子层；拓扑与资源管理演进发生在 HCOMM；芯片差异被封装在基础通信层之下。

## 2. 三层结构总览

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

## 3. 五层对外 API

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

- **L2-res（rank_graph）+ L3-prim = 自定义通信算子开发接口**：你可以查拓扑、拿通道，然后用自己的编排调用搬运与同步原语——这就是 `examples/04_custom_ops_p2p`、`05_custom_ops_allgather` 的玩法；
- **L3-res + L3-prim = 通信库开发接口**：面向要自己写一个"HCCL"的场景。

::: warning 位置澄清
L2/L3 的头文件与通信域接口位于 [hcomm 仓库](https://gitcode.com/cann/hcomm)，不要在 hccl 仓里找 `hccl_comm.h`。HCCL 仓的 `include/hccl.h` 只有 L1 算子 API。
:::

## 4. 三条架构约束

`architecture-brief.md` 末尾给出四条约束，其中三条最值得在读码前记住：

### 4.1 分层依赖方向

上层依赖下层，**禁止反向依赖**：

```text
base_comm ✗→ coll_communicator_mgr   （基础层不能回头找管理层）
coll_communicator_mgr / base_comm ✗→ coll_comm_ops（管理层/基础层不能依赖算子层）
```

读码时的用途：在 `base_comm` 里看到引用上层符号，要么是理解错了，要么是历史遗留（去 `legacy/` 找线索）。

### 4.2 控制面 / 数据面分离

| 平面 | 内容 | 特点 |
| --- | --- | --- |
| **控制面** | 资源管理、拓扑查询 | 建域/建链时执行，频率低 |
| **数据面** | Write / Read / Reduce / Notify | 通信热路径，频率极高 |

分离的价值：数据面接口可以按极致性能优化（H01-4 会看到它只剩几个"动词"），而不用拖着资源管理的复杂度。

### 4.3 dlsym 动态加载解耦

HCCL 算子层**编译期不链接 HCOMM**，运行时通过 `dlsym` 动态加载其接口（对应 `src/common/hcomm_dlsym`）。好处：

- 两仓独立编译、独立版本演进，发布节奏互不阻塞；
- HCOMM 不在时 HCCL 仍可完成编译检查（运行时才报加载失败）。

第四条约束（legacy 不持续演进）在 H01-1 已说明，读码时把 `legacy/` 当"博物馆"即可。

## 5. 用这张分层图重看仓库地图

现在把 H01-1 的地图按本单元视角重新标注：

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

后续单元的任务就是逐层展开这张图：H01-3 讲 rank_graph，H01-4 讲 primitives，H01-5 讲执行侧的引擎，H01-7 把整条链串起来。

## 6. 自测题

1. 三层软件结构分别叫什么，各自职责一句话？
2. L2-res + L3-prim 组合面向谁？L3-res + L3-prim 又面向谁？
3. 为什么 `base_comm` 不允许依赖 `coll_communicator_mgr`？
4. HCCL 与 HCOMM 是怎样做到独立编译、独立发版的？
5. 想给算子加新特性，应该改 `legacy/` 吗？

::: details 自测答案

1. HCCL 集合通信算子层（算子入口→算法选择→执行）；HCOMM 集合通信域管理层（通信域、拓扑、资源管理）；HCOMM 基础通信层（基础资源与通信原语执行）。
2. L2-res + L3-prim 是自定义通信算子开发接口（查拓扑拿资源 + 用原语编排数据搬运）；L3-res + L3-prim 是通信库开发接口（面向自研集合通信库）。
3. 保持依赖方向单向。基础层反向依赖管理层会让上层演进被下层锁死，分层失去意义；这类依赖只允许出现在 legacy 兼容代码中。
4. HCCL 通过 dlsym 在运行时动态加载 HCOMM 接口（src/common/hcomm_dlsym），编译期不链接，因此两仓可独立编译与版本演进。
5. 不应该。`legacy/` 只做历史版本兼容、不承接新特性；新能力一律落在标准目录。

:::

## 本单元小结

- **三层**：算子层（hccl 仓）→ 域管理层 → 基础通信层（hcomm 仓）；
- **五层 API**：L1 算子 / L2 通信域 / L2 拓扑资源 / L3 原语 / L3 基础资源，组合出"自定义算子"与"通信库"两条开发路线；
- **约束**：依赖单向、控制面/数据面分离、dlsym 解耦、legacy 冻结；
- **读码地图**：从此看任何文件，先定位它在哪一层、属于哪个平面。

## 参考资料

- [HCCL & HCOMM 软件架构简介（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/architecture/architecture-brief.md)
- [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm)
- [HCCL 头文件与库文件说明](https://gitcode.com/cann/hccl/blob/master/docs/zh/api_ref/hccl_header_and_lib.md)

---

下一单元进入 **H01-3：通信域、Rank 与 RankGraph**。

[返回专题导学 →](../hccl-source.md)
