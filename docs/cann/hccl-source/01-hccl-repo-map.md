# 单元 H01-1｜HCCL 全景与仓库地图

> 所属专题：[HCCL 源码学习](../hccl-source.md)

::: info 本单元目标
读完后，你能够说清 **HCCL 的定位与能力边界**，并拿到一张**仓库目录地图**——知道每类源码放在哪里、examples 和测试在哪里，为后续所有单元建立空间感。
:::

## 先记住 3 个结论

1. **HCCL 不只是一个库，而是"HCCL 算子层 + HCOMM 基础通信层"的组合**：前者提供 AllReduce 等算子入口，后者管理通信域、拓扑与底层原语。
2. **仓库目录与架构分层一一对应**：`src/ops/<算子名>` 是每个算子的实现，`src/ops/op_common` 是所有算子共享的 selector / executor / template / topo 组件。
3. **开源仓库自带完整中文文档与可运行样例**：`docs/zh` 是第一手资料，`examples` 是最小可运行入口，`test/{ut,st}` 是单元与系统测试。

## 1. HCCL 是什么

HCCL（Huawei Collective Communication Library）是**基于昇腾硬件的高性能集合通信库**，为 NPU 集群提供单机多卡、多机多卡的通信方案。它在 CANN 软件栈中的位置介于 **AI 框架与硬件驱动之间**：

```text
PyTorch / MindSpore / TensorFlow
        ↓  框架适配层（torch_npu 等）
      HCCL   ←—— 本专题
        ↓
   HCOMM（通信基础库）
        ↓
  驱动 / 硬件（HCCS、RoCE、UB 等）
```

从训练视角看它的必要性：数据并行下每张卡算不同样本，反向传播后必须 AllReduce 同步梯度；模型并行、专家并行还依赖 AllGather、ReduceScatter、AlltoAll。**通信是扩展瓶颈**，所以它必须"并行、高效、有序"。

HCCL 向上支持多种 AI 框架，向下使能多款昇腾处理器之间的通信：

![HCCL 集合通信库软件架构图（图源：HCCL 官方文档）](/images/cann/hccl/official/hccl-architecture.png)

### 核心能力一览

| 维度 | 能力 |
| --- | --- |
| 集合通信原语 | AllReduce、Broadcast、AllGather、AllGatherV、ReduceScatter、ReduceScatterV、AlltoAll(V/VC)、Reduce、Scatter 等 |
| 点对点通信 | Send、Recv、BatchSendRecv |
| 通信算法 | Ring、Mesh、RHD、NHR、NB、Pairwise、Star、自研算法等 |
| 通信协议 / 链路 | HCCS、RoCE（v2）、PCIe、UB 系列 |
| 执行模式 | 单算子模式 + 图模式 |
| 扩展能力 | 基于 HCOMM 接口自定义通信算子 |

## 2. 必须先认识的组网与概念

读 HCCL 文档时反复出现这几个词，先一次说清（对应官方"相关概念"一节）。HCCL 的典型通信组网如下图：多台 AI Server 经交换设备互联，Server 内 NPU 通过 HCCS 直连：

![HCCL 典型通信组网示例：AI Server、AI 集群与超节点（图源：HCCL 官方文档）](/images/cann/hccl/official/typical-network.png)

上图涉及的基本概念：

| 概念 | 一句话解释 |
| --- | --- |
| **AI Server** | 计算节点，通常是 8 卡或 16 卡昇腾 NPU 服务器 |
| **AI 集群** | 多台 AI Server 经交换设备互联后用于分布式训练/推理的系统 |
| **超节点组网** | AI Server 间通过灵衢总线交换设备连接形成的组网 |
| **rank** | 参与通信的最小逻辑实体（通常对应一个 NPU），域内拥有唯一 ID（从 0 开始） |
| **通信域** | 一组 rank 的组合，描述通信范围；一个任务可创建多个域，一个 rank 可加入多个域 |
| **通信算子** | 在通信域内完成通信任务的算子，如 AllReduce、Broadcast |
| **通信算法** | 同一算子在不同拓扑、数据量、硬件下选择的不同实现 |

::: tip 与 PyTorch 的对应
`init_process_group` 建的进程组 ≈ 通信域；进程的 `rank` ≈ 通信域成员；`dist.all_reduce` ≈ 通信算子。区别在于 HCCL 的通信域由专门的 HCOMM 层管理，且 rank 语义直接绑定 NPU。
:::

## 3. 仓库地图

HCCL 于 2025 年 11 月在 [GitCode](https://gitcode.com/cann/hccl) 开源。拿到仓库后，先按下面的地图建立空间感：

```text
hccl
├── include/                  # 对外头文件：hccl.h（算子 API）、hccl_mc2.h
├── src/
│   ├── common/               # 通用逻辑：类型、日志、参数检查、
│   │                         #   SAL/ACL 适配、hcomm_dlsym（动态加载 HCOMM）
│   └── ops/                  # ★ 算子实现主目录
│       ├── all_reduce/       #   每个算子一个目录，内部固定拆成
│       │                     #   all_reduce_op.cc + selector/ + template/ + executor/
│       ├── all_gather/  all_gather_v/  all_to_all_v/
│       ├── reduce/  reduce_scatter/  reduce_scatter_v/
│       ├── broadcast/  scatter/  barrier/
│       ├── send/  recv/  batch_send_recv/
│       ├── interface_graph_mode/     # 图模式接口实现
│       └── op_common/        #   算子共享组件：
│           ├── selector/     #     算法选择器
│           ├── executor/     #     算法执行器（含 channel、registry）
│           ├── template/     #     算法模板（aicpu / aiv / ccu / dpu / wrapper）
│           └── topo/         #     rankGraph 拓扑信息适配
├── examples/                 # 可运行样例：点对点、集合通信、框架集成、自定义算子
├── test/
│   ├── ut/                   # 单元测试（googletest）
│   └── st/                   # 系统测试
├── experimental/             # 社区试验性代码（结构同 src，不保证兼容，不进商用）
├── docs/zh/                  # ★ 官方中文文档（本专题的参考底本）
│   ├── architecture/         #   架构简介（最重要的一篇）
│   ├── user_guide/           #   用户指南：概念、快速入门、API 用法、
│   │                         #   集合算法、性能分析、故障诊断、环境变量
│   ├── api_ref/              #   API 参考：头文件与库、各算子接口
│   ├── build/                #   源码构建指南
│   └── rfcs/                 #   RFC 设计文档
└── build.sh                  # 构建入口脚本
```

::: warning 两个易混点
1. **`examples` 里的通信域样例部分位于 [hcomm 仓库](https://gitcode.com/cann/hcomm)**——因为通信域创建接口属于 HCOMM 层，HCCL 仓库只负责算子。
2. **`experimental/` 不是弃用目录**，而是社区试验代码的隔离区；读主线请以 `src/` 为准。
:::

## 4. 怎么把源码跑起来

对照 `docs/zh/build/build.md`，构建要点如下（细节以官方文档为准）：

- **前置依赖**：python ≥ 3.7、gcc/g++ 7.3.0～13.3.x、cmake ≥ 3.16.0；跑 UT 需要 googletest（建议 release-1.14.0）；
- **CANN 软件**：Toolkit 开发包必装；驱动/固件与 ops 算子包是**运行态依赖**，只编译源码时可不装；
- **验证环境**：`npu-smi info` 确认驱动正常，再检查 toolkit / ops 包的 `version` 信息；
- **构建入口**：仓库根目录 `build.sh`，配合 `test/ut`、`test/st` 可做两级验证。

快速体验 AllReduce 则不需要自己写代码：官方"快速入门"给出基于 root 节点信息建域 → `HcclAllReduce` → 销毁域的完整样例（8 卡各初始化 0～7，AllReduce 后每个 rank 都拿到 `[0 8 16 ... 56]`）。这个样例就是 H01-7 走读调用链的"使用侧剧本"。

## 5. 官方文档怎么用

`docs/zh` 各目录的读法建议：

| 目录 | 什么时候读 |
| --- | --- |
| `architecture/architecture-brief.md` | **本专题的主轴**，H01-2～H01-5 逐节消化 |
| `user_guide/hccl_intro.md`、`concepts.md` | 入门第一小时，建立术语表 |
| `user_guide/coll_algo_intro/` | 配合 H01-6 学算法，含每个算法的耗时公式 |
| `user_guide/api_comm_impl.md` | 写集成代码 / 读 H01-7 之前 |
| `api_ref/comm_op_interface/` | 查每个算子的参数与约束 |
| `user_guide/hccl_env/` | 调优与排障时按变量名查 |
| `user_guide/perf_analysis/`、`fault_diagnosis/` | 专题总结后的进阶方向 |
| `rfcs/` | 想理解"为什么这样设计"时 |

## 6. 自测题

先用自己的话回答，再展开答案。

1. HCCL 和 HCOMM 的分工是什么？
2. `src/ops/all_reduce/` 目录下有哪四类固定内容？
3. `op_common` 里的 selector、executor、template 分别解决什么问题？
4. 只想把 HCCL 源码编译通过，必须安装驱动固件吗？
5. `experimental/` 和 `src/` 的代码是什么关系？

::: details 自测答案

1. HCCL 提供通信算子入口与算法实现（集合/点对点算子层）；HCOMM 是通信基础库，管理通信域、拓扑与底层通信原语，分为控制面与数据面。
2. `all_reduce_op.cc/.h`（算子入口）、`selector/`（算法选择）、`template/`（算法模板，按引擎分 aicpu/aiv/ccu）、`executor/`（执行器实现）。
3. selector 负责"用哪个算法/引擎"；template 提供算法在不同引擎上的模板化流程；executor 负责把选定的算法真正执行（编排数据面原语）；三者被所有算子目录复用。
4. 不必须。驱动与固件是运行态依赖；只编译源码可以不装，但运行/系统测试需要。
5. `experimental/` 是社区贡献的试验性代码，目录结构与 `src/` 保持一致，但不保证接口兼容、不会被商用版本采纳；读主线以 `src/` 为准。

:::

## 本单元小结

- **定位**：HCCL 是昇腾集群的集合通信库，上承框架、下接驱动，由"算子层 + HCOMM 基础层"组成；
- **地图**：`src/ops/<op>/{selector,template,executor}` + `op_common` 是源码阅读的主战场；`docs/zh` 与 `examples` 是官方学习材料；
- **工程**：构建依赖 CANN Toolkit，驱动固件只在运行态需要；
- **方法**：先读 `architecture-brief.md` 和 `concepts.md` 建立术语，再进代码。

## 参考资料

- [HCCL 开源仓库 README](https://gitcode.com/cann/hccl)
- [HCCL 简介（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_intro.md)
- [相关概念（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/concepts.md)
- [源码构建（官方文档）](https://gitcode.com/cann/hccl/blob/master/docs/zh/build/build.md)
- [快速入门：AllReduce 样例](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/quick_start.md)

---

下一单元进入 **H01-2：软件架构——HCCL 与 HCOMM 怎样分层**。

[返回专题导学 →](../hccl-source.md)
