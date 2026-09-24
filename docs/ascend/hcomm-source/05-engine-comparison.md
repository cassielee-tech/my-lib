# 单元 5｜三引擎对比：AICPU / AIV / CCU

> 所属课程：[HCOMM 源码学习](../hcomm-source.md) · 第 5 单元（共 7 单元）

::: info 本单元目标
读完后，你能够用"**编排时机**"一句话区分三类引擎的编程模型，说出 CCU 的架构组件、C++ 资源抽象与三大优势（含访存优化的量化数字），并解释 KernelArg/TaskArg 的分工。
:::

## 先记住 3 个结论

1. **三引擎 = 三种编排时机**：AICPU——编排在 Kernel 启动**后**动态生成（[单元 4](04-aicpu-op-dev.md)）；AIV——编排逻辑随 Kernel **静态下发**（先编排后下发）；CCU——**没有编排步骤**，指令序列注册后由专用硬件执行。
2. **CCU 是 950 时代的集合通信协处理器**（位于 IO Die）：4KB 分片片上缓存 + 三类寄存器（通用/同步/地址）+ 并发执行引擎 + 指令空间 + Channel 表。
3. **CCU 三优势**：片上缓存把 Reduce 访存从"2(n−1) 读 + (n−1) 写"降到"n 读 + 1 写"（**约一个数量级**）；独立片上缓存保证归约**精度与顺序确定性**；硬件描述符构造**低时延且不占计算核**。

## 1. 编排时机：一张表看清三引擎

| 步骤 | AI CPU | AIV | CCU |
| --- | --- | --- | --- |
| 定义算子接口 | ✓ | ✓ | ✓ |
| 查询拓扑信息 | ✓ | ✓ | ✓ |
| 算法选择（可选） | ✓ | ✓ | ✓ |
| 创建资源 | ✓（+序列化到 Device） | ✓ | ✓（**含 Kernel 注册**） |
| 任务编排 | **Kernel 启动后动态编排** | **Host 侧编排完成后下发** | ——（无此步骤） |
| 算子下发 | ✓ | ✓ | ✓ |
| 算法执行 | 编排即执行 | Kernel 执行 | **专用硬件执行指令** |

```text
AICPU：  准备 → 下发 Kernel → [Kernel 内动态编排+执行]
AIV：    准备 → [编排固化进 Kernel] → 下发执行
CCU：    准备+注册 → 下发 → [CCU 硬件按指令序列执行]
```

一句话：**越往下（越专用），"编排"越早固化，运行时自由度越低、确定性越高**。引擎的物理特性（谁在执行、谁在调度）见 [HCCL 源码 4：通信引擎与任务执行](../hccl-source/04-comm-engines.md)。

## 2. CCU 架构与资源抽象

![CCU 通信模型（图源：HCCL 官方文档）](/images/cann/hccl/official/ccu-communication.png)

**硬件组件**（`CCU_models_concepts`）：

| 组件 | 职责 |
| --- | --- |
| 片上缓存 | 4KB 分片为基本操作单元，支持多片片上归约 |
| 通用寄存器 | 存参数/数据，支持赋值与加法 |
| 同步寄存器 | 信号量式 Wait/Set |
| 地址寄存器 | 存通信地址（未来并入通用寄存器） |
| 并发执行引擎 | 提供并发与循环执行指令 |
| 指令空间 / Channel 表 | 指令序列存储；UB 通信上下文 |

**C++ 资源抽象**（`include/ccu/*.hpp`）——与 C 风格的 HCOMM 接口完全不同的世界：

| 抽象 | 对应硬件 |
| --- | --- |
| `ccu::Variable` | 通用寄存器 |
| `ccu::Event` | 同步寄存器 |
| `ccu::Address` / `ccu::LocalAddr` / `ccu::RemoteAddr` | 地址寄存器（地址 + token） |
| `ccu::CcuBuffer` | 4KB 片上缓存分片 |

```cpp
ccu::Variable var;                 // 单个资源：默认构造即创建
ccu::Array<ccu::Variable> vars(10); // 批量资源：保证连续性
// 注意：原生数组 Variable vars[10] 不保证连续——CCU 并发操作对资源连续性有要求
```

`ccu/*.hpp` 还提供指令级控制流：`CCU_IF / CCU_WHILE / Loop / LoopGroup / CallFunc`——**CCU 的"编程"是把控制流编译成硬件指令**。

## 3. KernelArg 与 TaskArg

CCU Kernel 从 Host 接收两种参数：

| 参数 | 内容 | 传入方式 |
| --- | --- | --- |
| **KernelArg** | 编排参数：rankId、rankSize、归约类型等 | Kernel 函数入参 |
| **TaskArg** | 执行参数：input/output 地址、token 等 | `HcommCcuKernelLaunch` 传入，Kernel 内 `ccu::LoadArg` **动态加载** |

算法执行的典型骨架：初始化资源（`ccu::GetResByChannel<ccu::Variable>` 从 Channel 取绑定的 Variable）→ LoadArg 加载 TaskArg → 按指令序列搬运/同步/归约。

## 4. CCU 的访存账（为什么快）

以 Reduce 为例（n 个成员各有一份本地数据）：

```text
不用片上缓存：每归约一个对端都要读一次本地 + 写一次本地
              = 本地侧 2(n-1) 读 + (n-1) 写
用片上缓存：  本地数据一次读入片上 → 各对端数据依次汇入片上归约 → 一次写回
              = n 读 + 1 写
```

Broadcast 同理（n−1 读 + n−1 写 → 1 读 + n−1 写）。**访存需求降约一个数量级**——这正是 [《集合通信》05 章](../../collective/05-topology-hierarchical-overlap.md#_05-3-chunk-与-channel-拆小、铺满)"数据搬运是最贵的"在硬件层的回响。

## 5. 自测题

1. 三引擎的"编排时机"分别是什么？
2. CCU 位于芯片什么位置？六类硬件组件是什么？
3. `ccu::Variable/Event/CcuBuffer` 分别对应什么硬件？
4. 为什么批量资源要用 `ccu::Array` 而不是原生数组？
5. KernelArg 与 TaskArg 的区别是什么？
6. 用片上缓存做 Reduce，访存次数怎么变？

::: details 自测答案

1. AICPU：Kernel 启动后动态编排；AIV：Host 侧编排完成后随 Kernel 静态下发；CCU：无编排步骤，指令注册后由专用硬件执行。
2. IO Die 上的专用集合通信协处理器；片上缓存、通用/同步/地址寄存器、并发执行引擎、指令空间、Channel 表。
3. Variable→通用寄存器；Event→同步寄存器；CcuBuffer→4KB 片上缓存分片。
4. CCU 并发操作（Loop/LoopGroup）要求资源地址连续；原生数组不保证连续性。
5. KernelArg 是编排参数（rankId/rankSize/归约类型），走函数入参；TaskArg 是执行参数（地址/token），经 HcommCcuKernelLaunch 传入、Kernel 内 LoadArg 动态加载。
6. 本地侧从 2(n−1) 读 + (n−1) 写降为 n 读 + 1 写——约一个数量级。

:::

## 本单元小结

- 三引擎三种编排时机：动态（AICPU）→ 静态（AIV）→ 硬件固化（CCU）；
- CCU：IO Die 协处理器，4KB 片上缓存 + 三类寄存器 + 并发指令引擎；
- C++ 资源抽象（ccu:: 族）与控制流宏 = 指令级编程；
- KernelArg（编排）/TaskArg（执行）分离，LoadArg 动态加载；
- 访存账：片上缓存把集合通信的内存流量降一个数量级。

## 参考资料

- [CCU 编程模型与概念](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/prog_models_concepts/CCU_models_concepts.md)
- [CCU 算子开发：算法执行](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/ccu_comm_op_dev/algo_exec.md)
- [HCCL 源码 4：通信引擎与任务执行](../hccl-source/04-comm-engines.md)

---

下一单元进入 **[6｜实战：examples 与第一个自定义算子](06-examples-first-op.md)**。

[返回课程导学 →](../hcomm-source.md)
