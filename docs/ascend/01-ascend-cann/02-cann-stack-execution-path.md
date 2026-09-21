# 单元 01-2｜CANN 软件栈与模型执行路径

> 所属章节：[第 1 章｜认识昇腾与 CANN](../01-ascend-cann.md)

::: info 本单元目标
读完后，你能够说出 **CANN 软件栈的分层组成与各组件职责**（torch_npu、算子库、GE、Ascend C、毕昇编译器、Runtime、Driver、HCCL），并完整描述一行 `torch.matmul(a.npu(), b.npu())` 从 Python 到 AI Core 的执行路径。
:::

## 先记住 3 个结论

1. **NPU 只认识自己的硬件指令，不理解 Python 和 PyTorch。** CANN 是让 NPU"活起来"的软件栈，解决"框架代码怎样在 NPU 上高效运行"这个问题。
2. **CANN 采用分层解耦设计**：对上适配 AI 框架（torch_npu），对下驱动硬件（Driver），中间提供算子库、图引擎、编译器与运行时。
3. **一行代码的执行要穿过整个软件栈**：框架适配 → 算子映射 → 图优化 → 编译 → 运行时调度 → 驱动下发 → AI Core 执行。后面所有章节都在放大其中某一层。

## 1. CANN 解决什么问题

上一个单元拆开了 NPU 硬件：几十上百个 AI Core、Cube/Vector 计算单元、片上存储。但硬件只认识底层指令，你写的却是 Python：

| CANN 扮演的角色 | 解决什么问题 |
| --- | --- |
| **翻译官** | 把 PyTorch/MindSpore 的算子调用翻译成 NPU 能执行的指令 |
| **调度员** | 优化计算顺序、融合算子、分配内存、调度多核并行 |
| **工具箱** | 提供预置算子库、通信库、编译器与调试调优工具 |

从全栈视角看，昇腾平台从上到下分为：**应用层**（大模型、推荐、CV）→ **框架层**（PyTorch、MindSpore）→ **CANN** → **硬件层**（昇腾 AI 处理器）。CANN 是**承上启下的枢纽**——这正是《昇腾与 HCCL》首页那张全景图的出处：

![模型计算与分布式通信在 CANN 中的概念路径](/images/cann/cann-learning-map.svg)

## 2. CANN 分层架构总览

```text
PyTorch / MindSpore / TensorFlow
        ↓  框架适配层（torch_npu 等）
---------------- CANN ----------------
算子库  ops-math / ops-nn / ops-cv / ops-transformer
通信库  HCCL（集合通信）/ HIXL（单边通信）
图引擎  GE（图编译 / 图执行 / metadef）
领域加速库  FFT / BLAS 等
编程语言  Ascend C
编译器  毕昇 Bisheng
运行时  Runtime（Device/Memory/Context/Stream/...）
---------------- CANN ----------------
        ↓
驱动 Driver + 固件 Firmware
        ↓
昇腾 NPU 硬件
```

| 层次 | 组件 | 一句话职责 |
| --- | --- | --- |
| 框架适配层 | torch_npu 等 | 让 PyTorch 代码一行改动就能跑在 NPU 上 |
| 算子库 | ops-math / ops-nn / ops-cv / ops-transformer | 海量预置高性能算子，覆盖主流 AI 场景 |
| 通信库 | HCCL / HIXL | 多卡分布式训练与推理的通信保障 |
| 图引擎 | GE | 整网图优化（融合、内存、流水），最大化性能 |
| 领域加速库 | FFT / BLAS 等 | 特定领域的算子 + 算法组合 |
| 编程语言 | Ascend C | 自定义算子开发语言 |
| 编译器 | 毕昇 Bisheng | 把算子源码编译成 NPU 可执行指令 |
| 运行时 | Runtime | 管理 NPU 设备、内存、任务调度 |
| 驱动 | Driver | 与 NPU 硬件直接交互的最底层软件 |

## 3. 逐层认识关键组件

### 3.1 框架适配层：torch_npu

你用 PyTorch 写的代码默认跑在 CPU/GPU 上，`torch_npu` 把 `npu` 注册为 PyTorch 后端设备：

- `import torch_npu` 后，把 `.cuda()` 改成 `.npu()` 即可切换设备；
- 核心原理：将 PyTorch 的 `aten` 算子调用映射到 CANN 内置算子，内存和流管理通过底层运行时接口实现。

### 3.2 算子库：预置的"标准件仓库"

| 子库 | 提供什么 | 典型算子 |
| --- | --- | --- |
| **ops-math** | 基础数学与张量操作 | Add、Mul、ReduceSum |
| **ops-nn** | 神经网络常用算子 | Conv2D、BatchNorm、LayerNorm、Softmax |
| **ops-cv** | 视觉任务核心操作 | NMS、ROIAlign、Resize |
| **ops-transformer** | Transformer 相关算子 | FlashAttention、KVCache |

当预置算子不够用时，就用 **Ascend C**（本专栏第 4、5 章的主角）开发自定义算子。

### 3.3 图引擎 GE：全局优化的大脑

GE（Graph Engine）包含三部分：

- **图编译**：图融合/算子融合（减少访存开销）、自动融合、极致内存（复用中间 tensor 内存降低峰值显存）、自动流水；
- **图执行**：**计算图执行下沉**——把整个计算图一次性"下沉"到 NPU 侧执行，而不是逐个算子下发，减少 Host 与 Device 的交互开销；
- **metadef**：CANN 图和算子的基础数据结构，定义算子输入输出描述与属性，是图编译和图执行的"通用语言"。

::: tip 两条执行路径
- **Eager 逐算子路径**：框架每次调用一个算子，就映射一个 CANN 算子并下发一次，灵活但 Host 交互多；
- **图模式路径**：整网进入 GE 编译优化后整体下沉执行（如 `torch.compile` 使用 NPU 图后端）。

这解释了"为什么同样一行代码，图模式往往更快"——优化空间从单个算子扩大到整张图。
:::

### 3.4 毕昇编译器与 Runtime、Driver

- **毕昇 Bisheng**：Host-Device 异构编程编译（同时编译 Tiling 函数等 Host 侧代码和 Kernel 侧代码）、微架构精准优化、完备调试信息；
- **Runtime**：NPU 的"管家"，管理 Device（选哪台设备）、Memory（内存申请/释放/拷贝）、Context（执行上下文）、Stream（任务队列）、Kernel（注册与下发）、Event/Notify（同步信号）；
- **Driver**：与硬件对话的最底层软件，负责统一内存管理、Host↔Device 与 Device↔Device 通信通路、使能硬件调度器、设备全生命周期管理（`npu-smi` 就来自这一层）。

## 4. 一行代码的完整旅程

理解组件之后，用一个例子把整条链路串起来：

```python
output = torch.matmul(tensor_a.npu(), tensor_b.npu())
```

| 步骤 | 组件 | 做了什么 |
| --- | --- | --- |
| 1 | torch_npu | `.npu()` 把数据从 Host 内存搬到 Device 内存 |
| 2 | torch_npu | 把 `torch.matmul` 映射到 CANN 的 MatMul 算子 |
| 3 | GE 图引擎 | 算子进入计算图，做融合、内存、流水优化（图模式下） |
| 4 | 毕昇编译器 | 将算子编译成 NPU 可执行的 Kernel 指令 |
| 5 | Runtime | 分配 Device 内存、创建 Stream、下发 Kernel |
| 6 | Driver | 通过驱动把指令下发到 NPU 硬件 |
| 7 | AI Core | Cube 单元执行矩阵乘法，结果写回 Device 内存 |
| 8 | —— | `output` 留在 Device 上，后续操作继续在 NPU 执行 |

把这张表与上一单元的 **AI Core 数据流**（Global Memory → Local Memory → 计算 → 写回）接起来，就是"从 Python 代码到晶体管动作"的完整链条：

```text
Python 代码 → torch_npu → 算子映射 → GE 优化 → 毕昇编译 → Runtime 调度 → Driver 下发
                                                                          ↓
        Global Memory → DMA/MTE 搬入 → UB/L1 → Cube/Vector 计算 → 写回 Global Memory
```

## 5. 与 CUDA 生态的对照

如果你用过 NVIDIA GPU + CUDA，概念几乎一一对应：

| 维度 | CANN（昇腾） | CUDA（NVIDIA） |
| --- | --- | --- |
| 框架适配 | torch_npu | torch.cuda |
| 算子库 | ops-math / ops-nn 等 | cuDNN / cuBLAS |
| 通信库 | HCCL + HIXL | NCCL |
| 图引擎 | GE | TensorRT |
| 编程语言 | Ascend C | CUDA C++ |
| 编译器 | 毕昇 Bisheng | NVCC |
| Profiling | msprof / MindStudio | Nsight |

完整对照表见笔记 [CANN Learning Hub](../cann-learning-hub.md)。**概念相通，只是名字不同**——这也是本专栏反复用 CUDA 概念类比的原因。

## 6. 这与集合通信有什么关系

通信库 HCCL / HIXL 与算子库、GE 同处 CANN 生态之中：

1. 分布式训练中，`torch.distributed` 的 AllReduce 等调用经 `torch_npu` 进入 **HCCL**；
2. HCCL 组织数据交换，Reduce 类操作最终要靠设备侧的计算与搬运单元完成规约（呼应上一单元的 AI Core 数据流）；
3. 通信任务与计算任务通过 Runtime 的 **Stream / Event / Notify** 协同——第 2 章将展开这些概念。

至此，第 3 章"从 PyTorch 走向 HCCL"的每个环节都在本单元露过面了。

## 7. 自测题

先用自己的话回答，再展开答案。

1. CANN 在昇腾全栈中处于什么位置？它解决的核心问题是什么？
2. `torch_npu` 做了哪两件事，让 PyTorch 代码跑在 NPU 上？
3. GE 的"计算图执行下沉"解决什么问题？
4. 毕昇编译器、Runtime、Driver 三者的职责边界是什么？
5. ops-transformer 子库提供哪些与大模型推理直接相关的算子？

::: details 自测答案

1. CANN 位于框架层与硬件层之间，是昇腾 AI 处理器的核心软件栈；解决上层 AI 框架代码在 NPU 上高效运行的问题（翻译、调度、工具箱）。
2. 一是注册 `npu` 为 PyTorch 后端设备（`.npu()` 搬运数据到 Device）；二是把 PyTorch 的 `aten` 算子调用映射到 CANN 内置算子，并通过底层接口管理内存与流。
3. 把整个计算图一次性下沉到 NPU 侧执行，而不是逐个算子下发，减少 Host 与 Device 之间的交互开销。
4. 毕昇负责把算子源码编译成 NPU 指令；Runtime 负责设备、内存、Context、Stream、任务下发与同步；Driver 负责与硬件直接交互（内存管理、通信通路、硬件调度使能、设备管理）。
5. FlashAttention、KVCache 等 Transformer 相关算子。

:::

## 本单元小结

- **定位**：CANN 是连接 AI 框架与 NPU 硬件的软件栈，承上启下；
- **分层**：框架适配（torch_npu）→ 算子库/通信库/GE → Ascend C + 毕昇 → Runtime → Driver；
- **执行路径**：一行代码经过 8 步旅程，从 Python 落到 AI Core 的 Cube/Vector；
- **两种模式**：Eager 逐算子下发 vs 图模式整网编译下沉；
- **岗位联系**：HCCL 与算子库、GE 同层共生，通信与计算在 Runtime 的 Stream/Event 上交汇。

## 参考资料

- [CANN Learning Hub：什么是 CANN](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/03_what_is_cann.ipynb)
- [昇腾社区：CANN 主页](https://www.hiascend.com/cann)
- [昇腾社区：CANN 算子库 AOL](https://www.hiascend.com/cann/aol)
- [昇腾社区：GE 图引擎](https://www.hiascend.com/cann/graph-engine)
- [昇腾社区：Ascend C](https://www.hiascend.com/cann/ascend-c)
- [昇腾社区：毕昇编译器](https://www.hiascend.com/cann/bisheng)
- [昇腾社区：CANN Runtime](https://www.hiascend.com/cann/runtime)
- [昇腾社区：CANN Driver](https://www.hiascend.com/cann/driver)

---

下一单元将进入 **01-3：版本、驱动、固件与开发环境**。

[返回第 1 章 →](../01-ascend-cann.md)
