# 第 0 章｜认识昇腾与 CANN

> 本章目标：建立从 NPU 硬件到 CANN 软件栈的全景认识；分清 AI Core、计算单元、内存与数据搬运；理解驱动、固件、CANN 和框架插件为什么必须版本匹配。

## 本章导学

::: tip 本章只记住 3 件事
1. **NPU 是硬件，CANN 是让软件使用这块硬件的软件栈**，两者不能混为一谈。
2. AI Core 不只负责计算：数据要先从 Global Memory 搬到 Local Memory，再由 Cube/Vector 处理。
3. 学习资料可以跨版本理解概念，运行代码和阅读源码必须回到实际环境的配套版本。
:::

**学习节奏：** 本章拆为 **3 个学习单元，每个约 15 分钟**。先进入芯片内部，再回到软件栈，最后处理最容易踩坑的环境与版本问题。

- [ ] 我能解释 NPU、AI Core、Cube、Vector 和 Scalar 的关系
- [ ] 我能画出模型代码进入 CANN 并在 NPU 执行的路径
- [ ] 我能说清驱动、固件、CANN、PyTorch 与 `torch_npu` 为什么要配套

## 本章单元

- **0-0（约 15 分钟）**：[NPU、达芬奇架构与 AI Core](#_0-0-npu、达芬奇架构与-ai-core)
- **0-1（约 15 分钟）**：[CANN 软件栈与模型执行路径](#_0-1-cann-软件栈与模型执行路径)
- **0-2（约 15 分钟）**：[版本、驱动、固件与开发环境](#_0-2-版本、驱动、固件与开发环境)
- **本章总结**：[画出一段模型代码到 NPU 的完整路径](#本章总结)

## 这一章在整条路线中的位置

```text
大模型基础：模型要计算什么
          ↓
AI Infra：计算系统有哪些通用层次
          ↓
第 0 章（昇腾）：这些层次在昇腾上分别是什么
          ↓
Ascend C：怎样为 AI Core 编写算子
          ↓
HCCL：多个 NPU 怎样协作完成集合通信
```

---

[开始单元 0-0 →](#_0-0-npu、达芬奇架构与-ai-core)

## 0-0｜NPU、达芬奇架构与 AI Core

::: info 本单元目标
读完后，你能够分清 **NPU、AI Core、Cube、Vector、Scalar、Global Memory 和 Local Memory**，并解释一个算子的计算为什么必然伴随数据搬运。
:::

### 先记住 3 个结论

1. **NPU 是设备，AI Core 是 NPU 中执行主要张量计算的核心资源。** 一块 NPU 通常包含多个 AI Core，以及控制、调度、内存等其他模块。
2. **AI Core 内部不只有“算力”。** Cube/Vector 负责计算，Scalar 负责控制与指令发射，DMA/MTE 负责在不同层次的存储之间搬数据。
3. **数据靠近计算单元之后才能高效计算。** 典型过程是 `Global Memory → Local Memory → 计算 → Local Memory → Global Memory`。

### 1. 为什么需要 NPU

CPU 的每个核心功能很强，擅长复杂控制、分支和通用任务，但核心数量有限。大模型的主要工作却是大量结构规则、彼此相似的张量运算，例如：

- 矩阵乘法：Attention 和 MLP 中的大部分计算；
- 向量运算：激活函数、归一化、逐元素加法；
- 数据搬运：在设备内存与片上存储之间反复移动张量。

NPU 为这些工作准备了大量并行计算资源和专用数据通路。它追求的不是让单个复杂线程跑得最快，而是让**大量规则的数据块同时被处理**。

::: warning 不要只用“CPU 核更多”理解 NPU
NPU 不是简单堆叠很多缩小版 CPU。它的计算单元、存储层次、指令发射和数据搬运都围绕张量计算设计。
:::

### 2. NPU、达芬奇架构与 AI Core 是什么关系

可以先按三个层次理解：

| 层次 | 含义 | 当前需要关注什么 |
| --- | --- | --- |
| **NPU 设备** | 完整的 AI 加速设备 | 多个计算核心、设备内存、任务调度和数据通路 |
| **达芬奇架构** | 昇腾处理器采用的计算架构 | 怎样组织计算、存储、搬运和控制资源 |
| **AI Core** | 执行主要计算密集型算子的核心资源 | Cube、Vector、Scalar、Local Memory、DMA/MTE |

除了 AI Core，设备还可能包含控制 CPU、AI CPU、任务调度器及其他专用模块。它们分别承担设备控制、辅助算子和任务调度等工作。**具体模块数量与组织方式依产品型号而异，不要把某一型号的框图当成所有昇腾设备的固定结构。**

### 3. 打开 AI Core：计算、存储与搬运怎样配合

![AI Core 中 Vector 与 Cube 两条典型数据流，以及 Scalar 的控制作用](/images/cann/npu-ai-core-dataflow.svg)

这张图不是组件清单，而是两条**真实的数据依赖链**：

#### 3.1 Vector 路径

逐元素加法、激活函数和部分归约等向量运算，概念上的典型路径是：

```text
Global Memory → DMA/MTE 搬入 → UB → Vector 计算 → UB → DMA/MTE 搬出 → Global Memory
```

**UB（Unified Buffer）**属于片上 Local Memory，容量远小于设备内存，但离 Vector 更近、访问更快。大张量通常无法一次全部放入 UB，因此算子要把数据切成多个 Tile，分批完成“搬入—计算—搬出”。

#### 3.2 Cube 路径

矩阵乘法和卷积等计算会使用 Cube。典型数据会经过 L1、L0A、L0B、L0C 等片上存储：

```text
Global Memory → L1 → L0A / L0B → Cube → L0C → FixPipe → Global Memory 或 L1
```

- **L0A、L0B**：为矩阵乘法的左右输入准备数据；
- **L0C**：保存矩阵乘加的结果或中间累加结果；
- **FixPipe**：处理结果搬出路径上的格式转换、量化等工作，具体能力随硬件而异；
- **Cube**：执行高吞吐矩阵运算。

这说明“设备理论算力很高”并不自动等于“算子很快”。如果 Cube 经常等数据，真正的瓶颈可能是搬运、切分或流水调度。

### 4. 三种计算单元如何分工

| 单元 | 主要职责 | 典型工作 |
| --- | --- | --- |
| **Cube** | 矩阵类计算 | MatMul、卷积中的矩阵乘加 |
| **Vector** | 向量和逐元素计算 | Add、激活、归一化、部分归约 |
| **Scalar** | 标量运算、地址计算、循环与控制、指令发射 | 计算偏移、控制 Tile 循环、向其他单元发指令 |

Scalar 可以粗略看作 AI Core 内的控制者，但它**不是 Host CPU，也不是 AI CPU**：

- **Host CPU**运行应用、框架和 Host 侧调度；
- **AI CPU**是设备侧相对独立的 CPU 核，可承担不适合 AI Core 的辅助算子；
- **Scalar**位于 AI Core 的执行流水中，主要服务当前 Kernel 的地址计算、循环控制和指令发射。

### 5. 为什么图里有多条并行流水

Scalar 发出搬运、Vector/Cube 计算和同步指令后，不同执行单元可以异步工作。理想情况下，一个算子的时间线类似：

```text
时间 →
搬运单元： [搬入 Tile 0] [搬入 Tile 1] [搬入 Tile 2]
计算单元：             [计算 Tile 0] [计算 Tile 1] [计算 Tile 2]
搬出单元：                         [搬出 Tile 0] [搬出 Tile 1]
```

当搬入、计算、搬出互相重叠，计算单元就不必在每个 Tile 之间空等。这正是后续 Ascend C 中 **Queue、Pipeline、Double Buffer** 等概念的硬件根源。

### 6. 耦合架构与分离架构

不同昇腾产品的硬件组织并不完全相同：

- **耦合架构**：Cube 和 Vector 位于同一个核心中；
- **分离架构**：矩阵计算与向量计算拆成 AIC（AI Cube）和 AIV（AI Vector），各自拥有 Scalar 和相关存储、搬运资源。

当前阶段不需要背哪些产品属于哪种架构。更重要的是理解：**Ascend C 提供统一的硬件抽象，但做深度性能优化时仍要回到目标芯片的实际资源和数据通路。**

### 7. 这与集合通信有什么关系

一次集合通信不仅是“网卡把数据发出去”。从单个 NPU 观察，还会涉及：

1. 待通信张量存放在设备 Global Memory；
2. 通信任务与计算任务通过 Stream、同步信号等机制协调；
3. Reduce 类操作需要对收到的数据执行加法、最大值等本地规约计算；
4. 数据搬运、规约计算和链路传输能否形成流水，会直接影响性能。

因此，学习 AI Core 不是偏离 HCCL，而是在补齐“**通信数据到达设备后怎样被搬运和计算**”这一层。

### 8. 自测题

先用自己的话回答，再展开答案。

1. NPU 和 AI Core 是什么关系？
2. 为什么数据不能一直放在 Global Memory 中直接让 Vector/Cube 计算？
3. Cube、Vector、Scalar 各自负责什么？
4. AI CPU 与 AI Core 内的 Scalar 有什么区别？
5. 为什么一个理论算力很高的 Cube 仍可能利用率很低？

::: details 自测答案

1. NPU 是完整设备；AI Core 是设备中执行主要张量计算的核心资源。一块 NPU 还包含设备内存、控制和任务调度等其他模块。
2. 计算单元依赖片上 Local Memory 提供更高带宽和更低延迟的数据。大张量需要从 Global Memory 分块搬入 Local Memory，计算后再搬出。
3. Cube 主要执行矩阵类运算；Vector 主要执行向量和逐元素运算；Scalar 负责地址与标量计算、循环控制以及向其他单元发射指令。
4. AI CPU 是设备内相对独立、能执行较通用代码的 CPU 核；Scalar 是 AI Core 流水中的一部分，主要服务当前 Kernel 的控制与指令发射。
5. 如果数据搬运速度不足、Tile 切分不合适、流水存在空隙或任务规模太小，Cube 会等待数据或无法获得足够并行工作，因而达不到理论吞吐。

:::

### 本单元小结

- **设备层级**：NPU 包含 AI Core，但不只有 AI Core；
- **计算分工**：Cube 做矩阵，Vector 做向量，Scalar 做控制与发射；
- **数据路径**：Global Memory 中的数据要先进入 Local Memory，才能由计算单元高效处理；
- **性能直觉**：算得快的前提是搬得及时，搬运与计算最好形成流水；
- **岗位联系**：HCCL 的传输、设备侧规约和 Stream 协同最终也会落到这些硬件资源。

### 参考资料

- [CANN Learning Hub：什么是 NPU](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/02_what_is_npu.ipynb)
- [昇腾官方文档：AI Core 硬件架构](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [昇腾官方文档：计算单元](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0009.html)
- [昇腾官方文档：硬件架构抽象](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)

---

下一单元将进入 **01-2：CANN 软件栈与模型执行路径**。

## 0-1｜CANN 软件栈与模型执行路径

::: info 本单元目标
读完后，你能够说出 **CANN 软件栈的分层组成与各组件职责**（torch_npu、算子库、GE、Ascend C、毕昇编译器、Runtime、Driver、HCCL），并完整描述一行 `torch.matmul(a.npu(), b.npu())` 从 Python 到 AI Core 的执行路径。
:::

### 先记住 3 个结论

1. **NPU 只认识自己的硬件指令，不理解 Python 和 PyTorch。** CANN 是让 NPU"活起来"的软件栈，解决"框架代码怎样在 NPU 上高效运行"这个问题。
2. **CANN 采用分层解耦设计**：对上适配 AI 框架（torch_npu），对下驱动硬件（Driver），中间提供算子库、图引擎、编译器与运行时。
3. **一行代码的执行要穿过整个软件栈**：框架适配 → 算子映射 → 图优化 → 编译 → 运行时调度 → 驱动下发 → AI Core 执行。后面所有章节都在放大其中某一层。

### 1. CANN 解决什么问题

上一个单元拆开了 NPU 硬件：几十上百个 AI Core、Cube/Vector 计算单元、片上存储。但硬件只认识底层指令，你写的却是 Python：

| CANN 扮演的角色 | 解决什么问题 |
| --- | --- |
| **翻译官** | 把 PyTorch/MindSpore 的算子调用翻译成 NPU 能执行的指令 |
| **调度员** | 优化计算顺序、融合算子、分配内存、调度多核并行 |
| **工具箱** | 提供预置算子库、通信库、编译器与调试调优工具 |

从全栈视角看，昇腾平台从上到下分为：**应用层**（大模型、推荐、CV）→ **框架层**（PyTorch、MindSpore）→ **CANN** → **硬件层**（昇腾 AI 处理器）。CANN 是**承上启下的枢纽**——这正是《昇腾与 HCCL》首页那张全景图的出处：

![模型计算与分布式通信在 CANN 中的概念路径](/images/cann/cann-learning-map.svg)

### 2. CANN 分层架构总览

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

### 3. 逐层认识关键组件

#### 3.1 框架适配层：torch_npu

你用 PyTorch 写的代码默认跑在 CPU/GPU 上，`torch_npu` 把 `npu` 注册为 PyTorch 后端设备：

- `import torch_npu` 后，把 `.cuda()` 改成 `.npu()` 即可切换设备；
- 核心原理：将 PyTorch 的 `aten` 算子调用映射到 CANN 内置算子，内存和流管理通过底层运行时接口实现。

#### 3.2 算子库：预置的"标准件仓库"

| 子库 | 提供什么 | 典型算子 |
| --- | --- | --- |
| **ops-math** | 基础数学与张量操作 | Add、Mul、ReduceSum |
| **ops-nn** | 神经网络常用算子 | Conv2D、BatchNorm、LayerNorm、Softmax |
| **ops-cv** | 视觉任务核心操作 | NMS、ROIAlign、Resize |
| **ops-transformer** | Transformer 相关算子 | FlashAttention、KVCache |

当预置算子不够用时，就用 **Ascend C**（本专栏第 4、5 章的主角）开发自定义算子。

#### 3.3 图引擎 GE：全局优化的大脑

GE（Graph Engine）包含三部分：

- **图编译**：图融合/算子融合（减少访存开销）、自动融合、极致内存（复用中间 tensor 内存降低峰值显存）、自动流水；
- **图执行**：**计算图执行下沉**——把整个计算图一次性"下沉"到 NPU 侧执行，而不是逐个算子下发，减少 Host 与 Device 的交互开销；
- **metadef**：CANN 图和算子的基础数据结构，定义算子输入输出描述与属性，是图编译和图执行的"通用语言"。

::: tip 两条执行路径
- **Eager 逐算子路径**：框架每次调用一个算子，就映射一个 CANN 算子并下发一次，灵活但 Host 交互多；
- **图模式路径**：整网进入 GE 编译优化后整体下沉执行（如 `torch.compile` 使用 NPU 图后端）。

这解释了"为什么同样一行代码，图模式往往更快"——优化空间从单个算子扩大到整张图。
:::

#### 3.4 毕昇编译器与 Runtime、Driver

- **毕昇 Bisheng**：Host-Device 异构编程编译（同时编译 Tiling 函数等 Host 侧代码和 Kernel 侧代码）、微架构精准优化、完备调试信息；
- **Runtime**：NPU 的"管家"，管理 Device（选哪台设备）、Memory（内存申请/释放/拷贝）、Context（执行上下文）、Stream（任务队列）、Kernel（注册与下发）、Event/Notify（同步信号）；
- **Driver**：与硬件对话的最底层软件，负责统一内存管理、Host↔Device 与 Device↔Device 通信通路、使能硬件调度器、设备全生命周期管理（`npu-smi` 就来自这一层）。

### 4. 一行代码的完整旅程

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

### 5. 与 CUDA 生态的对照

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

完整对照表见笔记 [CANN Learning Hub](cann-learning-hub.md)。**概念相通，只是名字不同**——这也是本专栏反复用 CUDA 概念类比的原因。

### 6. 这与集合通信有什么关系

通信库 HCCL / HIXL 与算子库、GE 同处 CANN 生态之中：

1. 分布式训练中，`torch.distributed` 的 AllReduce 等调用经 `torch_npu` 进入 **HCCL**；
2. HCCL 组织数据交换，Reduce 类操作最终要靠设备侧的计算与搬运单元完成规约（呼应上一单元的 AI Core 数据流）；
3. 通信任务与计算任务通过 Runtime 的 **Stream / Event / Notify** 协同——第 1 章将展开这些概念。

至此，第 0 章（HCCL）"从 PyTorch 走向 HCCL"的每个环节都在本单元露过面了。

### 7. 自测题

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

### 本单元小结

- **定位**：CANN 是连接 AI 框架与 NPU 硬件的软件栈，承上启下；
- **分层**：框架适配（torch_npu）→ 算子库/通信库/GE → Ascend C + 毕昇 → Runtime → Driver；
- **执行路径**：一行代码经过 8 步旅程，从 Python 落到 AI Core 的 Cube/Vector；
- **两种模式**：Eager 逐算子下发 vs 图模式整网编译下沉；
- **岗位联系**：HCCL 与算子库、GE 同层共生，通信与计算在 Runtime 的 Stream/Event 上交汇。

### 参考资料

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

## 0-2｜版本、驱动、固件与开发环境

::: info 本单元目标
读完后，你能够说出**昇腾开发环境的版本链组成与配套关系**（驱动/固件 → CANN → PyTorch + torch_npu），知道安装顺序与常见坑，并掌握登录服务器后的**环境检查与版本查询方法**。
:::

### 先记住 3 个结论

1. **开发环境是一条版本链**：硬件 → 驱动 + 固件 → CANN Toolkit → Python + PyTorch + `torch_npu`。每一环都对上下环有配套要求，任何一环错配都可能导致环境不可用。
2. **安装有顺序，使用有前提**：驱动/固件最先装（通常需要 root 并重启），CANN Toolkit 随后，框架与 Python 包最后；此后**每个新 Shell 都要先 `source set_env.sh`** 加载 CANN 环境。
3. **学概念可以跨版本，跑代码必须对齐版本**。概念（分层架构、执行路径）跨版本基本稳定；命令、路径、接口以实际环境的配套版本为准。

### 1. 开发环境由哪几层组成

对照上一单元的软件栈，从下往上数：

| 层次 | 内容 | 谁来安装/管理 |
| --- | --- | --- |
| 硬件 | 昇腾 NPU（Atlas A2/A3 系列等） | 服务器自带 |
| 驱动 + 固件 | Driver + Firmware，直接对接硬件 | 管理员（root），通常随整机交付 |
| CANN Toolkit | 算子库、编译器、Runtime、工具链 | 管理员或用户安装 |
| Python 环境 | Python 3.11、虚拟环境 | 用户 |
| 框架与适配 | PyTorch + `torch_npu` | 用户（`pip install`） |

::: warning CANN 不是单一包
CANN Toolkit 按能力拆分为多个包（如 toolkit、kernels 等），安装形态（root/非 root、全量/最小）会影响目录布局与 `set_env.sh` 位置。这直接解释了速查表里为什么有两种加载路径。
:::

### 2. 为什么必须版本配套

各层之间通过接口和二进制约定协作，版本升级可能改变这些约定：

1. **驱动/固件 ↔ CANN**：每个 CANN 版本都有配套的驱动/固件版本区间，Runtime 通过驱动访问硬件，接口不匹配会直接失败；
2. **CANN ↔ torch_npu**：`torch_npu` 的每个版本绑定特定 CANN 版本区间——动态库（如 `libascendcl.so`、`libhccl.so`）来自 CANN，符号或版本对不上就会出现 `ImportError`；
3. **torch_npu ↔ PyTorch**：`torch_npu` 按具体 PyTorch 版本构建，混装不兼容版本会在导入或运行时报错。

常见错配症状对照：

| 症状 | 优先怀疑 |
| --- | --- |
| `ImportError: libascendcl.so` / `libhccl.so` | 未 `source set_env.sh`、CANN 版本与 `torch_npu` 不配套 |
| `torch.npu.is_available()` 为 `False` | 版本配套、Python 环境混用、容器设备未挂载 |
| `npu-smi` 正常但 CANN 命令找不到 | 驱动可用，但当前 Shell 未加载 CANN 环境 |

### 3. 典型安装顺序

```text
① 确认硬件与 OS（uname -m、ls /dev/davinci*）
        ↓
② 安装驱动 + 固件（root，通常需重启）—— 一般服务器出厂已装
        ↓
③ 安装 CANN Toolkit（root 或用户目录）
        ↓
④ 每次开 Shell：source <CANN 路径>/set_env.sh
        ↓
⑤ 创建 Python 虚拟环境，按配套表安装 torch 与 torch-npu
        ↓
⑥ 验证：npu-smi info → torch.npu.is_available() → 最小张量测试
```

完整命令细节不在本单元展开——**环境检查、版本查询、加载环境变量的全部命令都在 [CANN 常用命令速查](cann常用命令.md)**，那里按"环境确认 → 版本查看 → 常见问题"组织，建议搭配使用。

### 4. 登录服务器后的 30 秒检查

按顺序执行，可以定位绝大多数环境问题：

```bash
uname -m                      # CPU 架构：aarch64 / x86_64
npu-smi info                  # 驱动可用？设备健康？
echo "$ASCEND_HOME_PATH"      # CANN 环境是否已加载
python3 --version             # Python 版本
```

再用 Python 做框架层验证：

```python
import torch, torch_npu
print(torch.npu.is_available())   # True 才说明整条版本链打通
print(torch_npu.__version__)
```

**分层排障思路**：`npu-smi` 异常 → 驱动/固件层；`ASCEND_HOME_PATH` 为空 → CANN 环境层；`is_available()` 为 `False` → 框架配套层。自上而下缩小范围，不要一上来就重装。

### 5. 三种获得环境的方式

| 方式 | 适合谁 | 说明 |
| --- | --- | --- |
| 在线体验 / CANNLab 云环境 | 零基础、快速上手 | Learning Hub 教程支持浏览器在线运行；CANNLab 提供云 NPU 环境，选择 Python 3.11 内核 |
| 本地/服务器部署 | 长期学习开发 | 按 CANN 版本的安装指南与配套表操作 |
| 容器镜像 | 团队统一环境 | 注意容器需挂载 NPU 设备节点与驱动目录，否则一切正常却看不见设备 |

### 6. 常见坑清单

1. **忘记 `source`**：直接执行 `set_env.sh` 或开新终端未加载，导致 CANN 命令和动态库全部找不到；
2. **多个 Python 环境混用**：`torch` 与 `torch_npu` 不在同一虚拟环境，导入的是两份包；
3. **容器内看不到设备**：`ls /dev/davinci*` 为空通常是挂载问题，而不是 Python 问题；
4. **用错版本的资料**：照抄旧版本文档的命令参数在新环境执行失败——先 `--help` 确认当前版本支持项；
5. **擅自执行危险命令**：共享服务器上不要随意复位设备、升级固件或终止他人进程。

### 7. 自测题

先用自己的话回答，再展开答案。

1. 昇腾开发环境的版本链有哪几环？安装顺序是什么？
2. `npu-smi info` 正常，但 `ASCEND_HOME_PATH` 为空，说明什么？
3. 为什么 `torch`、`torch_npu`、CANN 三者必须版本配套？
4. `torch.npu.is_available()` 返回 `False`，你的排障顺序是什么？
5. 学概念和跑代码对版本的要求有何不同？

::: details 自测答案

1. 硬件 → 驱动 + 固件 → CANN Toolkit → Python + PyTorch + torch_npu。安装顺序：驱动/固件（root，可能需重启）→ CANN Toolkit → 虚拟环境中安装框架与适配包。
2. 驱动可用，但当前 Shell 没有加载 CANN Toolkit 环境，需要 `source` 正确的 `set_env.sh`。
3. torch_npu 按特定 PyTorch 版本构建，并依赖特定 CANN 版本提供的动态库（libascendcl.so、libhccl.so 等）；任何一环错配都会导致接口或符号不匹配，出现导入失败或运行异常。
4. 先查 CANN 环境是否加载（ASCEND_HOME_PATH），再查 torch/torch_npu 是否同一虚拟环境，再查版本配套关系，再查容器设备挂载与权限——按"环境 → 框架 → 配套 → 挂载"分层缩小范围。
5. 概念跨版本基本稳定，可用最新资料学习；命令、路径、接口、源码结构随版本变化，运行时必须以实际环境的配套版本为准。

:::

### 本单元小结

- **版本链**：驱动/固件 → CANN → Python → torch + torch_npu，环环配套；
- **使用前提**：每个 Shell 先 `source set_env.sh`，命令细节见 [速查表](cann常用命令.md)；
- **排障思路**：按硬件驱动 → CANN 环境 → 框架配套分层定位，从现象缩小范围；
- **环境获取**：在线体验 / CANNLab / 本地部署 / 容器，按阶段选择；
- **版本意识**：这一章的所有概念在后面章节反复用到，但操作命令永远以实际环境为准。

### 参考资料

- [CANN Learning Hub：什么是 CANN（环境验证小节）](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/03_what_is_cann.ipynb)
- [CANN 9.0 软件安装与环境配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/softwareinst/instg/instg_0054.html)
- [昇腾社区：CANN 下载](https://www.hiascend.com/cann/download)
- [昇腾社区：PyTorch 框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)
- [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)
- [CANN 常用命令速查](cann常用命令.md)

---

下一单元将进入 **本章总结：画出一段模型代码到 NPU 的完整路径**。

## 本章总结

三个单元走完，把硬件、软件栈和环境三张图拼成一张完整的路径图，并核对本章开头立下的目标。

### 一张图收束整章

```text
你的代码            output = torch.matmul(a.npu(), b.npu())
                          │
框架适配层          torch_npu：注册 npu 设备；aten 算子映射到 CANN 算子
                          │
图引擎 GE           图融合 / 内存复用 / 自动流水 / 执行下沉（图模式）
                          │
毕昇编译器          算子源码 ──► NPU 可执行的 Kernel 指令
                          │
Runtime            分配 Device 内存、创建 Stream、下发 Kernel
                          │
Driver             指令直达硬件，设备全生命周期管理
                          ↓  ───────── 以上：软件栈（01-2）
设备内数据流        Global Memory → DMA/MTE 搬入 → UB / L1 → L0A / L0B
                          │
计算单元            Cube 算矩阵 · Vector 算向量 · Scalar 控制与发射
                          │
写回                L0C → FixPipe → Global Memory
                          ↓  ───────── 以下：硬件数据流（01-1）
```

支撑这张图运行的环境链（01-3）：

```text
驱动 + 固件 ──► CANN Toolkit ──► Python venv ──► torch + torch_npu
（root，重启）   （set_env.sh）     （3.11）        （pip，版本配套）
```

### 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| NPU vs AI Core | 设备 vs 设备内的核心计算资源 | 01-1 |
| Cube / Vector / Scalar | 矩阵计算 / 向量计算 / 控制与指令发射 | 01-1 |
| 存储层次 | 数据要从 Global Memory 搬进 Local Memory（UB/L1/L0）才能高效计算 | 01-1 |
| 流水重叠 | 搬入、计算、搬出重叠执行——Queue、Double Buffer 的硬件根源 | 01-1 |
| CANN 定位 | 框架与硬件之间的软件栈：翻译官 + 调度员 + 工具箱 | 01-2 |
| 分层组件 | torch_npu / 算子库 / HCCL / GE / Ascend C / 毕昇 / Runtime / Driver | 01-2 |
| 两条执行路径 | Eager 逐算子下发 vs 图模式整网编译下沉 | 01-2 |
| 一行代码旅程 | 搬数据 → 算子映射 → 图优化 → 编译 → 调度 → 下发 → 执行 → 写回 | 01-2 |
| 版本链 | 驱动/固件 → CANN → torch_npu → torch，环环配套 | 01-3 |
| `source set_env.sh` | 每个 Shell 使用 CANN 的前提 | 01-3 |
| 分层排障 | `npu-smi` → `ASCEND_HOME_PATH` → `is_available()` 逐层缩小范围 | 01-3 |

### 章节自检清单

回到导学立下的三个目标，逐条核对：

- [ ] 我能解释 NPU、AI Core、Cube、Vector 和 Scalar 的关系（01-1）
- [ ] 我能画出模型代码进入 CANN 并在 NPU 执行的路径（01-2，见上图）
- [ ] 我能说清驱动、固件、CANN、PyTorch 与 `torch_npu` 为什么要配套（01-3）

额外加三条实操向的检查：

- [ ] 我能不看笔记说出 `.npu()` 之后到 Cube 开始计算之间发生了什么
- [ ] 我拿到一台新服务器，知道 30 秒环境检查要跑哪几条命令
- [ ] 我知道遇到 `is_available()` 为 `False` 时按什么顺序排查

### 下一章预告

本章多次按下不表的概念——**Stream、Event、Memory、任务下发与同步**——正是第 1 章「Runtime 与任务执行」的主角：

- 02-1：Host、Device 与异构计算；
- 02-2：设备内存、数据搬运与生命周期；
- 02-3：Stream、Event、Task 与异步执行；
- 02-4：一次算子调用怎样被下发和完成。

这些概念也是 HCCL 中"通信任务与计算任务协同"的基础。

### 最终参考资料

- [CANN Learning Hub：quick_start 公共基础](https://gitcode.com/cann/cann-learning-hub/tree/master/quick_start/cann_basics)（本章三个单元的原生课程）
- [昇腾社区：CANN 主页](https://www.hiascend.com/cann)
- [Ascend C 算子开发指南：硬件架构](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0018.html)
- [CANN 9.0 软件安装与环境配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/softwareinst/instg/instg_0054.html)
- [CANN 常用命令速查](cann常用命令.md)
