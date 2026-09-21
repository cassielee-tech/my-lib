# 单元 01-1｜NPU、达芬奇架构与 AI Core

> 所属章节：[第 1 章｜认识昇腾与 CANN](../01-ascend-cann.md)

::: info 本单元目标
读完后，你能够分清 **NPU、AI Core、Cube、Vector、Scalar、Global Memory 和 Local Memory**，并解释一个算子的计算为什么必然伴随数据搬运。
:::

## 先记住 3 个结论

1. **NPU 是设备，AI Core 是 NPU 中执行主要张量计算的核心资源。** 一块 NPU 通常包含多个 AI Core，以及控制、调度、内存等其他模块。
2. **AI Core 内部不只有“算力”。** Cube/Vector 负责计算，Scalar 负责控制与指令发射，DMA/MTE 负责在不同层次的存储之间搬数据。
3. **数据靠近计算单元之后才能高效计算。** 典型过程是 `Global Memory → Local Memory → 计算 → Local Memory → Global Memory`。

## 1. 为什么需要 NPU

CPU 的每个核心功能很强，擅长复杂控制、分支和通用任务，但核心数量有限。大模型的主要工作却是大量结构规则、彼此相似的张量运算，例如：

- 矩阵乘法：Attention 和 MLP 中的大部分计算；
- 向量运算：激活函数、归一化、逐元素加法；
- 数据搬运：在设备内存与片上存储之间反复移动张量。

NPU 为这些工作准备了大量并行计算资源和专用数据通路。它追求的不是让单个复杂线程跑得最快，而是让**大量规则的数据块同时被处理**。

::: warning 不要只用“CPU 核更多”理解 NPU
NPU 不是简单堆叠很多缩小版 CPU。它的计算单元、存储层次、指令发射和数据搬运都围绕张量计算设计。
:::

## 2. NPU、达芬奇架构与 AI Core 是什么关系

可以先按三个层次理解：

| 层次 | 含义 | 当前需要关注什么 |
| --- | --- | --- |
| **NPU 设备** | 完整的 AI 加速设备 | 多个计算核心、设备内存、任务调度和数据通路 |
| **达芬奇架构** | 昇腾处理器采用的计算架构 | 怎样组织计算、存储、搬运和控制资源 |
| **AI Core** | 执行主要计算密集型算子的核心资源 | Cube、Vector、Scalar、Local Memory、DMA/MTE |

除了 AI Core，设备还可能包含控制 CPU、AI CPU、任务调度器及其他专用模块。它们分别承担设备控制、辅助算子和任务调度等工作。**具体模块数量与组织方式依产品型号而异，不要把某一型号的框图当成所有昇腾设备的固定结构。**

## 3. 打开 AI Core：计算、存储与搬运怎样配合

![AI Core 中 Vector 与 Cube 两条典型数据流，以及 Scalar 的控制作用](/images/cann/npu-ai-core-dataflow.svg)

这张图不是组件清单，而是两条**真实的数据依赖链**：

### 3.1 Vector 路径

逐元素加法、激活函数和部分归约等向量运算，概念上的典型路径是：

```text
Global Memory → DMA/MTE 搬入 → UB → Vector 计算 → UB → DMA/MTE 搬出 → Global Memory
```

**UB（Unified Buffer）**属于片上 Local Memory，容量远小于设备内存，但离 Vector 更近、访问更快。大张量通常无法一次全部放入 UB，因此算子要把数据切成多个 Tile，分批完成“搬入—计算—搬出”。

### 3.2 Cube 路径

矩阵乘法和卷积等计算会使用 Cube。典型数据会经过 L1、L0A、L0B、L0C 等片上存储：

```text
Global Memory → L1 → L0A / L0B → Cube → L0C → FixPipe → Global Memory 或 L1
```

- **L0A、L0B**：为矩阵乘法的左右输入准备数据；
- **L0C**：保存矩阵乘加的结果或中间累加结果；
- **FixPipe**：处理结果搬出路径上的格式转换、量化等工作，具体能力随硬件而异；
- **Cube**：执行高吞吐矩阵运算。

这说明“设备理论算力很高”并不自动等于“算子很快”。如果 Cube 经常等数据，真正的瓶颈可能是搬运、切分或流水调度。

## 4. 三种计算单元如何分工

| 单元 | 主要职责 | 典型工作 |
| --- | --- | --- |
| **Cube** | 矩阵类计算 | MatMul、卷积中的矩阵乘加 |
| **Vector** | 向量和逐元素计算 | Add、激活、归一化、部分归约 |
| **Scalar** | 标量运算、地址计算、循环与控制、指令发射 | 计算偏移、控制 Tile 循环、向其他单元发指令 |

Scalar 可以粗略看作 AI Core 内的控制者，但它**不是 Host CPU，也不是 AI CPU**：

- **Host CPU**运行应用、框架和 Host 侧调度；
- **AI CPU**是设备侧相对独立的 CPU 核，可承担不适合 AI Core 的辅助算子；
- **Scalar**位于 AI Core 的执行流水中，主要服务当前 Kernel 的地址计算、循环控制和指令发射。

## 5. 为什么图里有多条并行流水

Scalar 发出搬运、Vector/Cube 计算和同步指令后，不同执行单元可以异步工作。理想情况下，一个算子的时间线类似：

```text
时间 →
搬运单元： [搬入 Tile 0] [搬入 Tile 1] [搬入 Tile 2]
计算单元：             [计算 Tile 0] [计算 Tile 1] [计算 Tile 2]
搬出单元：                         [搬出 Tile 0] [搬出 Tile 1]
```

当搬入、计算、搬出互相重叠，计算单元就不必在每个 Tile 之间空等。这正是后续 Ascend C 中 **Queue、Pipeline、Double Buffer** 等概念的硬件根源。

## 6. 耦合架构与分离架构

不同昇腾产品的硬件组织并不完全相同：

- **耦合架构**：Cube 和 Vector 位于同一个核心中；
- **分离架构**：矩阵计算与向量计算拆成 AIC（AI Cube）和 AIV（AI Vector），各自拥有 Scalar 和相关存储、搬运资源。

当前阶段不需要背哪些产品属于哪种架构。更重要的是理解：**Ascend C 提供统一的硬件抽象，但做深度性能优化时仍要回到目标芯片的实际资源和数据通路。**

## 7. 这与集合通信有什么关系

一次集合通信不仅是“网卡把数据发出去”。从单个 NPU 观察，还会涉及：

1. 待通信张量存放在设备 Global Memory；
2. 通信任务与计算任务通过 Stream、同步信号等机制协调；
3. Reduce 类操作需要对收到的数据执行加法、最大值等本地规约计算；
4. 数据搬运、规约计算和链路传输能否形成流水，会直接影响性能。

因此，学习 AI Core 不是偏离 HCCL，而是在补齐“**通信数据到达设备后怎样被搬运和计算**”这一层。

## 8. 自测题

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

## 本单元小结

- **设备层级**：NPU 包含 AI Core，但不只有 AI Core；
- **计算分工**：Cube 做矩阵，Vector 做向量，Scalar 做控制与发射；
- **数据路径**：Global Memory 中的数据要先进入 Local Memory，才能由计算单元高效处理；
- **性能直觉**：算得快的前提是搬得及时，搬运与计算最好形成流水；
- **岗位联系**：HCCL 的传输、设备侧规约和 Stream 协同最终也会落到这些硬件资源。

## 参考资料

- [CANN Learning Hub：什么是 NPU](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/02_what_is_npu.ipynb)
- [昇腾官方文档：AI Core 硬件架构](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [昇腾官方文档：计算单元](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0009.html)
- [昇腾官方文档：硬件架构抽象](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)

---

下一单元将进入 **01-2：CANN 软件栈与模型执行路径**。

[返回第 1 章 →](../01-ascend-cann.md)
