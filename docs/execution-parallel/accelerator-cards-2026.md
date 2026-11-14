---
title: 2026 主流 AI 加速卡全景
description: 对比 NVIDIA、昇腾、AMD 与 Intel 数据中心 AI 加速器的计算精度、显存、带宽、互联和软件生态。
---

# 2026 主流 AI 加速卡全景：NVIDIA、昇腾、AMD 与 Intel

> 更新于 2026 年 9 月。本文聚焦大模型训练与推理使用的数据中心加速器。产品迭代和不同厂商服务器配置变化很快，采购或部署时应以对应型号最新技术规格为准。

## 本章导学

::: tip 本文只记住 3 件事
1. 峰值算力必须统一精度、Dense/Sparse 和单卡/整机口径后才能比较。
2. 显存决定能否放下，HBM 与互联带宽决定能否及时搬完，软件生态决定纸面能力能否兑现。
3. 选卡必须用自己的模型、Shape、精度和拓扑做端到端 Benchmark。
:::

**建议节奏：** 核心路径 10 分钟；完整查阅 25～35 分钟。它更适合作为速查表，不必从头背到尾。

**15 分钟主线：** 第 1～4 节、两张总览图，以及与你实际设备厂商对应的一节。

- [ ] 我能识别一个数字是单卡还是整机
- [ ] 我能检查 Dense/Sparse 和精度口径
- [ ] 我能列出选卡时除 TFLOPS 外的三个指标

## 1. 先说结论：不能只看“多少 TFLOPS”

评价一张 AI 加速卡，至少要同时看五个维度：

1. **计算能力**：BF16/FP16、FP8、INT8、FP4 等不同精度的矩阵算力；
2. **显存容量**：决定单卡能放多少权重、激活或 KV Cache；
3. **显存带宽**：决定计算单元能多快拿到权重和数据；
4. **卡间互联**：决定多卡训练与推理中的集合通信效率；
5. **软件生态**：框架、编译器、算子库、通信库和推理引擎是否成熟。

```text
峰值算力高
   │
   ├─ 数据喂不满 ──▶ 受 HBM 带宽限制
   ├─ 模型放不下 ──▶ 需要更多卡，引入通信
   ├─ 卡间传得慢 ──▶ 多卡扩展效率下降
   └─ Kernel 不成熟 ─▶ 理论峰值无法兑现
```

所以“哪张卡最快”不是一个完整问题。更准确的问题应该是：

> 对某个模型、某种精度、某个 Batch 和序列长度，在指定软件栈与集群拓扑下，哪种设备的端到端性能和成本更合适？

## 2. TFLOPS、TOPS 和 PFLOPS 分别是什么

- **FLOPS**：每秒浮点运算次数；
- **OPS**：每秒操作次数，常用于 INT8 等整数运算；
- **TFLOPS**：每秒 $10^{12}$ 次浮点运算；
- **PFLOPS**：每秒 $10^{15}$ 次浮点运算，$1\ PFLOPS=1000\ TFLOPS$；
- **TOPS**：每秒 $10^{12}$ 次整数或一般操作。

同一张卡可能同时标注多个完全不同的峰值：

| 口径 | 常见用途 | 特点 |
| --- | --- | --- |
| FP32 Vector | 科学计算、通用浮点 | 精度高，数字通常较小 |
| TF32 Tensor/Matrix | 深度学习训练 | 保留较大数值范围，利用矩阵单元 |
| BF16/FP16 Matrix | 大模型训练和推理 | 当前训练中最常见的比较口径之一 |
| FP8 Matrix | 低精度训练和推理 | 吞吐更高，但依赖硬件与软件支持 |
| INT8 | 量化推理 | 厂商常使用 TOPS 表示 |
| FP4/MXFP4 | 新一代低精度推理 | 峰值很高，但不能与 BF16 直接比较 |

## 3. Dense 与 Sparse：最容易看错的地方

结构化稀疏允许硬件跳过一部分满足特定模式的零值，理论峰值经常是 Dense 的 2 倍。厂商规格表中的星号、`with sparsity` 或 `sparse` 非常重要。

例如 NVIDIA H100 SXM 官方页面列出的 FP16 Tensor Core 峰值是 1,979 TFLOPS，但该数字带有稀疏标记；对应 Dense 峰值约为 989 TFLOPS。[NVIDIA H100 官方规格](https://www.nvidia.com/en-us/data-center/h100/)

::: warning 比较规则
必须保持“同精度、同 Dense/Sparse 口径、同产品粒度”。不能拿一张卡的 Dense BF16，与另一台 8 卡服务器的 Sparse FP8 比较。
:::

## 4. 卡、模组、服务器和超节点不是一回事

![单卡、模组、服务器与超节点的层级关系](/images/ai-infra/accelerator-product-levels.svg)

| 层级 | 示例 | 数字代表什么 |
| --- | --- | --- |
| PCIe 卡 / OAM / SXM 模组 | H100 SXM、MI325X OAM、Atlas 300I A2 | 单个加速设备或模组 |
| Baseboard | HGX B200、AMD MI325X Platform | 通常集成 8 个加速器和高速互联 |
| 服务器 | DGX B200、Atlas 800T A3 | 加速器、CPU、内存、网络和电源组成的整机 |
| SuperPOD / 超节点 | GB200 NVL72、Atlas 900 A3 | 多服务器或大量加速器构成的 Scale-up/Scale-out 系统 |

“GB200”还可能指 Grace CPU 与 Blackwell GPU 组成的 Superchip，而 B200 指 GPU。看到型号时，先确认讨论的是芯片、模组还是系统。

## 5. NVIDIA：当前最完整的数据中心 AI 生态

NVIDIA 主流数据中心产品大致经历：

```text
Ampere              Hopper                    Blackwell / Blackwell Ultra
A100        →        H100 / H200       →       B200 / B300
```

### 5.1 主流型号对比

下表优先采用 SXM 数据中心形态；计算峰值统一尽量使用 **Dense Tensor/Matrix** 口径。不同形态、功耗配置和官方修订可能改变数值。

| 型号 | 架构 | 显存 | HBM 带宽 | BF16/FP16 Dense | FP8 Dense | 定位 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A100 80GB | Ampere | 80 GB HBM2e | 约 2.0 TB/s | 约 312 TFLOPS | 不支持原生 FP8 | 大量存量训练集群 |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 约 989 TFLOPS | 约 1.98 PFLOPS | 训练与推理主力 |
| H200 SXM | Hopper | 141 GB HBM3e | 4.8 TB/s | 约 989 TFLOPS | 约 1.98 PFLOPS | 更大容量、长上下文和推理 |
| B200 SXM | Blackwell | 180 GB HBM3e | 最高约 8 TB/s | 约 2.25 PFLOPS | 约 4.5 PFLOPS | 新一代训练和低精度推理 |
| B300 SXM | Blackwell Ultra | 288 GB HBM3e | 最高约 8 TB/s | 依具体官方口径 | 约 4.5 PFLOPS | 大模型推理与更大显存需求 |

H100/H200 的主要差别不是 Tensor Core 峰值，而是 H200 将单卡显存提升到 141 GB、带宽提升到 4.8 TB/s。B200 提升到 180 GB 和最高约 8 TB/s；B300 进一步提升到 288 GB。[NVIDIA HGX H200/B200/B300 规格](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)

### 5.2 为什么 H200 的算力没大涨，却仍很有价值

Decode 阶段经常需要反复读取权重和 KV Cache，容易受显存容量和带宽限制。H200 相比 H100：

- 能在单卡放入更大的模型分片或更多 KV Cache；
- 可以减少 Tensor Parallel 卡数；
- 更高 HBM 带宽有利于带宽受限的推理；
- 卡数减少后，某些通信和调度开销也随之下降。

这说明大模型推理中，**容量和带宽本身就是核心性能指标**。

### 5.3 NVIDIA 多卡互联

- H100/H200 SXM：单 GPU NVLink 双向带宽最高约 900 GB/s；
- B200：单 GPU NVLink 带宽最高约 1.8 TB/s；
- NVSwitch：让 8 卡或更大系统形成高带宽 Scale-up 域；
- ConnectX / InfiniBand / Spectrum-X：负责跨服务器 Scale-out 网络。

HGX B200 8 GPU 系统拥有 1.44 TB HBM3e 和最高 64 TB/s 聚合 HBM 带宽。[NVIDIA HGX 参考架构](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)

### 5.4 NVIDIA 软件栈

```text
PyTorch / JAX
  → CUDA
  → cuBLAS / cuDNN / CUTLASS / Transformer Engine
  → TensorRT-LLM / Triton
  → NCCL
  → GPU、NVLink、InfiniBand / Ethernet
```

NVIDIA 的优势不仅是单卡峰值，更在于 CUDA、算子库、NCCL、Profiler 和推理框架组成的成熟生态。

## 6. 昇腾：不要只搜索“910B/910C”

昇腾在公开产品和软件文档中，更多使用 **Atlas 产品系列**描述可采购和可部署形态：

- Atlas A2 训练系列；
- Atlas A2 推理系列；
- Atlas A3 训练系列；
- Atlas A3 推理系列；
- Atlas 200I/500 A2 边缘推理系列。

“910B”“910C”等名称在行业讨论中很常见，但实际开发、兼容性和部署时，应以 `Atlas 800T A2`、`Atlas 300I A2`、`Atlas 800T A3` 等官方产品名以及对应 CANN 版本为准。[昇腾官方产品形态总览](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)

### 6.1 当前主要 Atlas 产品

| 产品 | 形态 | 公开算力 | 内存/互联 | 主要场景 |
| --- | --- | ---: | --- | --- |
| Atlas 300I A2 | PCIe 推理卡 | 最高 280 TFLOPS FP16；560 TOPS INT8 | 32 GB / 0.8 TB/s，或 64 GB / 1.6 TB/s HBM | 大模型推理、内容生成与审核 |
| Atlas 800I A2 | 8 NPU 推理服务器 | 依具体服务器配置 | 常见 32/64 GB HBM 每 NPU | 中心侧大模型推理 |
| Atlas 800T A2 | 8 NPU 训练服务器 | 官方版本资料可见 376T 产品形态 | 依具体配置 | 训练、微调和推理 |
| Atlas 800T A3 | 8 NPU 超节点服务器 | 单机最高 6.0 PFLOPS FP16 | 最大 384 NPU 高速互联；官方标注 784 GB/s 互联带宽 | 大模型预训练、后训练 |
| Atlas 900 A3 | SuperPOD / 超节点 | 以系统配置为准 | 大规模统一互联域 | 超大规模训练和推理 |

Atlas 300I A2 的官方页面给出了 280 TFLOPS FP16、560 TOPS INT8、最高 64 GB HBM 和 1.6 TB/s 带宽。[Atlas 300I A2 官方规格](https://www.hiascend.com/zh/hardware/accelerator-card)

Atlas 800T A3 的官方页面给出整机 8 个昇腾处理器、最高 6.0 PFLOPS FP16，并支持扩展到最大 384 NPU 的超节点。若仅作平均理解，6.0 PFLOPS ÷ 8 约为 750 TFLOPS/NPU；但这是由**整机标称值推算**，不能替代单 NPU 的完整规格表。[Atlas 800T A3 官方页面](https://www.hiascend.com/hardware/ai-server/)

### 6.2 昇腾软件与互联

```text
PyTorch / MindSpore
  → torch_npu / 框架适配
  → CANN 图引擎、算子库与 Runtime
  → HCCL
  → Ascend NPU、HCCS / RoCE 等互联
```

- **CANN**：涵盖编译、算子、Runtime、开发与性能工具；
- **Ascend C**：用于开发昇腾自定义算子；
- **HCCL**：完成 AllReduce、AllGather、ReduceScatter、Alltoall 等集合通信；
- **MindCluster**：面向集群调度、运维和监控。

对昇腾开发最重要的不是背下一张参数表，而是确认：**具体 Atlas 型号、NPU 数量、显存版本、CANN 版本、服务器拓扑和 HCCL 配置**。

## 7. AMD Instinct：以大显存和高 HBM 带宽竞争

AMD 数据中心 AI 产品采用 CDNA 架构，配套软件生态是 ROCm。

| 型号 | 架构 | 显存 | HBM 带宽 | BF16/FP16 Dense | FP8 Dense |
| --- | --- | ---: | ---: | ---: | ---: |
| MI300X | CDNA 3 | 192 GB HBM3 | 5.3 TB/s | 约 1.31 PFLOPS | 约 2.61 PFLOPS |
| MI325X | CDNA 3 | 256 GB HBM3e | 6.0 TB/s | 约 1.31 PFLOPS | 约 2.61 PFLOPS |
| MI350X | CDNA 4 | 288 GB HBM3e | 8.0 TB/s | 约 2.3 PFLOPS | 约 4.6 PFLOPS |
| MI355X | CDNA 4 | 288 GB HBM3e | 8.0 TB/s | 约 2.5 PFLOPS | 约 5.0 PFLOPS |

MI325X 的显存容量达到 256 GB，峰值带宽为 6 TB/s；MI350X 则达到 288 GB、8 TB/s，并支持 MXFP4/MXFP6 等新数据类型。[AMD MI325X 官方规格](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)、[AMD MI350X 官方规格](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)

AMD 的典型优势：

- 单卡 HBM 容量大，可能用更少卡容纳模型；
- HBM 带宽高，适合权重和 KV 读取密集的任务；
- ROCm、HIP、RCCL 及主流框架支持持续完善；
- OAM/UBB 平台通过 Infinity Fabric 连接多个 GPU。

选型时还要验证目标模型在 ROCm、推理引擎、算子和量化方案上的实际成熟度，不能只按纸面峰值判断。

## 8. Intel Gaudi 3：以标准以太网扩展为特色

Gaudi 3 是 Intel 面向训练和推理的 AI 加速器。PCIe 产品公开规格包括：

- 128 GB HBM；
- 3.7 TB/s HBM 峰值带宽；
- 支持 BF16、FP16、FP8 和 FP32；
- 集成以太网接口，强调基于标准网络的 Scale-out；
- 软件栈以 SynapseAI 为核心。

[Intel Gaudi 3 PCIe 产品说明](https://cdrdv2-public.intel.com/817488/Gaudi%203%20PCIe%20Product%20Brief_RB_1_V6.pdf)

Gaudi 的思路与 NVIDIA NVLink/NVSwitch 或昇腾超节点路线不同：它强调以标准以太网构建集群。实际是否合适，取决于目标框架支持、算子覆盖、集群网络设计和供应条件。

## 9. 把主流产品放到同一张“能力地图”

![主流 AI 加速器显存容量与 HBM 带宽对比](/images/ai-infra/accelerator-memory-bandwidth-map.svg)

这张图只比较显存容量和 HBM 峰值带宽，不代表端到端性能排名。可以观察到：

- H100 的峰值计算很强，但 80 GB 容量在超大模型推理中可能先成为限制；
- H200 主要通过更大、更快的 HBM 改善推理能力；
- B200/B300 与 MI350 系列同时向 8 TB/s 带宽发展；
- MI325X 的 256 GB 和 MI350X/B300 的 288 GB 有利于降低模型并行度；
- Atlas 产品必须按卡、服务器和超节点具体形态分析，不能用一个“昇腾算力”概括。

## 10. 不同工作负载最看重什么

### 10.1 大模型预训练

优先关注：

- BF16/FP16/FP8 训练吞吐；
- HBM 容量与带宽；
- Scale-up 和 Scale-out 网络；
- AllReduce/ReduceScatter 效率；
- 框架稳定性、算子覆盖和故障恢复。

单卡快但多卡扩展差，无法高效完成大规模预训练。

### 10.2 大模型在线推理

优先关注：

- 权重是否能以较低 TP 放入显存；
- KV Cache 能容纳多少并发请求；
- HBM 带宽和低精度 Kernel；
- TTFT、TPOT、吞吐与调度能力；
- Tensor Parallel 通信延迟。

Decode 经常 Memory-bound，因此不能只看 FP8/FP4 峰值。

### 10.3 微调和研发

优先关注：

- 单卡显存是否足够；
- PyTorch、LoRA/QLoRA、Checkpoint 和常用算子是否可用；
- 调试、Profiler 和社区资料是否完善；
- 云资源是否容易获得，按小时价格是否合理。

### 10.4 集合通信开发

对集合通信算子开发，还要额外关注：

- 单机内每张卡之间的物理拓扑；
- Scale-up 链路带宽和并发通道；
- 跨机 RoCE/InfiniBand/Ethernet 带宽；
- NUMA、PCIe Root Complex 与 NIC 亲和性；
- 通信库的算法、协议、Stream 和错误诊断能力。

## 11. 为什么峰值算力不能换算成 Tokens/s

假设某卡有 2 PFLOPS BF16，也不能直接得出它能生成多少 Token。还缺少：

- 模型参数量和架构；
- 实际使用 BF16、FP8 还是量化权重；
- Batch Size 和 Sequence Length；
- Prefill 与 Decode 的比例；
- KV Cache 数据类型；
- Kernel 有效利用率；
- Tensor Parallel 数量和通信；
- 推理引擎调度策略。

近似上限可以写成：

$$
Tokens/s\le\frac{Effective\ FLOPs/s}{FLOPs/Token}
$$

但 Decode 如果受 HBM 带宽限制，更相关的近似是：

$$
Tokens/s\le\frac{Effective\ Memory\ Bandwidth}{Bytes\ Read/Token}
$$

真实上限取这两种约束以及通信、调度约束中的较小者。

## 12. 一套实用选卡流程

1. **确定工作负载**：训练、微调、离线推理还是在线服务；
2. **确定精度**：BF16、FP8、INT8、INT4 是否经过验证；
3. **计算容量**：权重、激活、优化器状态和 KV Cache 能否放下；
4. **判断瓶颈**：更可能受计算、HBM 带宽还是通信限制；
5. **检查多卡拓扑**：需要多少 TP/PP/DP，卡间和跨机链路如何；
6. **验证软件支持**：目标模型、算子、量化和推理引擎是否成熟；
7. **跑真实 Benchmark**：使用自己的模型、Shape、并发和服务指标；
8. **计算总体成本**：硬件、网络、机房、功耗、迁移和运维都要计入。

## 13. 容易混淆的点

1. **TFLOPS 越高就一定越快吗？** 不一定，程序可能受显存、通信或软件效率限制。
2. **FP8 峰值能代表 BF16 训练性能吗？** 不能，精度和执行路径不同。
3. **Sparse 峰值能与 Dense 峰值直接比吗？** 不能，模型必须满足对应稀疏模式才能利用。
4. **显存容量大就一定吞吐高吗？** 不一定，容量决定放得下多少，带宽和算力决定处理速度。
5. **8 卡服务器算力等于单卡算力吗？** 不是，整机数字通常是多卡聚合值。
6. **H200 是 H100 的纯算力升级吗？** 主要升级是 HBM 容量与带宽。
7. **910B/910C 是所有官方文档中的产品名吗？** 不是，部署和兼容性更常按 Atlas A2/A3 产品形态区分。
8. **纸面参数接近就代表迁移成本低吗？** 不代表，软件生态和算子成熟度非常关键。

## 14. 本文数据应该怎样使用

本文数据适合建立产品全景和数量级认知，不适合作为采购合同中的唯一依据。使用时应记录：

```text
厂商与准确型号：
PCIe / SXM / OAM / 整机形态：
Dense / Sparse：
数据类型：
功耗配置：
显存容量与带宽：
卡间与跨机互联：
软件版本：
Benchmark 模型与 Shape：
```

## 15. 小结

- NVIDIA 当前主线从 Hopper H100/H200 进入 Blackwell B200/B300，优势是完整软硬件生态；
- H200、B300 等产品说明显存容量和 HBM 带宽对大模型推理极其重要；
- 昇腾应按 Atlas A2/A3 的卡、服务器和超节点形态分析，CANN 与 HCCL 是工程落地核心；
- AMD MI300/MI350 系列通过大容量 HBM、高带宽和 ROCm 生态参与竞争；
- Intel Gaudi 3 强调标准以太网扩展，也需要结合目标软件栈验证；
- 任何峰值算力比较都必须统一精度、Dense/Sparse 和单卡/整机口径；
- 最终选型应以真实模型的端到端 Benchmark、稳定性和总体成本为准。

## 主要官方资料

- [NVIDIA H100 产品规格](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA HGX H200/B200/B300 参考架构](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
- [昇腾产品形态总览](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
- [Atlas 300I A2 产品规格](https://www.hiascend.com/zh/hardware/accelerator-card)
- [Atlas 800T A3 产品规格](https://www.hiascend.com/hardware/ai-server/)
- [AMD Instinct MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- [AMD Instinct MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)
- [Intel Gaudi 3 PCIe Product Brief](https://cdrdv2-public.intel.com/817488/Gaudi%203%20PCIe%20Product%20Brief_RB_1_V6.pdf)
