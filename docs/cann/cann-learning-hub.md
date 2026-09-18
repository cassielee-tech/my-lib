---
title: cann学习笔记
description:
---

算子库：
ops-math / ops-nn / ops-cv / ops-transformer


通信库：
HCCL / HIXL



---

CANN 与 CUDA 的对照

如果你之前用过 NVIDIA GPU + CUDA，下表帮你快速建立对应关系：

<table style="text-align: left; margin-left: 0;">
<tr>
<th style="text-align: left;">维度</th>
<th style="text-align: left;">CANN（昇腾）</th>
<th style="text-align: left;">CUDA（NVIDIA）</th>
<th style="text-align: left;">说明</th>
</tr>
<tr>
<td style="text-align: left;">框架适配</td>
<td style="text-align: left;">torch_npu</td>
<td style="text-align: left;">torch.cuda</td>
<td style="text-align: left;">把框架调用适配到自家硬件</td>
</tr>
<tr>
<td style="text-align: left;">算子库</td>
<td style="text-align: left;">ops-math / ops-nn / ops-cv / ops-transformer</td>
<td style="text-align: left;">cuDNN / cuBLAS / cuSPARSE</td>
<td style="text-align: left;">预置高性能算子</td>
</tr>
<tr>
<td style="text-align: left;">通信库</td>
<td style="text-align: left;">HCCL + HIXL</td>
<td style="text-align: left;">NCCL</td>
<td style="text-align: left;">多卡分布式通信</td>
</tr>
<tr>
<td style="text-align: left;">图引擎</td>
<td style="text-align: left;">GE</td>
<td style="text-align: left;">TensorRT</td>
<td style="text-align: left;">整网图优化与编译</td>
</tr>
<tr>
<td style="text-align: left;">领域加速库</td>
<td style="text-align: left;">FFT / BLAS 等</td>
<td style="text-align: left;">cuFFT / cuBLAS</td>
<td style="text-align: left;">特定领域极致加速</td>
</tr>
<tr>
<td style="text-align: left;">编程语言</td>
<td style="text-align: left;">Ascend C</td>
<td style="text-align: left;">CUDA C++</td>
<td style="text-align: left;">自定义算子开发语言</td>
</tr>
<tr>
<td style="text-align: left;">编译器</td>
<td style="text-align: left;">毕昇 Bisheng</td>
<td style="text-align: left;">NVCC</td>
<td style="text-align: left;">源码编译为硬件指令</td>
</tr>
<tr>
<td style="text-align: left;">运行时</td>
<td style="text-align: left;">Runtime</td>
<td style="text-align: left;">CUDA Runtime</td>
<td style="text-align: left;">设备/内存/任务管理</td>
</tr>
<tr>
<td style="text-align: left;">驱动</td>
<td style="text-align: left;">Driver</td>
<td style="text-align: left;">NVIDIA Driver</td>
<td style="text-align: left;">最底层硬件交互</td>
</tr>
<tr>
<td style="text-align: left;">Profiling</td>
<td style="text-align: left;">msprof / MindStudio</td>
<td style="text-align: left;">Nsight</td>
<td style="text-align: left;">性能分析工具</td>
</tr>
</table>
