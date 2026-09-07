# 大模型基础

::: tip 学习方式
这里把原来的 12 篇长课重构为 **46 个约 15 分钟的学习单元**。一次只点击一个单元、回答一个问题；同一章的单元学完后，再做练习与自测。
:::

从传统后端开发转向 AI Infra 与昇腾集合通信开发，这一阶段要建立一条清晰主线：

> 文本 → 张量 → Transformer → 训练与推理 → 并行切分 → 集合通信

## 阶段一：先看懂模型在算什么

### 第 1 章｜从模型到集合通信

- **01-1（15 分钟）**：[模型训练的主线与系统分层](./01-landscape/01-system-stack.md)
- **01-2（15 分钟）**：[并行策略为什么产生通信](./01-landscape/02-parallelism-communication.md)
- **01-3（15 分钟）**：[从 PyTorch 调用到 HCCL](./01-landscape/03-pytorch-to-hccl.md)

### 第 2 章｜张量、计算图与反向传播

- **02-1（15 分钟）**：[张量 Shape 与矩阵乘法](./02-tensor-autograd/01-tensor-matmul.md)
- **02-2（15 分钟）**：[广播与计算图](./02-tensor-autograd/02-broadcast-computation-graph.md)
- **02-3（15 分钟）**：[链式法则与反向传播](./02-tensor-autograd/03-backpropagation.md)
- **02-4（15 分钟）**：[梯度 Shape 与集合通信](./02-tensor-autograd/04-gradient-communication.md)

### 第 3 章｜文本怎样成为模型输入

- **03-1（15 分钟）**：[Tokenization：文本变成整数序列](./03-tokenization-embedding/01-tokenization.md)
- **03-2（15 分钟）**：[特殊 Token、Mask 与 Embedding](./03-tokenization-embedding/02-mask-embedding.md)
- **03-3（15 分钟）**：[训练标签与下一个 Token 预测](./03-tokenization-embedding/03-causal-lm-target.md)
- **03-4（15 分钟）**：[完整 Shape 流程与系统代价](./03-tokenization-embedding/04-shape-system-cost.md)

## 阶段二：拆开一个 Decoder Block

### 第 4 章｜Self-Attention

- **04-1（15 分钟）**：[Attention 直觉与 Q/K/V](./04-self-attention/01-qkv-intuition.md)
- **04-2（15 分钟）**：[缩放点积与 Causal Mask](./04-self-attention/02-scaled-dot-product-mask.md)
- **04-3（15 分钟）**：[多头注意力与 Shape](./04-self-attention/03-multi-head-shape.md)
- **04-4（15 分钟）**：[计算量、显存与 Flash Attention](./04-self-attention/04-attention-performance.md)

### 第 5 章｜完整 Decoder Block

- **05-1（15 分钟）**：[RMSNorm、残差与 Block 结构](./05-decoder-block/01-norm-residual-block.md)
- **05-2（15 分钟）**：[MLP 与 SwiGLU 数据流](./05-decoder-block/02-mlp-swiglu.md)
- **05-3（15 分钟）**：[参数量、FLOPs 与显存对象](./05-decoder-block/03-parameters-flops-memory.md)
- **05-4（15 分钟）**：[张量并行怎样切分 Block](./05-decoder-block/04-tensor-parallel-block.md)

### 第 6 章｜RoPE 与长上下文

- **06-1（15 分钟）**：[注意力为什么需要位置信息](./06-rope-long-context/01-position-information.md)
- **06-2（15 分钟）**：[二维旋转与相对位置](./06-rope-long-context/02-rotation-relative-position.md)
- **06-3（15 分钟）**：[RoPE 的 Shape 与长上下文限制](./06-rope-long-context/03-rope-shape-long-context.md)

## 阶段三：模型怎样训练和生成

### 第 7 章｜训练循环

- **07-1（15 分钟）**：[一次训练 Step 与梯度](./07-training-loop/01-training-step-gradient.md)
- **07-2（15 分钟）**：[梯度累积与 AdamW 状态](./07-training-loop/02-gradient-accumulation-adamw.md)
- **07-3（15 分钟）**：[学习率、混合精度与梯度裁剪](./07-training-loop/03-lr-mixed-precision-clipping.md)
- **07-4（15 分钟）**：[训练显存与梯度通信](./07-training-loop/04-training-memory-communication.md)

### 第 8 章｜自回归推理与 KV Cache

- **08-1（15 分钟）**：[生成循环、Prefill 与 Decode](./08-inference-kv-cache/01-generation-prefill-decode.md)
- **08-2（15 分钟）**：[KV Cache 保存什么、占多少显存](./08-inference-kv-cache/02-kv-cache-memory.md)
- **08-3（15 分钟）**：[采样与停止条件](./08-inference-kv-cache/03-sampling-stop.md)
- **08-4（15 分钟）**：[推理指标、服务与通信](./08-inference-kv-cache/04-inference-service-metrics.md)

### 第 9 章｜后训练与对齐

- **09-1（15 分钟）**：[预训练、继续预训练与 SFT](./09-post-training-alignment/01-pretrain-sft.md)
- **09-2（15 分钟）**：[Full Fine-Tuning、LoRA 与 QLoRA](./09-post-training-alignment/02-full-finetune-lora.md)
- **09-3（15 分钟）**：[RLHF 与 DPO](./09-post-training-alignment/03-rlhf-dpo.md)
- **09-4（15 分钟）**：[数据质量、显存与通信](./09-post-training-alignment/04-data-memory-communication.md)

## 阶段四：现代架构与系统分析

### 第 10 章｜MoE 与 Expert Parallel

- **10-1（15 分钟）**：[MoE、Router 与 Top-k](./10-moe-expert-parallel/01-moe-router-topk.md)
- **10-2（15 分钟）**：[负载均衡、Capacity 与丢 Token](./10-moe-expert-parallel/02-load-balance-capacity.md)
- **10-3（15 分钟）**：[Token 重排与 Alltoall](./10-moe-expert-parallel/03-token-alltoall.md)
- **10-4（15 分钟）**：[MoE 的并行组合与性能瓶颈](./10-moe-expert-parallel/04-moe-parallel-performance.md)

### 第 11 章｜效率设计

- **11-1（15 分钟）**：[MHA、MQA 与 GQA](./11-efficient-llm-design/01-mha-mqa-gqa.md)
- **11-2（15 分钟）**：[量化原理、粒度与显存收益](./11-efficient-llm-design/02-quantization.md)
- **11-3（15 分钟）**：[长上下文的四条优化路线](./11-efficient-llm-design/03-long-context.md)
- **11-4（15 分钟）**：[把模型优化映射到算子与通信](./11-efficient-llm-design/04-optimization-infra.md)

### 第 12 章｜综合拆解一个现代 LLM

- **12-1（15 分钟）**：[读配置并估算参数量](./12-llm-systems-analysis/01-config-parameters.md)
- **12-2（15 分钟）**：[权重、KV Cache 与训练显存](./12-llm-systems-analysis/02-model-memory.md)
- **12-3（15 分钟）**：[FLOPs 与完整数据流](./12-llm-systems-analysis/03-flops-dataflow.md)
- **12-4（15 分钟）**：[模型切分、通信与性能模型](./12-llm-systems-analysis/04-sharding-performance.md)

::: info 阶段完成标准
不要求背完公式。能够沿着 **Shape → 计算 → 显存 → 切分 → 通信** 解释一个 Decoder-only 模型，就可以进入 AI Infra。
:::

## 学习资料

- [【闪客】一小时从函数到 Transformer](https://www.bilibili.com/video/BV1NCgVzoEG9/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
