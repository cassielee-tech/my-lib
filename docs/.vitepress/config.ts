import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Cassie’s Stack',
  description: '记录 HCCL 开发主线课程、AI Agent、后端、效率工具与个人思考',
  base: '/my-lib/',
  cleanUrls: true,
  markdown: {
    math: true
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '模型全景', link: '/model/' },
      { text: '训练与推理系统', link: '/systems/' },
      { text: '单卡执行系统', link: '/device/' },
      { text: '并行策略', link: '/parallel/' },
      { text: '集合通信', link: '/collective/' },
      { text: '昇腾与 HCCL', link: '/ascend/' },
      {
        text: '更多',
        items: [
          { text: '课程总览', link: '/roadmap' },
          { text: 'AI Agent', link: '/ai-agent/' },
          { text: '后端知识', link: '/backend/' },
          { text: '工具与效率', link: '/tools-productivity/' },
          { text: '个人杂谈', link: '/essays/' }
        ]
      }
    ],

    sidebar: {
      '/roadmap': [
        {
          text: 'HCCL 开发主线',
          items: [
            { text: '课程总览', link: '/roadmap' },
            { text: '① 模型全景', link: '/model/' },
            { text: '② 训练与推理系统', link: '/systems/' },
            { text: '③ 单卡执行系统', link: '/device/' },
            { text: '④ 并行策略', link: '/parallel/' },
            { text: '⑤ 集合通信（岗位核心）', link: '/collective/' },
            { text: '⑥ 昇腾与 HCCL', link: '/ascend/' }
          ]
        }
      ],
      '/model/': [
        {
          text: '模型全景',
          items: [
            { text: '专栏目录', link: '/model/' },
            { text: '← 课程总览', link: '/roadmap' },
            {
              text: '第 1 章｜从模型到集合通信', link: '/model/01-landscape', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/model/01-landscape/quick' },
                { text: '01-1｜系统主线与分层', link: '/model/01-landscape/01-system-stack' },
                { text: '01-2｜并行为什么产生通信', link: '/model/01-landscape/02-parallelism-communication' },
                { text: '01-3｜PyTorch 到 HCCL', link: '/model/01-landscape/03-pytorch-to-hccl' },
                { text: '本章总结', link: '/model/01-landscape/summary' }
              ]
            },
            {
              text: '第 2 章｜张量、计算图与反向传播', link: '/model/02-tensor-autograd', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/model/02-tensor-autograd/quick' },
                { text: '02-1｜张量与矩阵乘法', link: '/model/02-tensor-autograd/01-tensor-matmul' },
                { text: '02-2｜广播与计算图', link: '/model/02-tensor-autograd/02-broadcast-computation-graph' },
                { text: '02-3｜反向传播', link: '/model/02-tensor-autograd/03-backpropagation' },
                { text: '02-4｜梯度与通信', link: '/model/02-tensor-autograd/04-gradient-communication' },
                { text: '本章总结', link: '/model/02-tensor-autograd/summary' }
              ]
            },
            {
              text: '第 3 章｜模型输入', link: '/model/03-tokenization-embedding', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/model/03-tokenization-embedding/quick' },
                { text: '03-1｜Tokenization', link: '/model/03-tokenization-embedding/01-tokenization' },
                { text: '03-2｜Mask 与 Embedding', link: '/model/03-tokenization-embedding/02-mask-embedding' },
                { text: '03-3｜语言模型目标', link: '/model/03-tokenization-embedding/03-causal-lm-target' },
                { text: '03-4｜Shape 与系统代价', link: '/model/03-tokenization-embedding/04-shape-system-cost' },
                { text: '本章总结', link: '/model/03-tokenization-embedding/summary' }
              ]
            },
            {
              text: '第 4 章｜Self-Attention', link: '/model/04-self-attention', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/model/04-self-attention/quick' },
                { text: '04-1｜Q/K/V 直觉', link: '/model/04-self-attention/01-qkv-intuition' },
                { text: '04-2｜缩放点积与 Mask', link: '/model/04-self-attention/02-scaled-dot-product-mask' },
                { text: '04-3｜多头与 Shape', link: '/model/04-self-attention/03-multi-head-shape' },
                { text: '04-4｜Attention 性能', link: '/model/04-self-attention/04-attention-performance' },
                { text: '本章总结', link: '/model/04-self-attention/summary' }
              ]
            },
            {
              text: '第 5 章｜完整 Decoder Block', link: '/model/05-decoder-block', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/model/05-decoder-block/quick' },
                { text: '05-1｜Norm 与残差', link: '/model/05-decoder-block/01-norm-residual-block' },
                { text: '05-2｜MLP 与 SwiGLU', link: '/model/05-decoder-block/02-mlp-swiglu' },
                { text: '05-3｜参数与 FLOPs', link: '/model/05-decoder-block/03-parameters-flops-memory' },
                { text: '05-4｜Block 张量并行', link: '/model/05-decoder-block/04-tensor-parallel-block' },
                { text: '本章总结', link: '/model/05-decoder-block/summary' }
              ]
            },
            {
              text: '第 6 章｜RoPE 与长上下文（选修）', link: '/model/06-rope-long-context', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/model/06-rope-long-context/quick' },
                { text: '06-1｜位置信息', link: '/model/06-rope-long-context/01-position-information' },
                { text: '06-2｜旋转与相对位置', link: '/model/06-rope-long-context/02-rotation-relative-position' },
                { text: '06-3｜RoPE 与长上下文', link: '/model/06-rope-long-context/03-rope-shape-long-context' },
                { text: '本章总结', link: '/model/06-rope-long-context/summary' }
              ]
            },
            { text: '下一专栏 → 训练与推理系统', link: '/systems/' }
          ]
        }
      ],
      '/systems/': [
        {
          text: '训练与推理系统',
          items: [
            { text: '专栏目录', link: '/systems/' },
            { text: '← 课程总览', link: '/roadmap' },
            {
              text: '第 1 章｜训练循环', link: '/systems/01-training-loop', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/systems/01-training-loop/quick' },
                { text: '01-1｜训练 Step', link: '/systems/01-training-loop/01-training-step-gradient' },
                { text: '01-2｜梯度累积与 AdamW', link: '/systems/01-training-loop/02-gradient-accumulation-adamw' },
                { text: '01-3｜精度与梯度裁剪', link: '/systems/01-training-loop/03-lr-mixed-precision-clipping' },
                { text: '01-4｜显存与通信', link: '/systems/01-training-loop/04-training-memory-communication' },
                { text: '本章总结', link: '/systems/01-training-loop/summary' }
              ]
            },
            {
              text: '第 2 章｜训练显存与计算优化', link: '/systems/02-training-memory-compute', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/systems/02-training-memory-compute/quick' },
                { text: '02-1｜训练显存账本', link: '/systems/02-training-memory-compute/01-memory-ledger' },
                { text: '02-2｜Activation Checkpointing', link: '/systems/02-training-memory-compute/02-activation-checkpointing' },
                { text: '02-3｜累积与 Offload', link: '/systems/02-training-memory-compute/03-accumulation-offload' },
                { text: '02-4｜组合拳与规划', link: '/systems/02-training-memory-compute/04-combined-planning' },
                { text: '本章总结', link: '/systems/02-training-memory-compute/summary' }
              ]
            },
            {
              text: '第 3 章｜推理与 KV Cache（选修）', link: '/systems/03-inference-kv-cache', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/systems/03-inference-kv-cache/quick' },
                { text: '03-1｜Prefill 与 Decode', link: '/systems/03-inference-kv-cache/01-generation-prefill-decode' },
                { text: '03-2｜KV Cache 显存', link: '/systems/03-inference-kv-cache/02-kv-cache-memory' },
                { text: '03-3｜采样与停止', link: '/systems/03-inference-kv-cache/03-sampling-stop' },
                { text: '03-4｜服务与指标', link: '/systems/03-inference-kv-cache/04-inference-service-metrics' },
                { text: '本章总结', link: '/systems/03-inference-kv-cache/summary' }
              ]
            },
            {
              text: '第 4 章｜后训练与对齐（选修）', link: '/systems/04-post-training-alignment', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/systems/04-post-training-alignment/quick' },
                { text: '04-1｜预训练与 SFT', link: '/systems/04-post-training-alignment/01-pretrain-sft' },
                { text: '04-2｜LoRA 与 QLoRA', link: '/systems/04-post-training-alignment/02-full-finetune-lora' },
                { text: '04-3｜RLHF 与 DPO', link: '/systems/04-post-training-alignment/03-rlhf-dpo' },
                { text: '04-4｜数据、显存与通信', link: '/systems/04-post-training-alignment/04-data-memory-communication' },
                { text: '本章总结', link: '/systems/04-post-training-alignment/summary' }
              ]
            },
            { text: '第 5-6 章｜推理系统两讲（选修·待写）', link: '/systems/' },
            { text: '下一专栏 → 单卡执行系统', link: '/device/' }
          ]
        }
      ],
      '/device/': [
        {
          text: '单卡执行系统',
          items: [
            { text: '专栏目录', link: '/device/' },
            { text: '← 课程总览', link: '/roadmap' },
            {
              text: '第 1 章｜全景与性能分析', link: '/device/01-landscape-performance', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/01-landscape-performance/quick' },
                { text: '01-1｜分层与执行对象', link: '/device/01-landscape-performance/01-stack-execution' },
                { text: '01-2｜指标与时间线', link: '/device/01-landscape-performance/02-metrics-timeline' },
                { text: '01-3｜三类瓶颈', link: '/device/01-landscape-performance/03-performance-bottlenecks' },
                { text: '01-4｜算术强度', link: '/device/01-landscape-performance/04-arithmetic-intensity' },
                { text: '01-5｜分析流程', link: '/device/01-landscape-performance/05-analysis-workflow' },
                { text: '本章总结', link: '/device/01-landscape-performance/summary' }
              ]
            },
            {
              text: '第 2 章｜GPU/NPU 执行模型', link: '/device/02-execution-model', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/02-execution-model/quick' },
                { text: '02-1｜执行哲学', link: '/device/02-execution-model/01-execution-philosophy' },
                { text: '02-2｜SM、Warp 与 SIMT', link: '/device/02-execution-model/02-gpu-sm-warp-simt' },
                { text: '02-3｜达芬奇与计算单元', link: '/device/02-execution-model/03-npu-davinci-core' },
                { text: '02-4｜利用率与尾效应', link: '/device/02-execution-model/04-utilization-tail-effects' },
                { text: '本章总结', link: '/device/02-execution-model/summary' }
              ]
            },
            {
              text: '第 3 章｜存储层次与数据搬运', link: '/device/03-memory-hierarchy', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/03-memory-hierarchy/quick' },
                { text: '03-1｜存储金字塔', link: '/device/03-memory-hierarchy/01-memory-pyramid' },
                { text: '03-2｜HBM、Cache 与 Buffer', link: '/device/03-memory-hierarchy/02-hbm-cache-buffer-register' },
                { text: '03-3｜数据搬运的代价', link: '/device/03-memory-hierarchy/03-data-movement-cost' },
                { text: '03-4｜局部性与复用', link: '/device/03-memory-hierarchy/04-locality-reuse' },
                { text: '本章总结', link: '/device/03-memory-hierarchy/summary' }
              ]
            },
            {
              text: '第 4 章｜Kernel、Tiling 与流水线', link: '/device/04-kernel-tiling-pipeline', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/04-kernel-tiling-pipeline/quick' },
                { text: '04-1｜Kernel 的解剖', link: '/device/04-kernel-tiling-pipeline/01-kernel-anatomy' },
                { text: '04-2｜Tiling 切分策略', link: '/device/04-kernel-tiling-pipeline/02-tiling-strategy' },
                { text: '04-3｜流水线与 Double Buffer', link: '/device/04-kernel-tiling-pipeline/03-pipeline-double-buffer' },
                { text: '04-4｜最小算子走读', link: '/device/04-kernel-tiling-pipeline/04-minimal-kernel-walkthrough' },
                { text: '本章总结', link: '/device/04-kernel-tiling-pipeline/summary' }
              ]
            },
            {
              text: '第 5 章｜FLOPs、带宽与 Roofline', link: '/device/05-flops-bandwidth-roofline', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/05-flops-bandwidth-roofline/quick' },
                { text: '05-1｜算子的三本账', link: '/device/05-flops-bandwidth-roofline/01-operator-accounting' },
                { text: '05-2｜Roofline 模型', link: '/device/05-flops-bandwidth-roofline/02-roofline-model' },
                { text: '05-3｜Roofline 案例分析', link: '/device/05-flops-bandwidth-roofline/03-roofline-case-studies' },
                { text: '05-4｜从判断到行动', link: '/device/05-flops-bandwidth-roofline/04-from-diagnosis-to-action' },
                { text: '本章总结', link: '/device/05-flops-bandwidth-roofline/summary' }
              ]
            },
            {
              text: '第 6 章｜Eager、计算图与图编译', link: '/device/06-eager-graph-compilation', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/06-eager-graph-compilation/quick' },
                { text: '06-1｜Eager 执行', link: '/device/06-eager-graph-compilation/01-eager-execution' },
                { text: '06-2｜动态图与静态图', link: '/device/06-eager-graph-compilation/02-computation-graph' },
                { text: '06-3｜图编译流水线', link: '/device/06-eager-graph-compilation/03-graph-compilation' },
                { text: '06-4｜图模式的边界', link: '/device/06-eager-graph-compilation/04-limits-and-tradeoffs' },
                { text: '本章总结', link: '/device/06-eager-graph-compilation/summary' }
              ]
            },
            {
              text: '第 7 章｜Stream、Event 与异步执行', link: '/device/07-stream-event-async', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/07-stream-event-async/quick' },
                { text: '07-1｜异步执行', link: '/device/07-stream-event-async/01-async-execution' },
                { text: '07-2｜Stream 单行道', link: '/device/07-stream-event-async/02-streams' },
                { text: '07-3｜Event 与同步', link: '/device/07-stream-event-async/03-events-sync' },
                { text: '07-4｜重叠的艺术', link: '/device/07-stream-event-async/04-overlap' },
                { text: '本章总结', link: '/device/07-stream-event-async/summary' }
              ]
            },
            {
              text: '第 8 章｜数据类型、布局与融合', link: '/device/08-dtype-layout-fusion', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/device/08-dtype-layout-fusion/quick' },
                { text: '08-1｜数据类型三角', link: '/device/08-dtype-layout-fusion/01-dtype-basics' },
                { text: '08-2｜混合精度实践', link: '/device/08-dtype-layout-fusion/02-mixed-precision' },
                { text: '08-3｜Layout 与数据摆放', link: '/device/08-dtype-layout-fusion/03-layout' },
                { text: '08-4｜融合：三类手艺', link: '/device/08-dtype-layout-fusion/04-fusion' },
                { text: '本章总结', link: '/device/08-dtype-layout-fusion/summary' }
              ]
            },
            { text: '专题｜主流 AI 加速卡全景', link: '/device/accelerator-cards-2026' },
            { text: '下一专栏 → 并行策略', link: '/parallel/' }
          ]
        }
      ],
      '/parallel/': [
        {
          text: '并行策略',
          items: [
            { text: '专栏目录', link: '/parallel/' },
            { text: '← 课程总览', link: '/roadmap' },
            {
              text: '第 1 章｜MoE 与 Expert Parallel', link: '/parallel/01-moe-expert-parallel', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/01-moe-expert-parallel/quick' },
                { text: '01-1｜Router 与 Top-k', link: '/parallel/01-moe-expert-parallel/01-moe-router-topk' },
                { text: '01-2｜负载与 Capacity', link: '/parallel/01-moe-expert-parallel/02-load-balance-capacity' },
                { text: '01-3｜Token 与 Alltoall', link: '/parallel/01-moe-expert-parallel/03-token-alltoall' },
                { text: '01-4｜并行与性能', link: '/parallel/01-moe-expert-parallel/04-moe-parallel-performance' },
                { text: '本章总结', link: '/parallel/01-moe-expert-parallel/summary' }
              ]
            },
            {
              text: '第 2 章｜效率设计（选修）', link: '/parallel/02-efficient-llm-design', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/02-efficient-llm-design/quick' },
                { text: '02-1｜MHA/MQA/GQA', link: '/parallel/02-efficient-llm-design/01-mha-mqa-gqa' },
                { text: '02-2｜量化', link: '/parallel/02-efficient-llm-design/02-quantization' },
                { text: '02-3｜长上下文', link: '/parallel/02-efficient-llm-design/03-long-context' },
                { text: '02-4｜算子与通信', link: '/parallel/02-efficient-llm-design/04-optimization-infra' },
                { text: '本章总结', link: '/parallel/02-efficient-llm-design/summary' }
              ]
            },
            {
              text: '第 3 章｜系统分析', link: '/parallel/03-llm-systems-analysis', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/03-llm-systems-analysis/quick' },
                { text: '03-1｜配置与参数量', link: '/parallel/03-llm-systems-analysis/01-config-parameters' },
                { text: '03-2｜模型显存', link: '/parallel/03-llm-systems-analysis/02-model-memory' },
                { text: '03-3｜FLOPs 与数据流', link: '/parallel/03-llm-systems-analysis/03-flops-dataflow' },
                { text: '03-4｜切分与性能模型', link: '/parallel/03-llm-systems-analysis/04-sharding-performance' },
                { text: '本章总结', link: '/parallel/03-llm-systems-analysis/summary' }
              ]
            },
            {
              text: '第 4 章｜分布式基础', link: '/parallel/04-distributed-basics', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/04-distributed-basics/quick' },
                { text: '04-1｜从进程到 Rank', link: '/parallel/04-distributed-basics/01-process-rank' },
                { text: '04-2｜通信域与 PG', link: '/parallel/04-distributed-basics/02-process-groups' },
                { text: '04-3｜拓扑感知排布', link: '/parallel/04-distributed-basics/03-topology-aware-layout' },
                { text: '04-4｜实操与排障', link: '/parallel/04-distributed-basics/04-launch-and-debug' },
                { text: '本章总结', link: '/parallel/04-distributed-basics/summary' }
              ]
            },
            {
              text: '第 5 章｜数据并行与 DDP', link: '/parallel/05-ddp', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/05-ddp/quick' },
                { text: '05-1｜DP 语义', link: '/parallel/05-ddp/01-dp-semantics' },
                { text: '05-2｜朴素版到 DDP', link: '/parallel/05-ddp/02-naive-to-ddp' },
                { text: '05-3｜Bucket 分桶', link: '/parallel/05-ddp/03-buckets' },
                { text: '05-4｜通信计算重叠', link: '/parallel/05-ddp/04-overlap' },
                { text: '本章总结', link: '/parallel/05-ddp/summary' }
              ]
            },
            {
              text: '第 6 章｜Tensor Parallel', link: '/parallel/06-tensor-parallel', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/06-tensor-parallel/quick' },
                { text: '06-1｜为什么是 TP', link: '/parallel/06-tensor-parallel/01-why-tp' },
                { text: '06-2｜Column 与 Row 切法', link: '/parallel/06-tensor-parallel/02-column-row-parallel' },
                { text: '06-3｜通信账与位置账', link: '/parallel/06-tensor-parallel/03-comm-cost' },
                { text: '06-4｜TP+SP 与工程要点', link: '/parallel/06-tensor-parallel/04-tp-sp-and-practice' },
                { text: '本章总结', link: '/parallel/06-tensor-parallel/summary' }
              ]
            },
            {
              text: '第 7 章｜Pipeline Parallel', link: '/parallel/07-pipeline-parallel', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/07-pipeline-parallel/quick' },
                { text: '07-1｜为什么是 PP', link: '/parallel/07-pipeline-parallel/01-why-pp' },
                { text: '07-2｜Micro-batch 与 Bubble', link: '/parallel/07-pipeline-parallel/02-microbatch-bubble' },
                { text: '07-3｜1F1B', link: '/parallel/07-pipeline-parallel/03-1f1b' },
                { text: '07-4｜三本账与 3D 组合', link: '/parallel/07-pipeline-parallel/04-pp-engineering' },
                { text: '本章总结', link: '/parallel/07-pipeline-parallel/summary' }
              ]
            },
            {
              text: '第 8 章｜Context Parallel', link: '/parallel/08-context-parallel', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/08-context-parallel/quick' },
                { text: '08-1｜为什么是 CP', link: '/parallel/08-context-parallel/01-why-cp' },
                { text: '08-2｜AllGather 还是 Ring', link: '/parallel/08-context-parallel/02-attention-comm' },
                { text: '08-3｜Ring 与负载均衡', link: '/parallel/08-context-parallel/03-ring-attention' },
                { text: '08-4｜账与 4D 组合', link: '/parallel/08-context-parallel/04-cp-engineering' },
                { text: '本章总结', link: '/parallel/08-context-parallel/summary' }
              ]
            },
            {
              text: '第 9 章｜ZeRO、FSDP 与混合并行', link: '/parallel/09-zero-fsdp', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/parallel/09-zero-fsdp/quick' },
                { text: '09-1｜ZeRO 洞察', link: '/parallel/09-zero-fsdp/01-zero-insight' },
                { text: '09-2｜FSDP', link: '/parallel/09-zero-fsdp/02-fsdp' },
                { text: '09-3｜五刀组合', link: '/parallel/09-zero-fsdp/03-combination' },
                { text: '09-4｜收官地图', link: '/parallel/09-zero-fsdp/04-wrapup' },
                { text: '本章总结', link: '/parallel/09-zero-fsdp/summary' }
              ]
            },
            { text: '下一专栏 → 集合通信', link: '/collective/' }
          ]
        }
      ],
      '/collective/': [
        {
          text: '集合通信',
          items: [
            { text: '专栏目录', link: '/collective/' },
            { text: '← 课程总览', link: '/roadmap' },
            {
              text: '第 1 章｜Collective 语义与代价模型', link: '/collective/01-collective-semantics-cost', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/collective/01-collective-semantics-cost/quick' },
                { text: '01-1｜Rank 与通信域', link: '/collective/01-collective-semantics-cost/01-rank-domain-message' },
                { text: '01-2｜六大原语卡片', link: '/collective/01-collective-semantics-cost/02-primitive-cards' },
                { text: '01-3｜α-β-γ 代价模型', link: '/collective/01-collective-semantics-cost/03-alpha-beta-gamma' },
                { text: '01-4｜从语义到选型', link: '/collective/01-collective-semantics-cost/04-from-semantics-to-selection' },
                { text: '本章总结', link: '/collective/01-collective-semantics-cost/summary' }
              ]
            },
            {
              text: '第 2 章｜Ring AllReduce 推导', link: '/collective/02-ring-allreduce', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/collective/02-ring-allreduce/quick' },
                { text: '02-1｜为什么是环', link: '/collective/02-ring-allreduce/01-why-ring' },
                { text: '02-2｜RS 轮转', link: '/collective/02-ring-allreduce/02-reduce-scatter-phase' },
                { text: '02-3｜AG 轮转', link: '/collective/02-ring-allreduce/03-allgather-phase' },
                { text: '02-4｜账本与变体', link: '/collective/02-ring-allreduce/04-ledger-and-variants' },
                { text: '本章总结', link: '/collective/02-ring-allreduce/summary' }
              ]
            },
            {
              text: '第 3 章｜Tree、RD 与算法选择', link: '/collective/03-tree-algorithms-selection', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/collective/03-tree-algorithms-selection/quick' },
                { text: '03-1｜Recursive Doubling', link: '/collective/03-tree-algorithms-selection/01-recursive-doubling' },
                { text: '03-2｜Tree 算法', link: '/collective/03-tree-algorithms-selection/02-tree-algorithms' },
                { text: '03-3｜选型地图', link: '/collective/03-tree-algorithms-selection/03-selection-map' },
                { text: '03-4｜selector 与实测', link: '/collective/03-tree-algorithms-selection/04-selector-engineering' },
                { text: '本章总结', link: '/collective/03-tree-algorithms-selection/summary' }
              ]
            },
            {
              text: '第 4 章｜AG、RS 与 AlltoAll', link: '/collective/04-gather-scatter-alltoall', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/collective/04-gather-scatter-alltoall/quick' },
                { text: '04-1｜AllGather', link: '/collective/04-gather-scatter-alltoall/01-allgather' },
                { text: '04-2｜ReduceScatter', link: '/collective/04-gather-scatter-alltoall/02-reduce-scatter' },
                { text: '04-3｜AlltoAll', link: '/collective/04-gather-scatter-alltoall/03-alltoall' },
                { text: '04-4｜MoE 的失衡', link: '/collective/04-gather-scatter-alltoall/04-moe-imbalance' },
                { text: '本章总结', link: '/collective/04-gather-scatter-alltoall/summary' }
              ]
            },
            {
              text: '第 5 章｜拓扑、分层与重叠', link: '/collective/05-topology-hierarchical-overlap', collapsed: true,
              items: [
                { text: '⚡ 速通版（5 分钟）', link: '/collective/05-topology-hierarchical-overlap/quick' },
                { text: '05-1｜拓扑地图', link: '/collective/05-topology-hierarchical-overlap/01-topology-map' },
                { text: '05-2｜分层算法', link: '/collective/05-topology-hierarchical-overlap/02-hierarchical' },
                { text: '05-3｜Chunk 与 Channel', link: '/collective/05-topology-hierarchical-overlap/03-chunk-channel' },
                { text: '05-4｜重叠实践', link: '/collective/05-topology-hierarchical-overlap/04-overlap-practice' },
                { text: '本章总结', link: '/collective/05-topology-hierarchical-overlap/summary' }
              ]
            },
            { text: '下一专栏 → 昇腾与 HCCL', link: '/ascend/' }
          ]
        }
      ],
      '/ascend/': [
        {
          text: '昇腾与 HCCL',
          items: [
            { text: '专栏目录', link: '/ascend/' },
            { text: '← 课程总览', link: '/roadmap' },
            {
              text: '第 1 章｜认识昇腾与 CANN', link: '/ascend/01-ascend-cann', collapsed: true,
              items: [
                { text: '01-1｜NPU 与 AI Core', link: '/ascend/01-ascend-cann/01-npu-davinci-ai-core' },
                { text: '01-2｜CANN 软件栈与执行路径', link: '/ascend/01-ascend-cann/02-cann-stack-execution-path' },
                { text: '01-3｜版本、驱动与开发环境', link: '/ascend/01-ascend-cann/03-version-driver-firmware-env' },
                { text: '本章总结｜模型到 NPU 的完整路径', link: '/ascend/01-ascend-cann/summary' }
              ]
            },
            {
              text: '第 2 章｜Runtime 与任务执行', link: '/ascend/02-runtime-task-execution', collapsed: true,
              items: [
                { text: '02-1｜Host、Device 与异构计算', link: '/ascend/02-runtime-task-execution/01-host-device' },
                { text: '02-2｜设备内存与数据搬运', link: '/ascend/02-runtime-task-execution/02-memory-copy-lifecycle' },
                { text: '02-3｜Stream、Event 与异步执行', link: '/ascend/02-runtime-task-execution/03-stream-event-task' },
                { text: '02-4｜一次算子调用的旅程', link: '/ascend/02-runtime-task-execution/04-op-call-lifecycle' },
                { text: '本章总结｜双时间线读性能', link: '/ascend/02-runtime-task-execution/summary' }
              ]
            },
            {
              text: '第 3 章｜从 PyTorch 走向 HCCL', link: '/ascend/03-pytorch-to-hccl', collapsed: true,
              items: [
                { text: '03-1｜torch_npu：框架之桥', link: '/ascend/03-pytorch-to-hccl/01-torch-npu-bridge' },
                { text: '03-2｜ProcessGroup 门面', link: '/ascend/03-pytorch-to-hccl/02-process-group' },
                { text: '03-3｜Stream 与 Notify 桥接', link: '/ascend/03-pytorch-to-hccl/03-stream-notify-bridge' },
                { text: '03-4｜AllReduce 完整旅程', link: '/ascend/03-pytorch-to-hccl/04-allreduce-journey' },
                { text: '本章总结｜计算与通信交汇', link: '/ascend/03-pytorch-to-hccl/summary' }
              ]
            },
            { text: '第 4-6 章｜Ascend C 算子开发（待写·二梯队）', link: '/ascend/' },
            {
              text: '专题｜HCCL 源码学习', link: '/ascend/hccl-source', collapsed: true,
              items: [
                { text: 'H01-1｜HCCL 全景与仓库地图', link: '/ascend/hccl-source/01-hccl-repo-map' },
                { text: 'H01-2｜HCCL 与 HCOMM 分层架构', link: '/ascend/hccl-source/02-architecture-layering' },
                { text: 'H01-3｜通信域、Rank 与 RankGraph', link: '/ascend/hccl-source/03-comm-domain-rank-graph' },
                { text: 'H01-4｜通信原语与同步机制', link: '/ascend/hccl-source/04-primitives-and-sync' },
                { text: 'H01-5｜通信引擎与任务执行', link: '/ascend/hccl-source/05-comm-engines' },
                { text: 'H01-6｜集合通信算法与代价模型', link: '/ascend/hccl-source/06-coll-algorithms' },
                { text: 'H01-7｜AllReduce 调用链走读', link: '/ascend/hccl-source/07-allreduce-call-chain' },
                { text: '专题总结｜源码阅读地图', link: '/ascend/hccl-source/summary' }
              ]
            },
            { text: '速查｜CANN 常用命令', link: '/ascend/cann常用命令' },
            { text: '笔记｜CANN Learning Hub', link: '/ascend/cann-learning-hub' }
          ]
        }
      ],
      '/ai-agent/': [
        {
          text: 'AI Agent',
          items: [
            { text: '目录', link: '/ai-agent/' },
            { text: '2026 热门 AI Agent 产品地图', link: '/ai-agent/01-popular-agents' },
            { text: 'AI Agent 工程化知识树', link: '/ai-agent/agent-engineering-knowledge-tree' },
            { text: 'AI Coding：从代码补全到智能体协作', link: '/ai-agent/ai-coding-codearts' }
          ]
        }
      ],
      '/backend/': [{ text: '后端知识', items: [{ text: '目录', link: '/backend/' }] }],
      '/tools-productivity/': [
        {
          text: '工具与效率',
          items: [
            { text: '目录', link: '/tools-productivity/' },
            { text: 'VS Code + SSH + Jupyter', link: '/tools-productivity/vscode-remote-ssh-jupyter' },
            { text: 'opencode 使用教程', link: '/tools-productivity/opencode-tutorial' },
            { text: 'Ghostty 快捷键速查', link: '/tools-productivity/ghostty-shortcuts' },
            { text: 'macOS 快捷键速查', link: '/tools-productivity/macos-shortcuts' }
          ]
        }
      ],
      '/essays/': [
        {
          text: '个人杂谈',
          items: [
            { text: '目录', link: '/essays/' },
            { text: '《金刚经》研读笔记', link: '/essays/《金刚经》研读笔记' }
          ]
        }
      ]
    },

    search: { provider: 'local' },
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新于' },
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    socialLinks: [{ icon: 'github', link: 'https://github.com/cassielee-tech' }]
  }
})
