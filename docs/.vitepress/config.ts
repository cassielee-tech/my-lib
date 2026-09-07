import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Cassie’s Stack',
  description: '记录大模型、AI Infra、AI Agent、后端、效率工具与个人思考',
  base: '/my-lib/',
  cleanUrls: true,
  markdown: {
    math: true
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/roadmap' },
      { text: '大模型基础', link: '/llm-foundations/' },
      { text: 'AI Infra', link: '/ai-infra/' },
      { text: 'AI Agent', link: '/ai-agent/' },
      { text: '后端知识', link: '/backend/' },
      { text: '工具与效率', link: '/tools-productivity/' },
      { text: '个人杂谈', link: '/essays/' }
    ],

    sidebar: {
      '/backend/': [{ text: '后端知识', items: [{ text: '目录', link: '/backend/' }] }],
      '/llm-foundations/': [
        {
          text: '大模型基础',
          items: [
            { text: '目录', link: '/llm-foundations/' },
            {
              text: '第 1 章｜从模型到集合通信', link: '/llm-foundations/01-landscape', collapsed: true,
              items: [
                { text: '01-1｜系统主线与分层', link: '/llm-foundations/01-landscape/01-system-stack' },
                { text: '01-2｜并行为什么产生通信', link: '/llm-foundations/01-landscape/02-parallelism-communication' },
                { text: '01-3｜PyTorch 到 HCCL', link: '/llm-foundations/01-landscape/03-pytorch-to-hccl' },
                { text: '本章总结', link: '/llm-foundations/01-landscape/summary' }
              ]
            },
            {
              text: '第 2 章｜张量与反向传播', link: '/llm-foundations/02-tensor-autograd', collapsed: true,
              items: [
                { text: '02-1｜张量与矩阵乘法', link: '/llm-foundations/02-tensor-autograd/01-tensor-matmul' },
                { text: '02-2｜广播与计算图', link: '/llm-foundations/02-tensor-autograd/02-broadcast-computation-graph' },
                { text: '02-3｜反向传播', link: '/llm-foundations/02-tensor-autograd/03-backpropagation' },
                { text: '02-4｜梯度与通信', link: '/llm-foundations/02-tensor-autograd/04-gradient-communication' },
                { text: '本章总结', link: '/llm-foundations/02-tensor-autograd/summary' }
              ]
            },
            {
              text: '第 3 章｜模型输入', link: '/llm-foundations/03-tokenization-embedding', collapsed: true,
              items: [
                { text: '03-1｜Tokenization', link: '/llm-foundations/03-tokenization-embedding/01-tokenization' },
                { text: '03-2｜Mask 与 Embedding', link: '/llm-foundations/03-tokenization-embedding/02-mask-embedding' },
                { text: '03-3｜语言模型目标', link: '/llm-foundations/03-tokenization-embedding/03-causal-lm-target' },
                { text: '03-4｜Shape 与系统代价', link: '/llm-foundations/03-tokenization-embedding/04-shape-system-cost' },
                { text: '本章总结', link: '/llm-foundations/03-tokenization-embedding/summary' }
              ]
            },
            {
              text: '第 4 章｜Self-Attention', link: '/llm-foundations/04-self-attention', collapsed: true,
              items: [
                { text: '04-1｜Q/K/V 直觉', link: '/llm-foundations/04-self-attention/01-qkv-intuition' },
                { text: '04-2｜缩放点积与 Mask', link: '/llm-foundations/04-self-attention/02-scaled-dot-product-mask' },
                { text: '04-3｜多头与 Shape', link: '/llm-foundations/04-self-attention/03-multi-head-shape' },
                { text: '04-4｜Attention 性能', link: '/llm-foundations/04-self-attention/04-attention-performance' },
                { text: '本章总结', link: '/llm-foundations/04-self-attention/summary' }
              ]
            },
            {
              text: '第 5 章｜Decoder Block', link: '/llm-foundations/05-decoder-block', collapsed: true,
              items: [
                { text: '05-1｜Norm 与残差', link: '/llm-foundations/05-decoder-block/01-norm-residual-block' },
                { text: '05-2｜MLP 与 SwiGLU', link: '/llm-foundations/05-decoder-block/02-mlp-swiglu' },
                { text: '05-3｜参数与 FLOPs', link: '/llm-foundations/05-decoder-block/03-parameters-flops-memory' },
                { text: '05-4｜Block 张量并行', link: '/llm-foundations/05-decoder-block/04-tensor-parallel-block' },
                { text: '本章总结', link: '/llm-foundations/05-decoder-block/summary' }
              ]
            },
            {
              text: '第 6 章｜RoPE 与长上下文', link: '/llm-foundations/06-rope-long-context', collapsed: true,
              items: [
                { text: '06-1｜位置信息', link: '/llm-foundations/06-rope-long-context/01-position-information' },
                { text: '06-2｜旋转与相对位置', link: '/llm-foundations/06-rope-long-context/02-rotation-relative-position' },
                { text: '06-3｜RoPE 与长上下文', link: '/llm-foundations/06-rope-long-context/03-rope-shape-long-context' },
                { text: '本章总结', link: '/llm-foundations/06-rope-long-context/summary' }
              ]
            },
            {
              text: '第 7 章｜训练循环', link: '/llm-foundations/07-training-loop', collapsed: true,
              items: [
                { text: '07-1｜训练 Step', link: '/llm-foundations/07-training-loop/01-training-step-gradient' },
                { text: '07-2｜梯度累积与 AdamW', link: '/llm-foundations/07-training-loop/02-gradient-accumulation-adamw' },
                { text: '07-3｜精度与梯度裁剪', link: '/llm-foundations/07-training-loop/03-lr-mixed-precision-clipping' },
                { text: '07-4｜显存与通信', link: '/llm-foundations/07-training-loop/04-training-memory-communication' },
                { text: '本章总结', link: '/llm-foundations/07-training-loop/summary' }
              ]
            },
            {
              text: '第 8 章｜推理与 KV Cache', link: '/llm-foundations/08-inference-kv-cache', collapsed: true,
              items: [
                { text: '08-1｜Prefill 与 Decode', link: '/llm-foundations/08-inference-kv-cache/01-generation-prefill-decode' },
                { text: '08-2｜KV Cache 显存', link: '/llm-foundations/08-inference-kv-cache/02-kv-cache-memory' },
                { text: '08-3｜采样与停止', link: '/llm-foundations/08-inference-kv-cache/03-sampling-stop' },
                { text: '08-4｜服务与指标', link: '/llm-foundations/08-inference-kv-cache/04-inference-service-metrics' },
                { text: '本章总结', link: '/llm-foundations/08-inference-kv-cache/summary' }
              ]
            },
            {
              text: '第 9 章｜后训练与对齐', link: '/llm-foundations/09-post-training-alignment', collapsed: true,
              items: [
                { text: '09-1｜预训练与 SFT', link: '/llm-foundations/09-post-training-alignment/01-pretrain-sft' },
                { text: '09-2｜LoRA 与 QLoRA', link: '/llm-foundations/09-post-training-alignment/02-full-finetune-lora' },
                { text: '09-3｜RLHF 与 DPO', link: '/llm-foundations/09-post-training-alignment/03-rlhf-dpo' },
                { text: '09-4｜数据、显存与通信', link: '/llm-foundations/09-post-training-alignment/04-data-memory-communication' },
                { text: '本章总结', link: '/llm-foundations/09-post-training-alignment/summary' }
              ]
            },
            {
              text: '第 10 章｜MoE', link: '/llm-foundations/10-moe-expert-parallel', collapsed: true,
              items: [
                { text: '10-1｜Router 与 Top-k', link: '/llm-foundations/10-moe-expert-parallel/01-moe-router-topk' },
                { text: '10-2｜负载与 Capacity', link: '/llm-foundations/10-moe-expert-parallel/02-load-balance-capacity' },
                { text: '10-3｜Token 与 Alltoall', link: '/llm-foundations/10-moe-expert-parallel/03-token-alltoall' },
                { text: '10-4｜并行与性能', link: '/llm-foundations/10-moe-expert-parallel/04-moe-parallel-performance' },
                { text: '本章总结', link: '/llm-foundations/10-moe-expert-parallel/summary' }
              ]
            },
            {
              text: '第 11 章｜效率设计', link: '/llm-foundations/11-efficient-llm-design', collapsed: true,
              items: [
                { text: '11-1｜MHA/MQA/GQA', link: '/llm-foundations/11-efficient-llm-design/01-mha-mqa-gqa' },
                { text: '11-2｜量化', link: '/llm-foundations/11-efficient-llm-design/02-quantization' },
                { text: '11-3｜长上下文', link: '/llm-foundations/11-efficient-llm-design/03-long-context' },
                { text: '11-4｜算子与通信', link: '/llm-foundations/11-efficient-llm-design/04-optimization-infra' },
                { text: '本章总结', link: '/llm-foundations/11-efficient-llm-design/summary' }
              ]
            },
            {
              text: '第 12 章｜系统分析', link: '/llm-foundations/12-llm-systems-analysis', collapsed: true,
              items: [
                { text: '12-1｜配置与参数量', link: '/llm-foundations/12-llm-systems-analysis/01-config-parameters' },
                { text: '12-2｜模型显存', link: '/llm-foundations/12-llm-systems-analysis/02-model-memory' },
                { text: '12-3｜FLOPs 与数据流', link: '/llm-foundations/12-llm-systems-analysis/03-flops-dataflow' },
                { text: '12-4｜切分与性能模型', link: '/llm-foundations/12-llm-systems-analysis/04-sharding-performance' },
                { text: '本章总结', link: '/llm-foundations/12-llm-systems-analysis/summary' }
              ]
            }
          ]
        }
      ],
      '/ai-infra/': [
        {
          text: 'AI Infra',
          items: [
            { text: '目录', link: '/ai-infra/' },
            {
              text: '第 1 章｜全景与性能分析', link: '/ai-infra/01-landscape-performance', collapsed: true,
              items: [
                { text: 'I01-1｜分层与执行对象', link: '/ai-infra/01-landscape-performance/01-stack-execution' },
                { text: 'I01-2｜指标与时间线', link: '/ai-infra/01-landscape-performance/02-metrics-timeline' },
                { text: 'I01-3｜三类瓶颈', link: '/ai-infra/01-landscape-performance/03-performance-bottlenecks' },
                { text: 'I01-4｜算术强度', link: '/ai-infra/01-landscape-performance/04-arithmetic-intensity' },
                { text: 'I01-5｜分析流程', link: '/ai-infra/01-landscape-performance/05-analysis-workflow' },
                { text: '本章总结', link: '/ai-infra/01-landscape-performance/summary' }
              ]
            },
            { text: '专题｜主流 AI 加速卡全景', link: '/ai-infra/accelerator-cards-2026' }
          ]
        }
      ],
      '/ai-agent/': [
        {
          text: 'AI Agent',
          items: [
            { text: '目录', link: '/ai-agent/' },
            { text: '2026 热门 AI Agent 产品地图', link: '/ai-agent/01-popular-agents' },
            { text: 'AI Agent 工程化知识树', link: '/ai-agent/agent-engineering-knowledge-tree' }
          ]
        }
      ],
      '/tools-productivity/': [
        {
          text: '工具与效率',
          items: [
            { text: '目录', link: '/tools-productivity/' },
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
            { text: 'AI Coding：从代码补全到智能体协作', link: '/essays/ai-coding-codearts' }
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
