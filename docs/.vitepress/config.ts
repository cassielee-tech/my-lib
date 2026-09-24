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
            { text: '第 1 章｜从模型到集合通信', link: '/model/01-landscape' },
            { text: '第 2 章｜张量、计算图与反向传播', link: '/model/02-tensor-autograd' },
            { text: '第 3 章｜模型输入', link: '/model/03-tokenization-embedding' },
            { text: '第 4 章｜Self-Attention', link: '/model/04-self-attention' },
            { text: '第 5 章｜完整 Decoder Block', link: '/model/05-decoder-block' },
            { text: '第 6 章｜RoPE 与长上下文（选修）', link: '/model/06-rope-long-context' },
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
            { text: '第 1 章｜训练循环', link: '/systems/01-training-loop' },
            { text: '第 2 章｜训练显存与计算优化', link: '/systems/02-training-memory-compute' },
            { text: '第 3 章｜推理与 KV Cache（选修）', link: '/systems/03-inference-kv-cache' },
            { text: '第 4 章｜后训练与对齐（选修）', link: '/systems/04-post-training-alignment' },
            { text: '第 5 章｜推理引擎与 KV Cache（选修）', link: '/systems/05-inference-engine' },
            { text: '第 6 章｜PD 调度与推理指标（选修）', link: '/systems/06-pd-scheduling' },
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
            { text: '第 1 章｜全景与性能分析', link: '/device/01-landscape-performance' },
            { text: '第 2 章｜GPU/NPU 执行模型', link: '/device/02-execution-model' },
            { text: '第 3 章｜存储层次与数据搬运', link: '/device/03-memory-hierarchy' },
            { text: '第 4 章｜Kernel、Tiling 与流水线', link: '/device/04-kernel-tiling-pipeline' },
            { text: '第 5 章｜FLOPs、带宽与 Roofline', link: '/device/05-flops-bandwidth-roofline' },
            { text: '第 6 章｜Eager、计算图与图编译', link: '/device/06-eager-graph-compilation' },
            { text: '第 7 章｜Stream、Event 与异步执行', link: '/device/07-stream-event-async' },
            { text: '第 8 章｜数据类型、布局与融合', link: '/device/08-dtype-layout-fusion' },
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
            { text: '第 1 章｜MoE 与 Expert Parallel', link: '/parallel/01-moe-expert-parallel' },
            { text: '第 2 章｜效率设计（选修）', link: '/parallel/02-efficient-llm-design' },
            { text: '第 3 章｜系统分析', link: '/parallel/03-llm-systems-analysis' },
            { text: '第 4 章｜分布式基础', link: '/parallel/04-distributed-basics' },
            { text: '第 5 章｜数据并行与 DDP', link: '/parallel/05-ddp' },
            { text: '第 6 章｜Tensor Parallel', link: '/parallel/06-tensor-parallel' },
            { text: '第 7 章｜Pipeline Parallel', link: '/parallel/07-pipeline-parallel' },
            { text: '第 8 章｜Context Parallel', link: '/parallel/08-context-parallel' },
            { text: '第 9 章｜ZeRO、FSDP 与混合并行', link: '/parallel/09-zero-fsdp' },
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
            { text: '第 1 章｜Collective 语义与代价模型', link: '/collective/01-collective-semantics-cost' },
            { text: '第 2 章｜Ring AllReduce 推导', link: '/collective/02-ring-allreduce' },
            { text: '第 3 章｜Tree、RD 与算法选择', link: '/collective/03-tree-algorithms-selection' },
            { text: '第 4 章｜AG、RS 与 AlltoAll', link: '/collective/04-gather-scatter-alltoall' },
            { text: '第 5 章｜拓扑、分层与重叠', link: '/collective/05-topology-hierarchical-overlap' },
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
              text: '昇腾', collapsed: true,
              items: [
                { text: '第 0 章｜认识昇腾与 CANN', link: '/ascend/00-ascend-cann' },
                { text: '第 1 章｜Runtime 与任务执行', link: '/ascend/01-runtime-task-execution' },
                { text: '速查｜CANN 常用命令', link: '/ascend/cann常用命令' },
                { text: '笔记｜CANN Learning Hub', link: '/ascend/cann-learning-hub' }
              ]
            },
            {
              text: 'Ascend C', collapsed: true,
              items: [
                { text: '第 0-2 章｜算子开发（待写·二梯队）', link: '/ascend/' }
              ]
            },
            {
              text: 'HCCL', collapsed: true,
              items: [
                { text: '第 0 章｜从 PyTorch 走向 HCCL', link: '/ascend/00-pytorch-to-hccl' },
                {
                  text: '课程｜HCCL 源码学习', link: '/ascend/hccl-source', collapsed: true,
                  items: [
                    { text: '0｜HCCL 全景与仓库地图', link: '/ascend/hccl-source/00-hccl-repo-map' },
                    { text: '1｜软件架构：分层与对外 API', link: '/ascend/hccl-source/01-architecture-layering' },
                    { text: '2｜通信域、Rank 与 RankGraph', link: '/ascend/hccl-source/02-comm-domain-rank-graph' },
                    { text: '3｜通信原语与同步机制', link: '/ascend/hccl-source/03-primitives-and-sync' },
                    { text: '4｜通信引擎与任务执行', link: '/ascend/hccl-source/04-comm-engines' },
                    { text: '5｜集合算法与 selector 选择器', link: '/ascend/hccl-source/05-coll-algorithms' },
                    { text: '6｜AllReduce 调用链走读', link: '/ascend/hccl-source/06-allreduce-call-chain' },
                    { text: '7｜executor 与 template', link: '/ascend/hccl-source/07-executor-template' },
                    { text: '8｜资源地基与 dlsym 解耦', link: '/ascend/hccl-source/08-resources-dlsym' },
                    { text: '9｜MC2 自定义算子框架', link: '/ascend/hccl-source/09-mc2-custom-ops' },
                    { text: '课程总结｜源码阅读地图', link: '/ascend/hccl-source/summary' }
                  ]
                }
              ]
            },
            {
              text: 'HCOMM', collapsed: true,
              items: [
                {
                  text: '课程｜HCOMM 源码学习', link: '/ascend/hcomm-source', collapsed: true,
                  items: [
                    { text: '0｜HCOMM 全景与仓库地图', link: '/ascend/hcomm-source/00-hcomm-repo-map' },
                    { text: '1｜编程模型：通信、并发与拓扑', link: '/ascend/hcomm-source/01-prog-models' },
                    { text: '2｜控制面：通信域一生', link: '/ascend/hcomm-source/02-control-plane-communicator' },
                    { text: '3｜数据面：原语与资源', link: '/ascend/hcomm-source/03-data-plane-primitives' },
                    { text: '4｜AICPU 算子七步', link: '/ascend/hcomm-source/04-aicpu-op-dev' },
                    { text: '5｜三引擎对比', link: '/ascend/hcomm-source/05-engine-comparison' },
                    { text: '6｜实战与第一个算子', link: '/ascend/hcomm-source/06-examples-first-op' },
                    { text: '课程总结｜源码阅读地图', link: '/ascend/hcomm-source/summary' }
                  ]
                }
              ]
            }
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
            { text: 'VS Code 快捷键速查', link: '/tools-productivity/vscode-shortcuts' },
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
            { text: '《金刚经》研读笔记', link: '/essays/《金刚经》研读笔记' },
            { text: '《金字塔原理》读书笔记', link: '/essays/《金字塔原理》读书笔记' }
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
