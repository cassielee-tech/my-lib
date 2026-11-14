import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  lang: 'zh-CN',
  title: 'Cassie’s Stack',
  description: '记录 HCCL 开发主线课程、后端、效率工具与个人思考',
  base: '/my-lib/',
  cleanUrls: true,
  lastUpdated: true,
  sitemap: {
    hostname: 'https://cassielee-tech.github.io/my-lib/'
  },
  markdown: {
    math: true
  },
  mermaid: {
    flowchart: { wrappingWidth: 360 }
  },
  vite: {
    optimizeDeps: {
      include: ['fastdom', 'mermaid']
    }
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/my-lib/favicon.svg' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'Cassie’s Stack' }],
    [
      'script',
      {},
      "(function(){try{var d=document.documentElement;if(localStorage.getItem('vp-sidebar-collapsed')==='1')d.classList.add('sidebar-collapsed');if(localStorage.getItem('vp-aside-collapsed')==='1')d.classList.add('aside-collapsed')}catch(e){}})()"
    ]
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      {
        text: '模型与系统',
        items: [
          { text: '专栏目录', link: '/model-systems/' },
          { text: '⚡ 速通版｜上篇·模型全景', link: '/model-systems/00-速通-上篇' },
          { text: '⚡ 速通版｜下篇·训练与推理', link: '/model-systems/00-速通-下篇' },
          { text: '下一专栏 → 执行与并行', link: '/execution-parallel/' }
        ]
      },
      {
        text: '执行与并行',
        items: [
          { text: '专栏目录', link: '/execution-parallel/' },
          { text: '⚡ 速通版｜上篇·单卡执行', link: '/execution-parallel/00-速通-上篇' },
          { text: '⚡ 速通版｜下篇·并行策略', link: '/execution-parallel/00-速通-下篇' },
          { text: '下一专栏 → 通信与昇腾', link: '/comm-ascend/' }
        ]
      },
      {
        text: '通信与昇腾',
        items: [
          { text: '专栏目录', link: '/comm-ascend/' },
          { text: '⚡ 速通版｜集合通信', link: '/comm-ascend/00-速通' },
          { text: 'HCCL 与 HCOMM 源码课程', link: '/comm-ascend/hccl-hcomm' },
          { text: '下一专栏 → 实战复盘', link: '/practice/' }
        ]
      },
      { text: '实战复盘', link: '/practice/' },
      { text: '后端知识', link: '/backend/' },
      { text: '工具与效率', link: '/tools-productivity/' },
      { text: '个人杂谈', link: '/essays/' }
    ],

    sidebar: {
      '/model-systems/': [
        {
          text: '模型与系统',
          items: [
            { text: '专栏目录', link: '/model-systems/' },
            { text: '⚡ 速通版｜上篇·模型全景', link: '/model-systems/00-速通-上篇' },
            { text: '⚡ 速通版｜下篇·训练与推理', link: '/model-systems/00-速通-下篇' },
            { text: '← 返回首页', link: '/' },
            {
              text: '上篇｜模型全景（第 1-6 章）', collapsed: false,
              items: [
                { text: '第 1 章｜从模型到集合通信', link: '/model-systems/01-landscape' },
                { text: '第 2 章｜张量、计算图与反向传播', link: '/model-systems/02-tensor-autograd' },
                { text: '第 3 章｜模型输入', link: '/model-systems/03-tokenization-embedding' },
                { text: '第 4 章｜Self-Attention', link: '/model-systems/04-self-attention' },
                { text: '第 5 章｜完整 Decoder Block', link: '/model-systems/05-decoder-block' },
                { text: '第 6 章｜RoPE 与长上下文（选修）', link: '/model-systems/06-rope-long-context' }
              ]
            },
            {
              text: '下篇｜训练与推理系统（第 1-6 章）', collapsed: false,
              items: [
                { text: '第 1 章｜训练循环', link: '/model-systems/01-training-loop' },
                { text: '第 2 章｜训练显存与计算优化', link: '/model-systems/02-training-memory-compute' },
                { text: '第 3 章｜推理与 KV Cache（选修）', link: '/model-systems/03-inference-kv-cache' },
                { text: '第 4 章｜后训练与对齐（选修）', link: '/model-systems/04-post-training-alignment' },
                { text: '第 5 章｜推理引擎与 KV Cache（选修）', link: '/model-systems/05-inference-engine' },
                { text: '第 6 章｜PD 调度与推理指标（选修）', link: '/model-systems/06-pd-scheduling' }
              ]
            },
            { text: '下一专栏 → 执行与并行', link: '/execution-parallel/' }
          ]
        }
      ],
      '/execution-parallel/': [
        {
          text: '执行与并行',
          items: [
            { text: '专栏目录', link: '/execution-parallel/' },
            { text: '⚡ 速通版｜上篇·单卡执行', link: '/execution-parallel/00-速通-上篇' },
            { text: '⚡ 速通版｜下篇·并行策略', link: '/execution-parallel/00-速通-下篇' },
            { text: '← 返回首页', link: '/' },
            {
              text: '上篇｜单卡执行系统（第 1-8 章）', collapsed: false,
              items: [
                { text: '第 1 章｜全景与性能分析', link: '/execution-parallel/01-landscape-performance' },
                { text: '第 2 章｜GPU/NPU 执行模型', link: '/execution-parallel/02-execution-model' },
                { text: '第 3 章｜存储层次与数据搬运', link: '/execution-parallel/03-memory-hierarchy' },
                { text: '第 4 章｜Kernel、Tiling 与流水线', link: '/execution-parallel/04-kernel-tiling-pipeline' },
                { text: '第 5 章｜FLOPs、带宽与 Roofline', link: '/execution-parallel/05-flops-bandwidth-roofline' },
                { text: '第 6 章｜Eager、计算图与图编译', link: '/execution-parallel/06-eager-graph-compilation' },
                { text: '第 7 章｜Stream、Event 与异步执行', link: '/execution-parallel/07-stream-event-async' },
                { text: '第 8 章｜数据类型、布局与融合', link: '/execution-parallel/08-dtype-layout-fusion' },
                { text: '专题｜主流 AI 加速卡全景', link: '/execution-parallel/accelerator-cards-2026' }
              ]
            },
            {
              text: '下篇｜并行策略（第 1-9 章）', collapsed: false,
              items: [
                { text: '第 1 章｜MoE 与 Expert Parallel', link: '/execution-parallel/01-moe-expert-parallel' },
                { text: '第 2 章｜效率设计（选修）', link: '/execution-parallel/02-efficient-llm-design' },
                { text: '第 3 章｜系统分析', link: '/execution-parallel/03-llm-systems-analysis' },
                { text: '第 4 章｜分布式基础', link: '/execution-parallel/04-distributed-basics' },
                { text: '第 5 章｜数据并行与 DDP', link: '/execution-parallel/05-ddp' },
                { text: '第 6 章｜Tensor Parallel', link: '/execution-parallel/06-tensor-parallel' },
                { text: '第 7 章｜Pipeline Parallel', link: '/execution-parallel/07-pipeline-parallel' },
                { text: '第 8 章｜Context Parallel', link: '/execution-parallel/08-context-parallel' },
                { text: '第 9 章｜ZeRO、FSDP 与混合并行', link: '/execution-parallel/09-zero-fsdp' }
              ]
            },
            { text: '下一专栏 → 通信与昇腾', link: '/comm-ascend/' }
          ]
        }
      ],
      '/comm-ascend/': [
        {
          text: '通信与昇腾',
          items: [
            { text: '专栏目录', link: '/comm-ascend/' },
            { text: '⚡ 速通版｜集合通信', link: '/comm-ascend/00-速通' },
            { text: '← 返回首页', link: '/' },
            {
              text: '上篇｜集合通信（岗位核心）', collapsed: false,
              items: [
                { text: '第 1 章｜Collective 语义与代价模型', link: '/comm-ascend/01-collective-semantics-cost' },
                { text: '第 2 章｜Ring AllReduce 推导', link: '/comm-ascend/02-ring-allreduce' },
                { text: '第 3 章｜Tree、RD 与算法选择', link: '/comm-ascend/03-tree-algorithms-selection' },
                { text: '第 4 章｜AG、RS 与 AlltoAll', link: '/comm-ascend/04-gather-scatter-alltoall' },
                { text: '第 5 章｜拓扑、分层与重叠', link: '/comm-ascend/05-topology-hierarchical-overlap' }
              ]
            },
            {
              text: '下篇｜昇腾平台', collapsed: false,
              items: [
                { text: '第 0 章｜认识昇腾与 CANN', link: '/comm-ascend/00-ascend-cann' },
                { text: '第 1 章｜Runtime 与任务执行', link: '/comm-ascend/01-runtime-task-execution' },
                { text: '第 2 章｜从 PyTorch 走向 HCCL', link: '/comm-ascend/02-pytorch-to-hccl' },
                { text: '速查｜CANN 常用命令', link: '/comm-ascend/cann常用命令' },
                { text: '笔记｜CANN Learning Hub', link: '/comm-ascend/cann-learning-hub' }
              ]
            },
            {
              text: '源码课程｜HCCL 与 HCOMM', collapsed: true,
              items: [
                { text: '课程导学｜HCCL 与 HCOMM 源码学习', link: '/comm-ascend/hccl-hcomm' },
                { text: '0｜双仓全景与调用链旅程', link: '/comm-ascend/hccl-hcomm/00-repo-map' },
                { text: '1｜软件架构与六族 API', link: '/comm-ascend/hccl-hcomm/01-architecture-layering' },
                { text: '2｜控制面：通信域的一生', link: '/comm-ascend/hccl-hcomm/02-control-plane' },
                { text: '3｜数据面：原语与资源', link: '/comm-ascend/hccl-hcomm/03-data-plane' },
                { text: '4｜通信引擎与三引擎编排', link: '/comm-ascend/hccl-hcomm/04-comm-engines' },
                { text: '5｜算法与 selector', link: '/comm-ascend/hccl-hcomm/05-coll-algorithms' },
                { text: '6｜AllReduce 调用链走读', link: '/comm-ascend/hccl-hcomm/06-allreduce-call-chain' },
                { text: '7｜executor 与 template', link: '/comm-ascend/hccl-hcomm/07-executor-template' },
                { text: '8｜资源地基与 dlsym', link: '/comm-ascend/hccl-hcomm/08-resources-dlsym' },
                { text: '9｜自定义算子七步流程', link: '/comm-ascend/hccl-hcomm/09-custom-op-dev' },
                { text: '10｜MC2 框架与官方样例', link: '/comm-ascend/hccl-hcomm/10-mc2-custom-ops' },
                { text: '11｜实战与第一个算子', link: '/comm-ascend/hccl-hcomm/11-examples-first-op' },
                { text: '课程总结｜源码阅读地图', link: '/comm-ascend/hccl-hcomm/summary' }
              ]
            },
            { text: '下一专栏 → 实战复盘', link: '/practice/' }
          ]
        }
      ],
      '/practice/': [
        {
          text: '实战复盘',
          items: [
            { text: '专栏目录', link: '/practice/' },
            { text: '← 返回首页', link: '/' },
            { text: '写作模板｜复盘手记', link: '/practice/template' },
            { text: '1｜UBOE/UB_RTP 建链端口注入实战', link: '/practice/01-uboe-ubrtp-port-injection' },
            { text: '2｜Reduce 64bit 特殊规约分级算法', link: '/practice/02-reduce-64bit-hierarchical-algorithm' },
            { text: '3｜AllGatherV/Reduce 零拷贝实战', link: '/practice/03-allgatherv-reduce-zerocopy' }
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
            { text: 'Linux 服务器常用命令速查', link: '/tools-productivity/linux-server-commands' },
            { text: 'Git 实用速查', link: '/tools-productivity/git-cheatsheet' },
            { text: 'VS Code 快捷键速查', link: '/tools-productivity/vscode-shortcuts' },
            { text: 'opencode 使用教程', link: '/tools-productivity/opencode-tutorial' },
            { text: 'Ghostty 快捷键速查', link: '/tools-productivity/ghostty-shortcuts' },
            { text: 'macOS 快捷键速查', link: '/tools-productivity/macos-shortcuts' },
            { text: 'Vim 常用按键与技巧', link: '/tools-productivity/vim-essentials' }
          ]
        }
      ],
      '/essays/': [
        {
          text: '个人杂谈',
          items: [
            { text: '专栏目录', link: '/essays/' },
            { text: '← 返回首页', link: '/' },
            {
              text: '读书笔记',
              collapsed: true,
              items: [
                { text: '《金刚经》研读笔记', link: '/essays/《金刚经》研读笔记' },
                { text: '《金字塔原理》读书笔记', link: '/essays/《金字塔原理》读书笔记' },
                { text: '《认知觉醒》读书笔记', link: '/essays/《认知觉醒》读书笔记' },
                { text: '《当下的力量》读书笔记', link: '/essays/《当下的力量》读书笔记' }
              ]
            },
            {
              text: 'AI 观察',
              collapsed: true,
              items: [
                { text: '2026 热门 AI Agent 产品地图', link: '/essays/01-popular-agents' },
                { text: 'AI Agent 工程化知识树', link: '/essays/agent-engineering-knowledge-tree' },
                { text: 'AI Coding：从代码补全到智能体协作', link: '/essays/ai-coding-codearts' },
                { text: '实用提示词收集', link: '/essays/实用提示词收集' }
              ]
            },
            {
              text: '生活与健康',
              collapsed: true,
              items: [
                { text: '《健身解剖完全手册》学习笔记', link: '/essays/《健身解剖完全手册》学习笔记' },
                { text: 'ADHD 日常管理指南', link: '/essays/ADHD日常管理指南' }
              ]
            }
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
}))
