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
          { text: '专栏目录', link: '/模型与系统/' },
          { text: '模型全景', link: '/模型与系统/01-landscape' },
          { text: '训练与推理系统', link: '/模型与系统/01-training-loop' }
        ]
      },
      {
        text: '执行与并行',
        items: [
          { text: '专栏目录', link: '/执行与并行/' },
          { text: '单卡执行系统', link: '/执行与并行/01-landscape-performance' },
          { text: '并行策略', link: '/执行与并行/01-moe-expert-parallel' }
        ]
      },
      {
        text: '通信与昇腾',
        items: [
          { text: '专栏目录', link: '/通信与昇腾/' },
          { text: '集合通信', link: '/通信与昇腾/01-collective-semantics-cost' },
          { text: '昇腾与 HCCL', link: '/通信与昇腾/00-ascend-cann' },
          { text: '源码课程', link: '/通信与昇腾/源码课程/' }
        ]
      },
      { text: '实战复盘', link: '/实战复盘/' },
      { text: '参考手册', link: '/通信与昇腾/参考手册/' },
      { text: '后端知识', link: '/后端知识/' },
      { text: '工具与效率', link: '/工具与效率/' },
      { text: '个人杂谈', link: '/个人杂谈/' }
    ],

    sidebar: {
      '/模型与系统/': [
        {
          text: '模型与系统',
          items: [
            { text: '专栏目录', link: '/模型与系统/' },
            { text: '⚡ 速通版｜上篇·模型全景', link: '/模型与系统/00-速通-上篇' },
            { text: '⚡ 速通版｜下篇·训练与推理', link: '/模型与系统/00-速通-下篇' },
            {
              text: '上篇｜模型全景（第 1-6 章）', collapsed: false,
              items: [
                { text: '第 1 章｜从模型到集合通信', link: '/模型与系统/01-landscape' },
                { text: '第 2 章｜张量、计算图与反向传播', link: '/模型与系统/02-tensor-autograd' },
                { text: '第 3 章｜模型输入', link: '/模型与系统/03-tokenization-embedding' },
                { text: '第 4 章｜Self-Attention', link: '/模型与系统/04-self-attention' },
                { text: '第 5 章｜完整 Decoder Block', link: '/模型与系统/05-decoder-block' },
                { text: '第 6 章｜RoPE 与长上下文（选修）', link: '/模型与系统/06-rope-long-context' }
              ]
            },
            {
              text: '下篇｜训练与推理系统（第 1-6 章）', collapsed: false,
              items: [
                { text: '第 1 章｜训练循环', link: '/模型与系统/01-training-loop' },
                { text: '第 2 章｜训练显存与计算优化', link: '/模型与系统/02-training-memory-compute' },
                { text: '第 3 章｜推理与 KV Cache（选修）', link: '/模型与系统/03-inference-kv-cache' },
                { text: '第 4 章｜后训练与对齐（选修）', link: '/模型与系统/04-post-training-alignment' },
                { text: '第 5 章｜推理引擎与 KV Cache（选修）', link: '/模型与系统/05-inference-engine' },
                { text: '第 6 章｜PD 调度与推理指标（选修）', link: '/模型与系统/06-pd-scheduling' }
              ]
            },
          ]
        }
      ],
      '/执行与并行/': [
        {
          text: '执行与并行',
          items: [
            { text: '专栏目录', link: '/执行与并行/' },
            { text: '⚡ 速通版｜上篇·单卡执行', link: '/执行与并行/00-速通-上篇' },
            { text: '⚡ 速通版｜下篇·并行策略', link: '/执行与并行/00-速通-下篇' },
            {
              text: '上篇｜单卡执行系统（第 1-8 章）', collapsed: false,
              items: [
                { text: '第 1 章｜全景与性能分析', link: '/执行与并行/01-landscape-performance' },
                { text: '第 2 章｜GPU/NPU 执行模型', link: '/执行与并行/02-execution-model' },
                { text: '第 3 章｜存储层次与数据搬运', link: '/执行与并行/03-memory-hierarchy' },
                { text: '第 4 章｜Kernel、Tiling 与流水线', link: '/执行与并行/04-kernel-tiling-pipeline' },
                { text: '第 5 章｜FLOPs、带宽与 Roofline', link: '/执行与并行/05-flops-bandwidth-roofline' },
                { text: '第 6 章｜Eager、计算图与图编译', link: '/执行与并行/06-eager-graph-compilation' },
                { text: '第 7 章｜Stream、Event 与异步执行', link: '/执行与并行/07-stream-event-async' },
                { text: '第 8 章｜数据类型、布局与融合', link: '/执行与并行/08-dtype-layout-fusion' },
                { text: '专题｜主流 AI 加速卡全景', link: '/执行与并行/accelerator-cards-2026' }
              ]
            },
            {
              text: '下篇｜并行策略（第 1-9 章）', collapsed: false,
              items: [
                { text: '第 1 章｜MoE 与 Expert Parallel', link: '/执行与并行/01-moe-expert-parallel' },
                { text: '第 2 章｜效率设计（选修）', link: '/执行与并行/02-efficient-llm-design' },
                { text: '第 3 章｜系统分析', link: '/执行与并行/03-llm-systems-analysis' },
                { text: '第 4 章｜分布式基础', link: '/执行与并行/04-distributed-basics' },
                { text: '第 5 章｜数据并行与 DDP', link: '/执行与并行/05-ddp' },
                { text: '第 6 章｜Tensor Parallel', link: '/执行与并行/06-tensor-parallel' },
                { text: '第 7 章｜Pipeline Parallel', link: '/执行与并行/07-pipeline-parallel' },
                { text: '第 8 章｜Context Parallel', link: '/执行与并行/08-context-parallel' },
                { text: '第 9 章｜ZeRO、FSDP 与混合并行', link: '/执行与并行/09-zero-fsdp' }
              ]
            },
          ]
        }
      ],
      '/通信与昇腾/': [
        {
          text: '通信与昇腾',
          items: [
            { text: '专栏目录', link: '/通信与昇腾/' },
            { text: '⚡ 速通版｜集合通信', link: '/通信与昇腾/00-速通' },
            {
              text: '上篇｜集合通信（岗位核心）', collapsed: false,
              items: [
                { text: '第 1 章｜Collective 语义与代价模型', link: '/通信与昇腾/01-collective-semantics-cost' },
                { text: '第 2 章｜Ring AllReduce 推导', link: '/通信与昇腾/02-ring-allreduce' },
                { text: '第 3 章｜Tree、RD 与算法选择', link: '/通信与昇腾/03-tree-algorithms-selection' },
                { text: '第 4 章｜AG、RS 与 AlltoAll', link: '/通信与昇腾/04-gather-scatter-alltoall' },
                { text: '第 5 章｜拓扑、分层与重叠', link: '/通信与昇腾/05-topology-hierarchical-overlap' }
              ]
            },
            {
              text: '下篇｜昇腾平台', collapsed: false,
              items: [
                { text: '第 0 章｜认识昇腾与 CANN', link: '/通信与昇腾/00-ascend-cann' },
                { text: '第 1 章｜Runtime 与任务执行', link: '/通信与昇腾/01-runtime-task-execution' },
                { text: '第 2 章｜从 PyTorch 走向 HCCL', link: '/通信与昇腾/02-pytorch-to-hccl' },
                { text: '速查｜CANN 常用命令', link: '/通信与昇腾/cann常用命令' },
                { text: '笔记｜CANN Learning Hub', link: '/通信与昇腾/cann-learning-hub' }
              ]
            },
            {
              text: '源码课程｜HCCL 与 HCOMM', collapsed: false,
              items: [
                { text: '课程导学｜HCCL 与 HCOMM 源码学习', link: '/通信与昇腾/源码课程/' },
                { text: '0｜双仓全景与调用链旅程', link: '/通信与昇腾/源码课程/00-repo-map' },
                { text: '1｜软件架构与六族 API', link: '/通信与昇腾/源码课程/01-architecture-layering' },
                { text: '2｜控制面：通信域的一生', link: '/通信与昇腾/源码课程/02-control-plane' },
                { text: '3｜数据面：原语与资源', link: '/通信与昇腾/源码课程/03-data-plane' },
                { text: '4｜通信引擎与三引擎编排', link: '/通信与昇腾/源码课程/04-comm-engines' },
                { text: '5｜算法与 selector', link: '/通信与昇腾/源码课程/05-coll-algorithms' },
                { text: '6｜AllReduce 调用链走读', link: '/通信与昇腾/源码课程/06-allreduce-call-chain' },
                { text: '7｜executor 与 template', link: '/通信与昇腾/源码课程/07-executor-template' },
                { text: '8｜资源地基与 dlsym', link: '/通信与昇腾/源码课程/08-resources-dlsym' },
                { text: '9｜自定义算子七步流程', link: '/通信与昇腾/源码课程/09-custom-op-dev' },
                { text: '10｜MC2 框架与官方样例', link: '/通信与昇腾/源码课程/10-mc2-custom-ops' },
                { text: '11｜实战与第一个算子', link: '/通信与昇腾/源码课程/11-examples-first-op' },
                { text: '课程总结｜源码阅读地图', link: '/通信与昇腾/源码课程/summary' }
              ]
            },
            { text: 'HCCL 参考手册 →', link: '/通信与昇腾/参考手册/' }
          ]
        }
      ],
      '/通信与昇腾/参考手册/': [
        {
          text: 'HCCL 参考手册',
              items: [
                { text: '导读', link: '/通信与昇腾/参考手册/' },
                {
                  text: '总览', collapsed: true,
                  items: [
                    { text: 'HCCL 概述', link: '/通信与昇腾/参考手册/总览/hccl-overview' },
                    { text: '算子与算法总览', link: '/通信与昇腾/参考手册/总览/ops-algos-overview' },
                    { text: 'L0/L1/L2 分级原理', link: '/通信与昇腾/参考手册/总览/l0l1l2-principle' },
                    { text: '集群组网总览', link: '/通信与昇腾/参考手册/总览/cluster-topology' }
                  ]
                },
                {
                  text: '算子参考', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/算子参考/' },
                    { text: 'AllReduce', link: '/通信与昇腾/参考手册/算子参考/all_reduce' },
                    { text: 'AllGather', link: '/通信与昇腾/参考手册/算子参考/all_gather' },
                    { text: 'AllGatherV', link: '/通信与昇腾/参考手册/算子参考/all_gather_v' },
                    { text: 'Reduce', link: '/通信与昇腾/参考手册/算子参考/reduce' },
                    { text: 'ReduceScatter', link: '/通信与昇腾/参考手册/算子参考/reduce_scatter' },
                    { text: 'ReduceScatterV', link: '/通信与昇腾/参考手册/算子参考/reduce_scatter_v' },
                    { text: 'Broadcast', link: '/通信与昇腾/参考手册/算子参考/broadcast' },
                    { text: 'Scatter', link: '/通信与昇腾/参考手册/算子参考/scatter' },
                    { text: 'AlltoAll/V/VC', link: '/通信与昇腾/参考手册/算子参考/all_to_all_v' },
                    { text: 'Send / Recv', link: '/通信与昇腾/参考手册/算子参考/send' },
                    { text: 'BatchSendRecv', link: '/通信与昇腾/参考手册/算子参考/batch_send_recv' },
                    { text: 'Barrier', link: '/通信与昇腾/参考手册/算子参考/barrier' }
                  ]
                },
                {
                  text: '算法参考', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/算法参考/' },
                    { text: 'Ring', link: '/通信与昇腾/参考手册/算法参考/ring' },
                    { text: 'Mesh/FullMesh', link: '/通信与昇腾/参考手册/算法参考/fullmesh' },
                    { text: 'RHD（递归倍增）', link: '/通信与昇腾/参考手册/算法参考/hdr' },
                    { text: 'Star', link: '/通信与昇腾/参考手册/算法参考/star' },
                    { text: 'NHR', link: '/通信与昇腾/参考手册/算法参考/nhr' },
                    { text: 'NB', link: '/通信与昇腾/参考手册/算法参考/nb' },
                    { text: 'AHC', link: '/通信与昇腾/参考手册/算法参考/ahc' },
                    { text: 'Pipeline', link: '/通信与昇腾/参考手册/算法参考/pipeline' },
                    { text: 'Pairwise', link: '/通信与昇腾/参考手册/算法参考/pairwise' },
                    { text: '机制｜selector', link: '/通信与昇腾/参考手册/算法参考/algo_selector' },
                    { text: '机制｜分层矩阵', link: '/通信与昇腾/参考手册/算法参考/algo_level_matrix' },
                    { text: '机制｜组合', link: '/通信与昇腾/参考手册/算法参考/combination' }
                  ]
                },
                {
                  text: '组网形态', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/组网形态/' },
                    { text: 'RoCE 集群', link: '/通信与昇腾/参考手册/组网形态/roce_cluster' },
                    { text: 'UB 超节点', link: '/通信与昇腾/参考手册/组网形态/ub_superpod' },
                    { text: '多平面网络', link: '/通信与昇腾/参考手册/组网形态/multi_plane' },
                    { text: '层次化混合组网', link: '/通信与昇腾/参考手册/组网形态/hierarchical' },
                    { text: '单机内部', link: '/通信与昇腾/参考手册/组网形态/server_internal' },
                    { text: '整机形态', link: '/通信与昇腾/参考手册/组网形态/board_form_factor' }
                  ]
                },
                {
                  text: '协议与传输', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/协议与传输/' },
                    { text: 'HCCS/PCIe/SIO', link: '/通信与昇腾/参考手册/协议与传输/hccs_pcie' },
                    { text: 'RoCE/RDMA', link: '/通信与昇腾/参考手册/协议与传输/roce_rdma' },
                    { text: '对称内存', link: '/通信与昇腾/参考手册/协议与传输/symmetric_mem' },
                    { text: 'UB_MEM', link: '/通信与昇腾/参考手册/协议与传输/ub_mem' },
                    { text: 'UB 网络（CTP/RTP/UBoE）', link: '/通信与昇腾/参考手册/协议与传输/ub_network' },
                    { text: 'URMA', link: '/通信与昇腾/参考手册/协议与传输/urma' }
                  ]
                },
                {
                  text: '通信域', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/通信域/' },
                    { text: '创建方式', link: '/通信与昇腾/参考手册/通信域/creation' },
                    { text: '配置', link: '/通信与昇腾/参考手册/通信域/config' }
                  ]
                },
                {
                  text: '通信引擎', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/通信引擎/' },
                    { text: 'AICPU_TS', link: '/通信与昇腾/参考手册/通信引擎/aicpu_ts' },
                    { text: 'AIV', link: '/通信与昇腾/参考手册/通信引擎/aiv' },
                    { text: 'CCU', link: '/通信与昇腾/参考手册/通信引擎/ccu' },
                    { text: 'Host CPU_TS', link: '/通信与昇腾/参考手册/通信引擎/host_cpu_ts' }
                  ]
                },
                {
                  text: '执行模式', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/执行模式/' },
                    { text: 'ACL Graph 图捕获', link: '/通信与昇腾/参考手册/执行模式/aclgraph' },
                    { text: '框架集成', link: '/通信与昇腾/参考手册/执行模式/framework_integration' }
                  ]
                },
                {
                  text: '排障手册', collapsed: true,
                  items: [
                    { text: '总入口', link: '/通信与昇腾/参考手册/排障手册/' },
                    { text: '定位流程', link: '/通信与昇腾/参考手册/排障手册/localization_flow' },
                    { text: 'EI 错误码全表', link: '/通信与昇腾/参考手册/排障手册/error_codes' },
                    { text: '案例｜精度', link: '/通信与昇腾/参考手册/排障手册/案例/精度/' },
                    { text: '案例｜建链', link: '/通信与昇腾/参考手册/排障手册/案例/建链/' },
                    { text: '案例｜整网', link: '/通信与昇腾/参考手册/排障手册/案例/整网/' },
                    { text: '案例｜coredump', link: '/通信与昇腾/参考手册/排障手册/案例/coredump/' },
                    { text: '案例｜资料配置', link: '/通信与昇腾/参考手册/排障手册/案例/资料配置/' },
                    { text: '案例｜拦截', link: '/通信与昇腾/参考手册/排障手册/案例/拦截/' },
                    { text: '案例｜AIC', link: '/通信与昇腾/参考手册/排障手册/案例/aic/' },
                    { text: '案例｜AIV', link: '/通信与昇腾/参考手册/排障手册/案例/aiv/' }
                  ]
                },
                {
                  text: '性能分析', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/性能分析/' },
                    { text: '采集与分析', link: '/通信与昇腾/参考手册/性能分析/collect_analyze' },
                    { text: '慢快与拥塞', link: '/通信与昇腾/参考手册/性能分析/slowfast_congestion' }
                  ]
                },
                {
                  text: '运行与约束', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/运行与约束/' },
                    { text: '约束', link: '/通信与昇腾/参考手册/运行与约束/constraints' },
                    { text: '环境变量', link: '/通信与昇腾/参考手册/运行与约束/env_vars' },
                    { text: '可靠性与安全', link: '/通信与昇腾/参考手册/运行与约束/reliability_security' },
                    { text: '支持与构建', link: '/通信与昇腾/参考手册/运行与约束/support_build' }
                  ]
                },
                {
                  text: '扩展机制', collapsed: true,
                  items: [
                    { text: '目录', link: '/通信与昇腾/参考手册/扩展机制/' },
                    { text: '算法插件', link: '/通信与昇腾/参考手册/扩展机制/algo_plugin' },
                    { text: 'MC2 与 RFC', link: '/通信与昇腾/参考手册/扩展机制/mc2_and_rfc' }
                  ]
                }
              ]
        }
      ],
      '/实战复盘/': [
        {
          text: '实战复盘',
          items: [
            { text: '专栏目录', link: '/实战复盘/' },
            {
              text: '模板与方法论', collapsed: false,
              items: [
                { text: '复盘写作模板｜复盘手记', link: '/实战复盘/00-模板与方法论/template' },
                { text: 'SRS/SD 文档模板', link: '/实战复盘/00-模板与方法论/01-SRS_SD文档模板' },
                { text: '需求实验方法论参考', link: '/实战复盘/00-模板与方法论/需求实验方法论参考' },
                { text: '实验用例设计与脚本写法参考', link: '/实战复盘/00-模板与方法论/实验用例设计与脚本写法经验参考' }
              ]
            },
            {
              text: '需求迭代', collapsed: false,
              items: [
                {
                  text: '26RP4/01｜Reduce-64bit 分级算法', collapsed: true,
                  items: [
                    { text: '复盘｜Reduce 64bit 特殊规约分级算法', link: '/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/' },
                    { text: '前置知识', link: '/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/02-前置知识' },
                    { text: 'SE 设计', link: '/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/03-SE设计' },
                    { text: 'SRS/SD', link: '/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/04-SRS_SD' },
                    { text: '问题对齐', link: '/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/08-问题对齐-DPU-Reduce写非root用户输出buffer' },
                    { text: '经验总结', link: '/实战复盘/01-需求迭代/26RP4/01-Reduce-64bit与特殊规约分级算法/09-经验总结' }
                  ]
                },
                {
                  text: '26RP4/02｜AllGatherV/Reduce 零拷贝', collapsed: true,
                  items: [
                    { text: '复盘｜AllGatherV/Reduce 零拷贝实战', link: '/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/' },
                    { text: '预调研', link: '/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/01-预调研' },
                    { text: '前置知识', link: '/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/02-前置知识' },
                    { text: 'SE 设计', link: '/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/03-SE设计' },
                    { text: 'SRS/SD', link: '/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/04-SRS_SD' },
                    { text: '经验总结', link: '/实战复盘/01-需求迭代/26RP4/02-AllGatherV与Reduce零拷贝模式/09-经验总结' }
                  ]
                }
              ]
            },
            {
              text: '问题修复', collapsed: false,
              items: [
                { text: '01｜UBOE/UB_RTP 建链端口注入实战', link: '/实战复盘/02-问题修复/01-uboe-ubrtp-port-injection' },
                {
                  text: '02｜HCCL aclgraph 零拷贝方案', collapsed: true,
                  items: [
                    { text: 'SE 设计', link: '/实战复盘/02-问题修复/02-HCCL-aclgraph模式零拷贝方案/03-SE设计' }
                  ]
                }
              ]
            },
            {
              text: '研讨笔记', collapsed: false,
              items: [
                {
                  text: 'AllGather Selector 算法选择策略研究', collapsed: true,
                  items: [
                    { text: '代价模型研讨（13 图）', link: '/实战复盘/03-研讨笔记/allgather-selector-算法选择策略研究/' },
                    { text: '决策树与 Corner Case', link: '/实战复盘/03-研讨笔记/allgather-selector-算法选择策略研究/allgather-selector-decision-tree' },
                    { text: '10 分钟分享', link: '/实战复盘/03-研讨笔记/allgather-selector-算法选择策略研究/分享_10分钟' }
                  ]
                }
              ]
            }
          ]
        }
      ],
      '/后端知识/': [{ text: '后端知识', items: [{ text: '目录', link: '/后端知识/' }] }],
      '/工具与效率/': [
        {
          text: '工具与效率',
          items: [
            { text: '目录', link: '/工具与效率/' },
            { text: 'VS Code + SSH + Jupyter', link: '/工具与效率/vscode-remote-ssh-jupyter' },
            { text: 'Linux 服务器常用命令速查', link: '/工具与效率/linux-server-commands' },
            { text: 'Git 实用速查', link: '/工具与效率/git-cheatsheet' },
            { text: 'VS Code 快捷键速查', link: '/工具与效率/vscode-shortcuts' },
            { text: 'opencode 使用教程', link: '/工具与效率/opencode-tutorial' },
            { text: 'Ghostty 快捷键速查', link: '/工具与效率/ghostty-shortcuts' },
            { text: 'macOS 快捷键速查', link: '/工具与效率/macos-shortcuts' },
            { text: 'Vim 常用按键与技巧', link: '/工具与效率/vim-essentials' }
          ]
        }
      ],
      '/个人杂谈/': [
        {
          text: '个人杂谈',
          items: [
            { text: '专栏目录', link: '/个人杂谈/' },
            {
              text: '读书笔记',
              collapsed: true,
              items: [
                { text: '《金刚经》研读笔记', link: '/个人杂谈/《金刚经》研读笔记' },
                { text: '《金字塔原理》读书笔记', link: '/个人杂谈/《金字塔原理》读书笔记' },
                { text: '《认知觉醒》读书笔记', link: '/个人杂谈/《认知觉醒》读书笔记' },
                { text: '《当下的力量》读书笔记', link: '/个人杂谈/《当下的力量》读书笔记' }
              ]
            },
            {
              text: 'AI 观察',
              collapsed: true,
              items: [
                { text: '2026 热门 AI Agent 产品地图', link: '/个人杂谈/01-popular-agents' },
                { text: 'AI Agent 工程化知识树', link: '/个人杂谈/agent-engineering-knowledge-tree' },
                { text: 'AI Coding：从代码补全到智能体协作', link: '/个人杂谈/ai-coding-codearts' },
                { text: '实用提示词收集', link: '/个人杂谈/实用提示词收集' }
              ]
            },
            {
              text: '生活与健康',
              collapsed: true,
              items: [
                { text: '《健身解剖完全手册》学习笔记', link: '/个人杂谈/《健身解剖完全手册》学习笔记' },
                { text: 'ADHD 日常管理指南', link: '/个人杂谈/ADHD日常管理指南' }
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
