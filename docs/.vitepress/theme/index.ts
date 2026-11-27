import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import './custom.css'

const BASE = '/my-lib/'

/**
 * 旧专栏 URL → 新合并专栏 URL。
 * 六专栏合并为三专栏后，外部引用的旧链接由 GitHub Pages 的 404.html
 * 加载本主题脚本，在此做客户端重定向。
 */
const REDIRECTS: Array<[string, string]> = [
  // 2026-11 目录中文化：英文路径 → 中文路径（前缀规则，具体在前）
  ['/comm-ascend/reference/debug/cases', '/通信与昇腾/参考手册/排障手册/案例'],
  ['/comm-ascend/reference/debug', '/通信与昇腾/参考手册/排障手册'],
  ['/comm-ascend/reference', '/通信与昇腾/参考手册'],
  ['/comm-ascend/hccl-hcomm', '/通信与昇腾/源码课程/'],
  ['/comm-ascend', '/通信与昇腾'],
  ['/model-systems', '/模型与系统'],
  ['/execution-parallel', '/执行与并行'],
  ['/practice', '/实战复盘'],
  ['/backend', '/后端知识'],
  ['/tools-productivity', '/工具与效率'],
  ['/essays', '/个人杂谈'],
  // 文件级中文化（2026-11）：旧英文文件路径 → 新中文路径（具体在前，先于目录级规则）
  ['/essays/01-popular-agents', '/个人杂谈/01-2026热门AIAgent产品地图'],
  ['/essays/agent-engineering-knowledge-tree', '/个人杂谈/AIAgent工程化知识树'],
  ['/essays/ai-coding-codearts', '/个人杂谈/AICoding-从代码补全到智能体协作'],
  ['/practice/00-模板与方法论/template', '/实战复盘/00-模板与方法论/复盘写作模板'],
  ['/practice/02-问题修复/01-uboe-ubrtp-port-injection', '/实战复盘/02-问题修复/01-UBOE与UB_RTP建链端口注入实战'],
  ['/tools-productivity/ghostty-shortcuts', '/工具与效率/Ghostty快捷键速查'],
  ['/tools-productivity/git-cheatsheet', '/工具与效率/Git实用速查'],
  ['/tools-productivity/linux-server-commands', '/工具与效率/Linux服务器常用命令速查'],
  ['/tools-productivity/macos-shortcuts', '/工具与效率/macOS快捷键速查'],
  ['/tools-productivity/opencode-tutorial', '/工具与效率/opencode使用教程'],
  ['/tools-productivity/vim-essentials', '/工具与效率/Vim常用按键与技巧'],
  ['/tools-productivity/vscode-shortcuts', '/工具与效率/VSCode快捷键速查'],
  ['/execution-parallel/01-landscape-performance', '/执行与并行/01-全景与性能分析'],
  ['/execution-parallel/02-efficient-llm-design', '/执行与并行/02-效率设计'],
  ['/execution-parallel/02-execution-model', '/执行与并行/02-GPU与NPU执行模型'],
  ['/execution-parallel/03-llm-systems-analysis', '/执行与并行/03-大模型系统拆解'],
  ['/execution-parallel/03-memory-hierarchy', '/执行与并行/03-存储层次与数据搬运'],
  ['/execution-parallel/04-distributed-basics', '/执行与并行/04-分布式基础'],
  ['/execution-parallel/04-kernel-tiling-pipeline', '/执行与并行/04-Kernel、Tiling与流水线'],
  ['/execution-parallel/05-ddp', '/执行与并行/05-数据并行与DDP'],
  ['/execution-parallel/05-flops-bandwidth-roofline', '/执行与并行/05-FLOPs、带宽与Roofline'],
  ['/execution-parallel/06-eager-graph-compilation', '/执行与并行/06-Eager、计算图与图编译'],
  ['/execution-parallel/07-stream-event-async', '/执行与并行/07-Stream、Event与异步执行'],
  ['/execution-parallel/08-dtype-layout-fusion', '/执行与并行/08-数据类型、布局与融合'],
  ['/execution-parallel/09-zero-fsdp', '/执行与并行/09-ZeRO、FSDP与混合并行'],
  ['/execution-parallel/accelerator-cards-2026', '/执行与并行/主流AI加速卡全景'],
  ['/model-systems/01-landscape', '/模型与系统/01-从模型到集合通信'],
  ['/model-systems/01-training-loop', '/模型与系统/01-训练循环'],
  ['/model-systems/02-tensor-autograd', '/模型与系统/02-张量、计算图与反向传播'],
  ['/model-systems/02-training-memory-compute', '/模型与系统/02-训练显存与计算优化'],
  ['/model-systems/03-inference-kv-cache', '/模型与系统/03-推理与KV-Cache'],
  ['/model-systems/03-tokenization-embedding', '/模型与系统/03-模型输入'],
  ['/model-systems/04-post-training-alignment', '/模型与系统/04-后训练与对齐'],
  ['/model-systems/05-decoder-block', '/模型与系统/05-完整Decoder-Block'],
  ['/model-systems/05-inference-engine', '/模型与系统/05-推理引擎与KV-Cache'],
  ['/model-systems/06-pd-scheduling', '/模型与系统/06-PD调度与推理指标'],
  ['/model-systems/06-rope-long-context', '/模型与系统/06-RoPE与长上下文'],
  ['/comm-ascend/00-ascend-cann', '/通信与昇腾/00-认识昇腾与CANN'],
  ['/comm-ascend/01-collective-semantics-cost', '/通信与昇腾/01-Collective语义与代价模型'],
  ['/comm-ascend/01-runtime-task-execution', '/通信与昇腾/01-Runtime与任务执行'],
  ['/comm-ascend/02-pytorch-to-hccl', '/通信与昇腾/02-从PyTorch走向HCCL'],
  ['/comm-ascend/02-ring-allreduce', '/通信与昇腾/02-Ring-AllReduce推导'],
  ['/comm-ascend/03-tree-algorithms-selection', '/通信与昇腾/03-Tree、RD与算法选择'],
  ['/comm-ascend/05-topology-hierarchical-overlap', '/通信与昇腾/05-拓扑、分层与重叠'],
  ['/comm-ascend/cann常用命令', '/通信与昇腾/CANN常用命令'],
  ['/comm-ascend/reference/comm/symmetric_mem', '/通信与昇腾/参考手册/协议与传输/对称内存'],
  ['/comm-ascend/reference/comm/ub_network', '/通信与昇腾/参考手册/协议与传输/UB网络'],
  ['/comm-ascend/reference/profiling/collect_analyze', '/通信与昇腾/参考手册/性能分析/采集与分析'],
  ['/comm-ascend/reference/profiling/slowfast_congestion', '/通信与昇腾/参考手册/性能分析/慢快与拥塞'],
  ['/comm-ascend/reference/overview/cluster-topology', '/通信与昇腾/参考手册/总览/集群组网总览'],
  ['/comm-ascend/reference/overview/hccl-overview', '/通信与昇腾/参考手册/总览/HCCL概述'],
  ['/comm-ascend/reference/overview/l0l1l2-principle', '/通信与昇腾/参考手册/总览/L0-L1-L2分级原理'],
  ['/comm-ascend/reference/overview/ops-algos-overview', '/通信与昇腾/参考手册/总览/算子与算法总览'],
  ['/comm-ascend/reference/execution/aclgraph', '/通信与昇腾/参考手册/执行模式/ACL-Graph图捕获'],
  ['/comm-ascend/reference/execution/framework_integration', '/通信与昇腾/参考手册/执行模式/框架集成'],
  ['/comm-ascend/reference/extension/algo_plugin', '/通信与昇腾/参考手册/扩展机制/算法插件'],
  ['/comm-ascend/reference/debug/error_codes', '/通信与昇腾/参考手册/排障手册/EI错误码全表'],
  ['/comm-ascend/reference/debug/localization_flow', '/通信与昇腾/参考手册/排障手册/定位流程'],
  ['/comm-ascend/reference/algorithms/algo_level_matrix', '/通信与昇腾/参考手册/算法参考/分层矩阵'],
  ['/comm-ascend/reference/algorithms/combination', '/通信与昇腾/参考手册/算法参考/算法组合'],
  ['/comm-ascend/reference/algorithms/hdr', '/通信与昇腾/参考手册/算法参考/RHD递归倍增'],
  ['/comm-ascend/reference/networking/board_form_factor', '/通信与昇腾/参考手册/组网形态/整机形态'],
  ['/comm-ascend/reference/networking/hierarchical', '/通信与昇腾/参考手册/组网形态/层次化混合组网'],
  ['/comm-ascend/reference/networking/multi_plane', '/通信与昇腾/参考手册/组网形态/多平面网络'],
  ['/comm-ascend/reference/networking/roce_cluster', '/通信与昇腾/参考手册/组网形态/RoCE集群'],
  ['/comm-ascend/reference/networking/server_internal', '/通信与昇腾/参考手册/组网形态/单机内部'],
  ['/comm-ascend/reference/networking/ub_superpod', '/通信与昇腾/参考手册/组网形态/UB超节点'],
  ['/comm-ascend/reference/runtime/constraints', '/通信与昇腾/参考手册/运行与约束/约束'],
  ['/comm-ascend/reference/runtime/env_vars', '/通信与昇腾/参考手册/运行与约束/环境变量'],
  ['/comm-ascend/reference/runtime/reliability_security', '/通信与昇腾/参考手册/运行与约束/可靠性与安全'],
  ['/comm-ascend/reference/runtime/support_build', '/通信与昇腾/参考手册/运行与约束/支持与构建'],
  ['/comm-ascend/reference/communicator/config', '/通信与昇腾/参考手册/通信域/配置'],
  ['/comm-ascend/reference/communicator/creation', '/通信与昇腾/参考手册/通信域/创建方式'],
  ['/comm-ascend/hccl-hcomm/00-repo-map', '/通信与昇腾/源码课程/00-双仓全景与调用链旅程'],
  ['/comm-ascend/hccl-hcomm/01-architecture-layering', '/通信与昇腾/源码课程/01-软件架构与六族API'],
  ['/comm-ascend/hccl-hcomm/02-control-plane', '/通信与昇腾/源码课程/02-控制面-通信域的一生'],
  ['/comm-ascend/hccl-hcomm/03-data-plane', '/通信与昇腾/源码课程/03-数据面-原语与资源'],
  ['/comm-ascend/hccl-hcomm/04-comm-engines', '/通信与昇腾/源码课程/04-通信引擎与三引擎编排'],
  ['/comm-ascend/hccl-hcomm/05-coll-algorithms', '/通信与昇腾/源码课程/05-算法与selector'],
  ['/comm-ascend/hccl-hcomm/06-allreduce-call-chain', '/通信与昇腾/源码课程/06-AllReduce调用链走读'],
  ['/comm-ascend/hccl-hcomm/08-resources-dlsym', '/通信与昇腾/源码课程/08-资源地基与dlsym'],
  ['/comm-ascend/hccl-hcomm/09-custom-op-dev', '/通信与昇腾/源码课程/09-自定义算子七步流程'],
  ['/comm-ascend/hccl-hcomm/10-mc2-custom-ops', '/通信与昇腾/源码课程/10-MC2框架与官方样例'],
  ['/comm-ascend/hccl-hcomm/11-examples-first-op', '/通信与昇腾/源码课程/11-实战与第一个算子'],
  ['/comm-ascend/hccl-hcomm/summary', '/通信与昇腾/源码课程/总结-源码阅读地图'],
  // 成对专栏的速通拆为上篇/下篇（须在前缀规则之前）
  ['/model/00-速通', '/模型与系统/00-速通-上篇'],
  ['/systems/00-速通', '/模型与系统/00-速通-下篇'],
  ['/device/00-速通', '/执行与并行/00-速通-上篇'],
  ['/parallel/00-速通', '/执行与并行/00-速通-下篇'],
  // practice 独立成 /实战复盘/（须在 /ascend 之前）
  ['/ascend/practice', '/实战复盘'],
  // 六专栏 → 三专栏
  ['/model', '/模型与系统'],
  ['/systems', '/模型与系统'],
  ['/device', '/执行与并行'],
  ['/parallel', '/执行与并行'],
  ['/collective', '/通信与昇腾'],
  ['/ascend', '/通信与昇腾']
]

function redirectFor(pathname: string): string | null {
  try {
    let p = decodeURIComponent(pathname)
    if (!p.startsWith(BASE)) return null
    p = p.slice(BASE.length - 1)          // 保留开头的 /
    p = p.replace(/\.html$/, '').replace(/\/$/, '')
    for (const [old, nu] of REDIRECTS) {
      if (p === old || p.startsWith(old + '/')) {
        const rest = p.slice(old.length)
        // 旧专栏目录页已合并进新专栏目录
        if (rest === '' || rest === '/index') return nu + '/'
        return nu + rest
      }
    }
  } catch {
    /* decode 失败则不重定向 */
  }
  return null
}

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp() {
    if (typeof window === 'undefined') return
    const to = redirectFor(window.location.pathname)
    if (to) {
      window.location.replace(BASE.slice(0, -1) + to + window.location.hash)
    }
  }
}
