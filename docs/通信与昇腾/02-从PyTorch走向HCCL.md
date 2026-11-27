# 第 2 章｜从 PyTorch 走向 HCCL

> 本章目标：打通 `torch.distributed` → torch_npu → HCCL 的完整调用链——说清 torch_npu 作为"桥"注册了什么、ProcessGroup 抽象怎样把框架的通信请求交给 HCCL、Stream/Notify 怎样桥接两个世界；最终能独立追踪一次 `dist.all_reduce` 从 Python 到通信库入口的每一步。

## 本章导学

::: tip 本章只记住 3 件事
1. **torch_npu 是一座三件套的桥**：注册 `npu` 设备（`.npu()`）、把 aten 算子映射到 aclnn 两段式（计算路径）、提供 `torch.npu.Stream/Event` 与 Runtime 的对应（异步路径）——[第 0 章](00-认识昇腾与CANN.md)分层图里那一层的放大。
2. **ProcessGroup 是通信的门面**：`init_process_group(backend="hccl")` 创建 `ProcessGroupHCCL`，其内部建立 HCCL 通信域（HcclComm）；`dist.all_reduce` 沿 Python → c10d → ProcessGroupHCCL 下降到 HCCL 入口。
3. **Stream 是两个世界的缝合线**：框架把当前流（或专用通信流）传给 HCCL，集合通信任务挂上流异步执行——"下发 ≠ 执行"与重叠编排都建立在这条线上。
:::

## 本章单元

- **0-0（约 15 分钟）**：[torch_npu：框架与 CANN 之间的桥](#_0-0-torch-npu-框架与-cann-之间的桥)
- **0-1（约 15 分钟）**：[torch.distributed 的门面：ProcessGroup](#_0-1-torch-distributed-的门面-processgroup)
- **0-2（约 15 分钟）**：[Stream 与 Notify 的桥接](#_0-2-stream-与-notify-的桥接)
- **0-3（约 15 分钟）**：[一次 AllReduce 的完整旅程](#_0-3-一次-allreduce-的完整旅程)
- **本章总结**：[追踪一次计算与集合通信的交汇点](#本章总结)

## 这一章在整条路线中的位置

```text
《并行策略》：并行策略为什么产生通信（需求侧）
《昇腾与 HCCL》昇腾篇第 0-1 章：CANN 分层与 Runtime 执行模型（平台地基）
        ↓
本章：框架调用链——torch_npu / ProcessGroup / Stream 三座桥（缝合处）
        ↓
HCCL 源码 6：AllReduce 在通信库内部的下潜（供给侧）
```

模型视角的同一条链在 [《模型全景》01-3：从 PyTorch 调用到 HCCL](/模型与系统/01-从模型到集合通信.md#_01-3-从-pytorch-调用到-hccl)——本章是它的工程完全体；读完后，[HCCL 源码 6 调用链走读](源码课程/06-AllReduce调用链走读.md) 的入口就亮了。

---

[开始单元 0-0 →](#_0-0-torch-npu-框架与-cann-之间的桥)

## 0-0｜torch_npu：框架与 CANN 之间的桥

::: info 本单元目标
读完后，你能够说出 **torch_npu 的三件注册（设备、算子、流/事件）与 `y = a + b` 的完整计算路径**，并对照 CUDA 生态说出每个组件的对应物。
:::

### 先记住 3 个结论

1. **`import torch_npu` 的瞬间发生三件事**：把 `npu` 注册为 PyTorch 后端设备、把 aten 算子映射到昇腾实现、把 `torch.npu.Stream/Event` 接到 CANN Runtime——桥就这么搭起来。
2. **计算路径的形态是"适配器 + 两段式"**：aten 算子 → torch_npu 的适配层（op-plugin）→ aclnn 两段式（[第 1 章](01-Runtime与任务执行.md#_1-3-一次算子调用怎样被下发和完成)）——框架不知道 NPU，torch_npu 负责翻译。
3. **生态对照一一映射**：torch.cuda ↔ torch.npu、cuBLAS/cuDNN ↔ AOL 算子库、NCCL ↔ HCCL——概念相通，只是名字不同。

### 1. 桥搭在哪：三件注册

[第 0 章 0-1](00-认识昇腾与CANN.md#_0-1-cann-软件栈与模型执行路径) 的分层图里，torch_npu 位于框架与 CANN 之间。它做的事可以归为三件注册：

| 注册 | 效果 | 你看到的 API |
| --- | --- | --- |
| **设备** | `npu` 成为合法后端 | `x.npu()`、`torch.npu.set_device()` |
| **算子** | aten 算子有 NPU 实现 | `y = a + b` 直接在 NPU 上跑 |
| **流/事件** | 异步原语对应 Runtime | `torch.npu.Stream`、`torch.npu.Event` |

注册的机制是 PyTorch 的 **dispatcher**：每个算子调用按设备键（device key）路由——`npu` 键的实现全部住在 torch_npu 里。框架本体不需要知道昇腾的存在。

安装自检（详见 [CANN 常用命令速查第 6 节](CANN常用命令.md)）：

```python
import torch, torch_npu
print(torch.npu.is_available())      # True 才有下文
print(torch_npu.__version__)
```

### 2. `y = a + b` 的计算路径

一个 npu tensor 的加法，从 Python 到硬件：

```text
y = a + b                    # Python：熟悉的写法，无任何 NPU 痕迹
  ↓ PyTorch dispatcher（设备键 = npu）
aten::add 的 NPU 实现          # torch_npu 适配层（op-plugin）
  ↓ 检查/布局处理后
aclnnAddGetWorkspaceSize      # ① 准备段（纯 Host，可缓存复用）
aclrtMalloc(workspace)
aclnnAdd(executor, stream)    # ② 执行段：任务挂上当前流
  ↓ Runtime → Driver（回扣第 1 章双时间线）
AI Core 执行
```

对照 [第 1 章 1-3](01-Runtime与任务执行.md#_1-3-一次算子调用怎样被下发和完成)：torch_npu 的适配层只是在这条已学过的路径**前面加了一层翻译**——它把 aten 的语义（shape 推导、广播规则、autograd 注册）翻译成对 aclnn 的调用。**框架侧的每一步，都落在第 1 章的地基上。**

### 3. 与 CUDA 生态的对照

完整对照表见笔记 [CANN Learning Hub](cann-learning-hub.md)，本章最常用的四行：

| 维度 | NVIDIA | 昇腾 |
| --- | --- | --- |
| 框架适配 | torch.cuda | **torch_npu** |
| 算子库 | cuBLAS / cuDNN | AOL（ops-math/nn/...） |
| 通信库 | NCCL（ProcessGroupNCCL） | **HCCL（ProcessGroupHCCL）** |
| 数据搬运 | cudaMemcpy | aclrtMemcpy |

### 本单元小结

- torch_npu = 设备注册 + 算子映射 + 流/事件对接，三件套把桥搭起来；
- 计算路径 = dispatcher → 适配层 → aclnn 两段式——第 1 章地基上的一层翻译；
- 生态四行对照：torch.cuda / cuBLAS / NCCL / cudaMemcpy 的昇腾对应物。

### 参考资料

- [昇腾社区：PyTorch 框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)
- [第 0 章 0-1：CANN 软件栈与执行路径](00-认识昇腾与CANN.md#_0-1-cann-软件栈与模型执行路径)
- [第 1 章 1-3：一次算子调用的旅程](01-Runtime与任务执行.md#_1-3-一次算子调用怎样被下发和完成)
- [CANN Learning Hub 笔记：CUDA 对照表](cann-learning-hub.md)

---

下一单元将进入 **03-2：torch.distributed 的门面：ProcessGroup**。

## 0-1｜torch.distributed 的门面：ProcessGroup

::: info 本单元目标
读完后，你能够说出 **torch.distributed 的四层结构、`init_process_group(backend="hccl")` 背后发生的三步、`dist.all_reduce` 的下降路径**——从框架 API 一路到 HCCL 入口。
:::

### 先记住 3 个结论

1. **四层结构**：Python API（`dist.*`）→ c10d 前端 → **ProcessGroup 抽象** → 后端实现（`ProcessGroupHCCL`）——抽象层让同一份训练代码跑在 NCCL/HCCL/Gloo 上。
2. **init 三步**：rendezvous（人齐）→ 创建 ProcessGroupHCCL → 内部 `HcclCommInitCluster` 建立通信域（[HCCL 源码 2](源码课程/02-控制面-通信域的一生.md) 的框架侧入口）。
3. **下降路径**：`dist.all_reduce` → `ProcessGroupHCCL::allreduce`（取流、检查）→ HCCL 集合通信接口——框架只决定"何时、对谁、在哪个流"，"怎么跑"全权交给 HCCL。

### 1. 为什么需要一层门面

[《并行策略》04-2](/执行与并行/04-分布式基础.md#_04-2-通信域-谁和谁是一伙的) 定义过 PG 的语义：**成员 + 后端 + 操作**。框架侧需要一个类承接这三样，并且**同一份代码不感知具体后端**——这层门面就是 `ProcessGroup`：

```text
torch.distributed（Python API）      ← 你写的代码
        ↓
c10d 前端（校验、分发）
        ↓
ProcessGroup（抽象基类）             ← "成员 + 操作"的接口
        ↓
ProcessGroupNCCL / ProcessGroupHCCL / ProcessGroupGloo   ← "后端"
```

`backend="hccl"` 这个字符串经过注册表（registry）解析到 `ProcessGroupHCCL`——与第 0 章的"框架适配层"同一套设计哲学：**接口归框架，实现归生态**。

### 2. init_process_group 背后

```python
dist.init_process_group(backend="hccl", init_method="env://")
```

三步展开：

1. **Rendezvous**：按 `MASTER_ADDR/PORT`（env://）签到，确定 rank/world_size——[《并行策略》04-1](/执行与并行/04-分布式基础.md#_04-1-从进程到-rank-人口登记) 的"人口登记"；
2. **构造 ProcessGroupHCCL**：注册表解析 backend 字符串，创建后端实例；
3. **建立通信域**：ProcessGroupHCCL 内部调用 HCCL 的初始化接口（`HcclCommInitCluster` 一族），为这个 PG 建立 **HcclCommunicator**——rank、world_size、成员关系在此固化（源码视角见 [HCCL 源码 2：通信域与 RankGraph](源码课程/02-控制面-通信域的一生.md)）。

`dist.new_group(ranks=[...])` 重复第 2、3 步——**一个子 PG 对应一个 HCCL 通信域**，`group=` 参数最终路由到对应的 communicator。

### 3. dist.all_reduce 的下降路径

```text
dist.all_reduce(t, op=SUM)                  # Python
  ↓ c10d 绑定层
ProcessGroupHCCL::allreduce(t, opts)        # C++ 后端实现
  ├─ 检查：dtype/contiguous/设备一致
  ├─ 取流：当前 npu 流（或 PG 管理的通信流）
  ↓
HcclAllReduce(dst, src, count, dtype, op, comm, stream)   # HCCL 入口
  ↓（进入通信库内部——HCCL 源码 6 的领域）
```

框架侧的职责到 `HcclAllReduce` 为止：**何时发（调用时机）、对谁发（comm）、在哪个流（stream）**；算法选择、任务编排、数据搬运全部下沉给 HCCL。这正是 [《并行策略》05-2](/执行与并行/05-数据并行与DDP.md#_05-2-从朴素-ddp-到生产级-ddp) 那句"框架侧何时发，通信库侧怎么跑"的分界线。

### 本单元小结

- ProcessGroup = "成员 + 后端 + 操作"的框架侧类，注册表把字符串解析成实现；
- init 三步：签到 → 建 PG → 建通信域；new_group 一组一域；
- 下降路径止于 HcclAllReduce——职责分界线清晰可述。

### 参考资料

- [PyTorch Distributed 文档](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [《并行策略》04-2：通信域（语义视角）](/执行与并行/04-分布式基础.md#_04-2-通信域-谁和谁是一伙的)
- [HCCL 源码 2：通信域与 RankGraph](源码课程/02-控制面-通信域的一生.md)

---

下一单元将进入 **03-3：Stream 与 Notify 的桥接**。

## 0-2｜Stream 与 Notify 的桥接

::: info 本单元目标
读完后，你能够说出 **框架流怎样传进 HCCL、Event 与 Notify 各管哪一层同步、通信计算重叠的完整闭环**——两个世界的异步体系如何缝合。
:::

### 先记住 3 个结论

1. **流是缝合线**：`torch.npu.Stream` 底层就是 `aclrtStream`（[第 1 章 1-2](01-Runtime与任务执行.md#_1-2-stream、event、task-与异步执行)）——ProcessGroupHCCL 把流对象原样传给 HCCL，通信任务挂上这条队列异步执行。
2. **两层同步，各管各的**：框架的 `Event`（record/wait）编排**计算流之间**的依赖；HCCL 内部的 **Notify**（[HCCL 源码 3](源码课程/03-数据面-原语与资源.md)）负责**设备间通信的握手**——名字不同、层次不同。
3. **重叠闭环三方各出一块**：框架分流（计算流/通信流）、通信库挂流（不占计算单元）、硬件并行（链路与 AI Core 独立）——缺一环都重不起来。

### 1. 流的传递：一条线穿三层

```text
torch.npu.Stream (Python)
  ↕ 同一对象
c10/npu 的 Stream 封装 (C++)
  ↕ 底层句柄
aclrtStream (Runtime)          ← 第 1 章的"任务单行道"
  ↕ 作为参数传入
HcclAllReduce(..., stream)     ← 通信任务挂上同一条单行道
```

关键推论：**`HcclAllReduce` 传哪个流，通信就排在哪个流的队列里**——

- 传**当前流**：通信与计算严格保序（同流 FIFO）——简单正确，但无重叠；
- 传**独立通信流**：通信与计算并行——需要 Event 编排依赖（[《单卡执行系统》07-4](/执行与并行/07-Stream、Event与异步执行.md#_07-4-重叠的艺术-让设备闲不下来) 的三条件）。

DDP 的选择是后者：反向 hook 触发时把桶的 AllReduce 发到通信流，与剩余层的反向计算重叠（[《并行策略》05-4](/执行与并行/05-数据并行与DDP.md#_05-4-通信计算重叠-把等待藏起来)）——**那条"通信流"就是 torch_npu Stream，最终以 aclrtStream 的身份进入 HCCL**。

### 2. Event 与 Notify：两层同步不要混

| | 框架 Event | HCCL Notify |
| --- | --- | --- |
| 属于 | torch.npu / Runtime | HCCL 内部机制 |
| 编排什么 | **同一设备内**流与流的依赖（record/wait） | **跨设备**通信参与方的握手 |
| 你直接用吗 | 用（`record_stream`/`wait_stream`） | 不用（通信库内部） |

一次跨流 AllReduce 的完整同步图：

```text
计算流:  [反向算梯度]──record(e)──............[wait(e2) 用结果]
通信流:        └──wait(e)──[AllReduce on stream]──record(e2)
                                   └ HCCL 内部：各 rank 的 Notify 握手 → 数据交换
```

外圈（Event）是框架看得见的编排；内圈（Notify）是 HCCL 在设备侧让 N 个 rank 步调一致的机制（[HCCL 源码 3](源码课程/03-数据面-原语与资源.md)）——**外圈管"什么时候能开始"，内圈管"大家到齐没有"**。

### 本单元小结

- 流是三层的缝合线：torch.npu.Stream = aclrtStream = HcclAllReduce 的 stream 参数；
- 传哪个流决定通信排在哪条队列——重叠的开关就在这个参数；
- Event（框架层）与 Notify（通信库层）各管一层同步，勿混用概念。

### 参考资料

- [第 1 章 1-2：Stream、Event 与异步执行](01-Runtime与任务执行.md#_1-2-stream、event、task-与异步执行)
- [《单卡执行系统》07-4：重叠的艺术](/执行与并行/07-Stream、Event与异步执行.md#_07-4-重叠的艺术-让设备闲不下来)
- [HCCL 源码 3：通信原语与同步机制](源码课程/03-数据面-原语与资源.md)

---

下一单元将进入 **03-4：一次 AllReduce 的完整旅程**。

## 0-3｜一次 AllReduce 的完整旅程

::: info 本单元目标
读完后，你能够**独立追踪 `dist.all_reduce` 从 DDP 的 backward hook 到 HCCL 入口的每一步**，说出每层职责与源码地图，为 HCCL 源码 6 的下潜做好准备。
:::

### 先记住 3 个结论

1. **旅程六站**：DDP hook 触发 → `dist.all_reduce` → ProcessGroupHCCL（检查+取流）→ `HcclAllReduce` 入口 → HCCL 内部（选算法→编排→传输）→ 任务挂流异步返回。
2. **每层只做自己的事**：框架管时机与对象、门面管检查与取流、HCCL 管执行——追踪调用链的过程就是反复问"这一步是谁的职责"。
3. **本章止步于入口**：入口之后的 selector/executor/复盘写作模板.md/Transport 下潜，全部属于 [HCCL 与 HCOMM 源码学习](源码课程/index.md)（单元 5/6）。

### 1. 总装：六站时序

以 DDP 训练一步中的一桶梯度为例（回扣 [《并行策略》05-3](/执行与并行/05-数据并行与DDP.md#_05-3-bucket-分桶的艺术)）：

```text
① DDP backward hook：某桶梯度就绪 → 触发该桶通信
        ↓
② dist.all_reduce(bucket, op=SUM)          # Python（c10d）
        ↓
③ ProcessGroupHCCL::allreduce              # 检查 dtype/contiguous/设备
   取流：当前流或通信流（03-3 的缝合线）
        ↓
④ HcclAllReduce(dst, src, count, dtype,    # HCCL 入口：comm + stream
                op, comm, stream)
        ↓
⑤ HCCL 内部（黑盒预览）：                    #
   通信域 → selector 选算法（Ring/Tree/...，HCCL 源码 5）
   → executor 编排任务 → transport 走链路 → 引擎执行（HCCL 源码 4）
        ↓
⑥ 任务挂上 stream 异步返回                    # Host 继续跑（第 1 章双时间线）
   后续：框架 Event 同步 / stream sync 兜底
```

追踪的技巧：**每一站问三个问题**——数据从哪来（参数）、交给谁（下一站）、挂在哪条时间线上（流）。

### 2. 源码阅读地图

本章（框架侧）与源码课程（库内侧）的分界线就在第 ④⑤ 站之间：

| 想看什么 | 去哪 |
| --- | --- |
| torch_npu 适配层、ProcessGroupHCCL | torch_npu 仓库（[框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)） |
| HCCL 公开 API（HcclAllReduce 等） | [HCCL API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html) |
| 入口之后的下潜 | **[HCCL 源码 6：AllReduce 调用链走读](源码课程/06-AllReduce调用链走读.md)** |
| 算法选择的依据 | [HCCL 源码 5：集合通信算法与代价模型](源码课程/05-算法与selector.md) |
| 通信域与 RankGraph | [HCCL 源码 2](源码课程/02-控制面-通信域的一生.md) |
| 引擎与执行 | [HCCL 源码 4](源码课程/04-通信引擎与三引擎编排.md) |

### 本单元小结

- 六站时序 = 框架五站 + 库内一站（黑盒预览）；
- 追踪技巧：每站问"数据从哪来、交给谁、挂哪条流"；
- 本章与 HCCL 源码 6 的分界线即两个仓库的分界线——继续下潜的路标已立好。

### 参考资料

- [HCCL 官方 API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)
- [HCCL 源码 6：AllReduce 调用链走读](源码课程/06-AllReduce调用链走读.md)
- [《并行策略》05-3：Bucket 分桶](/执行与并行/05-数据并行与DDP.md#_05-3-bucket-分桶的艺术)

---

下一单元将进入 **本章总结：追踪一次计算与集合通信的交汇点**。

## 本章总结

四个单元走完，把框架侧的调用链拼成一张总图，并核对本章开头立下的目标。

### 一张图收束整章

```text
你的代码            y = a + b          dist.all_reduce(bucket)
                        │                      │
框架适配层          torch_npu（三件注册）        torch.distributed
                    dispatcher → op-plugin   c10d → ProcessGroupHCCL
                        │                      │ 检查 · 取流 · comm
算子/通信入口       aclnnAdd 两段式          HcclAllReduce(…, comm, stream)
                        │                      │
昇腾执行栈          ──── Runtime：Stream / Event / Notify ────
                        │                      │
                    AI Core 计算            selector → executor
                    （第 1 章）             → transport → 引擎（HCCL 源码 4-7）
```

上半段（计算路径）由 03-1 铺设，下半段（通信路径）由 03-2/03-3 铺设，03-4 把两段在**同一条 Stream 时间线**上缝合——这就是"计算与集合通信的交汇点"的准确含义。

### 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| 三件注册 | 设备、算子、流/事件——torch_npu 的桥 | 03-1 |
| dispatcher | 按设备键路由，框架不感知后端 | 03-1 |
| op-plugin | aten 语义 → aclnn 两段式的翻译层 | 03-1 |
| 四层结构 | Python → c10d → ProcessGroup → 后端实现 | 03-2 |
| init 三步 | 签到 → 建 PG → 建通信域（HcclComm） | 03-2 |
| 职责分界线 | 框架管"何时/对谁/哪条流"，HCCL 管"怎么跑" | 03-2 |
| 流的缝合线 | torch.npu.Stream = aclrtStream = stream 参数 | 03-3 |
| Event vs Notify | 框架层流间依赖 vs 库内跨设备握手 | 03-3 |
| 六站时序 | hook → dist → PGHCCL → 入口 → 库内 → 挂流返回 | 03-4 |

### 章节自检清单

实操向追加：

- [ ] 我知道排障时哪一层看什么日志（torch_npu 日志 vs HCCL 日志，见[常用命令速查](CANN常用命令.md)第 8 节）

### 与主线的接口

本章是 **HCCL 开发主线全部主线材料的收官**：模型全景 → 训练与推理系统 → 单卡执行系统 → 并行策略 → 集合通信 → 昇腾平台与调用链——需求侧（为什么通信）与供给侧（怎么实现）在本章会师。

下一步的两个方向：

1. **下潜源码**：[HCCL 与 HCOMM 源码学习](源码课程/index.md)——从单元 0 双仓地图开始，单元 6 承接本章的第四站，单元 9～11 进阶到自定义通信算子开发；
2. **回望理论**：带着调用链的具体问题回读 [《集合通信》](/通信与昇腾/index.md)（例如"selector 在哪一步用到了 α-β-γ"）。

### 最终参考资料

- [HCCL 开源仓库](https://gitcode.com/cann/hccl) 与 [HCOMM 通信基础库](https://gitcode.com/cann/hcomm)
- [HCCL 官方 API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)
- [昇腾社区：PyTorch 框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)
- [《模型全景》01-3：从 PyTorch 调用到 HCCL（模型视角）](/模型与系统/01-从模型到集合通信.md#_01-3-从-pytorch-调用到-hccl)
- [《并行策略》第 4-5 章：分布式基础与 DDP](/执行与并行/04-分布式基础.md)
