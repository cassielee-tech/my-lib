# 第 4 章｜分布式基础：Rank、通信域与拓扑

> 本课目标：搞清进程、设备、Rank、World Size、Process Group 的确切含义与相互关系；掌握 node-major 排号与拓扑感知排布的因果；能用 torchrun 把一个最小分布式任务跑起来，并按清单排障。

## 本章导学

::: tip 本课只记住 3 件事
1. **一进程一卡，Rank 是全局工号**：进程是 Host 侧的"员工"，Rank 是它在全组的唯一编号，`local_rank` 是机内工位号——启动器负责"人口登记"（rendezvous + 环境变量注入）。
2. **Process Group 决定"谁和谁一伙"**：集合通信发生在组内而非全世界；混合并行的本质就是**一个 Rank 身兼多组身份**（TP 组在机内、DP 组跨机）。
3. **编号必须贴着拓扑排**：node-major 排号让通信组落进机内高速域——**乱排号的代价是每层通信都挤机间慢链路**。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。本章是第 5-9 章六种并行策略的全部地基，概念多而计算少，适合拿纸画出"编号表"边推边读。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](#⚡-速通-约-5-分钟)：一个"新公司开业"的类比讲完整章，再回来按单元深入。

- [ ] 我能画出 2 机 16 卡的 rank / local_rank / node_rank 编号表
- [ ] 我能为 TP=8 × DP=4 的混合并行写出每个 rank 的组身份
- [ ] 我知道 torchrun 卡在 init 时按什么顺序排查

## 本章单元

- **04-1（约 15 分钟）**：[从进程到 Rank：人口登记](#_04-1-从进程到-rank-人口登记)
- **04-2（约 15 分钟）**：[通信域：谁和谁是一伙的](#_04-2-通信域-谁和谁是一伙的)
- **04-3（约 15 分钟）**：[拓扑感知排布：把编号贴到物理上](#_04-3-拓扑感知排布-把编号贴到物理上)
- **04-4（约 15 分钟）**：[工程实操：跑起来与排障](#_04-4-工程实操-跑起来与排障)

- **本章总结**：[练习、自测与串联](#本章总结)

## 这一课在整条路线中的位置

```text
《模型全景》第 1 章 + 《并行策略》第 3 章：模型侧的并行速览
        ↓
本章：系统侧地基——进程 / 编号 / 分组 / 排布
        ↓
第 5-9 章：DDP / TP / PP / CP / ZeRO 逐一展开
        ↓
《集合通信》：这些"组"上跑的原语与算法
```

《集合通信》第 1 章给过 Rank 与通信域的**语义视角**（谁在说话、对谁说）；本章补上**系统视角**——这些编号和分组在进程、设备与物理网络上怎么落地。HCCL 的通信域与 RankGraph（[HCCL 源码 2](../ascend/hccl-source/02-comm-domain-rank-graph.md)）正是本章概念的源码化身。

---

[开始单元 04-1 →](#_04-1-从进程到-rank-人口登记)

## ⚡ 速通（约 5 分钟）

> 适用：时间紧，或完整版跟不动时。这页用"新公司开业"讲完整章，读完抓住 80% 的主干。任何一节想深入，拉到文末点对应单元。

### 1. 工号与签到：一进程一卡

新公司开业，第一件事是**人事登记**：

- 每位员工 = 一个**进程**；每人一张专属工位（一张卡）——**一进程一卡**。不让一个人占两张工位（管不过来），也不让两人挤一张（会打架/OOM）；
- 入职先到前台签到（`MASTER_ADDR:PORT` 就是前台地址），**人齐了才开工**——这就是 rendezvous。有人堵在路上没签到，全员干等，这就是最常见的"卡在 init"；
- 签完到每人领到身份牌：**全局工号**（rank：0 号、1 号……）、**本楼层工位号**（local_rank）、**公司总人数**（world_size）。工号 0 和工号 8 都可能是"3 楼 0 号工位"——不同楼层各有一套本地编号。

谁组织这一切？**启动器 torchrun**——它负责拉人（拉起进程）、发身份牌（注入环境变量）、守前台（组织签到）。

::: details 小测 1：为什么"一进程一卡"而不是一个进程管全部卡？
一个进程管多卡 = 所有 Host 侧工作（下发算子、管理内存）挤在一个脑袋里，还一崩全崩；一进程一卡各自独立调度、故障隔离——这是分布式训练的默认组织方式。
:::

### 2. 项目组：谁和谁一伙

公司默认有个**全员大群**（world group）——`init_process_group` 一建就是它，不指定组的集合通信都在大群里吼。

但真干活要按**项目组**（Process Group）：视觉组 8 个人拉个小群，群里的事（集合通信）只在群内广播，别的组根本听不见。关键三条：

1. 项目组 = **成员名单 + 沟通工具**（后端：HCCL 就是昇腾的"公司内线"）；
2. **一人可以身兼多组**：你可以同时在"视觉组"（机内 8 卡的 TP 组）和"跨部门评审组"（跨机的 DP 组）——混合并行的本质就是每人多重身份；
3. 组怎么划分，决定了**谁和谁之间要传数据**——这就引出下一节。

::: details 小测 2：`dist.all_reduce(t, group=tp_group)` 里不传 group 会怎样？
作用于默认的全员大群（world group）——全公司都参与这次 AllReduce，而不是只有你那 8 人小组。
:::

### 3. 就近组队：排号的隐形价值

同一层楼喊话**秒到**（机内 HCCS 专线，百 GB/s）；跨楼层要走电梯间（机间网卡，慢近一个量级）。

所以排工位时有大学问：**node-major 排号** = 按楼层整层分配（3 楼拿工号 0-7，4 楼拿 8-15）——项目组恰好同层，沟通全走楼内喊话 ✅。

反面教材：HR 图省事"轮流发牌"（round-robin），视觉组 8 人散在 4 个楼层——**每次组会全员跑电梯**，每层通信都挤慢链路 ❌。最阴险的是：**代码、算法、通信量一模一样，只是组员落位错了——任务能跑，就是慢**。这种 bug 不报错，只在 Profiling 里现形。

::: details 小测 3：任务能跑但通信耗时占了 60%，代码审查没问题——先查什么？
先查启动参数的排号方式：是不是 rank 分配让通信组横跨了机器（round-robin/乱序）？node-major 一行参数的事，量级的收益。
:::

### 3 句话带走

1. **一进程一卡 + 工号体系**：rank 全局、local_rank 机内，torchrun 负责拉人/发牌/签到；
2. **项目组（PG）决定谁和谁通信**：混合并行 = 一个 rank 身兼多组；
3. **node-major 排号让组落进机内快链路**：排号错误不报错、只变慢——免费的性能藏在启动参数里。

### 黑话小词典

| 术语 | 人话 |
| --- | --- |
| rank / local_rank | 全局工号 / 本楼层工位号 |
| world_size | 公司总人数（= 总卡数） |
| world group | 全员大群（默认组） |
| Process Group | 项目组：名单 + 沟通工具（后端） |
| rendezvous | 前台签到，人齐才开工 |
| node-major | 按楼层整层发工号，项目组恰好同层 |
| backend | 沟通工具：HCCL = 昇腾公司内线 |

### 想深入？

每节 15 分钟，按需点开，不必按顺序全读：

| 单元 | 讲什么（白话） | 什么时候需要它 |
| --- | --- | --- |
| [04-1 进程与 Rank](#_04-1-从进程到-rank-人口登记) | 一进程一卡的理由 + 编号表 + torchrun | 要搞懂启动和环境变量 |
| [04-2 通信域](#_04-2-通信域-谁和谁是一伙的) | PG 三要素 + 混合并行的身份系统 | 要读懂并行策略代码里的 group= |
| [04-3 拓扑排布](#_04-3-拓扑感知排布-把编号贴到物理上) | node-major vs round-robin 的账 | 要解释"能跑但慢" |
| [04-4 实操排障](#_04-4-工程实操-跑起来与排障) | 15 行骨架 + 排障清单 | 要动手跑分布式任务 |
| [本章总结](#本章总结) | 练习 + 10 道自测 | 检验是否真正掌握 |

## 04-1｜从进程到 Rank：人口登记

::: info 本单元目标
围绕 **"一进程一卡 + Rank 编号体系"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

分布式训练的第一步不是通信，而是**把人组织起来**：谁来干活、每人叫什么、总共多少人。

### 1. 为什么是"一进程一卡"

一次 16 卡训练 = 16 个**独立进程**，每个进程绑定一张卡。为什么不用一个进程管多张卡、或者多线程？

| 方案 | 问题 |
| --- | --- |
| 单进程多线程 | Python GIL 让 Host 侧调度串行化；一个线程崩溃全队陪葬 |
| 单进程多卡 | 所有卡的 Host 工作挤在一个进程里（算子下发、内存管理互相争抢）；故障域巨大 |
| **多进程，一进程一卡** ✅ | 每进程独立调度、独立故障域；与启动器/环境变量体系天然配合 |

回扣 [《昇腾与 HCCL》1-0](../ascend/01-runtime-task-execution.md#_1-0-host、device-与异构计算)：每个进程就是一套完整的 Runtime 样板——自己的 `aclrtSetDevice`、自己的 Context、自己的 Stream。**进程是 Host 侧的隔离单位，卡是 Device 侧的计算单位，绑定关系由编号决定**（04-3 展开）。

### 2. Rank 体系与启动器

四个编号一次记住：

| 名字 | 含义 | 范围 |
| --- | --- | --- |
| **rank** | 全局工号，全组唯一 | `0 .. world_size-1` |
| **local_rank** | 机内工位号 | `0 .. n-1`（n = 每机卡数） |
| **node_rank** | 机器编号 | `0 .. 节点数-1` |
| **world_size** | 总进程数 = 总卡数 | 例如 16 |

**谁来做人口登记？** 启动器（`torchrun` / `mpirun`）。它做三件事：

1. **拉起进程**：在每台机器上 spawn n 个训练进程；
2. **注入环境变量**：每个进程出生自带身份证——`RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`；
3. **组织签到（rendezvous）**：所有进程向 `MASTER_ADDR:PORT` 报到，人齐后训练才开始——没签上到的进程会一直等待，这是最常见的"卡住"现场（04-4 展开）。

```text
机器 0（node_rank=0）              机器 1（node_rank=1）
rank 0  local_rank 0              rank 8   local_rank 0
rank 1  local_rank 1              rank 9   local_rank 1
……                                ……
rank 7  local_rank 7              rank 15  local_rank 7
                world_size = 16
```

注意这张表：**同 local_rank 的进程在不同机器上**（rank 0 和 rank 8 都是 local 0）——它们在 DP 里常常是一组（04-2 展开）。

昇腾上跑 `torch_npu`：启动方式不变（`torchrun`），后端换成 `hccl`；`mpirun` 生态同样可用。语义层的第一课在 [《集合通信》01-1](../collective/01-collective-semantics-cost.md#_01-1-通信的坐标系-rank、通信域与消息)——本章给它补上进程与操作系统的落地层。

## 04-2｜通信域：谁和谁是一伙的

::: info 本单元目标
围绕 **Process Group 的定义与用途** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

人登记完了，但集合通信从不"对全世界广播"——它只发生在**组内**。

### 3. Process Group：成员 + 后端 + 操作

**Process Group（进程组，PG）= 一组成员 rank + 一个通信后端 + 一组可用的集合操作。**

- `init_process_group(backend=...)` 创建**默认组**（world group）：全体 rank 的大群；
- `new_group(ranks=[0,1,2,3])` 创建**子组**：只含指定成员的小群；
- 集合操作不传 `group` 参数时作用于默认组，传参时作用于子组：`dist.all_reduce(t, group=tp_group)`。

| 后端 | 适用 | 一句话 |
| --- | --- | --- |
| **HCCL** | 昇腾 NPU | 本课程的主角，`torch_npu` 下 `backend="hccl"` |
| NCCL | NVIDIA GPU | 生态对照物，概念几乎一一对应 |
| gloo | CPU | 调试/小规模 CPU 通信的备胎 |

::: tip 语义与实现的分工
《集合通信》01-1 讲过"通信域"的**语义**（谁在说话、对谁说）；本章的 PG 是它在框架层的**实现**；再往下一层，HCCL 里的通信域与 RankGraph（[HCCL 源码 2](../ascend/hccl-source/02-comm-domain-rank-graph.md)）是它在通信库里的**数据结构**——同一个概念的三层化身。
:::

### 4. 为什么需要子组：混合并行的身份系统

如果只有 world group，那么**每一次** AllReduce 都是全员参与——但 TP 只想让同机 8 卡通信，DP 只想让数据并行的伙伴同步梯度。子组是混合并行的骨架：

```text
32 卡 = TP 8 × DP 4，node-major 排号（每机 8 卡）：

tp_group[i] = [8i, 8i+1, …, 8i+7]     ← 每组恰好一台机器内部
dp_group[j] = [j, 8+j, 16+j, 24+j]    ← 每组 4 台机器各出一卡
```

于是一个 rank 有**双重身份**：`rank 13` = TP 组 1 的成员 5（13 = 8×1+5）**同时**是 DP 组 5 的成员 1（13 = 8×1+5 → dp 里它是 13//8=1 号）。它在这两个组里各自参与各自的集合通信，互不打扰。

**一个关键直觉**：组怎么划分，决定了通信发生在**哪几张卡之间**——而卡与卡之间的链路快慢悬殊。划分是逻辑决定，快慢是物理决定，两者怎么咬合？下一单元。

## 04-3｜拓扑感知排布：把编号贴到物理上

::: info 本单元目标
围绕 **"node-major 排号为什么快"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

编号是软件给的，链路是硬件长的。**排号 = 让逻辑分组对齐物理拓扑。**

### 5. 两层世界与 node-major 排号

回扣 [《集合通信》05-1](../collective/05-topology-hierarchical-overlap.md#_05-1-拓扑-通信的物理地图)：机内是 HCCS 全互联专线（百 GB/s），机间是网卡走交换网（几十 GB/s，差近一个量级）。

**node-major（按节点整块分配）**：机器 0 拿 rank 0-7，机器 1 拿 rank 8-15……

```text
node-major：tp_group[1] = [8,9,…,15] → 全部落在一台机器里 → 走机内专线 ✅
round-robin（轮流发牌）：rank 0,2,4,… 发给机器 0
                          → tp_group = [0,1,2,…,7] 横跨 4 台机器 → 层层通信挤机间 ❌
```

两种排号下，**算法、代码、通信量完全相同**，唯一区别是"组员落在哪"——而每层 TP 通信都要过一次机间链路时，慢一个量级不是修辞，是账本（回扣《集合通信》05-2 的分层流量账）。

::: warning 一句话结论
**排号错误不会报错，只会变慢**——这正是它危险的地方：任务"能跑"，Profiling 里通信占比刺眼，根因却在启动脚本的一行参数里。
:::

### 6. Local Rank 与设备绑定

编号落位后，每个进程的开场三步：

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch_npu.npu.set_device(local_rank)     # ① 绑卡：本进程认领本机第 local_rank 张卡
x = x.npu()                              # ② 此后所有 .npu() 都落到这张卡
dist.init_process_group(backend="hccl")  # ③ 进组：领工号、连后端
```

两个容易踩的坑：

1. **可见性重映射**：`ASCEND_RT_VISIBLE_DEVICES=4,5` 时，进程里的"卡 0"是物理 4 号卡（回扣 [《昇腾与 HCCL》1-0](../ascend/01-runtime-task-execution.md#_1-0-host、device-与异构计算)）——排查"数据怎么跑到别的卡上了"先查这个；
2. **一卡多进程是反模式**：两个进程绑同一张卡 = 显存翻倍 + 调度互踩，通常立刻 OOM 或性能崩塌。

顺带把链路接完整：**进程（Host 单位）→ local_rank 绑卡（Device 单位）→ PG 分组（通信单位）→ 拓扑对齐（物理约束）**。HCCL 启动时做的拓扑探测与分级通信（[HCCL 源码 4](../ascend/hccl-source/04-comm-engines.md)），本质就是自动替你做本章这套对齐。

## 04-4｜工程实操：跑起来与排障

::: info 本单元目标
围绕 **最小分布式骨架与排障清单** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

### 7. 最小可跑骨架（torch_npu + HCCL）

十五行代码把本章全部概念走一遍——**绑卡、进组、报身份、做一次集合通信验证**：

```python
import os
import torch
import torch_npu
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(local_rank)          # ① 绑卡
    dist.init_process_group(backend="hccl")       # ② 进组（签到 + 建 HCCL 通信域）

    rank, world = dist.get_rank(), dist.get_world_size()
    print(f"rank {rank}/{world} on device {local_rank}")   # ③ 报身份

    t = torch.ones(4, device=f"npu:{local_rank}")
    dist.all_reduce(t)                            # ④ 全员验证：结果应为 world
    assert t[0].item() == world
    print(f"rank {rank}: all_reduce OK")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

启动（单机 8 卡）：

```bash
torchrun --nproc_per_node=8 demo.py
```

多机时每台机器各跑一条 `torchrun`，多出三个参数：`--nnodes`、`--node_rank`、`--master_addr/--master_port`——这正是 04-1 的编号表从纸面变成进程的过程。

### 8. 排障清单：按依赖顺序查

分布式故障的第一原则：**先查"人口登记"，再查物理资源，最后查日志**。

| 症状 | 排查点 | 命令/动作 |
| --- | --- | --- |
| 卡在 init 几分钟无输出 | rendezvous 没签上：端口占用/网络不通 | `ss -tlnp \| grep 29500`；ping master；换 MASTER_PORT |
| 启动即报卡数不匹配 | `nproc_per_node × nnodes` ≠ 可见卡数 | `npu-smi info`；核对 `ASCEND_RT_VISIBLE_DEVICES` |
| 各进程行为错乱 | 环境变量缺失或不一致 | `env \| grep -E "RANK\|WORLD\|MASTER"`（详见 [CANN 常用命令速查](../ascend/cann常用命令.md)第 7 节） |
| OOM / 设备忙 | 一卡被多进程绑定 | `npu-smi info` 看每卡进程数；核对每进程 set_device |
| 报错栈驴唇不对马嘴 | 异步报错延迟浮出（回扣 [《昇腾与 HCCL》1-2](../ascend/01-runtime-task-execution.md#_1-2-stream、event、task-与异步执行)） | 开 HCCL 日志定位（速查第 8 节），看首次报错的 rank |

::: tip 本章收束
地基三件套就位：**编号**（rank/local_rank/world_size）、**分组**（Process Group）、**排布**（node-major 贴拓扑）。第 5 章起，每种并行策略的第一件事都是回答"怎么切张量"，第二件事就是"在本章的哪一组上发哪个集合通信"。
:::

## 本章总结

> 返回：[第 4 章首页](#)

### 把本章串成一条主线

请先合上各单元正文，沿"登记 → 分组 → 排布 → 实操"复述整章；遇到断点时，再回到对应单元查阅。

```text
进程与编号（04-1）：一进程一卡 · rank=全局工号 · local_rank=机内工位
   torchrun 三件事：拉进程 / 注环境变量 / rendezvous 签到
        ↓
通信域（04-2）：PG = 成员 + 后端 + 操作
   world group 默认大群 · new_group 建子组
   混合并行 = 一个 rank 身兼多组（TP 组 + DP 组）
        ↓
拓扑排布（04-3）：node-major 让组落进机内高速域
   排号错误不报错、只变慢（量级差异）
        ↓
实操（04-4）：15 行骨架 = 绑卡/进组/报身份/all_reduce 验证
   排障顺序：人口登记 → 物理资源 → 日志
```

### 9. 动手练习

#### 练习 1：编号表推演

4 机 × 8 卡（32 卡），node-major 排号。写出 rank 19 的 `node_rank`、`local_rank`；写出它所在的 TP=8 组与 DP=4 组的全部成员；判断两个组分别落在机内还是跨机。

#### 练习 2：分组设计

1 台 8 卡机器，要求 TP=4 × DP=2。列出全部 tp_group 与 dp_group 的成员表，并回答：两种组是否都能全落机内？rank 5 的双重身份是什么？

#### 练习 3：排障决策

`torchrun --nproc_per_node=8` 启动后卡在 `init_process_group` 约 3 分钟无任何输出。列出至少 3 个排查点，按你执行的顺序排列，并给出每步对应的命令。

### 10. 自测题

1. 为什么分布式训练选择"一进程一卡"而不是多线程或单进程多卡？
2. rank、local_rank、node_rank、world_size 四者的定义与取值范围？
3. torchrun 做的三件事是什么？rendezvous 发生在哪一步？
4. Process Group 由哪三个要素构成？
5. 默认组（world group）怎么创建？子组怎么创建？
6. 集合操作怎样指定作用于子组？不指定时作用于谁？
7. 32 卡 TP=8 × DP=4 时，rank 13 在 TP 组和 DP 组里分别是什么身份？
8. node-major 排号是什么？round-robin 排号会让 TP 组付出什么代价？
9. 为什么说"排号错误不会报错，只会变慢"？这带来什么排障难点？
10. 进程绑卡的三行开场代码是什么？`ASCEND_RT_VISIBLE_DEVICES` 会造成什么错觉？

::: details 自测答案

1. 多线程受 GIL 限制且故障域大；单进程多卡 Host 侧争抢且故障域大；一进程一卡调度独立、故障隔离、与启动器体系配合。
2. rank：全局编号 0..world_size-1；local_rank：机内编号 0..每机卡数-1；node_rank：机器编号 0..节点数-1；world_size：总进程数（= 总卡数）。
3. 拉起进程、注入环境变量（RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/PORT）、组织签到 rendezvous；签到发生在 init_process_group 阶段。
4. 成员 rank 集合 + 通信后端（HCCL/NCCL/gloo）+ 可用的集合操作集。
5. 默认组由 init_process_group 创建（全体成员）；子组由 new_group(ranks=[...]) 创建。
6. 传 group 参数（如 dist.all_reduce(t, group=tp_group)）；不指定时作用于默认 world group。
7. TP：13 = 8×1+5，是 TP 组 1 的成员 5；DP：13//8 = 1，是 DP 组 5 的成员 1。
8. node-major：按节点整块分配连续 rank（机器 0 拿 0-7）；round-robin 让 TP 组横跨多机，每层通信都挤机间慢链路（差近一个量级）。
9. 算法/代码/通信量都不变，只是组员落位错了——任务能跑但 Profiling 通信占比刺眼，根因藏在启动参数里而非代码里。
10. 读 LOCAL_RANK → set_device(local_rank) → 之后 .npu() 落到这张卡；可见设备列表会重映射编号：进程里的"卡 0"可能是物理 4 号卡。

:::

### 11. 本课小结

- 进程是 Host 隔离单位、卡是 Device 计算单位，rank 体系把两者钉在一起；
- PG 决定"谁和谁通信"：混合并行 = 一个 rank 身兼多组；
- node-major 排号让逻辑分组对齐物理拓扑——排号是免费的性能；
- 十五行骨架（绑卡/进组/报身份/all_reduce）是所有分布式代码的最小内核；
- 排障第一原则：先人口登记、再物理资源、最后日志。

### 参考资料

- [PyTorch Distributed 文档](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [《集合通信》01-1：Rank 与通信域（语义视角）](../collective/01-collective-semantics-cost.md#_01-1-通信的坐标系-rank、通信域与消息)
- [《集合通信》05-1：拓扑地图](../collective/05-topology-hierarchical-overlap.md#_05-1-拓扑-通信的物理地图)
- [HCCL 源码 2：通信域与 RankGraph](../ascend/hccl-source/02-comm-domain-rank-graph.md)
- [CANN 常用命令速查：分布式环境与排障](../ascend/cann常用命令.md)

下一章开始逐一展开六种并行策略，第一站是最经典的**数据并行与 DDP**——梯度为什么需要 AllReduce、Bucket 和通信计算重叠怎样工作，正是本章"world group 上的集体行动"。

---

<!-- chapter-navigation -->
