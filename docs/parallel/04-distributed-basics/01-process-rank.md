# 单元 04-1｜从进程到 Rank：人口登记

> 所属章节：[第 4 章｜分布式基础：Rank、通信域与拓扑](../04-distributed-basics.md)

::: info 本单元目标
围绕 **"一进程一卡 + Rank 编号体系"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

分布式训练的第一步不是通信，而是**把人组织起来**：谁来干活、每人叫什么、总共多少人。

## 1. 为什么是"一进程一卡"

一次 16 卡训练 = 16 个**独立进程**，每个进程绑定一张卡。为什么不用一个进程管多张卡、或者多线程？

| 方案 | 问题 |
| --- | --- |
| 单进程多线程 | Python GIL 让 Host 侧调度串行化；一个线程崩溃全队陪葬 |
| 单进程多卡 | 所有卡的 Host 工作挤在一个进程里（算子下发、内存管理互相争抢）；故障域巨大 |
| **多进程，一进程一卡** ✅ | 每进程独立调度、独立故障域；与启动器/环境变量体系天然配合 |

回扣 [《昇腾与 HCCL》02-1](../../ascend/02-runtime-task-execution/01-host-device.md)：每个进程就是一套完整的 Runtime 样板——自己的 `aclrtSetDevice`、自己的 Context、自己的 Stream。**进程是 Host 侧的隔离单位，卡是 Device 侧的计算单位，绑定关系由编号决定**（04-3 展开）。

## 2. Rank 体系与启动器

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

昇腾上跑 `torch_npu`：启动方式不变（`torchrun`），后端换成 `hccl`；`mpirun` 生态同样可用。语义层的第一课在 [《集合通信》01-1](../../collective/01-collective-semantics-cost/01-rank-domain-message.md)——本章给它补上进程与操作系统的落地层。

---

[继续单元 04-2 →](./02-process-groups.md)
