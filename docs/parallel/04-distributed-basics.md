# 第 4 章｜分布式基础：Rank、通信域与拓扑

> 本课目标：搞清进程、设备、Rank、World Size、Process Group 的确切含义与相互关系；掌握 node-major 排号与拓扑感知排布的因果；能用 torchrun 把一个最小分布式任务跑起来，并按清单排障。

## 本章导学

::: tip 本课只记住 3 件事
1. **一进程一卡，Rank 是全局工号**：进程是 Host 侧的"员工"，Rank 是它在全组的唯一编号，`local_rank` 是机内工位号——启动器负责"人口登记"（rendezvous + 环境变量注入）。
2. **Process Group 决定"谁和谁一伙"**：集合通信发生在组内而非全世界；混合并行的本质就是**一个 Rank 身兼多组身份**（TP 组在机内、DP 组跨机）。
3. **编号必须贴着拓扑排**：node-major 排号让通信组落进机内高速域——**乱排号的代价是每层通信都挤机间慢链路**。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。本章是第 5-9 章六种并行策略的全部地基，概念多而计算少，适合拿纸画出"编号表"边推边读。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](./04-distributed-basics/quick.md)：一个"新公司开业"的类比讲完整章，再回来按单元深入。

- [ ] 我能画出 2 机 16 卡的 rank / local_rank / node_rank 编号表
- [ ] 我能为 TP=8 × DP=4 的混合并行写出每个 rank 的组身份
- [ ] 我知道 torchrun 卡在 init 时按什么顺序排查

## 本章单元

- **04-1（约 15 分钟）**：[从进程到 Rank：人口登记](./04-distributed-basics/01-process-rank.md)
- **04-2（约 15 分钟）**：[通信域：谁和谁是一伙的](./04-distributed-basics/02-process-groups.md)
- **04-3（约 15 分钟）**：[拓扑感知排布：把编号贴到物理上](./04-distributed-basics/03-topology-aware-layout.md)
- **04-4（约 15 分钟）**：[工程实操：跑起来与排障](./04-distributed-basics/04-launch-and-debug.md)

- **本章总结**：[练习、自测与串联](./04-distributed-basics/summary.md)

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

《集合通信》第 1 章给过 Rank 与通信域的**语义视角**（谁在说话、对谁说）；本章补上**系统视角**——这些编号和分组在进程、设备与物理网络上怎么落地。HCCL 的通信域与 RankGraph（[H01-3](../ascend/hccl-source/03-comm-domain-rank-graph.md)）正是本章概念的源码化身。

---

[开始单元 04-1 →](./04-distributed-basics/01-process-rank.md)
