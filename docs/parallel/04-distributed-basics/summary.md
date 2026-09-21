# 第 4 章总结｜分布式基础：Rank、通信域与拓扑

> 返回：[第 4 章首页](../04-distributed-basics.md)

## 把本章串成一条主线

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

## 9. 动手练习

### 练习 1：编号表推演

4 机 × 8 卡（32 卡），node-major 排号。写出 rank 19 的 `node_rank`、`local_rank`；写出它所在的 TP=8 组与 DP=4 组的全部成员；判断两个组分别落在机内还是跨机。

### 练习 2：分组设计

1 台 8 卡机器，要求 TP=4 × DP=2。列出全部 tp_group 与 dp_group 的成员表，并回答：两种组是否都能全落机内？rank 5 的双重身份是什么？

### 练习 3：排障决策

`torchrun --nproc_per_node=8` 启动后卡在 `init_process_group` 约 3 分钟无任何输出。列出至少 3 个排查点，按你执行的顺序排列，并给出每步对应的命令。

## 10. 自测题

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

## 11. 本课小结

- 进程是 Host 隔离单位、卡是 Device 计算单位，rank 体系把两者钉在一起；
- PG 决定"谁和谁通信"：混合并行 = 一个 rank 身兼多组；
- node-major 排号让逻辑分组对齐物理拓扑——排号是免费的性能；
- 十五行骨架（绑卡/进组/报身份/all_reduce）是所有分布式代码的最小内核；
- 排障第一原则：先人口登记、再物理资源、最后日志。

## 参考资料

- [PyTorch Distributed 文档](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [《集合通信》01-1：Rank 与通信域（语义视角）](../../collective/01-collective-semantics-cost/01-rank-domain-message.md)
- [《集合通信》05-1：拓扑地图](../../collective/05-topology-hierarchical-overlap/01-topology-map.md)
- [HCCL 源码专题 H01-3：通信域与 RankGraph](../../ascend/hccl-source/03-comm-domain-rank-graph.md)
- [CANN 常用命令速查：分布式环境与排障](../../ascend/cann常用命令.md)

下一章开始逐一展开六种并行策略，第一站是最经典的**数据并行与 DDP**——梯度为什么需要 AllReduce、Bucket 和通信计算重叠怎样工作，正是本章"world group 上的集体行动"。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
