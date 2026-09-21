# 集合通信（岗位核心）

> 返回：[课程总览](../roadmap.md)

**专栏目标**：从 Collective 语义进入算法、代价模型与工程优化——**这是离 HCCL 岗位最近的一段**。手推算法、算清账本、看懂选型，是读 HCCL 源码前的全部理论准备。

## 章节列表

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [Collective 语义与代价模型](./01-collective-semantics-cost.md) | 六大原语的输入输出是什么？α-β-γ 账本怎么算？ | ✅ |
| 2 | [Ring AllReduce 完整推导](./02-ring-allreduce.md) | Ring 为什么等于 RS 加 AG？每轮发送哪一块数据？ | ✅ |
| 3 | [Tree、Recursive Doubling 与算法选择](./03-tree-algorithms-selection.md) | 小消息和大消息为什么选不同算法？分界线在哪？ | ✅ |
| 4 | [AllGather、ReduceScatter 与 AlltoAll](./04-gather-scatter-alltoall.md) | 数据布局怎样变化？MoE AlltoAll 为什么容易失衡？ | ✅ |
| 5 | [拓扑、分层通信与通信计算重叠](./05-topology-hierarchical-overlap.md) | 机内和机间链路怎样组合？Chunk、Channel、重叠各解决什么？ | ✅ |

## 阅读提示

- 学习顺序即章号顺序：语义账本（1）→ 算法（2/3）→ 原语场景（4）→ 拓扑与兑现（5）；
- **第 2 章务必手推**：4 rank 的 RS+AG 两阶段模拟是本专栏的验收硬指标；
- 每读一章，对照 [HCCL 源码专题 H01-6](../ascend/hccl-source/06-coll-algorithms.md) 的实现视角看一遍——理论/实现双轨并进。

## 验收清单

- [ ] 手工模拟 4~8 个 rank 的 Ring AllReduce（每轮谁发哪块）；
- [ ] 用 α-β-γ 模型按消息大小与 rank 数比较算法，算出分界点；
- [ ] 解释机内/机间分层通信为什么快（每节点一份 vs 每卡一份）；
- [ ] 说出 chunk/channel/重叠各自解决什么、代价是什么。

---

[返回课程总览 →](../roadmap.md) · [进入《昇腾与 HCCL》 →](../ascend/index.md)
