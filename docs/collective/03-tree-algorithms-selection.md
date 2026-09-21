# 第 3 章｜Tree、Recursive Doubling 与算法选择

> 本课目标：掌握轮次只有 O(log N) 的两大算法族；理解大小消息的分界线从哪来；了解通信库内部的算法选型机制，学会用实测校准理论。

## 本章导学

::: tip 本课只记住 3 件事
1. **RD（递归倍增）**：每轮与**距离翻倍**的伙伴交换全量部分和，log₂N 轮后**人人都有全量**——轮次极少，但每轮都发全量 S 字节。
2. **Tree（树）**：上行归约 + 下行广播各 log₂N 轮——轮次少、任意 N 都优雅，但**根节点链路有热点**。
3. **选型地图**：小消息看**轮次**（α 主导 → Tree/RD），大消息看**流量**（β 主导 → Ring）——分界点可以用公式估出来，但最终要靠实测。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。推荐边读边算：本章的魅力全在"数字会说话"。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](03-tree-algorithms-selection/quick.md)：一个"通知怎么传播"的类比讲完整章，再回来按单元深入。

- [ ] 我能写出 RD 的每一轮配对表并解释为什么 log₂N 轮就够
- [ ] 我能算出给定参数下 Ring 与 RD 的交叉点
- [ ] 我知道 HCCL/NCCL 里算法选择可以显式指定并实测

## 本章单元

- **03-1（约 15 分钟）**：[Recursive Doubling：距离翻倍的会师](03-tree-algorithms-selection/01-recursive-doubling.md)
- **03-2（约 15 分钟）**：[Tree：上行归约、下行广播](03-tree-algorithms-selection/02-tree-algorithms.md)
- **03-3（约 15 分钟）**：[选型地图：大小消息的分界线](03-tree-algorithms-selection/03-selection-map.md)
- **03-4（约 15 分钟）**：[工程视角：selector 与代价模型](03-tree-algorithms-selection/04-selector-engineering.md)

- **本章总结**：[练习、自测与串联](03-tree-algorithms-selection/summary.md)

## 这一课在整条路线中的位置

上一章 Ring 解决了大消息；本章补齐小消息的解法，并给出三族算法的统一选型框架。这也是阅读 [HCCL 源码专题 H01-6](../ascend/hccl-source/06-coll-algorithms.md) 算法族对比的必备前置。
