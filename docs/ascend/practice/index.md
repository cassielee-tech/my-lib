# 实践总结

> 返回：[昇腾与 HCCL](../index.md)

**专栏目标**：把课程里的概念、源码与真实需求对上——每篇复盘一个真实需求或 Issue/PR，讲清"问题怎么暴露、根因怎么定位、方案怎么演进、验证怎么做、检视怎么闭环"，沉淀可复用的工程方法。

## 文章列表

| 篇 | 主题 | 读完能够带走 | 状态 |
| ---: | --- | --- | --- |
| 1 | [UBOE/UB_RTP 建链端口注入：从 Issue 到合入](01-uboe-ubrtp-port-injection.md) | channel 建链知识链路（RA socket 模型 / 状态机 / hccpTag wire 契约）+ 影响面矩阵、分层验证、检视闭环方法论 | ✅ |
| 2 | [Reduce 64bit/特殊规约分级算法：集合通信与 HCCL 基础理论](02-reduce-64bit-hierarchical-algorithm.md) | Reduce 语义与 RS+AG 分级账本、selector/attrs/cost 选路链、64bit/PROD 执行分流 + 死锁排除法、计价同口径教训 | ✅ |

## 阅读提示

- **先读课程再读复盘**：两篇都以[HCCL 与 HCOMM 源码课程](../hccl-hcomm.md)的分层、注册与建链流程为底，跨不过去时先补[单元 1](../hccl-hcomm/01-architecture-layering.md)、[单元 2](../hccl-hcomm/02-control-plane.md)；
- **按"结论 → 全景 → 细节"读**：每篇开头都有先记住的结论清单，赶时间只读结论 + 方法论清单；
- **方法论可平移**：影响面矩阵、参考实现三问、wire 契约意识、对照实验、rebase 三同步、"计价必须与执行分流同口径"，换一个需求照样能用。

## 验收清单

- [ ] 能复述一次真实需求从 Issue 到合入的完整链路（定位 → 方案 → 验证 → 检视 → rebase → 合入）；
- [ ] 能画出 UBOE 建链路径，并说清"为什么只有一条路径命中写死端口"；
- [ ] 能说明 UT、上板对照实验、存量冒烟各自负责验证什么；
- [ ] 能解释 64bit/PROD 为什么只能走 AICPU 标量规约，以及分级算法为什么在框间只运部分和；
- [ ] 能说出 cost 三轮迭代各自的错误类型，以及"零计价的慢路径"为何最危险。

---

[返回昇腾与 HCCL →](../index.md) · [进入《HCCL 与 HCOMM 源码学习》 →](../hccl-hcomm.md)
