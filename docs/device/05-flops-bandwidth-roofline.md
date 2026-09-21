# 第 5 章｜FLOPs、带宽、算术强度与 Roofline

> 本课目标：掌握算子级的三本账（FLOPs、Bytes、算术强度）的系统算法；理解 Roofline 模型的两条上限与拐点；能用 Roofline 定位真实算子的瓶颈类型并给出优化方向；清楚模型的适用边界。

## 本章导学

::: tip 本课只记住 3 件事
1. 判断瓶颈之前先把账算对：FLOPs、Bytes、AI = FLOPs/Bytes，三个数定了，位置就定了。
2. Roofline 只有一条屋脊：左坡受带宽限制，屋顶受算力限制，拐点 = 峰值算力 ÷ 峰值带宽。
3. 优化只有两个方向：向上（更接近带宽上限）或向右（更高算术强度）——先判断自己卡在哪个方向。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**。第 1 章（01-4）给过三个下界公式，本课把它们升级成一套完整的定位方法：先会算账，再会读图，再分析真实算子，最后转成行动。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](05-flops-bandwidth-roofline/quick.md)：一家工厂的产能天花板类比讲完整章，再回来按单元深入。

- [ ] 我能为 GEMM、Attention、逐元素算子手工算出 FLOPs、Bytes 和 AI
- [ ] 我能画出设备的 Roofline 并解释拐点的含义
- [ ] 我能对一个给定算子判断 bound 类型并说出两个可行的优化方向

## 本章单元

- **05-1（约 15 分钟）**：[算子的三本账：FLOPs、Bytes 与 AI](05-flops-bandwidth-roofline/01-operator-accounting.md)
- **05-2（约 15 分钟）**：[Roofline 模型：一条线看清瓶颈](05-flops-bandwidth-roofline/02-roofline-model.md)
- **05-3（约 15 分钟）**：[用 Roofline 分析真实算子](05-flops-bandwidth-roofline/03-roofline-case-studies.md)
- **05-4（约 15 分钟）**：[从判断到行动：优化路线图](05-flops-bandwidth-roofline/04-from-diagnosis-to-action.md)

- **本章总结**：[练习、自测与上半程收官](05-flops-bandwidth-roofline/summary.md)

## 这一课在整条路线中的位置

本课是本专栏上半程的收官：第 2 章解释了计算资源，第 3 章解释了存储资源，第 4 章解释了任务怎样铺上去——本课给出判断"铺得好不好、下一步往哪改"的统一标尺。本专栏下半程（框架、编译与 Runtime）将带着这套标尺去看整网层面的执行。
