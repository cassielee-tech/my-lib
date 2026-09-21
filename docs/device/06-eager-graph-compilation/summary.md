# 第 6 章总结｜Eager、计算图与图编译

> 返回：[第 6 章首页](../06-eager-graph-compilation.md)

## 把本章串成一条主线

请先合上各单元正文，沿"Eager 链路 → 图视角 → 编译流水线 → 边界"复述整章；遇到断点时，再回到对应单元查阅。

```text
Eager：一行代码五段旅程，三层开销（06-1）
   Python → 算子拆解 → 分发 → Launch → 设备
   命门：只看得见当前算子
        ↓
计算图：全局视角（06-2）
   动态图 define-by-run  vs  静态图 define-then-run
   看得见相邻→融合 · 生命周期→内存复用 · 全图→整图下发
        ↓
图编译流水线（06-3）
   Dynamo 描图 + guard → Inductor 优化 → 生成 Kernel
   昇腾：GE 图编译 + 计算下沉 / 离线 ATC
        ↓
边界（06-4）：编译耗时 · recompile 风暴 · graph break · 行为差异
        ↓
第 7 章：图执行内部——Stream、Event 与异步并发
```

## 9. 动手练习

### 练习 1：数一数 Launch 开销

某模型前向有 1800 个小算子，每个算子设备执行 8 µs、Launch 开销 5 µs。估算 Eager 下每步前向中"组织成本"与"干活成本"各占多少。若图模式把 1800 个算子融合成 300 个并整图下发（Launch 摊薄为 300 次），每步省多少时间？

### 练习 2：找出融合机会

一段计算：`h = rmsnorm(x); y = h + residual; out = silu(y)`。用第 5 章的三本账说明：为什么把这三个算子融合成一个 Kernel 是"向右挪"？融合前后理想 AI 各怎么算？

### 练习 3：分析一次 recompile

训练任务用图模式后，日志显示 2000 步内发生了 380 次编译。给出至少三条可能原因，并按本章清单给出排查顺序。

## 10. 自测题

1. Eager 一次算子调用经过哪五段？
2. Eager 的三层开销分别是什么？哪类网络受害最深？
3. 动态图和静态图的"图何时确定"有何不同？
4. 图视角比 Eager 多看见什么？兑换成哪些优化？
5. torch.compile 流水线里 Dynamo 和 Inductor 各做什么？
6. guard 是什么？什么情况下失效？
7. graph break 是什么？为什么断得越碎收益越接近零？
8. GE 的"计算图执行下沉"回收的是哪笔钱？
9. 在线 JIT 和离线编译（ATC/export）各适合什么场景？
10. 一段变长输入的推理服务用图模式收益很差，最可能的原因是什么？

::: details 自测答案

1. Python 层（解释/调用栈）→ 算子拆解（aten）→ 分发（Dispatcher 查表选实现）→ Launch（任务入设备队列）→ 设备异步执行。
2. Python 解释开销、分发查表开销、Kernel Launch 开销。由大量小算子组成的网络（Norm/激活/逐元素链）受害最深——组织成本可超过干活成本。
3. 动态图执行时逐条形成路径；静态图执行前整体搭好再反复执行。
4. 看得见相邻（融合）、取值（常量折叠）、死路（死代码消除）、生命周期（内存复用）、全图（调度与整图下发）。
5. Dynamo 拦截 Python 字节码把执行轨迹描成 FX 图并记录 guard；Inductor 在图上做优化并生成 Kernel（或调用库）。
6. guard 是编译产物的适用条件（shape、dtype、属性等）；输入变化触发条件不满足即失效，引发 recompile。
7. 图中塞不进的部分（调试、动态 Python、不支持算子）把图切断，段间落回 Eager；图越碎，编译收益越少而切换成本占比越高。
8. 逐算子 Launch 的空隙：整图一次下发设备侧执行，把 Host 组织成本摊薄到整图（对应 06-1/04-1 的账）。
9. JIT 适合训练和科研（负载稳定后持续收益）；离线编译适合服务部署（编译耗时挪到交付前，运行时零负担）。
10. 变长输入使 guard 频繁失效、recompile 风暴——反复重编译的耗时吞掉了融合与下发省下的时间。应考虑动态 shape 编译或固定 padding 策略。

:::

## 11. 本课小结

- Eager 灵活即时，但每步付三层开销，且没有全局视角；
- 计算图是"工序依赖图"：动态图边跑边形成，静态图先建后跑；
- 图的价值在全局：融合、内存复用、整图下发——分别回收搬运、显存、Launch 三笔钱；
- torch.compile = Dynamo 描图 + guard、Inductor 优化；昇腾侧是 GE 编译与计算下沉；
- 图有边界：编译耗时、recompile 风暴、graph break、数值差异；
- 实践节奏：先 Eager 跑对，再图模式跑快；收益不达预期先查 recompile 和 graph break。

## 参考资料

- [PyTorch：torch.compile 教程](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch：TorchDynamo 深度解读](https://pytorch.org/docs/stable/dynamo/index.html)
- [《昇腾与 HCCL》的 01-2：CANN 软件栈与模型执行路径](../../ascend/01-ascend-cann/02-cann-stack-execution-path.md)
- [昇腾社区：GE 图引擎](https://www.hiascend.com/cann/graph-engine)
- [CANN Learning Hub：什么是 CANN（图模式体验小节）](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/03_what_is_cann.ipynb)

下一课进入 **Stream、Event 与异步执行**：Host、Device、计算和通信怎样并发，什么时候必须同步——图执行内部的任务调度就此展开。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
