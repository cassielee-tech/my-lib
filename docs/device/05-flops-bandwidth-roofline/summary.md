# 第 5 章总结｜FLOPs、带宽、算术强度与 Roofline

> 返回：[第 5 章首页](../05-flops-bandwidth-roofline.md)

## 把本章串成一条主线

请先合上各单元正文，沿"算账 → 落图 → 读案例 → 定行动"复述整章；遇到断点时，再回到对应单元查阅。

```text
三本账（05-1）：FLOPs · Bytes(min/实测) · AI = 两者之比
        ↓
一条屋脊（05-2）：P = min(峰值算力, AI × 峰值带宽)
   左坡 = Memory-bound   屋顶 = Compute-bound
   拐点 = 峰值算力 ÷ 峰值带宽
        ↓
四个案例（05-3）：GEMM 屋顶 · 逐元素深谷 · Norm 看实现 · FlashAttention 向右挪
        ↓
两个方向（05-4）：向上爬（对齐/流水/tile）· 向右挪（融合/复用/精度/算法）
        ↓
本专栏下半程：整网执行——Eager、计算图与图编译
```

## 9. 动手练习

### 练习 1：给 Attention 的一步算账

单头 Attention，$s=4096$、$d=128$、fp16，只看 $S = QK^T$ 这一步：

1. 计算 FLOPs、$Bytes_{min}$（读 Q、K，写 S）和理想 AI；
2. 设备峰值 400 TFLOPS、带宽 2 TB/s（拐点 200 FLOPs/B），判断这一步单独执行时的瓶颈与性能上限；
3. FlashAttention 把 QK^T、softmax、PV 融合后，S 不再落 HBM——说明这对 AI 分母的影响方向。

### 练习 2：画一条 Roofline

设备：峰值 700 TFLOPS（BF16）、峰值带宽 3.35 TB/s。

1. 计算拐点；
2. 分别计算 $4096^3$ BF16 GEMM（理想 AI ≈ 1365）和逐元素加（AI ≈ 0.17）的性能上限与算力利用率；
3. 若 GEMM 实测 420 TFLOPS，判断问题在"算子性质"还是"实现质量"，并给出排查方向。

### 练习 3：选优化方向

对下列三种情况，各选"向上爬"或"向右挪"并说明第一手段：

1. 逐元素 add 夹在两个 GEMM 之间，实测带宽利用率 85%；
2. 大 GEMM 实测只有屋顶的六成，Profiling 显示时间线后段利用率骤降；
3. 朴素 Attention 实现里 $s \times s$ 的中间矩阵反复进出 HBM。

## 10. 自测题

1. 三本账指什么？理想 AI 和实际 AI 的区别是什么？
2. GEMM 的 FLOPs 公式是什么？训练一步的总计算量约为前向的几倍？
3. 为什么 FLOPs 数与峰值算力必须同口径（精度、稀疏）？
4. 写出 Roofline 公式和拐点的定义。
5. 落在左坡和屋顶分别意味着什么？各自对应的优化方向？
6. 为什么说逐元素算子"优化计算无济于事"？
7. FlashAttention 是靠什么把算子挪向屋顶的？
8. 换用 FP8 为什么可能"一举两得"？代价是什么？
9. 列出三个读数陷阱。
10. Roofline 不覆盖哪些因素？完整的判断三件套是什么？

::: details 自测答案

1. FLOPs（计算量）、Bytes（搬运量，区分下界与实测）、AI（两者之比）。理想 AI 用理论最小 Bytes，反映算子天生性质；实际 AI 用实测 Bytes，反映实现质量。
2. $2MNK$；训练约为前向的 3 倍（前向 + 反向 + 参数更新）。
3. 不同精度的峰值算力可差数倍，稀疏与稠密口径也不同；口径不一致时，比值和落点结论全部失真。
4. $P_{max}=\min(峰值算力, AI \times 峰值带宽)$；拐点 = 峰值算力 ÷ 峰值带宽，是左坡与屋顶的分界。
5. 左坡：性能受带宽限制（Memory-bound），向右挪（提高 AI）；屋顶：受算力限制（Compute-bound），向上爬或接受，重点修实现质量。
6. 它的理想 AI 本身极低（约每 6 字节 1 次计算），计算占比可忽略，唯一出路是融合进邻居算子、搭搬运的"顺风车"。
7. 算子融合：把 QK^T、softmax、PV 在片上一次算完，中间矩阵不落 HBM——分母（Bytes）大减，AI 右移。
8. 字节数减半使 AI 翻倍（向右），同时低精度峰值更高（屋顶上移）。代价是精度与数值稳定性需要验证。
9. 实测 AI 误用理想 Bytes；屋顶误用峰值而非有效带宽；小 shape 下尾块与启动开销导致落点系统性偏低。
10. 延迟与同步、占用率与尾块、通信与多卡、Host 启动开销。三件套：Roofline 定方向，时间线找证据，约束账（容量/对齐/并行）定手段。

:::

## 11. 本课小结与本专栏上半程收官

- 三本账先行：FLOPs、Bytes、AI——账不对，图就是错的；
- Roofline 一条屋脊：左坡看带宽，屋顶看算力，拐点分界；
- 四类案例锚定直觉：GEMM 天生屋顶、逐元素天生深谷、Norm 输在实现、FlashAttention 靠融合向右挪；
- 优化两个方向：向上爬（访存质量）与向右挪（算术强度），先诊断再动手；
- 模型有边界：延迟、尾块、通信、启动开销需要时间线和约束账补充。

对照《单卡执行系统》上半程的产出自检：

- [ ] 我能为矩阵乘法或逐元素算子计算 FLOPs、访存量和算术强度；
- [ ] 我能用 Roofline 判断一个算子的理论瓶颈类型；
- [ ] 我能把瓶颈结论翻译成"向上/向右"的具体优化手段。

下一课进入本专栏下半程：**Eager、计算图与图编译**——从单算子视角上升到整网视角，看 Python 的一行代码怎样变成可执行计算图。

## 参考资料

- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [2026 主流 AI 加速卡全景](../accelerator-cards-2026.md)
- [CANN Learning Hub：NPU 实践](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/04_npu_practice.ipynb)

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
