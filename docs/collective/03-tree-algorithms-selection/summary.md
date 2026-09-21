# 第 3 章总结｜Tree、RD 与算法选择

> 返回：[第 3 章首页](../03-tree-algorithms-selection.md)

## 把本章串成一条主线

请先合上各单元正文，沿"RD → Tree → 地图 → selector"复述整章；遇到断点时，再回到对应单元查阅。

```text
RD（03-1）：距离翻倍配对，log₂N 轮人人有全量
   轮次极少 ✅ · 每轮全量 S ❌ · 需 2 的幂
        ↓
Tree（03-2）：上行归约 + 下行广播，2·log₂N 轮
   任意 N ✅ · 层级友好 ✅ · 根部热点 ❌
        ↓
地图（03-3）：S* = α·Δ轮次·BW / Δ流量
   小消息 → Tree/RD · 大消息 → Ring · N 越大分界点越右移
        ↓
工程（03-4）：静态表 + 代价模型 + 环境变量覆盖 + 实测校准
```

## 9. 动手练习

### 练习 1：RD 配对表

写出 N=16 时 RD AllReduce 的全部 4 轮配对表（每轮哪些 rank 互发），并标注每轮结束后每人的"知情圈"大小。

### 练习 2：算分界线

α=25 µs、BW=20 GB/s、N=16。分别写出 Ring 与 RD 的时间公式，解出分界消息大小 S*，并判断：256 KB 的梯度同步、64 MB 的梯度同步各选谁？

### 练习 3：排除法选型

三个场景各排除哪些算法、为什么：① N=48 的集群同步 32 MB 梯度；② N=8 同步 64 KB；③ 需要把通信卸载给专用引擎、且 rank 数不定。

## 10. 自测题

1. RD 第 k 轮的配对距离是多少？"知情圈"每轮怎么变？
2. 为什么 RD 做 AllReduce 不需要广播阶段？
3. RD 每 rank 的总流量是多少？为什么大消息吃亏？
4. Recursive Halving 相比 RD 省在哪？它的适用算子是什么？
5. Tree AllReduce 的两个阶段各几轮？
6. Tree 的"热点"出在哪、为什么？
7. N 越大，Ring 与 RD 的分界点怎么移动？为什么？
8. 什么情况下"消息小就用 Tree"这句话会失效？
9. selector 的两种实现风格是什么、各自的风险？
10. 怎么用一条实测曲线验证 selector 的选择？

::: details 自测答案

1. 距离 2^(k−1)（1, 2, 4, …）；知情圈每轮翻倍，log₂N 轮覆盖全组。
2. 配对交换是双向的，双方交换后各自加总——结束时每个 rank 都持有全量结果。
3. S·log₂N（每轮交换全量部分和）；大消息时 β 项远超 Ring 的 ~2S(N−1)/N。
4. RH 每轮把消息对半、只发一半——总流量 S(N−1)/N 带宽最优且 log₂N 轮；适用算子是 ReduceScatter。
5. 各 log₂N 轮，共 2·log₂N·α。
6. 根节点及近根链路：每轮都要收发聚合流量，负载不对称——非带宽最优，根还是单点。
7. 越往大消息方向移：N 增大时 Ring 的轮次 2(N−1) 线性膨胀，α 差距拉大，需要更大的 S 才能让 β 项追回来（N=8 时 S*≈4.4MB，N=64 时≈14.9MB，同参数）。
8. 约束一票否决时（如非 2 的幂排除朴素 RD）、拓扑异质时（分层场景重画地图）、有引擎加成时。
9. 静态表（快但换集群失准）与代价模型（灵活但依赖 α/β 校准新鲜度）；工程常混合。
10. allreduce 基准从 8B 扫到 1GB 画"大小 → 带宽/时延"曲线，观察分段拐点是否与 selector 的算法切换一致，也可用环境变量强制单算法对照。

:::

## 11. 本课小结

- RD：log₂N 轮配对会师，双向对称无需广播，但每轮全量、需 2 的幂；
- Tree：上行 + 下行各 log₂N 轮，任意 N 优雅、层级与引擎友好，代价是根部热点；
- 分界线 S* 可用 α、β、N 算出，但随硬件移动——地图背形状、数字靠实测；
- selector = 静态表 + 代价模型，环境变量可覆盖，基准曲线是最终裁判。

## 参考资料

- [HCCL 源码专题 H01-6：算法族与选择](../../ascend/hccl-source/06-coll-algorithms.md)
- [HCCL 源码专题 H01-5：通信引擎与任务执行](../../ascend/hccl-source/05-comm-engines.md)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)
- [第 1 章：α-β-γ 代价模型](../01-collective-semantics-cost/03-alpha-beta-gamma.md)
- [第 2 章：Ring 账本](../02-ring-allreduce/04-ledger-and-variants.md)

下一课把镜头从 AllReduce 移开，深潜三大单原语：**AllGather、ReduceScatter 与 AlltoAll**——它们是并行策略（TP/SP/FSDP/MoE）真正的日常动词。

---

<!-- chapter-navigation -->
[返回专栏目录 →](../index.md)
