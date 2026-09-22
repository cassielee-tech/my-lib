# 单元 06-1｜三指标与两种负载

> 所属章节：[第 6 章｜Prefill/Decode 调度与推理指标](../06-pd-scheduling.md)

::: info 本单元目标
围绕 **"指标与负载的对应关系"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 1. 三个指标

回扣 [第 3 章 03-4](../03-inference-kv-cache/04-inference-service-metrics.md)，推理服务的三件套：

| 指标 | 定义 | 由谁决定 |
| --- | --- | --- |
| **TTFT**（Time To First Token） | 从请求到达到第一个字返回 | 排队 + **Prefill** |
| **TPOT/ITL**（Time Per Output Token） | 生成阶段平均每个字的间隔 | **Decode 每步** |
| **吞吐** | 每秒全站产出的 token 数 | 并发 × (1/TPOT) 的聚合 |

用户的体感：**TTFT = "多久开始回我"，TPOT = "回复流得多快"**——前者决定"像不像卡死了"，后者决定"读起来顺不顺"。

## 2. 两种负载的性格

[第 3 章 03-1](../03-inference-kv-cache/01-generation-prefill-decode.md) 埋的伏笔，用 [《单卡执行系统》05 章的 Roofline](../../device/05-flops-bandwidth-roofline/02-roofline-model.md) 彻底说清：

| | Prefill | Decode |
| --- | --- | --- |
| 输入形态 | 整段 prompt 一次算（大矩阵 GEMM） | batch 个 token 逐个算（GEMV 形态） |
| 算术强度 | 高（每个权重元素被大量 token 复用） | **极低**（每个权重只被 1 个 token 用一次） |
| 瓶颈 | **Compute-bound**（屋顶区） | **Memory-bound**（山坡区：每步读全部权重） |
| 时间尺度 | 一次几十~几百 ms | 每步几~几十 ms，连续几百步 |

一句话：**Prefill 是"搬一批货"，Decode 是"每天摸一遍所有货架"**——前者恨 GPU 不够快，后者恨带宽不够大，**两者的最优配置（batch 大小、算子选择、甚至硬件配比）完全相反**。

::: tip 制约的根源
第 5 章的连续批让两种负载**同批混跑**——吞吐上去了，但"一山二虎"的矛盾种下了：Prefill 的几百毫秒大计算一旦插入，Decode 的每步节拍立刻被打乱。这正是下一单元的课题。
:::

---

[继续单元 06-2 →](./02-interference-chunked-prefill.md)
