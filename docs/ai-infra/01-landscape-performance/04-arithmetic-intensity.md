# 单元 I01-4｜数量级、算术强度与峰值

> 所属章节：[第 1 章｜AI Infra 全景与性能分析](../01-landscape-performance.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **数量级、算术强度与峰值** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

在打开 Profiling 工具前，可以先做三个粗估算。

### 7.1 计算时间下界

$$
T_{compute}^{min}=\frac{FLOPs}{Peak\ FLOPs/s}
$$

假设一个算子需要 20 TFLOPs，设备理论峰值为 200 TFLOPs/s：

$$
T_{compute}^{min}=\frac{20}{200}s=100ms
$$

这是理想下界。真实程序通常达不到理论峰值。

### 7.2 访存时间下界

$$
T_{memory}^{min}=\frac{Bytes}{MemoryBandwidth}
$$

若需要从 HBM 搬运 120 GB，有效带宽为 1.2 TB/s：

$$
T_{memory}^{min}=100ms
$$

### 7.3 通信时间下界

简单的延迟—带宽模型为：

$$
T_{comm}\approx\alpha\times R+\frac{Bytes}{EffectiveBandwidth}
$$

- $\alpha$：一次通信步骤的固定启动与同步代价；
- $R$：通信轮次；
- `Bytes`：实际经过链路的数据量；
- `EffectiveBandwidth`：考虑拓扑和协议后的有效带宽。

小消息通常更容易被 $\alpha$ 主导，大消息更容易被带宽项主导。这也是通信库需要根据消息大小选择不同算法的原因之一。

## 8. 算术强度：连接计算与访存

算术强度 Arithmetic Intensity 定义为：

$$
AI=\frac{FLOPs}{Bytes\ moved}
$$

它表示每搬运一个 Byte 数据完成多少次计算。

- AI 低：数据刚搬来只算几下，更可能 Memory-bound；
- AI 高：同一份数据被反复计算，更可能 Compute-bound。

下一阶段会专门学习 Roofline Model。现在先记住：**性能不仅由 FLOPs 决定，也由为了完成这些 FLOPs 必须搬多少数据决定。**

## 9. 为什么理论峰值不能直接代表实际性能

硬件规格中的峰值往往依赖理想条件：

- 特定数据类型；
- 足够大的规则矩阵；
- 完美的数据对齐；
- 没有依赖和气泡；
- 数据已经及时到达计算单元；
- 所有并行单元都获得均匀任务。

真实模型可能受到以下损失：

- Shape 太小或维度不友好；
- Kernel Launch 过于频繁；
- 数据布局转换；
- 分支和 Padding；
- Pipeline 气泡；
- HBM、片上 Buffer 或网络等待；
- 不同 Rank 负载不均衡。

所以性能优化的任务，是解释“理论能力为什么没有转化为有效吞吐”。

---

[继续单元 I01-5 →](./05-analysis-workflow.md)
