# 单元 05-4｜张量并行怎样切分 Block

> 所属章节：[第 5 章｜完整 Decoder Block](../05-decoder-block.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **张量并行怎样切分 Block** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

### Attention 投影切分

QKV 常按输出特征列切分，每个 Rank 计算部分 Head；输出投影再按相配方式切分，局部结果可能需要 AllReduce 或 ReduceScatter。

### MLP 切分

Gate/Up 可以按 intermediate 维切分，每个 Rank 只计算一部分 I；Down 投影使用相反方向切分，局部 `[B,S,H]` 贡献需要聚合。

```text
Gate/Up：各 Rank 获得不同 intermediate 分片
逐元素 SwiGLU：完全本地完成
Down：各 Rank 产生 H 维局部和
AllReduce / ReduceScatter：合成后续需要的结果
```

这正是 Megatron 风格张量并行的重要模式。后续会用矩阵分块严格推导通信量。

---

[进入本章总结 →](./summary.md)
