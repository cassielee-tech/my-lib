# 单元 11-4｜把模型优化映射到算子与通信

> 所属章节：[第 11 章｜现代模型的效率设计](../11-efficient-llm-design.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **把模型优化映射到算子与通信** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

假设服务遇到“长上下文下 Batch 开不大、Decode 又很慢”：

1. **GQA** 减少 KV Head 数，缓存容量和每步读取量一起下降；
2. **权重量化**减少每步 Decode 读取的模型权重；
3. **KV Cache 量化**进一步压缩历史 K/V；
4. **Paged Attention**减少动态请求造成的显存碎片；
5. **FlashAttention**提高 Prefill 的 Attention 执行效率；
6. **Chunked Prefill**改善长 Prompt 与短请求共同服务时的调度。

它们解决的是不同层的问题。性能优化应先找到瓶颈，再选择对应技术。

## 15. 与昇腾算子和集合通信开发的关系

阅读实现时，可以沿以下问题建立模型语义与底层执行的联系：

1. `num_attention_heads`、`num_key_value_heads` 怎样映射到每个 Rank？
2. GQA 的 K/V 是物理复制、广播读取，还是 Kernel 内完成 Head 映射？
3. 量化数据、Scale 和 Zero Point 采用什么布局与对齐？
4. 反量化发生在 HBM、片上 Buffer 还是矩阵计算流水中？
5. KV Cache 的 Block Table 由 Host 还是 Device 管理？
6. FlashAttention 的分块大小怎样受片上存储容量限制？
7. Context Parallel 交换的是 Q、K/V、中间结果还是输出？
8. 通信能否与当前分块的矩阵计算和 Softmax 重叠？

这些问题会让“GQA”“INT4”“长上下文”从产品参数变成可落到 Shape、内存地址、Kernel 和通信量上的工程对象。

---

[进入本章总结 →](./summary.md)
