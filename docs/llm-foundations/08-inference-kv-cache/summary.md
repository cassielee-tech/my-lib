# 第 8 章总结｜自回归推理与 KV Cache

> 返回：[第 8 章首页](../08-inference-kv-cache.md)

::: tip 本章核心结论
1. 自回归生成每次只产生一个新 Token，再把它追加到上下文继续计算。
2. Prefill 并行处理 Prompt，Decode 逐步生成；两者的计算 Shape 和性能瓶颈不同。
3. KV Cache 避免重复计算历史 K/V，但会随层数、序列长度、Batch 和 KV Head 数增长。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ⚠️ 易错点

1. **Prefill 会逐 Token 串行运行吗？** 因果关系通过 Mask 表达，但 Prompt 各位置的层内计算可以并行执行。
2. **KV Cache 会保存模型参数吗？** 不会，它保存每层历史 Token 的 K/V 激活；模型权重另行常驻或分片存储。
3. **有了 Cache，Decode 与上下文长度无关吗？** 不是。当前 Q 仍需读取并关注可见历史 K/V。
4. **为什么不缓存 Q？** 历史 Q 不会被未来 Token 再次查询；未来只复用历史 K/V。
5. **Temperature 越低就越准确吗？** 不一定。它只改变分布尖锐程度，不能修正模型知识或推理错误。
6. **吞吐最高就代表体验最好吗？** 不一定。高吞吐配置可能增加排队、TTFT 或 TPOT。
7. **PagedAttention 会降低模型参数量吗？** 不会，它优化 KV Cache 的内存管理。

## 14. 动手练习

### 练习 1：计算 Cache

选择一个开源模型，从配置中找出层数、KV Head 数和 Head Dim，分别估算 2K、8K、32K 上下文下单请求 BF16 KV Cache 大小。

### 练习 2：观察有无 Cache

使用同一模型生成相同数量 Token，分别设置 `use_cache=True/False`，记录总耗时和峰值显存。小模型和短序列差异可能不明显，应逐步增加上下文观察趋势。

### 练习 3：观察采样

固定 Prompt 和随机种子，分别尝试 Greedy、低 Temperature、高 Temperature、Top-p，比较输出稳定性、重复性和多样性。

## 15. 自测题

1. 自回归生成为什么不能一次直接得到整段答案？
2. Prefill 和 Decode 的输入 Shape 有什么核心差异？
3. TTFT 和 TPOT 分别对应哪一阶段？
4. KV Cache 保存哪些张量？为什么按层保存？
5. 为什么缓存 K/V 而不缓存历史 Q？
6. KV Cache 大小与哪些变量成正比？
7. 使用 Cache 后，Decode 是否完全与上下文长度无关？
8. Temperature、Top-k 和 Top-p 分别怎样改变采样？
9. Continuous Batching 解决什么问题？
10. Decode 阶段的集合通信为什么更关注延迟？

::: details 自测答案

1. 第 $t+1$ 个 Token 的输入包含此前已经选出的 Token，因此必须先得到第 $t$ 个结果才能继续。
2. Prefill 通常一次输入完整 Prompt `[B,S]`；Decode 每轮通常只输入最新 Token `[B,1]`。
3. TTFT 主要覆盖排队和 Prefill 到首 Token；TPOT/ITL 描述后续 Decode Token 的时间间隔。
4. 每层历史 Token 的 K 和 V，因为每层都有独立的 Q/K/V 投影与 Attention。
5. 未来的新 Q 会查询历史 K/V；历史 Q 完成本轮查询后不会被未来步骤复用。
6. 与层数、并发 Batch、缓存长度、KV Head 数、Head Dim 和每元素字节数成正比，并包含 K/V 两份。
7. 不是。它避免重算历史层输出，但当前 Q 仍需读取并处理可见历史 K/V。
8. Temperature 调整分布尖锐程度；Top-k 保留固定数量的高分候选；Top-p 保留累计概率达到阈值的最小候选集合。
9. 它在迭代边界动态加入新请求、移除已完成请求，避免整批等待最慢请求，提高设备利用率。
10. 每个 Decode Step 工作量较小且 Token 间串行依赖，通信启动和同步延迟会逐步累积到每 Token 延迟。

:::

## 16. 本章小结

- 自回归模型通过“前向、采样、追加 Token”循环生成文本；
- Prefill 并行处理 Prompt、建立 Cache，Decode 每轮生成一个新 Token；
- KV Cache 保存每层历史 K/V，以显存换取避免重复计算；
- Cache 显存随层数、并发、序列长度和 KV Head 数线性增长；
- Temperature、Top-k、Top-p 决定如何从 Logits 选择 Token；
- TTFT 衡量首 Token 体验，TPOT/ITL 衡量持续生成速度；
- Continuous Batching、PagedAttention 和 Prefix Caching 是服务系统对调度与 Cache 的优化；
- 多卡推理中的每 Token 通信更容易对延迟敏感。

## 参考资料

- [Hugging Face Transformers：Cache strategies](https://huggingface.co/docs/transformers/main/kv_cache)
- [Hugging Face Transformers：Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Hugging Face Transformers：Continuous batching](https://huggingface.co/docs/transformers/main/continuous_batching)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [vLLM：PagedAttention](https://docs.vllm.ai/en/latest/design/paged_attention/)

下一课将学习预训练、SFT、LoRA 与偏好对齐，理解同一个模型如何经历不同训练阶段，最终成为可用的对话模型。

---

<!-- chapter-navigation -->
[进入第 9 章 →](../09-post-training-alignment.md)
