# 第 1 章总结｜从模型到集合通信

> 返回：[第 1 章首页](../01-landscape.md)

::: tip 本章核心结论
1. 模型定义计算，并行策略把计算切开，集合通信重新连接被切开的数据。
2. DP、TP、PP、FSDP 和 EP 切分对象不同，因此需要的通信也不同。
3. HCCL 位于框架与硬件之间，负责把 Collective 语义变成真实传输任务。
:::

## 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

## ✅ 理解检查

不看上文，尝试回答：

1. 为什么单卡训练不需要跨设备集合通信？
2. DDP 为什么通常需要 AllReduce？
3. FSDP 为什么常见 AllGather 和 ReduceScatter？
4. MoE 为什么容易产生 AlltoAll？
5. 从 `loss.backward()` 到设备执行，中间大致经过哪些层？
6. 优化一个 AllReduce 算子时，为什么需要知道它来自 DP 还是 TP？

## 参考答案

建议先独立回答，再展开核对。

::: details 1. 为什么单卡训练不需要跨设备集合通信？

因为模型参数、输入、激活和梯度都在同一设备内，数据可以通过本地显存读写完成流动，没有其他 Rank 需要交换数据。

单卡内部仍然存在不同计算单元之间的数据搬运与同步，但这不属于“跨设备集合通信”。集合通信解决的是多个 Rank 共同参与的数据交换与聚合问题。

:::

::: details 2. DDP 为什么通常需要 AllReduce？

DDP 在每个 Rank 上保存相同的完整模型，但给不同 Rank 分配不同数据。不同数据会产生不同的本地梯度：

```text
Rank 0 → 数据 A → 梯度 dW₀
Rank 1 → 数据 B → 梯度 dW₁
```

如果两个 Rank 直接使用各自的梯度更新模型，参数会逐渐不同，模型副本便不再一致。AllReduce 可以把所有本地梯度求和或求平均，并让所有 Rank 得到相同结果，因此每个模型副本会执行相同更新。

:::

::: details 3. FSDP 为什么常见 AllGather 和 ReduceScatter？

FSDP 平时让每个 Rank 只保存一部分参数，以减少单卡显存占用。某一层开始计算前，需要把其他 Rank 的参数分片临时收集起来，这对应 AllGather。

反向传播得到完整梯度后，需要在多个 Rank 间完成梯度聚合，但每个 Rank 最后只保留自己负责的梯度分片，这对应 ReduceScatter。

```text
计算前：参数分片 --AllGather--> 临时完整参数
反向后：完整梯度 --ReduceScatter--> 聚合后的梯度分片
```

:::

::: details 4. MoE 为什么容易产生 AlltoAll？

MoE 把不同专家部署到不同设备，而一个 Rank 当前持有的 Token 可能被路由到任意专家。因此每个 Rank 都可能向其他多个 Rank 发送 Token，同时从其他 Rank 接收 Token。

专家计算完成后，还要把输出送回 Token 原来的位置。这个“多对多交换”的数据模式与 AlltoAll 的语义相符，通常会出现一次分发和一次回传。

:::

::: details 5. 从 loss.backward() 到设备执行，中间大致经过哪些层？

概念调用链是：

```text
loss.backward()
→ Autograd 沿计算图生成梯度
→ DDP Hook 发现梯度或 Bucket 就绪
→ ProcessGroup 发起 Collective
→ 昇腾框架适配层调用 HCCL
→ HCCL 选择算法、拓扑与传输方式
→ Runtime / 驱动提交任务
→ NPU 与互联链路执行
```

不同软件版本的具体函数名会变化，但各层职责基本保持一致。

:::

::: details 6. 优化 AllReduce 时，为什么需要知道它来自 DP 还是 TP？

因为来源决定通信所处的模型位置、消息大小、调用频率以及它能否与计算重叠：

- DP AllReduce 通常聚合参数梯度，可以按 Bucket 启动，并与反向计算重叠；
- TP AllReduce 常位于某一层的前向或反向关键路径，下一步计算可能必须等待结果，延迟更敏感；
- 两者的通信组规模和拓扑也可能不同。

只优化通信算子本身的峰值带宽，不一定能改善端到端训练性能。必须先知道它在整个计算图中为什么出现、谁在等待它。

:::

## 动手任务

画一张自己的“大模型训练软件栈”图，并为每一层写一句话：

- 它接收什么；
- 它输出什么；
- 它对下一层提出什么要求。

下一课将从张量、矩阵乘法、计算图和反向传播开始，为理解 Transformer 与通信量估算建立计算基础。

## 参考资料

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)
- [HCCL 官方 API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)

---

<!-- chapter-navigation -->
[进入第 2 章 →](../02-tensor-autograd.md)
