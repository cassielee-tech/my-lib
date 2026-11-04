# 第 3 章｜自回归推理与 KV Cache

> 本章目标：理解 Decoder-only 模型怎样逐 Token 生成文本；区分 Prefill 与 Decode；解释 KV Cache 缓存了什么、节省了什么、占用了什么；理解采样参数与推理性能指标。

## 本章导学

::: tip 本章核心结论
1. 自回归生成每次只产生一个新 Token，再把它追加到上下文继续计算。
2. Prefill 并行处理 Prompt，Decode 逐步生成；两者的计算 Shape 和性能瓶颈不同。
3. KV Cache 避免重复计算历史 K/V，但会随层数、序列长度、Batch 和 KV Head 数增长。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，从单次生成推进到推理服务。

**跟不动？** 先读 [⚡ 本专栏速通版](00-速通.md#第-3-章-自回归推理与-kv-cache)：一场开卷接龙考试类比讲完整章，再回来按单元深入。

**先看这几张核心图：** Autoregressive Loop、Prefill vs Decode、KV Cache Recompute 三张图。

- [ ] 我能画出逐 Token 生成循环
- [ ] 我能区分 TTFT 与 TPOT
- [ ] 我能解释 KV Cache 节省和消耗了什么

## 本章单元

- **03-1（约 15 分钟）**：[生成循环、Prefill 与 Decode](#_03-1-生成循环、prefill-与-decode)
- **03-2（约 15 分钟）**：[KV Cache 保存什么、占多少显存](#_03-2-kv-cache-保存什么、占多少显存)
- **03-3（约 15 分钟）**：[采样与停止条件](#_03-3-采样与停止条件)
- **03-4（约 15 分钟）**：[推理指标、服务与通信](#_03-4-推理指标、服务与通信)

- **本章总结**：[串联知识、练习与检查](#本章总结)

## 03-1｜生成循环、Prefill 与 Decode

::: info 本单元目标
围绕 **生成循环、Prefill 与 Decode** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

语言模型一次前向传播输出的是每个位置的下一个 Token 分布。生成答案时，程序只取最后一个位置的 Logits，选出一个 Token，把它追加到上下文末尾，再执行下一次前向传播。

假设输入是“天空为什么是蓝色”，生成过程可能是：

```text
天空为什么是蓝色 → 因
天空为什么是蓝色因 → 为
天空为什么是蓝色因为 → 瑞
天空为什么是蓝色因为瑞 → 利
……
```

每个新 Token 都依赖完整的已有上下文，因此这叫 **Autoregressive Generation，自回归生成**。

![Prompt 经过 Prefill 后进入逐 Token Decode 循环](/images/llm/autoregressive-inference-loop.svg)

完整过程可以分成：

1. Tokenize：文本变为 Token ID；
2. Prefill：并行处理全部 Prompt Token，建立 KV Cache；
3. 选择第一个输出 Token；
4. Decode：每次只输入最新 Token，复用历史 Cache；
5. Sampling：从当前概率分布选择下一个 Token；
6. 遇到停止条件后 Detokenize，返回文本。

### 2. Prefill 与 Decode 有什么不同

#### 2.1 Prefill

Prompt 有 $S$ 个 Token 时，Prefill 会一次处理 `[B,S]` 的输入。每一层都为这 $S$ 个位置计算 Q、K、V，并将 K/V 写入 Cache。

Prefill 的特点：

- 一次处理许多 Token；
- 大矩阵乘法较多，并行度高；
- Attention 需要处理 Prompt 内部的 Token 关系；
- 产生第一个输出 Token；
- 直接影响 **TTFT（Time To First Token，首 Token 延迟）**。

#### 2.2 Decode

进入 Decode 后，每轮只输入刚生成的一个 Token。模型为它计算新的 Q/K/V，再让当前 Q 查询全部历史 K，并用注意力权重汇聚历史 V。

Decode 的特点：

- 每轮、每个请求通常只新增一个 Token；
- 必须串行执行，下一轮依赖上一轮结果；
- 每轮都要读取模型权重和不断增长的 KV Cache；
- 单步并行度比 Prefill 小；
- 影响 **TPOT（Time Per Output Token）** 或 **ITL（Inter-Token Latency）**。

![Prefill 与 Decode 的输入规模和硬件行为不同](/images/llm/prefill-vs-decode.svg)

在许多常见部署条件下，Prefill 更容易体现计算密集特征；小 Batch Decode 更容易受模型权重和 KV Cache 的内存带宽限制。这不是绝对结论，具体瓶颈仍取决于 Batch、序列长度、模型结构、精度和硬件。

## 03-2｜KV Cache 保存什么、占多少显存

::: info 本单元目标
围绕 **KV Cache 保存什么、占多少显存** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

假设已有 Prompt `A B C`，模型生成了 `D`，接下来要生成 `E`。

如果没有 Cache，第二轮必须重新把 `A B C D` 全部送入模型，并再次计算 A、B、C 对应的 K/V。第三轮又会对 `A B C D E` 重算。历史越长，重复工作越多。

KV Cache 的做法是：

- Prefill 后保存 A、B、C 在每层产生的 K/V；
- Decode D 时只计算 D 的 Q/K/V，并追加 $K_D,V_D$；
- 当前 $Q_D$ 直接读取缓存的 $K_{A:D}$、$V_{A:D}$；
- 下一轮只计算 E 的新状态。

![无 Cache 会反复重算历史 K/V，使用 Cache 后每轮只追加一格](/images/llm/kv-cache-recompute.svg)

为什么缓存 K 和 V，却不缓存 Q？因为历史 Token 的 Q 只在它作为“当前查询”时使用一次；未来 Token 会产生自己的新 Q，去查询历史 K/V。历史 Q 不会再次参与后续 Attention。

### 4. KV Cache 具体存在哪里

KV Cache 不是只存一份，也不是保存 Attention 矩阵。**每个 Transformer 层**都保存过去 Token 的 Key 和 Value。

单层常见逻辑 Shape 为：

$$
K_{cache},V_{cache}\in[B,N_{kv},S,D_h]
$$

- $B$：并发序列数；
- $N_{kv}$：KV Head 数；
- $S$：当前已缓存序列长度；
- $D_h$：每个 Head 的维度。

不同框架的物理维度顺序可能不同，但元素数量一致。Decode 每生成一个 Token，通常会沿序列维追加一组 K/V。

![KV Cache 在每一层分别保存，并随序列逐 Token 增长](/images/llm/kv-cache-layer-layout.svg)

上一课讲过 RoPE。常见实现会先根据位置旋转 K，再把旋转后的 K 写入 Cache；Decode 时只需旋转当前 Q/K。一定要以具体模型实现为准。

### 5. KV Cache 显存怎样估算

不考虑内存对齐、分页和额外元数据，KV Cache 的近似大小为：

$$
M_{KV}=2\times L\times B\times S\times N_{kv}\times D_h\times Bytes
$$

最前面的 2 代表 K 和 V，$L$ 是 Transformer 层数。

例如：

- 层数 $L=32$
- Batch $B=1$
- 上下文长度 $S=4096$
- KV Head 数 $N_{kv}=8$
- Head Dim $D_h=128$
- BF16，每个元素 2 Bytes

则：

$$
M_{KV}=2\times32\times1\times4096\times8\times128\times2
=512\ MiB
$$

如果同样结构使用 32 个 KV Head，则约为 2 GiB。后面《并行策略》第 2 章会讲到，MQA/GQA 通过减少 KV Head 数显著降低 Cache 大小和读取量。

::: tip 自己估算时不要漏掉
层数、K/V 两份、数据类型字节数和并发 Batch。服务端同时运行多个请求时，每条序列都有自己的逻辑 KV Cache。
:::

### 6. 一次 Attention 怎样使用 Cache

设历史长度为 $S$，当前 Decode 只输入一个新 Token：

```text
当前输入 x_new
   ├─ Q_new：只用于本轮查询
   ├─ K_new：追加到 K_cache
   └─ V_new：追加到 V_cache

score = Q_new @ K_cacheᵀ
output = softmax(score) @ V_cache
```

对应 Shape 可以写为：

$$
Q_{new}:[B,N_q,1,D_h]
$$

$$
K_{cache},V_{cache}:[B,N_{kv},S+1,D_h]
$$

当前 Q 只有一个位置，但必须读取全部可见历史 K/V。因此使用 Cache 消除了历史投影和 Decoder Block 的重复计算，却没有让读取历史上下文变成常数成本。

## 03-3｜采样与停止条件

::: info 本单元目标
围绕 **采样与停止条件** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

模型输出最后一个位置的 Logits：

$$
z\in\mathbb{R}^{|V|}
$$

其中 $|V|$ 是词表大小。接下来会经过 Logits 处理、过滤与采样。

#### 7.1 Greedy Decoding

每次选择概率最大的 Token：

$$
y=\arg\max_i z_i
$$

结果确定、速度简单，但可能陷入重复或缺少多样性。

#### 7.2 Temperature

Softmax 前将 Logits 除以温度 $T$：

$$
p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T<1$：分布更尖锐，更偏向高概率 Token；
- $T>1$：分布更平坦，随机性更强；
- Greedy 通常直接关闭采样，而不是把 $T$ 真设为 0。

#### 7.3 Top-k

只保留概率最高的 $k$ 个候选，其余置为不可选。候选数量固定，但不同上下文中第 $k$ 名的概率可能差异很大。

#### 7.4 Top-p

按概率从高到低选择最小候选集合，使累计概率达到阈值 $p$。模型很确定时集合较小，不确定时集合会扩大。

![Temperature、Top-k 和 Top-p 依次改变候选分布](/images/llm/sampling-pipeline.svg)

Temperature、Top-k、Top-p 可以组合使用。参数不是“越大越好”，应结合任务评测：事实问答通常更保守，创意写作可以增加多样性。

### 8. 什么时候停止生成

常见停止条件包括：

- 生成 EOS Token；
- 达到 `max_new_tokens`；
- 命中指定 Stop String 或 Stop Token Sequence；
- 请求被取消或超时；
- 服务端达到资源或安全限制。

停止字符串可能跨越多个 Token，不能简单只检查最后一个 Token。流式输出时还要避免提前把停止序列的一部分发送给用户。

### 9. 最小生成循环

下面是强调数据流的伪代码，真实框架会处理 Cache 类型、Attention Mask、Position ID、Batch 和停止条件：

```python
input_ids = tokenizer(prompt)
cache = None

with torch.inference_mode():
    while True:
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]
        cache = outputs.past_key_values
        next_token = sample(logits, temperature=0.8, top_p=0.9)

        if should_stop(next_token):
            break

        stream_to_user(next_token)
        input_ids = next_token[:, None]  # Decode 后续每轮只输入新 Token
```

第一次循环输入完整 Prompt，属于 Prefill；之后 `input_ids` 只有一个 Token，属于 Decode。实际使用时优先使用框架提供的 `generate()` 或推理引擎，不要把教学伪代码直接用于生产服务。

## 03-4｜推理指标、服务与通信

::: info 本单元目标
围绕 **推理指标、服务与通信** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

![TTFT、TPOT 与端到端延迟位于生成时间线的不同区间](/images/llm/inference-latency-metrics.svg)

| 指标 | 含义 | 主要受什么影响 |
| --- | --- | --- |
| TTFT | 请求到第一个输出 Token 的时间 | 排队、Prompt 长度、Prefill、调度 |
| TPOT / ITL | 后续每个输出 Token 的平均间隔 | Decode、Batch、内存带宽、通信 |
| E2E Latency | 请求到完整响应结束 | TTFT、输出长度、每步 Decode |
| Token Throughput | 单位时间系统生成的 Token 数 | Batch、调度、并行、算子效率 |
| Request Throughput | 单位时间完成的请求数 | 输入输出长度分布、并发与资源 |

低延迟和高吞吐并不总能同时最大化。扩大 Batch 往往能提高总吞吐，却可能增加排队时间和单请求延迟。

### 11. 从单请求到推理服务

真实服务会同时面对许多长度不同、到达时间不同的请求：

#### 11.1 Continuous Batching

传统静态 Batch 要等整批请求全部结束才能换入新请求。Continuous Batching 可以在每个 Decode 迭代边界移除已完成请求、加入新请求，提高设备利用率。

#### 11.2 Paged KV Cache

为每个请求预留最大连续 Cache 会造成大量空闲和内存碎片。PagedAttention 将 KV Cache 划分为固定大小的物理块，通过块表把逻辑连续 Token 映射到非连续物理块，思想类似操作系统分页。

#### 11.3 Prefix Caching

如果大量请求共享相同系统 Prompt，可以复用其已经计算好的 KV Cache，减少重复 Prefill。但复用必须保证模型、Token、位置和相关配置完全一致。

这些优化不改变语言模型“逐 Token 生成”的语义，改变的是请求如何调度、Cache 如何管理以及 Kernel 如何执行。

### 12. 与 AI Infra 和集合通信的关系

#### 12.1 为什么 Decode 对延迟敏感

每生成一个 Token 都要依次通过所有层，还要等待采样结果才能开始下一步。这条依赖链很难跨 Token 并行，所以单次 Kernel 启动、同步和通信延迟都会累积到 TPOT。

#### 12.2 Tensor Parallel 通信

模型放不进单卡或需要提高算力时，可以做 Tensor Parallel。每层的分片矩阵计算之间通常需要 AllReduce 或 ReduceScatter/AllGather。Prefill 的消息对应多个 Token，Decode 每步只有少量 Token，通信更容易呈现延迟敏感特征。

#### 12.3 KV Cache 如何切分

KV Cache 可以随 KV Head 按 Tensor Parallel Rank 分片。每个 Rank 保存自己负责的 Head，减少单设备 Cache；但具体是否需要额外通信，取决于 Attention 与输出投影的切分方案。

#### 12.4 推理优化的三个对象

后续学习 AI Infra 时，可以把优化归纳为：

1. **权重**：量化、分片、减少重复读取；
2. **KV Cache**：分页、量化、复用、卸载或减少 KV Head；
3. **调度**：Batching、请求优先级、Prefill/Decode 协同和通信计算重叠。

## 本章总结

> 返回：[第 3 章首页](#)

::: tip 本章核心结论
1. 自回归生成每次只产生一个新 Token，再把它追加到上下文继续计算。
2. Prefill 并行处理 Prompt，Decode 逐步生成；两者的计算 Shape 和性能瓶颈不同。
3. KV Cache 避免重复计算历史 K/V，但会随层数、序列长度、Batch 和 KV Head 数增长。
:::

### 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

### ⚠️ 易错点

1. **Prefill 会逐 Token 串行运行吗？** 因果关系通过 Mask 表达，但 Prompt 各位置的层内计算可以并行执行。
2. **KV Cache 会保存模型参数吗？** 不会，它保存每层历史 Token 的 K/V 激活；模型权重另行常驻或分片存储。
3. **有了 Cache，Decode 与上下文长度无关吗？** 不是。当前 Q 仍需读取并关注可见历史 K/V。
4. **为什么不缓存 Q？** 历史 Q 不会被未来 Token 再次查询；未来只复用历史 K/V。
5. **Temperature 越低就越准确吗？** 不一定。它只改变分布尖锐程度，不能修正模型知识或推理错误。
6. **吞吐最高就代表体验最好吗？** 不一定。高吞吐配置可能增加排队、TTFT 或 TPOT。
7. **PagedAttention 会降低模型参数量吗？** 不会，它优化 KV Cache 的内存管理。

### 14. 动手练习

#### 练习 1：计算 Cache

选择一个开源模型，从配置中找出层数、KV Head 数和 Head Dim，分别估算 2K、8K、32K 上下文下单请求 BF16 KV Cache 大小。

#### 练习 2：观察有无 Cache

使用同一模型生成相同数量 Token，分别设置 `use_cache=True/False`，记录总耗时和峰值显存。小模型和短序列差异可能不明显，应逐步增加上下文观察趋势。

#### 练习 3：观察采样

固定 Prompt 和随机种子，分别尝试 Greedy、低 Temperature、高 Temperature、Top-p，比较输出稳定性、重复性和多样性。

### 15. 自测题

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

### 16. 本章小结

- 自回归模型通过“前向、采样、追加 Token”循环生成文本；
- Prefill 并行处理 Prompt、建立 Cache，Decode 每轮生成一个新 Token；
- KV Cache 保存每层历史 K/V，以显存换取避免重复计算；
- Cache 显存随层数、并发、序列长度和 KV Head 数线性增长；
- Temperature、Top-k、Top-p 决定如何从 Logits 选择 Token；
- TTFT 衡量首 Token 体验，TPOT/ITL 衡量持续生成速度；
- Continuous Batching、PagedAttention 和 Prefix Caching 是服务系统对调度与 Cache 的优化；
- 多卡推理中的每 Token 通信更容易对延迟敏感。

### 参考资料

- [Hugging Face Transformers：Cache strategies](https://huggingface.co/docs/transformers/main/kv_cache)
- [Hugging Face Transformers：Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Hugging Face Transformers：Continuous batching](https://huggingface.co/docs/transformers/main/continuous_batching)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [vLLM：PagedAttention](https://docs.vllm.ai/en/latest/design/paged_attention/)

下一课将学习预训练、SFT、LoRA 与偏好对齐，理解同一个模型如何经历不同训练阶段，最终成为可用的对话模型。

---

<!-- chapter-navigation -->
