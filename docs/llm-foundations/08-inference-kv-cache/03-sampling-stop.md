# 单元 08-3｜采样与停止条件

> 所属章节：[第 8 章｜自回归推理与 KV Cache](../08-inference-kv-cache.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **采样与停止条件** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

模型输出最后一个位置的 Logits：

$$
z\in\mathbb{R}^{|V|}
$$

其中 $|V|$ 是词表大小。接下来会经过 Logits 处理、过滤与采样。

### 7.1 Greedy Decoding

每次选择概率最大的 Token：

$$
y=\arg\max_i z_i
$$

结果确定、速度简单，但可能陷入重复或缺少多样性。

### 7.2 Temperature

Softmax 前将 Logits 除以温度 $T$：

$$
p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T<1$：分布更尖锐，更偏向高概率 Token；
- $T>1$：分布更平坦，随机性更强；
- Greedy 通常直接关闭采样，而不是把 $T$ 真设为 0。

### 7.3 Top-k

只保留概率最高的 $k$ 个候选，其余置为不可选。候选数量固定，但不同上下文中第 $k$ 名的概率可能差异很大。

### 7.4 Top-p

按概率从高到低选择最小候选集合，使累计概率达到阈值 $p$。模型很确定时集合较小，不确定时集合会扩大。

![Temperature、Top-k 和 Top-p 依次改变候选分布](/images/llm/sampling-pipeline.svg)

Temperature、Top-k、Top-p 可以组合使用。参数不是“越大越好”，应结合任务评测：事实问答通常更保守，创意写作可以增加多样性。

## 8. 什么时候停止生成

常见停止条件包括：

- 生成 EOS Token；
- 达到 `max_new_tokens`；
- 命中指定 Stop String 或 Stop Token Sequence；
- 请求被取消或超时；
- 服务端达到资源或安全限制。

停止字符串可能跨越多个 Token，不能简单只检查最后一个 Token。流式输出时还要避免提前把停止序列的一部分发送给用户。

## 9. 最小生成循环

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

---

[继续单元 08-4 →](./04-inference-service-metrics.md)
