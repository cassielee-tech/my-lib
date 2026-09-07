# 单元 08-2｜KV Cache 保存什么、占多少显存

> 所属章节：[第 8 章｜自回归推理与 KV Cache](../08-inference-kv-cache.md) · 预计用时：约 **15 分钟**

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

## 4. KV Cache 具体存在哪里

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

## 5. KV Cache 显存怎样估算

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

如果同样结构使用 32 个 KV Head，则约为 2 GiB。后面第 11 课会讲到，MQA/GQA 通过减少 KV Head 数显著降低 Cache 大小和读取量。

::: tip 自己估算时不要漏掉
层数、K/V 两份、数据类型字节数和并发 Batch。服务端同时运行多个请求时，每条序列都有自己的逻辑 KV Cache。
:::

## 6. 一次 Attention 怎样使用 Cache

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

---

[继续单元 08-3 →](./03-sampling-stop.md)
