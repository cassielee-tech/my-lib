# 第 1 章｜大模型训练循环

> 本章目标：把数据、前向传播、Loss、反向传播和参数更新连成完整闭环；理解 AdamW、学习率、梯度累积、混合精度与梯度裁剪各自解决的问题；知道分布式训练中的梯度通信发生在哪里。

## 本章导学

::: tip 本章核心结论
1. 一个训练 Step 是数据、前向、Loss、反向、梯度同步和参数更新的闭环。
2. 梯度累积扩大有效 Batch，混合精度减少资源，AdamW 根据梯度和历史统计更新参数。
3. 数据并行中，同一参数的梯度必须在更新前聚合。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，沿一次参数更新的时间顺序推进。

**跟不动？** 先读 [⚡ 速通版（约 5 分钟）](#⚡-速通-约-5-分钟)：一个学生的学习闭环类比讲完整章，再回来按单元深入。

**先看这几张核心图：** Training Step Lifecycle、Gradient Accumulation 和 AdamW State Update 三张图。

- [ ] 我能按顺序说出一次参数更新的步骤
- [ ] 我能区分 Micro-batch 与有效 Batch
- [ ] 我能指出梯度通信发生的位置

## 本章单元

- **01-1（约 15 分钟）**：[一次训练 Step 与梯度](#_01-1-一次训练-step-与梯度)
- **01-2（约 15 分钟）**：[梯度累积与 AdamW 状态](#_01-2-梯度累积与-adamw-状态)
- **01-3（约 15 分钟）**：[学习率、混合精度与梯度裁剪](#_01-3-学习率、混合精度与梯度裁剪)
- **01-4（约 15 分钟）**：[训练显存与梯度通信](#_01-4-训练显存与梯度通信)

- **本章总结**：[串联知识、练习与检查](#本章总结)

## ⚡ 速通（约 5 分钟）

> 适用：时间紧，或完整版跟不动时。这页用一个学生的学习闭环讲完整章，读完抓住 80% 的主干。任何一节想深入，拉到文末点对应单元。

### 1. 学生的一天：做题、对答案、改方法

训练一个模型，就是一个学生无限重复的学习循环：

```text
拿一批练习册（取数据）
  → 做题（前向传播）
  → 对答案算错多少（Loss）
  → 复盘每道错题错在哪（反向传播，得到梯度）
  → 调整学习方法（optimizer.step 更新参数）
  → 撕掉草稿纸（zero_grad 清空梯度）
  → 下一批
```

最容易犯的错：**复盘 ≠ 改方法**。`loss.backward()` 只是算出"每个知识点该怎么调"的清单；要等 `optimizer.step()` 才真的调。而且**复盘完不清零**，下次的清单会和这次的混在一起——所以 `zero_grad()` 不能省。

::: details 小测 1：一次最小的参数更新包含哪几步？
取数据 → 前向算 Loss → 反向算梯度 → （多卡时对账）→ `optimizer.step()` 更新参数 → `zero_grad()` 清空梯度。其中 backward 和 step 是两件事，中间还隔着优化器和可能的通信。
:::

### 2. 三个聪明的学习技巧

**技巧一：攒作业（梯度累积）。** 显存不够一次批改 32 本练习册？每次只批 4 本，批 8 次**攒在一起再统一调整**。效果上等效大 batch，代价是时间——**省的是桌面（峰值激活显存），不是功夫**。等效批量的公式：`4 × 8 × 卡数`。

**技巧二：聪明的复习计划（AdamW + 学习率）。** 普通学生（SGD）对每个知识点一视同仁；AdamW 给每个参数记**两本账**：

- m 账本：最近错题的**平均方向**（一阶矩）；
- v 账本：最近错题的**波动大小**（二阶矩）。

更新时看方向、除以波动——**常错的多改，偶尔错的少改**。学习率则是"每次敢改多大"：开学先小步试探（Warmup），期末再精雕细琢（Decay）。

**技巧三：铅笔草稿（混合精度）。** 正式作业用钢笔（FP32 保存关键数字），大量草稿用铅笔（FP16/BF16 算得快、存得省）。铅笔的坑：**特别小的数会被擦掉**（下溢）——所以先用放大镜把 Loss 放大再算，算完再缩回来（Loss Scaling）。

::: details 小测 2：梯度累积能省时间吗？
不能。它省的是显存：允许把大 batch 拆成多个小 micro-batch，峰值激活变小。计算总量一点没少，反而因为多轮循环略慢。它是"用时间换显存"。
:::

### 3. 教室里有很多同学：对账的时机

一个班几十个同学（多张卡），每人分到不同的练习册。各自的复盘结论不同，**调整方法前必须先对账**（AllReduce 求平均）——否则每人一套方法，模型分家。

聪明的对账时机：**复盘是从后往前逐题进行的，后面几题的清单一出来就先装箱发走**，对账和前面的复盘同时进行（通信与计算重叠）。

记住这条**正确的顺序链**，很多诡异 bug 都是它错了：

```text
反向 → 对账 → 放大还原（unscale）→ 裁剪 → 更新 → 清零
```

（裁剪是"防止某次情绪化大改"：梯度太大就整体按比例压小。注意必须在放大还原**之后**裁剪，否则裁的是被放大的假数值。）

::: details 小测 3：使用 Loss Scaling 时，为什么裁剪前必须先 unscale？
因为梯度当时还是被放大的（真实值 × scale）。直接裁剪，裁的是放大后的假象，真实的梯度大小判断全错。先 unscale 还原真实梯度，再算范数、再裁剪。
:::

### 3 句话带走

1. 训练循环 = **做题→对答案→复盘→改方法→清草稿**，`backward()` 和 `step()` 是两件事；
2. 三个技巧：**攒作业省显存不省时间**、**AdamW 记两本账**、**铅笔草稿配放大镜**；
3. 多卡必须**先对账再更新**，可以让对账和复盘并行——顺序链错了就是 bug。

### 黑话小词典

| 术语 | 人话 |
| --- | --- |
| Step / Micro-batch | 一轮完整学习 / 攒作业时的一小批 |
| 梯度累积 | 攒几批一起批改，等效大 batch |
| `backward()` / `step()` | 算复盘清单 / 真的改参数 |
| AdamW（m / v） | 记平均方向和波动两本账的复习法 |
| Warmup / Decay | 开学小步试探 / 期末精雕细琢 |
| 混合精度 / Loss Scaling | 铅笔草稿 / 先放大再算，防擦掉小数 |
| 梯度裁剪 | 情绪化大改的保护阀 |
| `zero_grad()` | 撕草稿，防新旧混账 |

### 想深入？

每节 15 分钟，按需点开，不必按顺序全读：

| 单元 | 讲什么（白话） | 什么时候需要它 |
| --- | --- | --- |
| [01-1 一次训练 Step 与梯度](#_01-1-一次训练-step-与梯度) | 学习闭环的完整拆解 | 想亲手跑通一个训练循环 |
| [01-2 梯度累积与 AdamW 状态](#_01-2-梯度累积与-adamw-状态) | 攒作业与两本账的细节 | 想算等效 batch / 理解优化器显存 |
| [01-3 学习率、混合精度与梯度裁剪](#_01-3-学习率、混合精度与梯度裁剪) | 三件学习技巧的原理与顺序 | 遇到精度/稳定性问题时 |
| [01-4 训练显存与梯度通信](#_01-4-训练显存与梯度通信) | 书桌都摆了什么、对账时机 | 想分析训练显存和重叠 |
| [本章总结](#本章总结) | 动手练习 + 自测 | 想检验整章掌握程度 |

## 01-1｜一次训练 Step 与梯度

::: info 本单元目标
围绕 **一次训练 Step 与梯度** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

前六课解释了一个 Decoder-only Transformer 如何把 Token 变成下一个 Token 的概率。模型刚初始化时，这些概率几乎没有意义。训练的任务，就是反复调整参数，使正确 Token 的概率逐渐增大。

一次最小参数更新包含六步：

1. 从数据集中取出一批 Token；
2. 前向传播得到 Logits；
3. 用标签计算 Loss；
4. 反向传播得到每个参数的梯度；
5. 优化器根据梯度更新参数；
6. 清空旧梯度，开始下一轮。

![一次参数更新中数据、激活、梯度和参数的流向](/images/llm/training-step-lifecycle.svg)

先区分三个容易混淆的计数单位：

- **Micro-batch**：一次前向和反向实际送入设备的数据；
- **Optimizer Step**：优化器真正修改一次参数；
- **Training Step**：有时指 Micro-batch，有时指 Optimizer Step，阅读日志和源码时必须确认定义。

### 2. 从 Loss 到梯度

设当前参数为 $\theta$，一个 Batch 的平均 Loss 为：

$$
L(\theta)=\frac{1}{N}\sum_{j=1}^{N}\ell_j(\theta)
$$

执行 `loss.backward()` 后，每个可训练参数会得到：

$$
g_t=\nabla_{\theta}L(\theta_t)
$$

梯度不是“参数应该变成多少”，而是 Loss 对参数变化的局部敏感程度。最简单的梯度下降为：

$$
\theta_{t+1}=\theta_t-\eta_t g_t
$$

$\eta_t$ 是第 $t$ 步的学习率。负号表示沿着使 Loss 下降的方向移动。

在 PyTorch 中，梯度默认会**累加**到 `parameter.grad`，而不是自动覆盖。因此标准循环必须显式清空梯度：

```python
for input_ids, labels in dataloader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
```

`zero_grad()` 放在循环开头或上一次 `step()` 之后都可以，关键是一次参数更新所需的梯度不能和上一轮意外混在一起。

## 01-2｜梯度累积与 AdamW 状态

::: info 本单元目标
围绕 **梯度累积与 AdamW 状态** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

设备显存可能只容纳较小的 Micro-batch，但训练又希望使用较大的有效 Batch。梯度累积会连续执行 $K$ 次前向和反向，先把梯度累加起来，最后只更新一次参数。

$$
B_{global}=B_{micro}\times K_{acc}\times N_{data\ parallel}
$$

其中：

- $B_{micro}$：每个设备每次前向的样本数；
- $K_{acc}$：梯度累积步数；
- $N_{data\ parallel}$：数据并行副本数。

![四个 Micro-batch 如何合成一次参数更新](/images/llm/gradient-accumulation.svg)

若每个 Micro-batch 的 Loss 都已经取平均，应将 Loss 除以累积步数，使最终梯度等价于大 Batch 的平均梯度：

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)

for micro_step, (input_ids, labels) in enumerate(dataloader, start=1):
    logits = model(input_ids)
    loss = loss_fn(logits, labels) / accum_steps
    loss.backward()

    if micro_step % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

梯度累积降低的是一次前向/反向所需的激活显存，并不会减少完成同样 Token 数所需的总计算量。累积步数过大还会减少单位时间内的参数更新次数。

### 4. AdamW 在保存什么

大模型训练通常不直接使用普通 SGD，而会使用 AdamW。它为每个参数维护两份状态：

- 一阶矩 $m_t$：梯度的指数移动平均，可理解为带平滑的方向；
- 二阶矩 $v_t$：梯度平方的指数移动平均，用来估计各参数方向的尺度。

忽略偏置修正后的核心形式为：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

其中 $\lambda$ 是 Weight Decay。AdamW 把参数衰减与 Loss 梯度更新分开，因此名字中的 `W` 指的是 decoupled weight decay。

![AdamW 如何利用梯度、动量和尺度更新参数](/images/llm/adamw-state-update.svg)

AdamW 的代价也很直观：除参数和梯度外，还要保存 $m$、$v$ 等优化器状态。若状态使用 FP32，仅两份矩状态就约为每参数 8 Bytes。后续学习 ZeRO/FSDP 时，会看到这些状态为什么值得跨设备分片。

实践中通常不会对所有参数做 Weight Decay。Bias 和归一化层的缩放参数经常放入 `weight_decay=0` 的参数组，具体规则以模型训练方案为准。

## 01-3｜学习率、混合精度与梯度裁剪

::: info 本单元目标
围绕 **学习率、混合精度与梯度裁剪** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

固定学习率很少贯穿整个大模型预训练过程。常见策略分为两个阶段：

#### 5.1 Warmup

训练开始时，参数和 AdamW 的矩估计都不稳定。Warmup 会让学习率从较小值逐步升至峰值，避免一开始更新过猛。

线性 Warmup 可写为：

$$
\eta_t=\eta_{max}\frac{t}{T_{warmup}},\qquad t\le T_{warmup}
$$

#### 5.2 Decay

Warmup 后逐渐降低学习率，让训练后期的更新更细致。常见方案有线性衰减和余弦衰减。余弦衰减示意为：

$$
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi(t-T_{warmup})}{T-T_{warmup}}\right)
$$

![Warmup 与余弦衰减控制不同训练阶段的更新幅度](/images/llm/learning-rate-schedule.svg)

Scheduler 应按 **Optimizer Step** 计数，而不是按 Micro-batch 计数，否则使用梯度累积后学习率曲线会被错误加速。

### 6. 混合精度解决什么问题

训练全部使用 FP32 通常会占用更多显存和内存带宽。混合精度的基本思想是：

- 适合低精度的矩阵乘法等算子使用 FP16 或 BF16；
- 对精度敏感的归一化、归约或部分状态保留 FP32；
- 由 `autocast` 根据算子选择合适精度，而不是简单把整个模型强制转换为同一种类型。

低精度通常能减少张量存储和搬运，并使用加速器的低精度计算单元。但低精度表示范围有限：

- **Underflow**：很小的梯度被舍入为 0；
- **Overflow**：数值太大，变成 `inf` 或 `nan`。

FP16 训练常使用 Loss Scaling：先把 Loss 乘以较大的缩放因子 $S$，反向得到放大的梯度；更新前再除以 $S$。动态 GradScaler 在检测到 `inf/nan` 时会跳过更新并减小缩放因子。BF16 的指数范围与 FP32 接近，通常不需要 GradScaler，但仍要监控数值稳定性。

### 7. 梯度裁剪为什么放在更新前

某一步的梯度可能突然变得很大，导致参数更新剧烈甚至训练发散。全局范数裁剪先计算所有梯度组成的整体范数：

$$
\lVert g\rVert_2=\sqrt{\sum_i\lVert g_i\rVert_2^2}
$$

若它超过阈值 $c$，则统一缩放：

$$
g_i\leftarrow g_i\cdot\frac{c}{\lVert g\rVert_2}
$$

裁剪不会分别把每个元素截断到 $[-c,c]$；它保持整体方向，只缩短梯度向量。使用 Loss Scaling 时，必须先 `unscale_`，再按真实梯度范数裁剪。

### 8. 一份顺序正确的训练骨架

下面用 FP16 展示关键顺序。实际 NPU/CUDA 设备类型、支持精度和 AMP 接口应以运行环境为准。

```python
import torch

accum_steps = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")
optimizer.zero_grad(set_to_none=True)

for micro_step, (input_ids, labels) in enumerate(dataloader, start=1):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids)
        loss = loss_fn(logits, labels) / accum_steps

    scaler.scale(loss).backward()

    if micro_step % accum_steps == 0:
        scaler.unscale_(optimizer)                 # 先还原真实梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)                     # inf/nan 时可能跳过更新
        scaler.update()
        scheduler.step()                           # 按 Optimizer Step 前进
        optimizer.zero_grad(set_to_none=True)
```

真实训练还要处理最后不足 `accum_steps` 的 Batch、Checkpoint 恢复、日志、评估以及分布式同步。不要只复制代码，要先确认每个计数器表示 Micro-step 还是 Optimizer Step。

## 01-4｜训练显存与梯度通信

::: info 本单元目标
围绕 **训练显存与梯度通信** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

训练显存不只是模型参数，大致还包括：

$$
M_{train}\approx M_{parameter}+M_{gradient}+M_{optimizer}+M_{activation}+M_{temporary}
$$

- **参数**：当前模型权重；
- **梯度**：与可训练参数对应；
- **优化器状态**：AdamW 的 $m$、$v$ 等；
- **激活值**：前向保存、反向使用，随 Batch、序列长度和层数增长；
- **临时缓冲区**：算子 Workspace、通信 Buffer、内存碎片等。

参数、梯度和优化器状态主要随参数量增长；激活值主要随 Micro-batch 和序列长度增长。所以减小 Micro-batch 或使用 Activation Checkpointing 能降低激活显存，但不能消除 AdamW 状态。

### 10. 与集合通信的关系

单设备训练的反向传播只产生本地梯度。进入数据并行后，每个 Rank 读取不同 Micro-batch，得到不同的本地梯度，必须在 `optimizer.step()` 前聚合为一致的平均梯度：

$$
g=\frac{1}{N}\sum_{r=0}^{N-1}g^{(r)}
$$

这通常通过 **AllReduce** 完成。DDP 会在反向传播过程中按 Bucket 触发梯度通信，并尝试与尚未完成的反向计算重叠。

梯度累积带来一个重要优化：中间 Micro-step 只在本地累加，最后一个 Micro-step 再同步，否则每个 Micro-step 都 AllReduce 会产生多余通信。PyTorch DDP 中通常借助 `no_sync()` 控制，但必须保证最后一次反向确实触发同步。

从时间线看，正确性约束是：

> 本地反向 → 梯度聚合 → 反缩放/裁剪 → Optimizer Step → 所有 Rank 获得一致的新参数

混合精度还会影响通信数据类型；Bucket 大小影响通信启动次数与重叠机会；梯度裁剪的全局范数在分片训练中可能需要额外归约。这些会在分布式训练和集合通信章节展开。

## 本章总结

> 返回：[第 1 章首页](#)

::: tip 本章核心结论
1. 一个训练 Step 是数据、前向、Loss、反向、梯度同步和参数更新的闭环。
2. 梯度累积扩大有效 Batch，混合精度减少资源，AdamW 根据梯度和历史统计更新参数。
3. 数据并行中，同一参数的梯度必须在更新前聚合。
:::

### 把本章串成一条主线

请先合上各单元正文，沿着核心概念画出输入、处理过程和输出；遇到断点时，再回到对应单元查阅。

### ⚠️ 易错点

1. **一个 `backward()` 就会更新参数吗？** 不会，它只计算并累积梯度；`optimizer.step()` 才修改参数。
2. **梯度累积能减少计算量吗？** 不能，它主要用更多时间换取较低的峰值激活显存。
3. **Loss 为什么除以累积步数？** 为了让累加结果对应有效大 Batch 的平均梯度。
4. **AdamW 的 Weight Decay 等于把 L2 项塞进 Loss 吗？** 对自适应优化器而言不等价；AdamW 将衰减与梯度更新解耦。
5. **混合精度就是整个模型都用 FP16 吗？** 不是，不同算子和状态可使用不同精度。
6. **裁剪应该在 GradScaler 反缩放之前吗？** 不应该，否则裁剪的是被放大的梯度。
7. **Scheduler 每个 Micro-step 都执行吗？** 通常按 Optimizer Step 执行。

### 12. 动手练习

#### 练习 1：观察梯度累加

连续执行两次 `loss.backward()`，中间不调用 `zero_grad()`，打印同一参数的 `.grad.norm()`；再清空梯度重复实验，观察差异。

#### 练习 2：计算有效 Batch

假设每个设备 Micro-batch 为 2，梯度累积 8 步，数据并行 16 个 Rank：

$$
B_{global}=2\times8\times16=256
$$

如果每条样本序列固定为 4096 Token，则每次参数更新处理：

$$
256\times4096=1,048,576\ Tokens
$$

#### 练习 3：记录一次训练时间线

为每个 Micro-step 打印 `loss`、梯度范数、学习率和是否执行 `optimizer.step()`。确认 Scheduler 只在参数更新时前进。

### 13. 自测题

1. 一次最小参数更新包含哪些步骤？
2. `backward()` 和 `optimizer.step()` 分别做什么？
3. 为什么每轮需要清空梯度？
4. 有效 Batch Size 如何计算？
5. 梯度累积主要节省哪部分显存？
6. AdamW 为每个参数维护哪两类主要状态？
7. Warmup 和 Decay 分别解决什么问题？
8. FP16 为什么常需要 Loss Scaling？
9. 使用 GradScaler 时，梯度裁剪前必须做什么？
10. 数据并行为什么需要 AllReduce？

::: details 自测答案

1. 取数据、前向、计算 Loss、反向得到梯度、优化器更新参数、清空梯度。
2. `backward()` 沿计算图计算并累积梯度；`optimizer.step()` 根据梯度和优化器状态修改参数。
3. PyTorch 默认把新梯度加到 `.grad`，不清空会意外混入上一轮梯度。
4. $B_{global}=B_{micro}\times K_{acc}\times N_{data\ parallel}$。
5. 它允许减小 Micro-batch，主要降低峰值激活显存；参数和优化器状态不会因此消失。
6. 梯度的一阶矩 $m$ 和平方梯度的二阶矩 $v$。
7. Warmup 避免训练初期更新过猛；Decay 让后期更新逐渐变小、更加细致。
8. FP16 的表示范围有限，小梯度可能 Underflow；先放大 Loss 能让反向梯度落入可表示范围。
9. 先调用 `unscale_` 恢复真实梯度，再计算范数并裁剪。
10. 各 Rank 使用不同数据得到不同本地梯度，AllReduce 将它们聚合，使各副本用一致梯度更新出一致参数。

:::

### 14. 本章小结

- 训练循环的核心闭环是前向、Loss、反向和参数更新；
- 梯度累积通过多个 Micro-batch 合成一次 Optimizer Step；
- AdamW 用一阶矩和二阶矩调节更新，并将 Weight Decay 解耦；
- Warmup 与 Decay 控制训练不同阶段的更新幅度；
- 混合精度降低存储与搬运成本，FP16 常配合动态 Loss Scaling；
- 正确顺序是反向、梯度同步、反缩放、裁剪、更新、清梯度；
- 数据并行训练中，梯度 AllReduce 位于反向传播与参数更新之间，并可与反向计算重叠。

### 参考资料

- [PyTorch：Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch：Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch：AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch：clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

下一课将进入自回归推理，拆解 Prefill、Decode、KV Cache 和采样如何协作生成一个 Token。

---

<!-- chapter-navigation -->
