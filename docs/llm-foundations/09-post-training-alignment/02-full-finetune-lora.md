# 单元 09-2｜Full Fine-Tuning、LoRA 与 QLoRA

> 所属章节：[第 9 章｜后训练与对齐](../09-post-training-alignment.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **Full Fine-Tuning、LoRA 与 QLoRA** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

全参数微调会更新模型的全部权重。假设一个线性层：

$$
Y=XW^T
$$

Full Fine-Tuning 需要为 $W$ 保存梯度，并为 AdamW 保存一阶、二阶矩等状态。对数十亿参数模型而言，即使训练数据不多，参数、梯度和优化器状态仍会带来巨大显存需求。

全参数微调的优势是自由度最大，适合数据和算力充足、需要显著改变模型能力的场景。代价则包括：

- 训练显存和分布式通信量大；
- 每个任务要保存一份完整模型；
- 小数据上更容易过拟合或损伤原有能力；
- 多任务、多租户管理成本高。

这推动了 PEFT（Parameter-Efficient Fine-Tuning），其中最常见的方法之一就是 LoRA。

## 5. LoRA：只学习低秩权重增量

LoRA 冻结原始权重 $W_0$，不直接训练完整的 $\Delta W$，而是假设任务适配所需的权重变化可以用两个小矩阵近似：

$$
W'=W_0+\Delta W
$$

$$
\Delta W=\frac{\alpha}{r}BA
$$

若原矩阵 $W_0\in\mathbb{R}^{d\times k}$，则：

$$
A\in\mathbb{R}^{r\times k},\qquad
B\in\mathbb{R}^{d\times r},\qquad r\ll\min(d,k)
$$

训练时只更新 $A$、$B$，原始矩阵保持冻结。

![Full Fine-Tuning 与 LoRA 的参数更新范围](/images/llm/lora-low-rank-update.svg)

例如 $W_0$ 的 Shape 为 `[4096, 4096]`，参数量是：

$$
4096\times4096=16,777,216
$$

若 LoRA Rank $r=8$：

$$
8\times4096+4096\times8=65,536
$$

这个单层适配器的参数量约为原矩阵的 $0.39\%$。这只是一个矩阵的示例，整个模型的比例还取决于 LoRA 插入哪些层。

### 5.1 Rank、Alpha 与 Target Modules

- **Rank $r$**：低秩通道宽度；越大表示适配自由度越高，同时参数和计算增加；
- **Alpha $\alpha$**：控制 LoRA 分支的缩放；常以 $\alpha/r$ 作用于增量；
- **Target Modules**：指定在哪些矩阵插入 LoRA，例如 Attention 的 Q/K/V/O 投影或 MLP 投影；
- **Dropout**：可对 LoRA 分支使用 Dropout，降低小数据过拟合风险。

这些参数不能脱离模型、数据和目标任务单独判断“最佳值”。

### 5.2 合并与多适配器

推理时可以：

- 保留 Base Model 与 LoRA Adapter 分离，按请求切换适配器；
- 将 $BA$ 合并到 $W_0$，得到普通权重矩阵，避免额外 LoRA 分支计算。

合并后要注意精度、版本和可逆性；多租户服务则常选择动态加载 Adapter。

## 6. QLoRA 又多做了什么

QLoRA 的核心组合是：

1. 将冻结的 Base Model 权重量化保存，例如 4-bit；
2. 前向计算时按需要反量化到计算精度；
3. 只训练较小的 LoRA Adapter；
4. 梯度不会更新量化后的 Base 权重。

它进一步降低了加载基础模型所需的显存，使单机微调更大的模型成为可能。但量化并不会消除激活值，低比特反量化也可能带来额外计算，具体速度取决于 Kernel 和硬件支持。

::: warning LoRA 不等于量化
LoRA 解决“训练哪些参数”，量化解决“用多少位保存或计算数值”。二者可以组合，但不是同一个概念。
:::

---

[继续单元 09-3 →](./03-rlhf-dpo.md)
