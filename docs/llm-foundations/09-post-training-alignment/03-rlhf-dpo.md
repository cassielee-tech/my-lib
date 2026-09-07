# 单元 09-3｜RLHF 与 DPO

> 所属章节：[第 9 章｜后训练与对齐](../09-post-training-alignment.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **RLHF 与 DPO** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

对同一个问题，可能存在许多语法正确的回答。交叉熵 SFT 只模仿给定示范，无法完整表达“回答 A 比回答 B 更好”的相对偏好。

偏好数据通常写为：

$$
(x,y_w,y_l)
$$

- $x$：Prompt；
- $y_w$：Preferred / Chosen Response；
- $y_l$：Rejected Response。

人或其他评审系统根据帮助性、正确性、安全性、风格等准则选出更好的回答。偏好对齐要让模型在相同 Prompt 下提高 $y_w$ 的相对概率、降低 $y_l$ 的相对概率。

## 8. 经典 RLHF 流水线

InstructGPT 展示了一条经典路径：SFT、Reward Model、强化学习优化。

### 8.1 训练 Reward Model

Reward Model 输入 Prompt 和回答，输出一个标量分数 $r_\phi(x,y)$。利用成对偏好训练：

$$
P(y_w\succ y_l\mid x)=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
$$

让 Preferred Response 的奖励高于 Rejected Response。

### 8.2 优化 Policy

当前语言模型作为 Policy 生成回答，Reward Model 为回答打分，再通过 PPO 等强化学习方法提高奖励。同时使用 KL 约束，防止 Policy 过度偏离参考模型：

$$
Reward=r_\phi(x,y)-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})
$$

![经典 RLHF 需要示范数据、偏好数据、奖励模型和在线采样](/images/llm/rlhf-pipeline.svg)

RLHF 能把复杂偏好转化为奖励信号，但系统较复杂：需要维护多个模型、在线生成回答、估计优势并稳定执行强化学习训练。

## 9. DPO 为什么更直接

DPO 使用同样的成对偏好数据，但不显式训练 Reward Model，也不在训练循环中用 PPO 在线采样。它直接优化 Policy，使 Preferred Response 相对于 Reference Model 获得更大的概率优势。

其目标可以写成：

$$
L_{DPO}=-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]\right)
$$

直观理解：

- 让当前模型比参考模型更偏向 Chosen；
- 让当前模型减少对 Rejected 的相对偏好；
- 参考模型充当锚点，限制行为漂移；
- $\beta$ 控制偏好强化和贴近参考模型之间的权衡。

![RLHF 与 DPO 如何使用同一类偏好数据](/images/llm/rlhf-vs-dpo.svg)

DPO 的训练形态更接近监督学习，因此实现和稳定性通常更简单。但“算法更简单”并不意味着数据质量不重要：偏好标准含糊、标注不一致、Chosen 本身有事实错误，都会直接影响模型行为。

## 10. 对齐不等于事实正确

偏好对齐优化的是特定标注准则下的行为。它可能改善帮助性、格式和拒答边界，但不能保证：

- 回答中的事实一定正确；
- 模型获得训练数据之外的新知识；
- 所有用户群体的价值偏好都一致；
- 模型不会学会迎合评审偏好；
- 安全行为不会牺牲正常问题的可用性。

Reward Model 还可能被 Policy 找到漏洞，即获得高奖励却没有真正满足人的意图，这类现象常被称为 Reward Hacking。

所以对齐训练必须配合多维度评测、红队测试、事实性检查和上线监控。

---

[继续单元 09-4 →](./04-data-memory-communication.md)
