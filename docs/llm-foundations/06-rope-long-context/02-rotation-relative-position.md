# 单元 06-2｜二维旋转与相对位置

> 所属章节：[第 6 章｜RoPE 与长上下文](../06-rope-long-context.md) · 预计用时：约 **15 分钟**

::: info 本单元目标
围绕 **二维旋转与相对位置** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

二维向量旋转角度 $\phi$，可以写成：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}
$$

$$
R(\phi)
\begin{bmatrix}x_1\\x_2\end{bmatrix}
=
\begin{bmatrix}
x_1\cos\phi-x_2\sin\phi\\
x_1\sin\phi+x_2\cos\phi
\end{bmatrix}
$$

旋转只改变方向，不改变向量长度。RoPE 将每个注意力头的维度两两配对，例如 $D=8$ 时形成 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)` 四个二维平面；位置越靠后，旋转角度越大。

![RoPE 在二维平面旋转向量，相对角度只由位置差决定](/images/llm/rope-relative-rotation.svg)

## 3. RoPE 怎样进入 Attention

输入仍然先通过线性投影得到 $Q$、$K$、$V$。RoPE 位于投影之后、计算 $QK^T$ 之前，通常只旋转 $Q$ 和 $K$，不旋转 $V$：

$$
Q'=\operatorname{RoPE}(Q, position),\qquad
K'=\operatorname{RoPE}(K, position)
$$

$$
Attention(Q,K,V)=\operatorname{softmax}\left(\frac{Q'K'^T}{\sqrt{D}}+Mask\right)V
$$

![RoPE 在注意力计算链路中的准确位置](/images/llm/rope-attention-flow.svg)

这一区分很重要：

- $Q$ 和 $K$ 决定“应该关注谁”，因此把位置写进二者的匹配分数；
- $V$ 携带被取回的内容，通常不需要同样旋转；
- RoPE 没有改变 Attention 的输出形状，也通常不引入可训练参数。

## 4. 为什么点积能得到相对位置

对同一组二维维度，设位置 $m$ 的 Query 旋转 $m\theta$，位置 $n$ 的 Key 旋转 $n\theta$：

$$
q'_m=R(m\theta)q_m,\qquad k'_n=R(n\theta)k_n
$$

二者点积为：

$$
\begin{aligned}
(q'_m)^T k'_n
&=q_m^T R(m\theta)^T R(n\theta)k_n\\
&=q_m^T R((n-m)\theta)k_n
\end{aligned}
$$

结果只通过 $n-m$ 感知位置。若把两个 Token 同时向后移动 $c$ 个位置，距离仍为：

$$
(n+c)-(m+c)=n-m
$$

所以对应的相对旋转不变。这就是 RoPE 的核心：**用绝对位置决定各自旋转多少，再通过点积把它转化为相对位置关系。**

注意，这并不表示整个模型对位置平移完全不变；因果 Mask、上下文边界和 Token 内容仍然会影响结果。这里说的是单个 RoPE 点积中的相对位置性质。

## 5. 为什么需要多种旋转频率

如果所有维度都使用同一个 $\theta$，模型只能用一种尺度观察距离。RoPE 为不同维度对设置不同频率。常见写法之一是：

$$
\theta_i=base^{-2i/D},\qquad i=0,1,\ldots,D/2-1
$$

高频维度对旋转得快，擅长区分邻近位置；低频维度对旋转得慢，可以表达更长距离。模型把多个尺度组合起来判断两个 Token 的位置关系。

![不同频率共同编码近距离和远距离](/images/llm/rope-multi-frequency.svg)

具体实现中的维度排列、频率公式和 `base` 可能不同，阅读源码时要以对应模型配置为准。

---

[继续单元 06-3 →](./03-rope-shape-long-context.md)
