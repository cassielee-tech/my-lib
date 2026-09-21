# 单元 01-2｜六个核心原语的语义卡片

> 所属章节：[第 1 章｜Collective 语义与代价模型](../01-collective-semantics-cost.md)

::: info 本单元目标
围绕 **六大原语的输入输出与恒等式** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

设 N 个 rank，每人持有向量的一份（长度 S 字节，或切好的 N 块）。六张卡片只回答：**开始时谁有什么、结束时谁有什么**。

### 3.1 六张语义卡片

| 原语 | 开始 | 结束 | 谁拿到结果 | 典型用途 |
| --- | --- | --- | --- | --- |
| **Broadcast** | rank 0 有 X | 全员有 X | 全员 | 参数同步、下发指令 |
| **Reduce** | 全员各有 Xᵢ | 归约结果（ΣXᵢ）在 root | 仅 root | 收集汇总到主节点 |
| **AllReduce** | 全员各有 Xᵢ | 全员有 ΣXᵢ | **全员** | DDP 梯度同步 |
| **AllGather** | rank i 只有第 i 块 | 全员有完整拼接结果 | 全员 | TP 拼激活、FSDP 收参数 |
| **ReduceScatter** | 全员各有完整 Xᵢ | rank i 有 ΣX 的第 i 块 | **各拿一块** | FSDP 分发梯度、TP 反向 |
| **AlltoAll** | rank i 持有发往各家的块 | 各家收到定向投递的块 | 定向交换 | MoE 的 Dispatch/Combine |

形状视角（把通信看成张量变形）：

```text
Broadcast:   [S]@root        → [S]@all
AllReduce:   [S]@each        → [S]@all（数值=归约）
AllGather:   [S/N]@each      → [S]@all（拼接）
ReduceScatter: [S]@each      → [S/N]@each（归约+切分）
AlltoAll:    [N 块]@each     → [N 块]@each（按目标重排，总量不变）
```

### 3.2 一条黄金恒等式

**AllReduce 有两种等价分解**：

```text
AllReduce = Reduce           + Broadcast     （先汇总到 root，再广播）
AllReduce = ReduceScatter    + AllGather     （先分头规约，再拼完整）
```

第二种是重点：它把"人人要全量"拆成两段**每人只经手 1/N 的中间态**——Ring 算法（第 2 章）正是它的直接实现。记住这张拆解图，后面所有算法都是在"怎么高效实现 RS + AG"上做文章。

### 4. 语义之间的对偶

- **AllGather ↔ ReduceScatter** 互为对偶：一个"拼"、一个"归约后切"；
- **AllReduce** 是"Reduce 的全员版"，也是"RS + AG 的组合体"；
- **AlltoAll 最特殊**：不是"拼/切"，是**定向重分发**——出发前就要知道每块发谁（MoE 的路由结果），这也是它容易失衡的根源（第 4 章）。

---

[继续单元 01-3 →](03-alpha-beta-gamma.md)
