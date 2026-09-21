# 单元 03-2｜torch.distributed 的门面：ProcessGroup

> 所属章节：[第 3 章｜从 PyTorch 走向 HCCL](../03-pytorch-to-hccl.md)

::: info 本单元目标
读完后，你能够说出 **torch.distributed 的四层结构、`init_process_group(backend="hccl")` 背后发生的三步、`dist.all_reduce` 的下降路径**——从框架 API 一路到 HCCL 入口。
:::

## 先记住 3 个结论

1. **四层结构**：Python API（`dist.*`）→ c10d 前端 → **ProcessGroup 抽象** → 后端实现（`ProcessGroupHCCL`）——抽象层让同一份训练代码跑在 NCCL/HCCL/Gloo 上。
2. **init 三步**：rendezvous（人齐）→ 创建 ProcessGroupHCCL → 内部 `HcclCommInitCluster` 建立通信域（[H01-3](../hccl-source/03-comm-domain-rank-graph.md) 的框架侧入口）。
3. **下降路径**：`dist.all_reduce` → `ProcessGroupHCCL::allreduce`（取流、检查）→ HCCL 集合通信接口——框架只决定"何时、对谁、在哪个流"，"怎么跑"全权交给 HCCL。

## 1. 为什么需要一层门面

[《并行策略》04-2](../../parallel/04-distributed-basics/02-process-groups.md) 定义过 PG 的语义：**成员 + 后端 + 操作**。框架侧需要一个类承接这三样，并且**同一份代码不感知具体后端**——这层门面就是 `ProcessGroup`：

```text
torch.distributed（Python API）      ← 你写的代码
        ↓
c10d 前端（校验、分发）
        ↓
ProcessGroup（抽象基类）             ← "成员 + 操作"的接口
        ↓
ProcessGroupNCCL / ProcessGroupHCCL / ProcessGroupGloo   ← "后端"
```

`backend="hccl"` 这个字符串经过注册表（registry）解析到 `ProcessGroupHCCL`——与第 1 章的"框架适配层"同一套设计哲学：**接口归框架，实现归生态**。

## 2. init_process_group 背后

```python
dist.init_process_group(backend="hccl", init_method="env://")
```

三步展开：

1. **Rendezvous**：按 `MASTER_ADDR/PORT`（env://）签到，确定 rank/world_size——[《并行策略》04-1](../../parallel/04-distributed-basics/01-process-rank.md) 的"人口登记"；
2. **构造 ProcessGroupHCCL**：注册表解析 backend 字符串，创建后端实例；
3. **建立通信域**：ProcessGroupHCCL 内部调用 HCCL 的初始化接口（`HcclCommInitCluster` 一族），为这个 PG 建立 **HcclCommunicator**——rank、world_size、成员关系在此固化（源码视角见 [H01-3：通信域与 RankGraph](../hccl-source/03-comm-domain-rank-graph.md)）。

`dist.new_group(ranks=[...])` 重复第 2、3 步——**一个子 PG 对应一个 HCCL 通信域**，`group=` 参数最终路由到对应的 communicator。

## 3. dist.all_reduce 的下降路径

```text
dist.all_reduce(t, op=SUM)                  # Python
  ↓ c10d 绑定层
ProcessGroupHCCL::allreduce(t, opts)        # C++ 后端实现
  ├─ 检查：dtype/contiguous/设备一致
  ├─ 取流：当前 npu 流（或 PG 管理的通信流）
  ↓
HcclAllReduce(dst, src, count, dtype, op, comm, stream)   # HCCL 入口
  ↓（进入通信库内部——H01-7 的领域）
```

框架侧的职责到 `HcclAllReduce` 为止：**何时发（调用时机）、对谁发（comm）、在哪个流（stream）**；算法选择、任务编排、数据搬运全部下沉给 HCCL。这正是 [《并行策略》05-2](../../parallel/05-ddp/02-naive-to-ddp.md) 那句"框架侧何时发，通信库侧怎么跑"的分界线。

## 4. 自测题

先用自己的话回答，再展开答案。

1. torch.distributed 的四层结构是什么？抽象层解决了什么问题？
2. `backend="hccl"` 怎样变成 `ProcessGroupHCCL`？
3. `init_process_group` 的三步是什么？第三步建立的对象在 HCCL 里叫什么？
4. `new_group` 与通信域是什么关系？
5. 框架侧与通信库侧的职责分界线在哪？

::: details 自测答案

1. Python API → c10d 前端 → ProcessGroup 抽象 → 后端实现；让同一份训练代码不感知具体后端（NCCL/HCCL/Gloo 可切换）。
2. 经注册表（registry）解析——后端字符串映射到注册的实现类。
3. Rendezvous 签到定 rank/world_size；创建 ProcessGroupHCCL；内部 HcclCommInitCluster 建立通信域（HcclCommunicator）。
4. 一个子 PG 对应一个独立的 HCCL 通信域；`group=` 参数路由到对应 communicator。
5. `HcclAllReduce` 入口：框架管"何时、对谁、在哪个流"；HCCL 管"怎么跑"（算法/编排/搬运）。

:::

## 本单元小结

- ProcessGroup = "成员 + 后端 + 操作"的框架侧类，注册表把字符串解析成实现；
- init 三步：签到 → 建 PG → 建通信域；new_group 一组一域；
- 下降路径止于 HcclAllReduce——职责分界线清晰可述。

## 参考资料

- [PyTorch Distributed 文档](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [《并行策略》04-2：通信域（语义视角）](../../parallel/04-distributed-basics/02-process-groups.md)
- [HCCL 源码专题 H01-3：通信域与 RankGraph](../hccl-source/03-comm-domain-rank-graph.md)

---

下一单元将进入 **03-3：Stream 与 Notify 的桥接**。

[返回第 3 章 →](../03-pytorch-to-hccl.md)
