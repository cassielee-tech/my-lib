# 本章总结｜追踪一次计算与集合通信的交汇点

> 所属章节：[第 3 章｜从 PyTorch 走向 HCCL](../03-pytorch-to-hccl.md)

四个单元走完，把框架侧的调用链拼成一张总图，并核对本章开头立下的目标。

## 一张图收束整章

```text
你的代码            y = a + b          dist.all_reduce(bucket)
                        │                      │
框架适配层          torch_npu（三件注册）        torch.distributed
                    dispatcher → op-plugin   c10d → ProcessGroupHCCL
                        │                      │ 检查 · 取流 · comm
算子/通信入口       aclnnAdd 两段式          HcclAllReduce(…, comm, stream)
                        │                      │
昇腾执行栈          ──── Runtime：Stream / Event / Notify ────
                        │                      │
                    AI Core 计算            selector → executor
                    （第 2 章）             → transport → 引擎（H01-5/6/7）
```

上半段（计算路径）由 03-1 铺设，下半段（通信路径）由 03-2/03-3 铺设，03-4 把两段在**同一条 Stream 时间线**上缝合——这就是"计算与集合通信的交汇点"的准确含义。

## 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| 三件注册 | 设备、算子、流/事件——torch_npu 的桥 | 03-1 |
| dispatcher | 按设备键路由，框架不感知后端 | 03-1 |
| op-plugin | aten 语义 → aclnn 两段式的翻译层 | 03-1 |
| 四层结构 | Python → c10d → ProcessGroup → 后端实现 | 03-2 |
| init 三步 | 签到 → 建 PG → 建通信域（HcclComm） | 03-2 |
| 职责分界线 | 框架管"何时/对谁/哪条流"，HCCL 管"怎么跑" | 03-2 |
| 流的缝合线 | torch.npu.Stream = aclrtStream = stream 参数 | 03-3 |
| Event vs Notify | 框架层流间依赖 vs 库内跨设备握手 | 03-3 |
| 六站时序 | hook → dist → PGHCCL → 入口 → 库内 → 挂流返回 | 03-4 |

## 章节自检清单

- [ ] 我能画出 `y = a + b` 从 Python 到 aclnn 的路径（03-1）
- [ ] 我能说出 init_process_group 三步与通信域的关系（03-2）
- [ ] 我能解释传"当前流"与"通信流"的后果差异（03-3）
- [ ] 我能独立复述 AllReduce 六站时序并指出分界线（03-4）

实操向追加：

- [ ] 我能在源码里找到 ProcessGroupHCCL 的 allreduce 实现（torch_npu 仓库）
- [ ] 我知道排障时哪一层看什么日志（torch_npu 日志 vs HCCL 日志，见[常用命令速查](../cann常用命令.md)第 8 节）

## 与主线的接口

本章是 **HCCL 开发主线全部主线材料的收官**：模型全景 → 训练与推理系统 → 单卡执行系统 → 并行策略 → 集合通信 → 昇腾平台与调用链——需求侧（为什么通信）与供给侧（怎么实现）在本章会师。

下一步的三个方向：

1. **下潜源码**：[HCCL 源码专题](../hccl-source.md)——从 H01-1 仓库地图开始，H01-7 承接本章的第四站；
2. **补齐平台**：[第 4-6 章](../index.md)（Ascend C 算子开发，二梯队）——进入引擎 template 开发时回读；
3. **回望理论**：带着调用链的具体问题回读 [《集合通信》](../../collective/index.md)（例如"selector 在哪一步用到了 α-β-γ"）。

## 最终参考资料

- [HCCL 开源仓库](https://gitcode.com/cann/hccl) 与 [HCOMM 通信基础库](https://gitcode.com/cann/hcomm)
- [HCCL 官方 API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0001.html)
- [昇腾社区：PyTorch 框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)
- [《模型全景》01-3：从 PyTorch 调用到 HCCL（模型视角）](../../model/01-landscape/03-pytorch-to-hccl.md)
- [《并行策略》第 4-5 章：分布式基础与 DDP](../../parallel/04-distributed-basics.md)

---

[返回第 3 章 →](../03-pytorch-to-hccl.md) · [返回专栏目录 →](../index.md)
