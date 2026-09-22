# 训练与推理系统

> 返回：[课程总览](../roadmap.md)

**专栏目标**：训练循环全景、显存账本、梯度通信的时机与重叠；推理与后训练作为选修视角补全模型生命周期。

## 章节列表

| 章 | 主题 | 学完能够回答的问题 | 状态 |
| ---: | --- | --- | --- |
| 1 | [训练循环](./01-training-loop.md) | 一次训练 Step 发生什么？梯度累积、AdamW、混合精度各自做什么？ | ✅ 主线 |
| 2 | [训练显存与计算优化](./02-training-memory-compute.md) | 训练显存的四个大件各占多少？Checkpointing/累积/Offload 分别省什么？ | ✅ 主线 |
| 3 | [推理与 KV Cache](./03-inference-kv-cache.md)【选修】 | Prefill 与 Decode 有何不同？KV Cache 占多少显存？ | ✅ |
| 4 | [后训练与对齐](./04-post-training-alignment.md)【选修】 | SFT/LoRA/RLHF/DPO 分别解决什么？ | ✅ |
| 5 | [推理引擎与 KV Cache 管理](./05-inference-engine.md)【选修】 | Continuous Batching、Paged Attention 和 Prefix Cache 怎样提高并发？ | ✅ |
| 6 | [Prefill/Decode 调度与推理指标](./06-pd-scheduling.md)【选修】 | TTFT、TPOT、吞吐为什么互相制约？PD 分离解决什么？ | ✅ |

## 阅读提示

- **主线只需第 1、2 章**：先看模型侧的训练循环（第 1 章），再用系统侧的显存账本（第 2 章）落地；
- 第 3-6 章为推理/后训练视角（选修）：第 3 章讲单请求生成，第 5 章讲多请求服务，第 6 章讲指标与 PD 分离——对 HCCL 岗位，推理通信是训练通信互补的另一半战场；
- 第 1 章的 01-4（显存与通信）与第 2 章是同一问题的两个视角，建议连读。

## 验收清单

- [ ] 算出 7.5B 模型的训练显存账本（16 字节/参数）；
- [ ] 说出梯度 AllReduce 发生在反向与更新之间、为什么能与反向计算重叠；
- [ ] 能为一次训练任务画出"计算-通信"时间线草图。

---

[返回课程总览 →](../roadmap.md) · [进入《单卡执行系统》 →](../device/index.md)
