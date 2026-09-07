# 第 9 章｜预训练、SFT、LoRA 与偏好对齐

> 本章目标：理解 Base Model 怎样变成能够对话和遵循指令的 Assistant；区分预训练、继续预训练、SFT 与偏好对齐；看懂 LoRA 的低秩增量；建立 RLHF 和 DPO 的基本数据流。

## 本章导学

::: tip 本章核心结论
1. 预训练学习续写能力，SFT 用示范塑造指令行为，偏好对齐进一步调整回答倾向。
2. LoRA 冻结基础权重，只训练低秩增量；QLoRA 再把冻结的基础权重量化。
3. 对齐改善行为偏好，但不自动保证事实正确。
:::

**学习节奏：** 本章拆为 **4 个学习单元，每个约 15 分钟**，先分清训练阶段，再比较实现代价。

**先看这几张核心图：** Model Training Stages、LoRA Low-Rank Update、RLHF vs DPO 三张图。

- [ ] 我能区分预训练、SFT 和偏好对齐
- [ ] 我能解释 LoRA 训练了什么、没训练什么
- [ ] 我知道对齐不能替代事实评测

## 本章单元

- **09-1（约 15 分钟）**：[预训练、继续预训练与 SFT](./09-post-training-alignment/01-pretrain-sft.md)
- **09-2（约 15 分钟）**：[Full Fine-Tuning、LoRA 与 QLoRA](./09-post-training-alignment/02-full-finetune-lora.md)
- **09-3（约 15 分钟）**：[RLHF 与 DPO](./09-post-training-alignment/03-rlhf-dpo.md)
- **09-4（约 15 分钟）**：[数据质量、显存与通信](./09-post-training-alignment/04-data-memory-communication.md)

- **本章总结**：[串联知识、练习与检查](./09-post-training-alignment/summary.md)
