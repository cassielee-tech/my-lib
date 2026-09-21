# 第 1 章｜认识昇腾与 CANN

> 本章目标：建立从 NPU 硬件到 CANN 软件栈的全景认识；分清 AI Core、计算单元、内存与数据搬运；理解驱动、固件、CANN 和框架插件为什么必须版本匹配。

## 本章导学

::: tip 本章只记住 3 件事
1. **NPU 是硬件，CANN 是让软件使用这块硬件的软件栈**，两者不能混为一谈。
2. AI Core 不只负责计算：数据要先从 Global Memory 搬到 Local Memory，再由 Cube/Vector 处理。
3. 学习资料可以跨版本理解概念，运行代码和阅读源码必须回到实际环境的配套版本。
:::

**学习节奏：** 本章拆为 **3 个学习单元，每个约 15 分钟**。先进入芯片内部，再回到软件栈，最后处理最容易踩坑的环境与版本问题。

- [ ] 我能解释 NPU、AI Core、Cube、Vector 和 Scalar 的关系
- [ ] 我能画出模型代码进入 CANN 并在 NPU 执行的路径
- [ ] 我能说清驱动、固件、CANN、PyTorch 与 `torch_npu` 为什么要配套

## 本章单元

- **01-1（约 15 分钟）**：[NPU、达芬奇架构与 AI Core](01-ascend-cann/01-npu-davinci-ai-core.md)
- **01-2（约 15 分钟）**：[CANN 软件栈与模型执行路径](01-ascend-cann/02-cann-stack-execution-path.md)
- **01-3（约 15 分钟）**：[版本、驱动、固件与开发环境](01-ascend-cann/03-version-driver-firmware-env.md)
- **本章总结**：[画出一段模型代码到 NPU 的完整路径](01-ascend-cann/summary.md)

## 这一章在整条路线中的位置

```text
大模型基础：模型要计算什么
          ↓
AI Infra：计算系统有哪些通用层次
          ↓
第 1 章：这些层次在昇腾上分别是什么
          ↓
Ascend C：怎样为 AI Core 编写算子
          ↓
HCCL：多个 NPU 怎样协作完成集合通信
```

---

[开始单元 01-1 →](01-ascend-cann/01-npu-davinci-ai-core.md)
