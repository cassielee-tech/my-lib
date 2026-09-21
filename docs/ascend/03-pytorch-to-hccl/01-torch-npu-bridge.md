# 单元 03-1｜torch_npu：框架与 CANN 之间的桥

> 所属章节：[第 3 章｜从 PyTorch 走向 HCCL](../03-pytorch-to-hccl.md)

::: info 本单元目标
读完后，你能够说出 **torch_npu 的三件注册（设备、算子、流/事件）与 `y = a + b` 的完整计算路径**，并对照 CUDA 生态说出每个组件的对应物。
:::

## 先记住 3 个结论

1. **`import torch_npu` 的瞬间发生三件事**：把 `npu` 注册为 PyTorch 后端设备、把 aten 算子映射到昇腾实现、把 `torch.npu.Stream/Event` 接到 CANN Runtime——桥就这么搭起来。
2. **计算路径的形态是"适配器 + 两段式"**：aten 算子 → torch_npu 的适配层（op-plugin）→ aclnn 两段式（[第 2 章](../02-runtime-task-execution/04-op-call-lifecycle.md)）——框架不知道 NPU，torch_npu 负责翻译。
3. **生态对照一一映射**：torch.cuda ↔ torch.npu、cuBLAS/cuDNN ↔ AOL 算子库、NCCL ↔ HCCL——概念相通，只是名字不同。

## 1. 桥搭在哪：三件注册

[第 1 章 01-2](../01-ascend-cann/02-cann-stack-execution-path.md) 的分层图里，torch_npu 位于框架与 CANN 之间。它做的事可以归为三件注册：

| 注册 | 效果 | 你看到的 API |
| --- | --- | --- |
| **设备** | `npu` 成为合法后端 | `x.npu()`、`torch.npu.set_device()` |
| **算子** | aten 算子有 NPU 实现 | `y = a + b` 直接在 NPU 上跑 |
| **流/事件** | 异步原语对应 Runtime | `torch.npu.Stream`、`torch.npu.Event` |

注册的机制是 PyTorch 的 **dispatcher**：每个算子调用按设备键（device key）路由——`npu` 键的实现全部住在 torch_npu 里。框架本体不需要知道昇腾的存在。

安装自检（详见 [CANN 常用命令速查第 6 节](../cann常用命令.md)）：

```python
import torch, torch_npu
print(torch.npu.is_available())      # True 才有下文
print(torch_npu.__version__)
```

## 2. `y = a + b` 的计算路径

一个 npu tensor 的加法，从 Python 到硬件：

```text
y = a + b                    # Python：熟悉的写法，无任何 NPU 痕迹
  ↓ PyTorch dispatcher（设备键 = npu）
aten::add 的 NPU 实现          # torch_npu 适配层（op-plugin）
  ↓ 检查/布局处理后
aclnnAddGetWorkspaceSize      # ① 准备段（纯 Host，可缓存复用）
aclrtMalloc(workspace)
aclnnAdd(executor, stream)    # ② 执行段：任务挂上当前流
  ↓ Runtime → Driver（回扣第 2 章双时间线）
AI Core 执行
```

对照 [第 2 章 02-4](../02-runtime-task-execution/04-op-call-lifecycle.md)：torch_npu 的适配层只是在这条已学过的路径**前面加了一层翻译**——它把 aten 的语义（shape 推导、广播规则、autograd 注册）翻译成对 aclnn 的调用。**框架侧的每一步，都落在第 2 章的地基上。**

## 3. 与 CUDA 生态的对照

完整对照表见笔记 [CANN Learning Hub](../cann-learning-hub.md)，本章最常用的四行：

| 维度 | NVIDIA | 昇腾 |
| --- | --- | --- |
| 框架适配 | torch.cuda | **torch_npu** |
| 算子库 | cuBLAS / cuDNN | AOL（ops-math/nn/...） |
| 通信库 | NCCL（ProcessGroupNCCL） | **HCCL（ProcessGroupHCCL）** |
| 数据搬运 | cudaMemcpy | aclrtMemcpy |

## 4. 自测题

先用自己的话回答，再展开答案。

1. `import torch_npu` 之后发生的三件注册分别是什么？
2. dispatcher 在算子调用中起什么作用？为什么说"框架本体不需要知道昇腾"？
3. 写出 `y = a + b`（npu tensor）的完整路径（六步）。
4. op-plugin 适配层负责翻译什么？
5. `torch.npu.is_available()` 为 `False` 时，按什么顺序排查？

::: details 自测答案

1. 注册 `npu` 设备后端；把 aten 算子映射到昇腾实现（适配层 + aclnn）；把 `torch.npu.Stream/Event` 对接到 CANN Runtime。
2. 按设备键把算子调用路由到对应实现——`npu` 键的实现都在 torch_npu 里，PyTorch 本体无需修改。
3. Python 调用 → dispatcher 路由到 aten::add 的 NPU 实现 → op-plugin 适配层 → aclnnAddGetWorkspaceSize（准备段）→ 申请 workspace → aclnnAdd 入队 → AI Core 执行。
4. aten 的语义（shape 推导、广播、autograd 注册、布局处理）翻译成对 aclnn 两段式的调用。
5. 驱动/固件 → CANN 环境 → torch/torch_npu 版本配套 → `npu-smi` 可见性（详见速查第 6 节的分层排障）。

:::

## 本单元小结

- torch_npu = 设备注册 + 算子映射 + 流/事件对接，三件套把桥搭起来；
- 计算路径 = dispatcher → 适配层 → aclnn 两段式——第 2 章地基上的一层翻译；
- 生态四行对照：torch.cuda / cuBLAS / NCCL / cudaMemcpy 的昇腾对应物。

## 参考资料

- [昇腾社区：PyTorch 框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)
- [第 1 章 01-2：CANN 软件栈与执行路径](../01-ascend-cann/02-cann-stack-execution-path.md)
- [第 2 章 02-4：一次算子调用的旅程](../02-runtime-task-execution/04-op-call-lifecycle.md)
- [CANN Learning Hub 笔记：CUDA 对照表](../cann-learning-hub.md)

---

下一单元将进入 **03-2：torch.distributed 的门面：ProcessGroup**。

[返回第 3 章 →](../03-pytorch-to-hccl.md)
