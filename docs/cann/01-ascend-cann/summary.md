# 本章总结｜画出一段模型代码到 NPU 的完整路径

> 所属章节：[第 1 章｜认识昇腾与 CANN](../01-ascend-cann.md)

三个单元走完，把硬件、软件栈和环境三张图拼成一张完整的路径图，并核对本章开头立下的目标。

## 一张图收束整章

```text
你的代码            output = torch.matmul(a.npu(), b.npu())
                          │
框架适配层          torch_npu：注册 npu 设备；aten 算子映射到 CANN 算子
                          │
图引擎 GE           图融合 / 内存复用 / 自动流水 / 执行下沉（图模式）
                          │
毕昇编译器          算子源码 ──► NPU 可执行的 Kernel 指令
                          │
Runtime            分配 Device 内存、创建 Stream、下发 Kernel
                          │
Driver             指令直达硬件，设备全生命周期管理
                          ↓  ───────── 以上：软件栈（C01-2）
设备内数据流        Global Memory → DMA/MTE 搬入 → UB / L1 → L0A / L0B
                          │
计算单元            Cube 算矩阵 · Vector 算向量 · Scalar 控制与发射
                          │
写回                L0C → FixPipe → Global Memory
                          ↓  ───────── 以下：硬件数据流（C01-1）
```

支撑这张图运行的环境链（C01-3）：

```text
驱动 + 固件 ──► CANN Toolkit ──► Python venv ──► torch + torch_npu
（root，重启）   （set_env.sh）     （3.11）        （pip，版本配套）
```

## 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| NPU vs AI Core | 设备 vs 设备内的核心计算资源 | C01-1 |
| Cube / Vector / Scalar | 矩阵计算 / 向量计算 / 控制与指令发射 | C01-1 |
| 存储层次 | 数据要从 Global Memory 搬进 Local Memory（UB/L1/L0）才能高效计算 | C01-1 |
| 流水重叠 | 搬入、计算、搬出重叠执行——Queue、Double Buffer 的硬件根源 | C01-1 |
| CANN 定位 | 框架与硬件之间的软件栈：翻译官 + 调度员 + 工具箱 | C01-2 |
| 分层组件 | torch_npu / 算子库 / HCCL / GE / Ascend C / 毕昇 / Runtime / Driver | C01-2 |
| 两条执行路径 | Eager 逐算子下发 vs 图模式整网编译下沉 | C01-2 |
| 一行代码旅程 | 搬数据 → 算子映射 → 图优化 → 编译 → 调度 → 下发 → 执行 → 写回 | C01-2 |
| 版本链 | 驱动/固件 → CANN → torch_npu → torch，环环配套 | C01-3 |
| `source set_env.sh` | 每个 Shell 使用 CANN 的前提 | C01-3 |
| 分层排障 | `npu-smi` → `ASCEND_HOME_PATH` → `is_available()` 逐层缩小范围 | C01-3 |

## 章节自检清单

回到导学立下的三个目标，逐条核对：

- [ ] 我能解释 NPU、AI Core、Cube、Vector 和 Scalar 的关系（C01-1）
- [ ] 我能画出模型代码进入 CANN 并在 NPU 执行的路径（C01-2，见上图）
- [ ] 我能说清驱动、固件、CANN、PyTorch 与 `torch_npu` 为什么要配套（C01-3）

额外加三条实操向的检查：

- [ ] 我能不看笔记说出 `.npu()` 之后到 Cube 开始计算之间发生了什么
- [ ] 我拿到一台新服务器，知道 30 秒环境检查要跑哪几条命令
- [ ] 我知道遇到 `is_available()` 为 `False` 时按什么顺序排查

## 下一章预告

本章多次按下不表的概念——**Stream、Event、Memory、任务下发与同步**——正是第 2 章「Runtime 与任务执行」的主角：

- C02-1：Host、Device 与异构计算；
- C02-2：设备内存、数据搬运与生命周期；
- C02-3：Stream、Event、Task 与异步执行；
- C02-4：一次算子调用怎样被下发和完成。

这些概念也是 HCCL 中"通信任务与计算任务协同"的基础。

## 最终参考资料

- [CANN Learning Hub：quick_start 公共基础](https://gitcode.com/cann/cann-learning-hub/tree/master/quick_start/cann_basics)（本章三个单元的原生课程）
- [昇腾社区：CANN 主页](https://www.hiascend.com/cann)
- [Ascend C 算子开发指南：硬件架构](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0018.html)
- [CANN 9.0 软件安装与环境配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/softwareinst/instg/instg_0054.html)
- [CANN 常用命令速查](../cann常用命令.md)

---

[返回第 1 章 →](../01-ascend-cann.md) · [返回 CANN 目录 →](../index.md)
