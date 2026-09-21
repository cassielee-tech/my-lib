# 单元 01-3｜版本、驱动、固件与开发环境

> 所属章节：[第 1 章｜认识昇腾与 CANN](../01-ascend-cann.md)

::: info 本单元目标
读完后，你能够说出**昇腾开发环境的版本链组成与配套关系**（驱动/固件 → CANN → PyTorch + torch_npu），知道安装顺序与常见坑，并掌握登录服务器后的**环境检查与版本查询方法**。
:::

## 先记住 3 个结论

1. **开发环境是一条版本链**：硬件 → 驱动 + 固件 → CANN Toolkit → Python + PyTorch + `torch_npu`。每一环都对上下环有配套要求，任何一环错配都可能导致环境不可用。
2. **安装有顺序，使用有前提**：驱动/固件最先装（通常需要 root 并重启），CANN Toolkit 随后，框架与 Python 包最后；此后**每个新 Shell 都要先 `source set_env.sh`** 加载 CANN 环境。
3. **学概念可以跨版本，跑代码必须对齐版本**。概念（分层架构、执行路径）跨版本基本稳定；命令、路径、接口以实际环境的配套版本为准。

## 1. 开发环境由哪几层组成

对照上一单元的软件栈，从下往上数：

| 层次 | 内容 | 谁来安装/管理 |
| --- | --- | --- |
| 硬件 | 昇腾 NPU（Atlas A2/A3 系列等） | 服务器自带 |
| 驱动 + 固件 | Driver + Firmware，直接对接硬件 | 管理员（root），通常随整机交付 |
| CANN Toolkit | 算子库、编译器、Runtime、工具链 | 管理员或用户安装 |
| Python 环境 | Python 3.11、虚拟环境 | 用户 |
| 框架与适配 | PyTorch + `torch_npu` | 用户（`pip install`） |

::: warning CANN 不是单一包
CANN Toolkit 按能力拆分为多个包（如 toolkit、kernels 等），安装形态（root/非 root、全量/最小）会影响目录布局与 `set_env.sh` 位置。这直接解释了速查表里为什么有两种加载路径。
:::

## 2. 为什么必须版本配套

各层之间通过接口和二进制约定协作，版本升级可能改变这些约定：

1. **驱动/固件 ↔ CANN**：每个 CANN 版本都有配套的驱动/固件版本区间，Runtime 通过驱动访问硬件，接口不匹配会直接失败；
2. **CANN ↔ torch_npu**：`torch_npu` 的每个版本绑定特定 CANN 版本区间——动态库（如 `libascendcl.so`、`libhccl.so`）来自 CANN，符号或版本对不上就会出现 `ImportError`；
3. **torch_npu ↔ PyTorch**：`torch_npu` 按具体 PyTorch 版本构建，混装不兼容版本会在导入或运行时报错。

常见错配症状对照：

| 症状 | 优先怀疑 |
| --- | --- |
| `ImportError: libascendcl.so` / `libhccl.so` | 未 `source set_env.sh`、CANN 版本与 `torch_npu` 不配套 |
| `torch.npu.is_available()` 为 `False` | 版本配套、Python 环境混用、容器设备未挂载 |
| `npu-smi` 正常但 CANN 命令找不到 | 驱动可用，但当前 Shell 未加载 CANN 环境 |

## 3. 典型安装顺序

```text
① 确认硬件与 OS（uname -m、ls /dev/davinci*）
        ↓
② 安装驱动 + 固件（root，通常需重启）—— 一般服务器出厂已装
        ↓
③ 安装 CANN Toolkit（root 或用户目录）
        ↓
④ 每次开 Shell：source <CANN 路径>/set_env.sh
        ↓
⑤ 创建 Python 虚拟环境，按配套表安装 torch 与 torch-npu
        ↓
⑥ 验证：npu-smi info → torch.npu.is_available() → 最小张量测试
```

完整命令细节不在本单元展开——**环境检查、版本查询、加载环境变量的全部命令都在 [CANN 常用命令速查](../cann常用命令.md)**，那里按"环境确认 → 版本查看 → 常见问题"组织，建议搭配使用。

## 4. 登录服务器后的 30 秒检查

按顺序执行，可以定位绝大多数环境问题：

```bash
uname -m                      # CPU 架构：aarch64 / x86_64
npu-smi info                  # 驱动可用？设备健康？
echo "$ASCEND_HOME_PATH"      # CANN 环境是否已加载
python3 --version             # Python 版本
```

再用 Python 做框架层验证：

```python
import torch, torch_npu
print(torch.npu.is_available())   # True 才说明整条版本链打通
print(torch_npu.__version__)
```

**分层排障思路**：`npu-smi` 异常 → 驱动/固件层；`ASCEND_HOME_PATH` 为空 → CANN 环境层；`is_available()` 为 `False` → 框架配套层。自上而下缩小范围，不要一上来就重装。

## 5. 三种获得环境的方式

| 方式 | 适合谁 | 说明 |
| --- | --- | --- |
| 在线体验 / CANNLab 云环境 | 零基础、快速上手 | Learning Hub 教程支持浏览器在线运行；CANNLab 提供云 NPU 环境，选择 Python 3.11 内核 |
| 本地/服务器部署 | 长期学习开发 | 按 CANN 版本的安装指南与配套表操作 |
| 容器镜像 | 团队统一环境 | 注意容器需挂载 NPU 设备节点与驱动目录，否则一切正常却看不见设备 |

## 6. 常见坑清单

1. **忘记 `source`**：直接执行 `set_env.sh` 或开新终端未加载，导致 CANN 命令和动态库全部找不到；
2. **多个 Python 环境混用**：`torch` 与 `torch_npu` 不在同一虚拟环境，导入的是两份包；
3. **容器内看不到设备**：`ls /dev/davinci*` 为空通常是挂载问题，而不是 Python 问题；
4. **用错版本的资料**：照抄旧版本文档的命令参数在新环境执行失败——先 `--help` 确认当前版本支持项；
5. **擅自执行危险命令**：共享服务器上不要随意复位设备、升级固件或终止他人进程。

## 7. 自测题

先用自己的话回答，再展开答案。

1. 昇腾开发环境的版本链有哪几环？安装顺序是什么？
2. `npu-smi info` 正常，但 `ASCEND_HOME_PATH` 为空，说明什么？
3. 为什么 `torch`、`torch_npu`、CANN 三者必须版本配套？
4. `torch.npu.is_available()` 返回 `False`，你的排障顺序是什么？
5. 学概念和跑代码对版本的要求有何不同？

::: details 自测答案

1. 硬件 → 驱动 + 固件 → CANN Toolkit → Python + PyTorch + torch_npu。安装顺序：驱动/固件（root，可能需重启）→ CANN Toolkit → 虚拟环境中安装框架与适配包。
2. 驱动可用，但当前 Shell 没有加载 CANN Toolkit 环境，需要 `source` 正确的 `set_env.sh`。
3. torch_npu 按特定 PyTorch 版本构建，并依赖特定 CANN 版本提供的动态库（libascendcl.so、libhccl.so 等）；任何一环错配都会导致接口或符号不匹配，出现导入失败或运行异常。
4. 先查 CANN 环境是否加载（ASCEND_HOME_PATH），再查 torch/torch_npu 是否同一虚拟环境，再查版本配套关系，再查容器设备挂载与权限——按"环境 → 框架 → 配套 → 挂载"分层缩小范围。
5. 概念跨版本基本稳定，可用最新资料学习；命令、路径、接口、源码结构随版本变化，运行时必须以实际环境的配套版本为准。

:::

## 本单元小结

- **版本链**：驱动/固件 → CANN → Python → torch + torch_npu，环环配套；
- **使用前提**：每个 Shell 先 `source set_env.sh`，命令细节见 [速查表](../cann常用命令.md)；
- **排障思路**：按硬件驱动 → CANN 环境 → 框架配套分层定位，从现象缩小范围；
- **环境获取**：在线体验 / CANNLab / 本地部署 / 容器，按阶段选择；
- **版本意识**：这一章的所有概念在后面章节反复用到，但操作命令永远以实际环境为准。

## 参考资料

- [CANN Learning Hub：什么是 CANN（环境验证小节）](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start/cann_basics/03_what_is_cann.ipynb)
- [CANN 9.0 软件安装与环境配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/softwareinst/instg/instg_0054.html)
- [昇腾社区：CANN 下载](https://www.hiascend.com/cann/download)
- [昇腾社区：PyTorch 框架适配](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch)
- [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)
- [CANN 常用命令速查](../cann常用命令.md)

---

下一单元将进入 **本章总结：画出一段模型代码到 NPU 的完整路径**。

[返回第 1 章 →](../01-ascend-cann.md)
