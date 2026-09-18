# VS Code + Remote SSH + Jupyter

这套工作流适合这样的场景：代码和编辑界面在自己的电脑上操作，但 Python、模型、数据以及 GPU/NPU 计算都运行在远程服务器。

::: tip 推荐方案
优先使用 **VS Code Remote SSH + 远程 Python 环境 + VS Code Notebook 编辑器**。连接后直接打开远程目录并选择远程内核，通常不需要自己启动 Jupyter Server，也不需要把 `8888` 端口暴露到公网。
:::

## 1. 先建立正确的心智模型

```text
本地电脑
┌───────────────────────────────────┐
│ VS Code 界面                      │
│ 编辑器、快捷键、主题              │
└───────────────┬───────────────────┘
                │ 加密 SSH 连接
远程服务器      ▼
┌───────────────────────────────────┐
│ VS Code Server                    │
│ Python / Jupyter 扩展             │
│ 项目代码、数据、模型权重          │
│ Python 环境与 Notebook Kernel     │
│ CUDA / CANN / GPU / NPU           │
└───────────────────────────────────┘
```

**最关键的一点：** Remote SSH 窗口里的终端、Python 扩展、Notebook Kernel 和代码文件都属于远程服务器。你的 Mac 只负责显示和交互。

这意味着：

- 不需要在本地安装 CUDA、CANN 或模型依赖；
- `pip install`、`conda create` 等命令要在**远程终端**执行；
- Notebook 中看到的文件路径是远程路径；
- Notebook 能否使用 GPU/NPU，取决于所选远程 Python 环境，而不是本地环境。

## 2. 两种连接方式怎么选

| 方式 | 数据和内核在哪里 | 适用场景 | 推荐度 |
| --- | --- | --- | --- |
| Remote SSH 后直接选择 Python Kernel | 远程服务器 | 日常开发、调试、实验 | **首选** |
| 独立 Jupyter Server + SSH 隧道 | 远程服务器 | 长期运行的 Jupyter 服务、计算节点或浏览器访问 | 备选 |

本文先讲首选方案，最后再介绍 SSH 隧道。

## 3. 本地需要准备什么

在本地安装：

1. [Visual Studio Code](https://code.visualstudio.com/)；
2. Microsoft 的 [Remote - SSH 扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)；
3. 可用的 OpenSSH 客户端。macOS 通常已经自带，可用下面命令检查：

```bash
ssh -V
```

连接远程服务器前，先在普通终端确认 SSH 本身可以工作：

```bash
ssh username@server.example.com
```

如果这一步失败，应先解决账号、网络、VPN、密钥或服务器 SSH 服务问题。VS Code 建立在 SSH 之上，不会绕过底层连接错误。

## 4. 配置一个容易记住的 SSH Host

编辑本地文件：

```text
~/.ssh/config
```

加入：

```ssh-config
Host llm-lab
    HostName server.example.com
    User cassie
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

以后不需要记服务器地址：

```bash
ssh llm-lab
```

字段含义：

| 字段 | 作用 |
| --- | --- |
| `Host` | 给这台服务器起的本地别名 |
| `HostName` | 真实域名或 IP |
| `User` | 远程用户名 |
| `IdentityFile` | SSH 私钥路径 |
| `ServerAliveInterval` | 定期发送保活消息 |
| `ServerAliveCountMax` | 连续多少次无响应后断开 |

::: warning 不要把私钥上传到服务器或 Git 仓库
私钥留在自己的电脑中，服务器只保存对应的公钥。配置文件也不要记录密码、Token 或其他凭证。
:::

如果必须经过跳板机，可以使用：

```ssh-config
Host llm-bastion
    HostName bastion.example.com
    User cassie
    IdentityFile ~/.ssh/id_ed25519

Host llm-lab
    HostName 10.0.0.12
    User cassie
    ProxyJump llm-bastion
    IdentityFile ~/.ssh/id_ed25519
```

## 5. 用 VS Code 连接服务器

1. 打开命令面板：macOS 使用 `⇧ ⌘ P`，Windows/Linux 使用 `Ctrl Shift P`；
2. 执行 `Remote-SSH: Connect to Host...`；
3. 选择 `llm-lab`；
4. 第一次连接时确认远程系统类型，例如 Linux；
5. 连接完成后，选择 `File → Open Folder...`，打开远程项目目录。

VS Code 会在远程端安装 **VS Code Server**。窗口左下角应显示当前 SSH Host，这也是判断“现在到底在本地还是远程”的最快方法。

### 扩展安装在哪里

连接成功后，在这个远程窗口中安装：

- **Python**：`ms-python.python`
- **Jupyter**：`ms-toolsai.jupyter`

主题、图标等界面扩展通常运行在本地；Python、Jupyter 等需要读取代码和调用解释器的扩展通常运行在远程端。扩展面板会分别显示本地和 SSH Host 的安装状态。

## 6. 在远程服务器创建项目环境

先打开 VS Code 的远程终端：

```text
Terminal → New Terminal
```

确认终端确实位于远程服务器：

```bash
hostname
pwd
which python3
```

### 方案 A：使用 venv

```bash
cd ~/projects/llm-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ipykernel jupyterlab
```

### 方案 B：使用 Conda

```bash
conda create -n llm-lab python=3.11 -y
conda activate llm-lab
python -m pip install ipykernel jupyterlab
```

大模型框架不要盲目复制网上的一条安装命令：

- NVIDIA 环境根据服务器驱动和 CUDA 条件，从 [PyTorch 官方安装页](https://docs.pytorch.org/get-started/locally/)选择匹配版本；
- 昇腾环境应严格按照服务器当前的 **CANN、PyTorch、torch_npu 和 Python 配套关系**安装，优先使用团队提供的容器或环境；
- 公共服务器先查看已有 Module、Conda 环境或基础镜像，避免重复安装和破坏共享环境。

### 注册一个名称清楚的 Kernel

激活目标环境后执行：

```bash
python -m ipykernel install \
  --user \
  --name llm-lab \
  --display-name "Python (llm-lab)"
```

`--name` 是内部标识，`--display-name` 是 VS Code Kernel 列表中显示的名称。多个项目应使用不同名字，避免选错环境。

## 7. 打开 Notebook 并选择远程 Kernel

在远程项目中创建或打开一个 `.ipynb` 文件，然后：

1. 点击 Notebook 右上角的 **Select Kernel**；
2. 选择 `Select Another Kernel...`；
3. 选择 `Python Environments` 或 `Jupyter Kernels`；
4. 选择 `Python (llm-lab)`。

第一格先不要加载模型，先验证执行位置：

```python
import os
import socket
import sys

print("hostname:", socket.gethostname())
print("python:", sys.executable)
print("working directory:", os.getcwd())
```

你应该看到：

- `hostname` 是远程服务器；
- `sys.executable` 指向刚创建的 `.venv` 或 Conda 环境；
- 工作目录是远程项目路径。

::: info 排错第一原则
遇到 `ModuleNotFoundError` 或设备不可见时，先看 `sys.executable`。Notebook 选错 Kernel，比“包真的没装”更常见。
:::

## 8. 验证 GPU 或 NPU

### NVIDIA GPU

先在远程终端查看设备：

```bash
nvidia-smi
```

再在 Notebook 中验证 PyTorch：

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("tensor device:", x.device)
```

### 昇腾 NPU

先在远程终端检查设备和环境。具体命令以服务器 CANN 版本与管理员说明为准，常见检查方式是：

```bash
npu-smi info
```

Notebook 中可以做最小验证：

```python
import torch
import torch_npu

print("PyTorch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("device count:", torch.npu.device_count())

if torch.npu.is_available():
    x = torch.randn(1024, 1024, device="npu")
    print("tensor device:", x.device)
```

如果终端可以导入 `torch_npu`，Notebook 却失败，通常检查：

1. Notebook 是否选择了同一个 Python；
2. CANN 环境变量是否对 Kernel 进程生效；
3. `torch`、`torch_npu`、CANN 和 Python 版本是否匹配；
4. 当前用户是否拥有设备访问权限。

## 9. 一个最小的大模型实验模板

推荐每个实验 Notebook 都从四个区域开始。

### ① 记录环境

```python
import platform
import sys
import torch

print("Python:", sys.version)
print("OS:", platform.platform())
print("PyTorch:", torch.__version__)
```

### ② 固定随机性

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

### ③ 选择并确认设备

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch, "npu") and torch.npu.is_available():
    device = torch.device("npu")
else:
    device = torch.device("cpu")

print("device:", device)
```

### ④ 运行最小计算

```python
x = torch.randn(2048, 2048, device=device)
w = torch.randn(2048, 2048, device=device)
y = x @ w

print("shape:", y.shape)
print("device:", y.device)
print("dtype:", y.dtype)
```

先让小张量正确运行，再下载模型、扩大 Batch 或启用多卡。这样可以把“环境错误”和“实验代码错误”分开。

## 10. 推荐的项目目录

```text
llm-lab/
├── notebooks/          # 探索和图表
│   └── 01-baseline.ipynb
├── src/                # 可复用实现
├── scripts/            # 训练、评测和启动脚本
├── configs/            # 模型与实验配置
├── tests/              # 小规模正确性测试
├── outputs/            # 日志与结果，通常不提交
├── checkpoints/        # 权重，通常不提交
├── requirements.txt    # 或 pyproject.toml / environment.yml
├── .gitignore
└── README.md
```

Notebook 适合观察张量、验证公式和绘图；正式训练逻辑应逐步移动到 `src/` 与 `scripts/`。否则实验会依赖隐藏的单元执行顺序，很难复现。

建议加入 `.gitignore`：

```text
.venv/
.ipynb_checkpoints/
__pycache__/
outputs/
checkpoints/
data/
*.pt
*.pth
*.safetensors
```

是否忽略 Notebook 输出取决于项目用途：教学笔记可以保留少量结果；包含大量日志、图片或敏感数据时，应在提交前清理输出。

## 11. 需要独立 Jupyter Server 时

某些场景不适合让 VS Code 自动启动 Kernel，例如：

- Jupyter 需要在调度系统分配的计算节点上长期运行；
- 希望同一个 Jupyter Server 被浏览器和 VS Code 使用；
- SSH 会频繁断开，但实验会话需要由 `tmux` 保持；
- 团队已经提供统一的 Jupyter 服务。

### 第一步：在远程端启动服务

激活环境后：

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Jupyter 默认会打印带 Token 的地址。不要把 Token 发到聊天、Issue、截图或 Git 仓库。

### 第二步：在本地建立 SSH 隧道

另开一个本地终端：

```bash
ssh -N -L 8888:127.0.0.1:8888 llm-lab
```

这条命令表示：

```text
本地 127.0.0.1:8888
        ↓ SSH 加密隧道
远程 127.0.0.1:8888
```

### 第三步：让 VS Code 连接这个 Server

在 Notebook 右上角选择：

```text
Select Kernel
→ Select Another Kernel...
→ Existing Jupyter Server
→ Enter the URL of the running Jupyter server
```

输入远程服务打印的 Token URL，但主机使用本地隧道地址，例如：

```text
http://127.0.0.1:8888/lab?token=服务器生成的Token
```

::: danger 不要为省事直接暴露 Jupyter 端口
访问 Jupyter Server 等同于获得执行代码的能力。个人实验优先让它监听远程 `127.0.0.1`，再通过 SSH 隧道访问；不要随意使用 `0.0.0.0`、关闭 Token，或把 `8888` 暴露到公网。
:::

## 12. 在集群和计算节点上使用

如果服务器由 Slurm 等调度系统管理，不要直接在登录节点运行大模型。典型流程是：

```text
本地 VS Code
  → SSH 登录节点：编辑代码
  → 调度器申请计算节点
  → 在计算节点启动 Jupyter 或训练脚本
  → 按集群规则建立隧道或提交 Batch Job
```

不同集群对端口转发、计算节点登录、GPU/NPU 申请和最长运行时间的规则差异很大，应优先遵守管理员提供的命令。不要照搬其他集群的 `srun`、`salloc` 或端口配置。

长任务建议写成脚本交给调度器；Notebook 更适合小规模验证、分析结果和调试单个步骤。

## 13. 常见问题排查

### VS Code 一直停在 Connecting

按这个顺序检查：

1. 本地终端能否执行 `ssh llm-lab`；
2. VS Code 的 `Remote - SSH` Output 日志；
3. 远程端是否具备 Bash、`tar`、`curl` 或 `wget`；
4. 远程磁盘配额是否已满；
5. VS Code Server 下载是否被网络或代理阻止。

需要查看 SSH 细节时：

```bash
ssh -vvv llm-lab
```

### Kernel 列表里没有目标环境

在远程终端执行：

```bash
which python
python -m pip show ipykernel
jupyter kernelspec list
```

然后重新执行 Kernel 注册命令，并在 VS Code 中运行 `Developer: Reload Window`。

### 终端可以 import，Notebook 不可以

Notebook 中执行：

```python
import sys
print(sys.executable)
```

再在远程终端执行：

```bash
which python
python -m pip show 目标包名
```

两个 Python 路径必须属于同一个环境。安装包时使用 `python -m pip`，比直接使用 `pip` 更不容易装错环境。

### Notebook 占着显存不释放

删除 Python 变量不一定立刻结束 Kernel 进程。优先使用 Notebook 工具栏中的：

```text
Restart Kernel
```

然后用 `nvidia-smi` 或 `npu-smi info` 确认进程和显存状态。不要随意终止其他用户的进程。

### SSH 断开后任务也停了

交互终端中的前台任务通常会随连接中断。长任务使用调度系统、`tmux` 或 `systemd-run` 等服务器允许的持久化方式，不要依赖 VS Code 窗口一直在线。

## 14. 推荐的日常工作流

```text
1. Remote SSH 打开远程项目
2. 确认左下角 SSH Host
3. 选择项目对应的 Notebook Kernel
4. 打印 hostname、sys.executable 和设备信息
5. 用小张量验证环境
6. 再运行模型实验
7. 把稳定逻辑从 Notebook 移到 src/ 或 scripts/
8. 提交代码和配置，不提交模型、数据与凭证
9. 长任务交给 tmux 或集群调度器
```

::: tip 一句话总结
**VS Code 负责本地交互，SSH 负责安全连接，远程 Python Kernel 负责执行，GPU/NPU 负责计算。** 出现问题时，沿着这四层逐层确认，不要一开始就重装所有依赖。
:::

## 参考资料

- [VS Code 官方文档：Remote Development using SSH](https://code.visualstudio.com/docs/remote/ssh)
- [VS Code 官方文档：Remote Development Tips and Tricks](https://code.visualstudio.com/docs/remote/troubleshooting)
- [VS Code 官方文档：Jupyter Notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [VS Code 官方文档：Manage Jupyter Kernels](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)
- [Jupyter Server 官方文档：Security](https://jupyter-server.readthedocs.io/en/latest/operators/security.html)
- [IPython 官方文档：Installing the IPython Kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
- [Python 官方文档：venv](https://docs.python.org/3/library/venv.html)
- [PyTorch 官方文档：Start Locally](https://docs.pytorch.org/get-started/locally/)
- [OpenSSH 配置手册：ssh_config](https://man.openbsd.org/ssh_config)
