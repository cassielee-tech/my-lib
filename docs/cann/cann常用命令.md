---
title: CANN 常用命令速查
description: CANN 环境、版本、NPU 状态、PyTorch、日志、Profiling 与分布式排障命令。
---

# CANN 常用命令速查

这篇文档按实际工作流整理：**先确认硬件和版本，再检查框架，最后定位日志与性能问题**。示例默认在 Linux 服务器终端执行。

::: warning 使用前先确认
- `npu-smi` 的子命令与输出字段会因 **产品型号、驱动版本和部署形态**不同而变化，先用 `npu-smi --help` 查看当前环境支持项。
- 不要在共享服务器上擅自执行设备复位、升级、功耗设置或进程终止命令。
- 文中的设备 `0`、芯片 `0` 和端口 `29500` 都是示例，使用时替换成实际值。
:::

## 1. 30 秒环境检查

刚登录服务器时，先依次执行：

```bash
uname -m
command -v npu-smi
npu-smi info -l
npu-smi info
echo "$ASCEND_HOME_PATH"
python3 --version
```

分别确认：

1. CPU 架构是 `aarch64` 还是 `x86_64`；
2. 驱动工具是否可用；
3. 系统识别到多少 NPU；
4. NPU 健康状态、显存和进程概况；
5. CANN 环境变量是否已加载；
6. 当前使用哪个 Python。

如果 `npu-smi info` 正常，但 `ASCEND_HOME_PATH` 为空，通常表示**驱动可用，但当前 Shell 还没有加载 CANN Toolkit 环境**。

## 2. 加载 CANN 环境

### root 默认安装路径

CANN 8.5/9.x 的新目录布局通常使用：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

一些环境仍使用旧目录布局：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 非 root 默认安装路径

```bash
source "$HOME/Ascend/cann/set_env.sh"
```

旧目录布局可能是：

```bash
source "$HOME/Ascend/ascend-toolkit/set_env.sh"
```

### 不知道安装在哪里

先检查常见位置，不要凭空拼路径：

```bash
ls -ld /usr/local/Ascend/cann /usr/local/Ascend/ascend-toolkit 2>/dev/null
find /usr/local/Ascend "$HOME/Ascend" -maxdepth 3 -name set_env.sh -print 2>/dev/null
```

加载后检查关键变量：

```bash
env | grep -E '^ASCEND_(HOME_PATH|TOOLKIT_HOME|OPP_PATH|AICPU_PATH)='
command -v msprof
command -v bisheng
```

::: tip `source` 为什么不能省
`set_env.sh` 会为当前 Shell 配置可执行文件、动态库、Python 包和算子库的搜索路径。直接运行脚本会创建子进程，环境变量不会保留在当前 Shell，因此要使用 `source`。
:::

## 3. 查看 CANN、驱动和固件版本

### CANN Toolkit 版本

已加载环境变量时：

```bash
cat "$ASCEND_HOME_PATH/$(uname -m)-linux/ascend_toolkit_install.info"
```

如果路径布局不同，先定位文件：

```bash
find "$ASCEND_HOME_PATH" -maxdepth 3 -name ascend_toolkit_install.info -print
```

再查看其中的 `version` 字段：

```bash
grep -i '^version' "$ASCEND_HOME_PATH/$(uname -m)-linux/ascend_toolkit_install.info"
```

### CANN 安装记录

root 安装的新版本常见记录文件：

```bash
cat /etc/Ascend/ascend_cann_install.info
```

部分环境使用：

```bash
cat /etc/ascend_install.info
```

### Driver 与 Firmware

默认安装路径下可以查看：

```bash
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/firmware/version.info
```

如果驱动安装在自定义位置，先从安装记录中确认 `Driver_Install_Path_Param`，再进入对应目录查看 `version.info`。

### 编译器和性能工具

```bash
bisheng --version
msprof --version
```

如果命令存在但运行时报动态库缺失，先重新加载正确版本的 `set_env.sh`，不要急着手动拼接一长串 `LD_LIBRARY_PATH`。

## 4. 查看 NPU 状态

### 最常用的三条

```bash
npu-smi info
npu-smi info -l
npu-smi info -m
```

| 命令 | 作用 |
| --- | --- |
| `npu-smi info` | 查看设备概览、健康状态、功耗、温度、显存与进程 |
| `npu-smi info -l` | 列出 NPU 设备，得到 **NPU ID** |
| `npu-smi info -m` | 查看 NPU ID、Chip ID 与逻辑 ID 的映射 |

**NPU ID、Chip ID 和逻辑 ID 不是同一个概念。** 调用命令前先看它要求哪一种 ID，尤其是在物理编号不连续、单卡多芯片或容器环境中。

### 查看指定设备的常用信息

```bash
npu-smi info -t common -i 0
npu-smi info -t usages -i 0
```

查询设备 `0`、芯片 `0` 的详细统计和内存信息：

```bash
npu-smi info -t usages -i 0 -c 0
npu-smi info -t memory -i 0 -c 0
```

这些输出通常包含：

- AI Core、AI CPU、控制 CPU 利用率；
- 内存容量与使用率；
- 内存带宽使用率；
- 温度与功耗。

具体字段及 `-t` 类型是否支持，以当前产品的 `npu-smi` 帮助和对应硬件文档为准。

### 持续观察

通用 Linux 方式：

```bash
watch -n 1 npu-smi info
```

部分产品支持 `npu-smi` 自带监测模式。先查看帮助：

```bash
npu-smi info watch --help
```

再按当前产品支持的参数指定设备、芯片、刷新间隔和指标。不要直接照抄其他型号的参数组合。

### 查看设备节点

```bash
ls -l /dev/davinci* 2>/dev/null
ls -l /dev/vdavinci* 2>/dev/null
```

`davinci*` 通常是物理设备节点，`vdavinci*` 与虚拟化 NPU 场景有关。在容器里看不到设备节点时，往往是容器没有正确挂载设备，而不一定是 Python 环境问题。

## 5. 限制当前进程可见的 NPU

只让当前进程看到逻辑设备 0：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

让当前进程看到逻辑设备 0、1、2、3：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

恢复为不限制：

```bash
unset ASCEND_RT_VISIBLE_DEVICES
```

::: warning 映射会重新编号
例如设置 `ASCEND_RT_VISIBLE_DEVICES=4,5` 后，应用内部看到的设备索引通常从 `0,1` 重新开始。环境变量值中不要带空格，并确认当前 CANN 版本与框架是否支持这种设置。
:::

容器启动时常见的 `ASCEND_VISIBLE_DEVICES` 用于决定把哪些设备挂进容器，它与进程内的 `ASCEND_RT_VISIBLE_DEVICES` 作用层次不同，不要混用。

## 6. 检查 PyTorch 与 torch_npu

### 查看安装版本

```bash
python3 -m pip show torch torch-npu
```

检查 Python 实际导入的是哪一份包：

```bash
python3 -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('torch path:', torch.__file__); print('torch_npu path:', torch_npu.__file__)"
```

### 检查 NPU 是否可用

```bash
python3 -c "import torch, torch_npu; print('available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

### 最小张量测试

```bash
python3 -c "import torch, torch_npu; x=torch.arange(4, dtype=torch.float32).npu(); print(x); print('device:', x.device)"
```

### 查看当前设备与显存

```bash
python3 -c "import torch, torch_npu; print('device:', torch.npu.current_device()); print('allocated:', torch.npu.memory_allocated()); print('reserved:', torch.npu.memory_reserved())"
```

如果 `npu-smi` 正常，但 `torch.npu.is_available()` 为 `False`，按下面顺序排查：

1. 当前 Shell 是否加载了正确的 `set_env.sh`；
2. `torch` 与 `torch_npu` 是否来自同一个 Python 虚拟环境；
3. PyTorch、`torch_npu`、CANN 是否满足版本配套关系；
4. 容器是否挂载了 NPU 设备节点和驱动目录；
5. 当前用户是否拥有访问设备节点的权限。

## 7. 查看进程、端口与分布式配置

### 查找训练和推理进程

```bash
pgrep -af 'python|torchrun|mpirun'
```

查看指定进程：

```bash
ps -fp 12345
```

把 `12345` 替换为实际 PID。终止进程前必须确认它属于自己，并先尝试正常结束，避免直接使用强制信号。

### 查看分布式环境变量

```bash
env | grep -E '^(RANK|LOCAL_RANK|WORLD_SIZE|MASTER_ADDR|MASTER_PORT|HCCL_)='
```

### 检查 rendezvous 端口

假设使用端口 `29500`：

```bash
ss -lntp | grep ':29500'
```

### 检查网络配置

```bash
ip addr
ip route
cat /etc/hccn.conf
```

`/etc/hccn.conf` 并非所有环境都存在。多机 HCCL 问题还要核对各节点的网卡、IP、路由、Rank 配置和时间是否一致。

## 8. 查看 CANN 日志

Host 侧应用日志默认常见于：

```bash
find "$HOME/ascend/log" -maxdepth 4 -type f -name '*.log' 2>/dev/null | tail -n 30
```

查看最近修改的日志：

```bash
find "$HOME/ascend/log" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 20
```

在日志中查找错误：

```bash
grep -RniE 'ERROR|FATAL|EH[0-9]+|E[0-9]{4}' "$HOME/ascend/log" 2>/dev/null | tail -n 100
```

常见目录含义：

| 目录 | 内容 |
| --- | --- |
| `$HOME/ascend/log/run/plog/` | Host 侧运行日志，可能包含 GE、Runtime、HCCL 等模块 |
| `$HOME/ascend/log/run/device-*/` | Device 侧运行日志 |
| `$HOME/ascend/log/debug/` | Debug 级日志 |

### 临时调整日志级别

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

常用级别为：`0=DEBUG`、`1=INFO`、`2=WARNING`、`3=ERROR`、`4=NULL`。

让 Host 应用日志直接打印到终端：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

恢复默认设置：

```bash
unset ASCEND_GLOBAL_LOG_LEVEL
unset ASCEND_SLOG_PRINT_TO_STDOUT
```

::: warning 日志会影响性能
DEBUG/INFO 日志可能产生大量 I/O。不要带着高日志级别测性能；`ASCEND_SLOG_PRINT_TO_STDOUT=1` 开启后，日志通常不再按默认方式写入日志文件。
:::

### 导出设备侧故障信息

`msnpureport` 用于收集设备日志和维护信息。先查看当前版本帮助：

```bash
command -v msnpureport
msnpureport --help
```

某些环境中的工具位于：

```bash
/usr/local/Ascend/driver/tools/msnpureport --help
```

收集结果可能较大，也可能需要额外权限。遇到硬件故障时，应在有写权限的独立目录中执行，并遵循当前版本的故障处理文档。

## 9. 异步报错定位

NPU 算子默认异步执行，因此 Python 堆栈位置可能不是实际出错算子。临时改为同步执行：

```bash
ASCEND_LAUNCH_BLOCKING=1 python3 train.py
```

这比全局 `export` 更安全，只影响当前这一条命令。

::: danger 只用于定位，不用于性能测试
同步模式会改变执行方式并明显降低性能。找到真实报错位置后，应回到默认异步模式重新验证。
:::

## 10. 使用 msprof 做性能采集

先确认工具及参数：

```bash
command -v msprof
msprof --version
msprof --help
```

最小采集示例：

```bash
mkdir -p "$HOME/prof_output"
msprof --application="python3 train.py" --output="$HOME/prof_output"
```

采集前先做三件事：

1. 缩短任务，只保留少量预热和少量稳定 Step；
2. 确认输出目录空间充足；
3. 记录模型、输入 Shape、卡数、环境变量和 CANN 版本。

不同版本支持的采集项和导出流程会变化。先使用当前环境的 `msprof --help`，再查对应版本文档，不要把旧版本参数直接复制到新环境。

## 11. 编译 Ascend C 算子前的检查

```bash
echo "$ASCEND_HOME_PATH"
echo "$ASCEND_OPP_PATH"
command -v bisheng
bisheng --version
cmake --version
g++ --version
```

检查工程实际使用的 CANN 路径：

```bash
grep -RniE 'ASCEND_HOME_PATH|ASCEND_TOOLKIT_HOME|ASCEND_OPP_PATH|CMAKE_PREFIX_PATH' CMakeLists.txt cmake 2>/dev/null
```

查看动态库依赖是否缺失：

```bash
ldd ./your_executable | grep 'not found'
```

把 `./your_executable` 替换为真实可执行文件。若出现 `libascendcl.so`、`libhccl.so` 等缺失，优先检查加载的 CANN 环境和版本，不要随意复制动态库到系统目录。

## 12. 常见问题速查

| 现象 | 优先检查 |
| --- | --- |
| `npu-smi: command not found` | 驱动是否安装、命令路径是否进入 `PATH` |
| `npu-smi info` 报错或设备缺失 | 驱动/固件、设备节点、容器挂载、硬件健康状态 |
| `ASCEND_HOME_PATH` 为空 | 是否 `source` 了正确的 `set_env.sh` |
| `ImportError: libascendcl.so` | CANN 动态库路径、Toolkit 版本、容器驱动挂载 |
| `ImportError: libhccl.so` | CANN/HCCL 库路径和 `torch_npu` 配套版本 |
| `torch.npu.is_available()` 为 `False` | Python 环境、版本配套、设备权限、容器挂载 |
| NPU OOM | `npu-smi info`、其他进程、模型显存、Reserved/Allocated |
| Python 堆栈与真实错误不一致 | 临时使用 `ASCEND_LAUNCH_BLOCKING=1` 并查看 plog |
| HCCL 初始化或超时 | Rank/World Size、网卡/IP/路由、端口、RankTable、各节点时间与首个错误日志 |
| Profiling 文件巨大 | 缩短采集区间、减少采集项、检查输出目录空间 |

## 13. 终端与 Jupyter 的区别

终端中直接写命令：

```bash
npu-smi info
```

Jupyter Notebook 的代码单元格中，Shell 命令前需要加 `!`：

```python
!npu-smi info
```

同理，下面写法只适用于 Notebook：

```python
!cat $ASCEND_HOME_PATH/$(uname -m)-linux/ascend_toolkit_install.info
```

不要把开头的 `!` 复制到普通终端，也不要把配置文件路径本身当成命令执行。

## 参考资料

- [CANN 9.0 软件安装与环境配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/softwareinst/instg/instg_0054.html)
- [npu-smi：查询所有 NPU 设备](https://www.hiascend.com/document/detail/zh/Atlas%2520200I%2520A2/23.0.0/re/npu/npusmi_012.html)
- [npu-smi：查询设备统计信息](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/2520/re/npu/npusmi_019.html)
- [CANN 日志目录与查看方式](https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/logreference/logreference_0002.html)
- [PyTorch NPU：ASCEND_LAUNCH_BLOCKING](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/ASCEND_LAUNCH_BLOCKING.md)
- [msProf 快速入门](https://www.hiascend.com/document/detail/en/mindstudio/2600/TITools/msProf/docs/en/getting_started/quick_start.md)
