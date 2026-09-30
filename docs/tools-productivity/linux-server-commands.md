# Linux 服务器常用命令速查

::: tip 本篇定位
远程 Linux 服务器的日常运维与排障命令，按场景组织，每条命令给"最常用形态 + 一句话说明"。
NPU / CANN 侧命令（npu-smi、CANN 环境、分布式排障）见姊妹篇：[CANN 常用命令速查](../ascend/cann常用命令.md)。
:::

## 1. 使用原则

- **命令记不全时**：`man cmd` 看权威手册，`cmd --help` 看速览，`tldr cmd`（需安装）给最常用的几条例子；
- **组合优于单体**：Linux 的哲学是"小工具 + 管道"——`ps aux | grep python` 比"找一个能直接查 python 进程的命令"更快；
- **危险动作三连问**：`kill -9`、`rm -rf`、`sed -i` 执行前，先问自己——目标是谁？有没有通配符误伤？是不是生产环境？

## 2. 系统体检：先看总量

### 负载与进程总览：top

```bash
top                     # 打开即看：load average、CPU、内存、进程列表
```

进去之后是交互模式，记住五个键：

| 键 | 作用 |
| --- | --- |
| `P` | 按 CPU 排序（默认）——找吃算力的 |
| `M` | 按内存排序——找吃内存的 |
| `1` | 展开每个核的占用——区分"单核打满"与"整体打满" |
| `k` | 输入 PID 杀进程 |
| `q` | 退出 |

**load average 三个数**是 1/5/15 分钟的平均负载。经验法则：单机持续超过核数（`nproc` 可查）就是过载。

`htop` 是增强版 top（彩色、可滚动、F9 杀进程），多数发行版需自行安装。

### 内存与磁盘

```bash
uptime                  # 一行看负载与运行时长
free -h                 # 内存：看 available 列（才是真正可用），不是 free 列
df -h                   # 各分区磁盘用量——定位哪个分区快满了
du -sh /var/log         # 单个目录占多少（-h 人类可读，-s 只看总计不递归展开）
du -sh /var/* | sort -rh | head   # 逐层下钻：哪个子目录最大
```

::: tip free 的常见误读
`free` 列小不等于内存紧张——Linux 会拿空闲内存做 page cache。判断标准永远是 **available**。
:::

### CPU 与 IO 瓶颈

```bash
vmstat 1                # 每秒刷新：r 列（等待 CPU 的进程数）、b 列（阻塞在 IO 的进程数）、si/so（换页）
iostat -x 1             # 磁盘：%util 接近 100% = 磁盘打满；await 高 = IO 等待长（sysstat 包）
```

`vmstat` 的 r 持续大于核数 → CPU 瓶颈；`b` 持续不为 0、`iostat` 的 `%util` 打满 → IO 瓶颈。

## 3. 进程与服务

### 找进程与控制进程

```bash
ps aux | grep python    # 最常用组合：全量进程里过滤关键词（aux：所有用户 + 详细信息）
ps -ef | grep nginx     # 另一种风格，输出格式不同（-ef 能看父进程 PPID）
kill 12345              # 默认发 SIGTERM（15）：优雅退出，进程可做清理
kill -9 12345           # SIGKILL：立即强杀，进程没机会清理——最后手段
pkill -f "train.py"     # 按完整命令行匹配并杀（-f 很关键，不带只匹配进程名）
```

**15 与 9 的区别**：15 是"请你退出"，进程可以保存现场；9 是"原地蒸发"，可能留下半写的文件和锁。先 15，等几秒不退再 9。

### 后台运行

```bash
nohup python train.py > train.log 2>&1 &   # 挂后台 + 断开 SSH 也不死；日志重定向到文件
jobs                    # 看当前 shell 的后台任务
fg %1                   # 把 1 号任务调回前台（Ctrl+Z 暂停当前任务入 jobs）
```

`2>&1` 是"标准错误也并进标准输出"；漏了它，报错信息不会进日志文件。

### systemd 服务

```bash
systemctl status nginx          # 看状态（是否 running、最近几行日志）
systemctl start / stop / restart nginx
systemctl enable --now nginx    # 开机自启 + 立即启动
systemctl list-units --type=service --state=running   # 当前跑着哪些服务
```

### 服务日志：journalctl

```bash
journalctl -u nginx -f              # 跟随某个服务的日志（= 专属 tail -f）
journalctl -u nginx -n 200          # 最近 200 行
journalctl -u nginx --since "10 min ago"
journalctl -xe                      # 系统级排障入口：最近的报错 + 上下文
```

### lsof：谁在用这个文件/端口/设备

```bash
lsof -i:8080             # 谁占用了 8080 端口（起服务报 Address in use 时第一反应）
lsof -p 12345            # 这个进程打开了哪些文件/网络连接/设备
lsof /data/train.log     # 谁在写这个文件（删不掉的文件先查这个）
```

## 4. 网络：连接、端口与可达性

### 端口与连接：ss（首选）与 netstat（旧版）

```bash
ss -tlnp                 # 监听中的 TCP 端口 + 进程（新系统首选）
netstat -tlnp            # 同义旧命令（net-tools 包，老系统/容器里常见）
ss -tanp                 # 所有 TCP 连接（含已建立）——看谁连着我、我连着谁
ss -s                    # 连接数统计总览
```

参数拆解：`t`=TCP、`u`=UDP、`l`=仅监听、`a`=全部、`n`=数字形式（不做域名反解，快）、`p`=显示进程。

**连接状态一句话判读**：`TIME_WAIT` 多是正常的（主动关闭方的收尾阶段，系统自动回收）；`CLOSE_WAIT` 持续堆积说明**本机程序忘了调 close()**——是代码 bug 的信号，不是网络问题。

### 可达性逐层测试

```bash
ping -c 4 10.0.0.5               # 网络层通不通（IP 层）
nc -zv 10.0.0.5 8080             # 端口通不通（传输层）——服务起没起、防火墙放没放
telnet 10.0.0.5 8080             # 同上的老办法
curl -v http://10.0.0.5:8080/api # 应用层：完整请求过程逐行打印
curl -I https://example.com      # 只要响应头（看状态码、Content-Type、重定向）
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" https://example.com   # 只要状态码和耗时
```

`curl -v` 的三段式输出（→ 请求、← 响应、* 连接过程）是判断"卡在哪一层"的利器。

### 路径与 DNS

```bash
traceroute 10.0.0.5              # 包走了哪些跳（哪一跳开始延迟暴增）
mtr 10.0.0.5                     # traceroute + ping 持续刷新版，更直观（可能需安装）
dig example.com +short           # 只输出解析结果 IP
dig example.com                  # 完整解析过程（看 TTL、CNAME、权威服务器）
nslookup example.com             # 交互式 DNS 查询（老命令）
```

### 地址与路由

```bash
ip addr                  # 网卡与 IP（新命令）
ip route                 # 路由表（默认网关是哪）
ifconfig                 # 旧命令，容器/老系统常见
```

### 抓包：tcpdump（简要）

```bash
sudo tcpdump -i any -nn host 10.0.0.5 and port 8080     # 抓指定主机+端口的包（-i any 所有网卡）
sudo tcpdump -i any -nn -c 100 port 8080                # 抓 100 个包就停
sudo tcpdump -i any -nn -w dump.cap host 10.0.0.5       # 存文件，拿去 Wireshark 分析
```

`-nn` 禁用端口/协议名转换（快且干净）；过滤器语法 `host / port / and / or / not` 先会这几个就够。

### 防火墙（只读检查）

```bash
sudo iptables -L -n -v           # 经典 netfilter 规则（-n 数字化，-v 带计数）
sudo firewall-cmd --list-all     # firewalld 系（CentOS/RHEL/openEuler 默认）
sudo ufw status                  # ufw 系（Ubuntu 默认）
```

端口连不上时，先看这里的 DROP/REJECT 规则，再怀疑程序。

### 文件传输

```bash
scp file.tar.gz user@10.0.0.5:/data/            # 单文件/目录（-r 递归）
rsync -avz --progress src/ user@10.0.0.5:/data/ # 增量同步：断点续传、只传差异、带进度
```

大文件、断断续续的网络、要重复同步的场景，一律 `rsync`。

## 5. 文件与文本：grep 一族与日志

### 看文件与跟日志

```bash
tail -n 200 train.log        # 看最后 200 行
tail -f train.log            # 跟随新增（Ctrl+C 退出）——看日志的第一反应
tail -F train.log            # 文件被轮转（rename+新建）后依然继续跟——比 -f 更稳
head -n 50 train.log         # 看开头 50 行
```

`less` 是翻页看大文件的标准姿势（不会像 cat 一样刷屏）：

| 键 | 作用 |
| --- | --- |
| `/error` + Enter | 向下搜索 error；`n` 下一个、`N` 上一个 |
| `G` / `g` | 跳到文件末尾 / 开头 |
| `F` | 进入跟随模式（= tail -f，且可 Ctrl+C 后回看历史） |
| `q` | 退出 |

### grep：文本过滤主力

```bash
grep "ERROR" app.log                 # 基本过滤
grep -n "ERROR" app.log              # 带行号
grep -i "error" app.log              # 忽略大小写（Error/ERROR/error 都中）
grep -r "HCCL_BUFFER" /path/         # 递归搜目录下所有文件
grep -rn "import torch" --include="*.py" .          # 递归 + 只搜 .py 文件 + 带行号
grep -v "DEBUG" app.log              # 反选：排除 DEBUG 行
grep -E "ERROR|WARN" app.log         # 扩展正则（| 表示或）
grep -C 3 "Exception" app.log        # 命中行的前后各 3 行上下文（-A 后 / -B 前）
tail -f app.log | grep --line-buffered "ERROR"      # 实时日志只看 ERROR（必须 --line-buffered）
```

::: tip --line-buffered 是关键
管道里 grep 默认整块缓冲，不加 `--line-buffered` 时"实时过滤"会卡住几秒甚至不输出。
:::

### find：按条件找文件

```bash
find /data -name "*.log"                     # 按名字
find . -type f -mtime -7                     # 最近 7 天内修改过的普通文件
find . -type f -size +500M                   # 大于 500MB 的文件
find . -name "*.tmp" -delete                 # 找到即删（先去掉 -delete 跑一遍确认！）
find . -name "*.py" | xargs grep -l "torch"  # 找到的文件再交给 grep（-l 只列文件名）
find . -name "*.py" -print0 | xargs -0 grep -l "torch"      # 文件名带空格也安全
```

### 统计与列处理

```bash
wc -l access.log                             # 数行数
sort access.log | uniq -c | sort -rn | head  # 经典统计链：排序→去重计数→按次数倒序→Top 10
du -sh * | sort -rh | head -10               # 当前目录下最大的 10 个子目录
awk '{print $1}' access.log                  # 取第一列（默认按空白切分）
awk -F: '{print $1}' /etc/passwd             # 指定冒号分隔符，取第一列
awk '{sum += $1} END {print sum}' nums.txt   # 对第一列求和
```

`awk '{print $1}' | sort | uniq -c | sort -rn` 这条链能解决一半的"日志里谁最多"问题。

### sed：流式编辑

```bash
sed -n '100,200p' big.log        # 只打印 100-200 行（大文件截段看）
sed 's/foo/bar/g' file           # 替换所有 foo 为 bar（只输出不改文件）
sed -i 's/foo/bar/g' file        # 直接改写文件——改前先备份：sed -i.bak
```

### 权限与归档

```bash
chmod 755 run.sh                 # rwxr-xr-x：目录与可执行文件常用
chmod 644 config.yaml            # rw-r--r--：普通文件常用
chown -R user:group /data/proj   # 递归改属主（sudo）
tar -czf arch.tar.gz dir/        # 打包+gzip 压缩
tar -xzf arch.tar.gz             # 解压
which python                     # 命令的可执行文件在哪
```

## 6. 场景组合拳：现象 → 命令链

排障的本质是**二分**：每条命令排除一层可能性。以下是最常用的六条链。

### 服务器变慢了

```bash
uptime                                          # ① 负载多高？（对比 nproc 核数）
top                                             # ② 谁在吃 CPU/内存？（P / M 切排序）
ps -o pid,ppid,cmd -p 12345                     # ③ 确认这个 PID 是什么、谁起的
lsof -p 12345                                   # ④ 它在读写哪些文件/连着谁
strace -p 12345                                 # ⑤ 还看不懂就上系统调用（重炮，谨慎）
```

### 磁盘满了

```bash
df -h                                           # ① 哪个分区满了
sudo du -sh /* 2>/dev/null | sort -rh | head    # ② 逐层下钻找大目录
lsof +L1                                        # ③ 已删除但仍被占用、没释放空间的文件
                                                #    （日志被 rm 但进程还写着 → 重启该进程才释放）
```

### 端口起不来（Address already in use）

```bash
ss -tlnp | grep 8080                            # ① 谁在监听这个端口
lsof -i:8080                                    # ② 拿到 PID 和进程名
kill <旧PID>                                    # ③ 杀掉或给新服务换端口
```

### 服务连不上（五层排查）

```bash
ping 10.0.0.5                                   # ① IP 层：机器活着吗
nc -zv 10.0.0.5 8080                            # ② 传输层：端口开着吗
curl -v http://10.0.0.5:8080/                   # ③ 应用层：服务正常响应吗
mtr 10.0.0.5                                    # ④ 路径：哪一跳开始不通/变慢
sudo tcpdump -i any -nn host 10.0.0.5 and port 8080    # ⑤ 终极手段：两端同时抓包看包到没到
```

### 实时盯日志

```bash
tail -F app.log | grep --line-buffered -E "ERROR|WARN"   # 只看报错和告警
journalctl -u my-service -f --since "5 min ago"          # systemd 服务
```

### 找大文件 / 大目录

```bash
du -sh * | sort -rh | head -10                          # 目录维度
find / -xdev -type f -size +500M 2>/dev/null            # 文件维度（-xdev 不跨文件系统）
```

## 7. 参考资料

- 姊妹篇：[CANN 常用命令速查](../ascend/cann常用命令.md)——NPU 状态、CANN 环境、分布式排障
- 远程工作流：[VS Code + Remote SSH + Jupyter](vscode-remote-ssh-jupyter.md)——SSH 配置、隧道与远程实验
- [man7.org Linux man-pages](https://man7.org/linux/man-pages/)：权威手册在线版
- [tldr pages](https://tldr.sh/)：社区维护的命令示例集
- [Brendan Gregg：Linux Performance](https://www.brendangregg.com/linuxperf.html)：性能分析全景图（进阶）
