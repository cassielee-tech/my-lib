# 实践总结｜UBOE/UB_RTP 建链端口注入：从 Issue 到合入的完整实战

> 所属课程：[HCCL 与 HCOMM 源码学习](../hccl-hcomm.md) · 实践篇（第一个真实需求复盘）

::: info 本篇目标
复盘一个真实需求的完整生命周期：[Issue #966](https://gitcode.com/cann/hcomm/issues/966)（UBOE/UB_RTP 通道建链监听端口写死 60001，多 endpoint 冲突）→ 根因定位 → 方案三轮演进（两轮检视意见）→ 分层验证（UT + 上板对照实验）→ [PR #5614](https://gitcode.com/cann/hcomm/pull/5614) 合入（含三次上游冲突 rebase）。读完你能带走 **channel 建链的完整知识链路** 和一套 **可复用的需求处理方法论**。
:::

## 先记住 5 个结论

1. **改通信库代码前先画调用链影响面**：同一段代码往往有 3～4 条入口路径（HCCM 预建 socket / 直连 / legacy / 其他引擎），只有一条命中问题，其余必须证明不受影响——影响面矩阵是方案评审的第一输入。
2. **参考同类实现，但必须核对架构差异**：URMA 的显式建链不创建监听 socket，是因为它的 endpoint 级监听是"真"的；UB_RTP/UBOE 的 endpoint 级监听是 no-op，channel 必须自建监听——照抄行为而不理解差异，会直接把功能改挂。
3. **hccpTag 是跨版本建链的 wire 契约**：它同时被本端白名单与对端连接串使用，两侧镜像生成、逐字符匹配。改它的格式 = 新旧版本混跑集群无法建链，必须显式声明兼容策略。
4. **检视意见是方案演进的主线**：本需求从 v1 到 v3 的三次迭代全部由检视意见驱动（端口注入 → 仅传 port 也生效 → 仅 SERVER 监听 + wire 兼容 + 可观测性），逐条"事实核查 → 采纳/答复 → 回复原线程"是唯一正确姿势。
5. **验证分层，各层只测自己能测的**：UT 用 mock 验证分支逻辑与配置字段（快、可断言）；上板用 G0/G1 对照实验验证真实建链（基线组复现冲突，证明装置有效）；回归用存量冒烟证明零影响。

## 0. 需求全景

```text
现象（Issue #966，Atlas A5 / CANN 9.2.0）
  两卡间多次调用 HcommChannelCreate（直连 L2-comm 路径，UBOE/UBG 协议）
  同一 server 侧创建多个 endpoint
        ↓
根因
  AicpuTsUboeUbRtpChannelHelper::BuildSocket()
  serverSocket_ 监听端口写死 SERVER_LISTEN_PORT = 60001
  HcommChannelDesc.port（"监听端口或目标端口"）完全不生效
        ↓
后果
  多个 endpoint 全部抢绑 60001 → 多对 endpoint 之间端口冲突，建链失败
        ↓
修复（PR #5614）
  端口统一从 channelDesc_.port 读取（0 回退默认）；
  显式 role 仅 SERVER 端监听；RESERVED 自动模式角色推导单一来源、
  port=0 沿用旧构造保持 wire 兼容；监听失败日志带端口
```

## 1. 基础理论与本需求知识点

### 1.1 基础理论：RA socket 对象模型——从 TCP 到 hccp

UB 建链不走内核 TCP 栈，而是走 hccp 的 RA（RDMA Adapter）抽象。先把它与熟悉的 BSD socket 对齐：

| 传统 TCP | RA 层（hccp） | 关键差异 |
| --- | --- | --- |
| `socket()` 创建 fd | `HrtRaSocketInit` 创建 **socketHandle**（设备网卡上下文） | handle 由 SocketHandleManager 按 (devPhyId, 协议, IP) 全局去重——同 IP 同协议的多个 channel 共享一个 handle |
| `bind(ip:port)` + `listen()` | `Listen(port)`：在 handle 上启动监听（HDC 设备网络模式） | 监听发生在**设备网卡**，主机 `netstat/ss` 完全不可见——排障只能靠 plog |
| `accept()` 收任意来源连接 | `GetOneSocket(SERVER)`：等待**白名单匹配**的连接 | 不接受任意来源——`AddWhiteList(remoteIp, tag, connLimit)` 逐条登记准入 |
| `connect(ip:port)` | `ConnectAsync(remoteIp, port, tag)` | 连接请求携带 tag，RA 层按（来源 IP, tag）匹配白名单后才建立 fd |

![RA socket 建链模型：白名单与 hccpTag 匹配](/images/cann/hccl/uboe-socket-tag-matching.svg)

由这张模型图可以读出两个贯穿全篇的机制：

- **tag 匹配替代了"端口即业务"**：一根监听 handle 可以同时服务多对逻辑连接，hccpTag 是区分它们的业务标识——这就是 hccpTag 成为 wire 契约的原因（1.5 节展开）；
- **socketHandle 去重的边界**：去重维度是 IP、不含端口——不同 endpoint（不同 IP）各自 handle、各自 `Listen`，撞同一端口就冲突（issue #966 的机制根源）；同 endpoint 的多 channel 则共享 handle，工作 socket 由 SocketMgr 按 SocketConfig（tag + 端口）去重复用。

### 1.2 基础理论：建链状态机——GetStatus 如何"推动"建链

`HcommChannelCreate` 只创建对象不建链，`HcommChannelGetStatus` 每调用一次推进状态机一步——这是与集合通信异步模型配合的设计（上层可在建链等待期做别的事，轮询即推进）。UBOE 的状态机分两阶段：

```text
socket 阶段（本篇主题）
  INIT ──► [IsSocketReady?] ──► SOCKET_OK     ← 端口/角色/白名单在这里生效，冲突在这里暴露
元数据交换阶段
  ──► SEND_EID ─► RECV_EID ─► PROCESS_EID     ← UBOE：交换对端可达地址（UB_RTP 无此段，直接用 EID）
  ──► BUILD_CONN                              ← 创建 UB 连接对象（DevUbUboe/RtpConnection）
  ──► SEND_SIZE ─► 收发数据 ─► PROCESS_DATA   ← 交换 notify / buffer / conn 元数据
  ──► SEND_FIN ─► RECV_FIN ─► READY           ← 建链完成
```

两个实践含义：验证程序轮询 `GetStatus` 直到 READY/FAILED，本质就是持续驱动状态机；日志判定也按阶段看——`listen_socket_info` / `Listen failed` 属 socket 阶段，是端口问题的直接证据。

### 1.3 建链调用链：三条路径，谁会走到出问题的代码

![建链全景：谁会走到 BuildSocket 的写死端口](/images/cann/hccl/uboe-buildsocket-paths.svg)

这是本需求最重要的知识——`BuildSocket()` 不是所有建链都会走到：

| 路径 | 链路 | 是否触达写死端口 |
| --- | --- | --- |
| **HCCM 集合通信**（A5/V2） | `HcclChannelAcquire` → `MyRank::CreateChannels` → `BatchCreateSockets`（`QueryListenPort` 填 role/port + 预建 socket 填 `channelDesc.socket`）→ `EndpointPair::CreateChannel` | **否**：`BuildSocket` 开头 `socket_ != nullptr` 提前返回；监听由 `EndpointPair::ServerInit` → legacy `SocketManager::ServerListen` 完成 |
| **直连 L2-comm**（数据面用户） | `HcommChannelCreate`（`socket=NULL`）→ `Channel::CreateChannel` → channel `Init()` → `BuildSocket()` | **是**：问题路径，全部自建监听 |
| **非 V2 独立算子** | `ChannelManager::ChannelCommCreate` → legacy Transport/link 体系 | 否：不经过 base_comm channel |

另有两个易混点：**UBG 是 `COMM_PROTOCOL_UB_RTP` 的别名**（`hcomm_res_defs.h`）；AIV 引擎 + UB_RTP 走的是 `AivUrmaChannel`（独立实现），与 UBOE/UB_RTP 共用的 `AicpuTsUboeUbRtpChannelHelper` 无关。

### 1.4 监听机制：channel 自建 vs endpoint 级（最关键的架构差异）

HCOMM 有两套监听机制，理解它们的分工是本需求方案选型的核心：

| | URMA/RoCE | UBOE/UB_RTP |
| --- | --- | --- |
| endpoint 级监听 | `HcommEndpointStartListen` → `DeviceServerSocketContext` → **ServerSocketManager**（按 (PortData, port) 引用计数去重、端口复用） | `UbRtpUboeServerSocketContext` 是**设计上的 no-op**（迁移语义保留） |
| channel 侧行为 | 显式模式**不创建** serverSocket_，依赖上面 | **channel 自建** `serverSocket_` 并 `Listen()`——历史上每 channel 一次、无去重 |

所以 URMA 的 `BuildSocket` 显式分支里没有监听代码，不是"省略"而是"职责在 endpoint"；UB_RTP/UBOE 若照抄，监听就没人做了——这正是"参考实现不能只抄行为"的教训。

### 1.5 SocketConfig 构造族与 hccpTag（wire 契约）

`socket_config.h` 的三个构造函数是本需求的主战场，**hccpTag 的格式差异就是兼容性差异**（镜像匹配机制见 1.1 的模型图）：

| 构造 | hccpTag 格式 | listeningPort | 角色 |
| --- | --- | --- | --- |
| `(link, tag, noRankId)` 旧构造 | `tag_本端IP_对端IP`（无端口） | `DEFAULT_LISTENING_PORT`(60001) | 地址比较推导（IP 相等时按 rank） |
| `(link, port, tag)` | `tag_IP_IP_端口` | 传入 port | 地址比较推导 |
| `(link, port, tag, isServer)` | `tag_IP_IP_端口` | 传入 port | 显式指定 |

hccpTag 同时出现在**本端白名单**（`SocketMgr::AddWhiteList`）与**对端连接串**（`Socket::ConnectAsync`）里，server/client 两侧镜像生成（IP 顺序互换）、逐字符匹配——**同版本集群内自洽，新旧版本混跑则格式不匹配即建链失败**。改 tag 格式前必须想清楚混版本策略。

上游还有一条演进线要注意：hccpTag 超长修复（`SetHccpTag` 旁路，!5180 → revert → !5779 重落）反复横跳，rebase 时要么适配要么移除——**外部依赖的特性要跟着上游当前状态走**。

### 1.6 role/port 语义与三种使用模式

`HcommChannelDesc` 的 `role`（RESERVED=-1/CLIENT=0/SERVER=1）与 `port`（uint16_t）组合出三种模式：

![三种模式的建链行为](/images/cann/hccl/uboe-role-port-modes.svg)

| 模式 | 配置 | 行为 |
| --- | --- | --- |
| 自动（存量默认） | `HcommChannelDescInit` 默认 RESERVED + port=0 | 角色地址比较推导；仅推导 SERVER 端监听 60001；旧构造（tag 无后缀，wire 兼容） |
| 自动 + 仅传 port | RESERVED + port=N | 角色仍推导；端口生效（检视意见补充的场景） |
| 显式 | role=SERVER/CLIENT + port=N | 角色覆盖地址推导；仅 SERVER 端监听；CLIENT 只连接 |

**两端一致性约束**：同一链路两端须同模式；显式模式两端 role 相反、端口一致。混用（一端显式一端 RESERVED 不传 port）会建链超时。

## 2. 怎么定位

```text
① 现象入手      "多对 endpoint 之间端口冲突" → 端口相关的第一直觉：搜写死的端口号
② 常量搜索      grep 60001 → 命中 aicpu_ts_uboe_ub_rtp_channel_helper.cc 的
               SERVER_LISTEN_PORT，确认 BuildSocket 全路径使用该常量
③ 反向验证      channelDesc_.port 字段在 helper 中从未被读取 → "配置不生效" 实锤
④ 调用链梳理    谁会走到 BuildSocket？（1.3 的影响面矩阵）
               ——HCCM 预建 socket 提前返回、legacy 不经过、AIV 另一套实现
⑤ 冲突机制      SocketHandleManager 按 (devPhyId, proto, IP) 去重 socketHandle，
               但不同 endpoint 的 IP 不同 → 各自 handle 各自 Listen(60001) → RA 层端口绑定冲突
⑥ 影响面结论    只有直连路径受影响；存量 RESERVED 调用方行为必须保持不变
```

定位阶段最容易犯的错是跳过 ④ 直接改代码——不画影响面，就无法回答"这个改动会不会影响 MC²/集合通信"，评审第一问就会被打回。

## 3. 怎么处理

### 3.1 方案选型：为什么不能照抄 URMA

参考对象是 `aicpu_ts_urma_channel.cc::BuildSocket` 的显式分支（读 `channelDesc_.port`、0 回退默认、`SocketConfig(linkData, port, tag, isServer)`）。但有一个**致命差异**：URMA 显式分支不建监听 socket，因为它的监听在 endpoint 级（`StartListen` → `DeviceServerSocketContext`，带引用计数去重）；UB_RTP/UBOE 的 endpoint 级监听是 no-op。**正确姿势 = URMA 的端口/角色语义 + 保留 channel 自建监听**。

同时识别出一个更深的问题：即使端口可传，**RESERVED 存量调用方（不传 port）的多 endpoint 冲突依旧存在**——这是要不要在 hcomm 内部做端口自动分配/去重的架构决策点（可画三方案决策图交架构师：A 上层适配 / B hcomm 自动端口管理 / C 短期 A + 规划 B），本 PR 定位是"打通端口传入能力"。

### 3.2 三轮演进：检视意见驱动的方案迭代

| 版本 | 触发 | 关键变化 |
| --- | --- | --- |
| v1 | 初版 | 显式 role/port 分支生效；RESERVED 分支保持原样（60001 + 旧构造） |
| v2 | 内部检视意见 | RESERVED + 仅传 port 也生效：端口解析前置、新构造 + `noRankId=true` 手动回填（保持 SocketMgr 去重语义） |
| v3 | PR 检视 5 条（2 MEDIUM + 3 LOW） | ① 自动模式仅推导 SERVER 端建监听（角色推导单一来源：先构造 SocketConfig 再 `GetRole()` 决定）；② port=0 复用旧构造（wire 级兼容，混版本可建链）；③ 监听失败日志带 `listenPort/devPhyId`；④ PR 描述补两端 mode 约束 + 存量调用方排查结论；⑤ 补 7 个 UT 用例 |

v3 ① 的实现技巧值得记住：**角色推导逻辑不要复制**，先构造 SocketConfig（推导在其构造函数内，含 IP 相等按 rank 的分支），再用 `socketConfig.GetRole()` 决定是否建监听——单一来源，永不失配。

### 3.3 最终代码形态（骨架）

```cpp
uint16_t port = channelDesc_.port;
if (port == 0) { port = SERVER_LISTEN_PORT; HCCL_INFO(...); }   // 回退带日志

if (role == RESERVED) {
    // port=0 用旧构造（tag 无后缀，存量 wire 兼容）；port!=0 用带端口构造
    SocketConfig sc = (channelDesc_.port == 0) ? SocketConfig(link, tag, true)
                                               : SocketConfig(link, port, tag);
    sc.noRankId = true;                          // 保持自动模式去重语义
    if (channelName 非空) sc.SetHccpTag(tag);    // 跟随上游 !5779 超长旁路
    SaveSocketConfig(sc);
    if (sc.GetRole() == SERVER) BuildServerSocket(port);   // 仅推导 SERVER 端监听
    GetSocket(...);
} else {
    bool isServer = (role == SERVER);
    SocketConfig sc = SocketConfig(link, port, tag, isServer);
    if (channelName 非空) sc.SetHccpTag(tag);
    SaveSocketConfig(sc);
    if (isServer) BuildServerSocket(port);       // 显式模式仅 SERVER 监听
    GetSocket(...);
}
```

`BuildServerSocket(listenPort)` 抽成独立函数（参数化端口），失败路径打含端口的 ERROR 日志——修复目标就是端口冲突，冲突发生时必须能从日志一眼看到是哪个端口。

## 4. 怎么验证

### 4.1 分层验证策略

| 层 | 验证什么 | 手段 | 状态 |
| --- | --- | --- | --- |
| L1 UT | 分支逻辑、SocketConfig 字段（listeningPort/role/noRankId/hccpTag）、监听行为（调用次数/端口） | mockcpp + gtest | CI 通过 |
| L2 上板 | RA 层真实建链、端口实际生效、基线对照复现冲突 | 验证程序 + G0/G1 对照实验 | 设计完成，实验室执行 |
| L3 回归 | MC²/集合通信存量路径零影响 | ST/hccl_test 冒烟 vs 基线一致 | 同上 |

### 4.2 UT：mockcpp 的四个手法

本仓 UT 框架（`test/ut/framework/next/`，mockcpp）验证了几个可复用的模式：

```cpp
#define private public
#define protected public      // 访问 protected 成员与成员函数地址
#include "aicpu/aicpu_ts_uboe_channel.h"

// ① mock 自己类的成员函数（同 TU 亦可，先例 HostCpuRoceChannel::IbvPostRecv）
MOCKER_CPP(&AicpuTsUboeUbRtpChannelHelper::BuildServerSocket).stubs().will(invoke(StubBuildServerSocket));

// ② mock 带出参的成员函数：invoke 桩的首参是 this
HcclResult StubGetSocket(SocketMgr*, const Hccl::SocketConfig& sc, Hccl::Socket*& socket) {
    socket = (sc.GetRole() == CLIENT) ? &clientSock : &serverSock;  // 按配置角色给 fake
    return HCCL_SUCCESS;
}

// ③ fake Hccl::Socket：构造只做成员赋值、析构无副作用，静态实例安全
static Hccl::Socket serverSock(nullptr, IpAddress(), 0, IpAddress(), "ut", SERVER, DEVICE_NIC_TYPE);

// ④ 断言直接读 socketConfigHolder_（SaveSocketConfig 的拷贝），
//    比"只 mock 返回值"强——检视人批评过隔壁 PR 的 UT 只验返回码不验参数
EXPECT_EQ(ch.socketConfigHolder_->listeningPort, 60001);
EXPECT_EQ(ch.socketConfigHolder_->GetHccpTag(), "AUTOMATIC_SOCKET_TAG_127.0.0.1_127.0.0.2");
```

角色推导的确定性技巧：`IpAddress` 按地址字节 memcmp 比较，`127.0.0.1 < 127.0.0.2` 在第 4 字节即分出大小——用这对 IP 控制推导结果为 SERVER/CLIENT。

### 4.3 上板：对照实验设计

- **分组**：G0 基线（未修复，阴性对照——场景 A 必须复现冲突，证明实验装置有效）vs G1 修复组，同矩阵双跑；
- **矩阵 T1-T7**：正向（显式多端口 / RESERVED 回归 / 仅传 port）+ 负向（`--same-port` 冲突注入验证错误日志、两端模式混用验证约束）+ 回归（MC² 冒烟）；
- **测量点**：程序退出码 + `GetStatus` 轮询；plog 的 `listen_socket_info`（端口/tag）、`Listen failed, listenPort[...]`、`channelDesc port is 0`；
- **测量局限**：UB 监听在设备网卡 RA 层，主机 `netstat/ss` 不可见，日志 + 建链结果是唯一权威判据；
- **验证程序设计**：双实例经 TCP 控制通道交换 `EndpointDesc`（`HcommEndpointGetDescs` 查询本机 EID，仅 950PR/950DT 支持），再按场景建链轮询状态——把"复现 issue 的完整调用链"固化成一个可重复执行的二进制。

## 5. 合入流程经验

### 5.1 GitCode 贡献链路（hcomm-contribute skill）

```text
Issue 查重/创建 → feature 分支（基于最新 upstream/master，squash 成单 commit）
→ pre-commit（clang-format 18.1.8 + OAT）→ fork push（--force-with-lease）
→ PR（描述按模板六章节、必须关联 Issue）→ 评论 /compile 触发 CI
→ --ci-status --wait 轮询 → 失败则 --ci-logs + gate-analysis 报告定位
→ --list-review-comments 拿 discussion_id → 逐条回复原线程
```

三条纪律：**他人意见只回复不 resolve**（关闭权在提出者）；**push 后必须重新 /compile**（旧 passed 标签失效）；**检视回复必须进原线程**（独立顶层评论会丢失上下文）。

### 5.2 上游活跃期的三次 rebase

上游 master 在 PR 存续期前进了三轮，全部撞在 `BuildSocket` 同一区域：

| 轮次 | 上游动作 | 应对 |
| --- | --- | --- |
| 1（+50 commits） | hccpTag 超长修复 !5180 插入 `SetHccpTag` | 保留双分支结构 + 在两个分支适配旁路 |
| 2（+2 commits） | !5180 被 revert | 移除适配 + UT 断言改回拼接式 + PR 描述同步 |
| 3（+16 commits） | !5779 重落（与 !5180 内容一致） | 恢复适配 + UT 断言改回旁路式 + 检视线程补更新说明 |

教训：**rebase 不只是解冲突**——所有与外部特性相关的资产（UT 断言、PR 描述、检视回复中的表述）都要跟着翻转；对外承诺过的表述过时要在线程里主动更新，保持记录准确。

### 5.3 CI 失败的快速定位

`/compile` 失败后优先看 cann-robot 的 **gate-analysis 报告**（直接给"定界结论 + 根因 + 处置建议"，如本例 UT 的 `setenv` 缺第三参就是它定位的）；`mergeable: false` 时先 `git fetch upstream` 看新增提交是否触及自己的文件，再决定 rebase。

## 6. 可复用方法论清单

- [ ] **影响面先行**：改代码前画调用链矩阵，逐路径证明"受影响/不受影响"；
- [ ] **参考实现三问**：它为什么这么写？差异点在哪（架构/依赖/生命周期）？哪部分能抄哪部分必须改？
- [ ] **wire 契约意识**：凡涉及两侧匹配的字符串/端口/序列化格式，改动即兼容性事件，声明混版本策略；
- [ ] **推导逻辑单一来源**：角色推导只活在 SocketConfig 构造里，消费方用 `GetRole()`，永不复制；
- [ ] **存量兼容优先**：新能力走新分支，默认路径尽量走旧构造/旧行为（port=0 复用旧构造是本次最典型的取舍）；
- [ ] **可观测性随修复落地**：修什么就让什么可定位（端口冲突修复必须带含端口的错误日志）；
- [ ] **负向用例**：给验证工具加冲突注入开关（`--same-port`），错误路径的可观测性只能靠它验证；
- [ ] **对照实验**：阴性对照组（基线复现问题）是证明"实验装置有效 + 问题真实存在"的关键，缺了它 PASS 没有说服力；
- [ ] **检视意见闭环**：事实核查 → 采纳/答复 → 原线程回复 → 表述过时主动更新；
- [ ] **rebase 三同步**：代码、测试断言、文档/回复，一个都不能漏。

## 参考资料

- [Issue #966｜UBOE/UB_RTP 通道 BuildSocket 监听端口硬编码 60001](https://gitcode.com/cann/hcomm/issues/966)
- [PR #5614｜UBOE/UB_RTP 建链监听端口支持从 channelDesc_.port 传入](https://gitcode.com/cann/hcomm/pull/5614)
- 涉及源码：`src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_uboe_ub_rtp_channel_helper.{h,cc}`、`aicpu_ts_urma_channel.cc`、`sockets/socket_mgr.cc`、`endpoints/server_socket_context/`、legacy `socket_config.h`
- 课程关联：[单元 1 架构分层](../hccl-hcomm/01-architecture-layering.md) · [单元 2 控制面与 HCCM](../hccl-hcomm/02-control-plane.md) · [单元 3 数据面名词](../hccl-hcomm/03-data-plane.md) · [单元 11 实战起步](../hccl-hcomm/11-examples-first-op.md)

---

[返回课程导学 →](../hccl-hcomm.md) · [返回课程总结 →](../hccl-hcomm/summary.md) · [返回实践总结 →](index.md) · [返回专栏目录 →](../index.md)
