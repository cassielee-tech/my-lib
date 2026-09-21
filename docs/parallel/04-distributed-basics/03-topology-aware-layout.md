# 单元 04-3｜拓扑感知排布：把编号贴到物理上

> 所属章节：[第 4 章｜分布式基础：Rank、通信域与拓扑](../04-distributed-basics.md)

::: info 本单元目标
围绕 **"node-major 排号为什么快"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

编号是软件给的，链路是硬件长的。**排号 = 让逻辑分组对齐物理拓扑。**

## 5. 两层世界与 node-major 排号

回扣 [《集合通信》05-1](../../collective/05-topology-hierarchical-overlap/01-topology-map.md)：机内是 HCCS 全互联专线（百 GB/s），机间是网卡走交换网（几十 GB/s，差近一个量级）。

**node-major（按节点整块分配）**：机器 0 拿 rank 0-7，机器 1 拿 rank 8-15……

```text
node-major：tp_group[1] = [8,9,…,15] → 全部落在一台机器里 → 走机内专线 ✅
round-robin（轮流发牌）：rank 0,2,4,… 发给机器 0
                          → tp_group = [0,1,2,…,7] 横跨 4 台机器 → 层层通信挤机间 ❌
```

两种排号下，**算法、代码、通信量完全相同**，唯一区别是"组员落在哪"——而每层 TP 通信都要过一次机间链路时，慢一个量级不是修辞，是账本（回扣《集合通信》05-2 的分层流量账）。

::: warning 一句话结论
**排号错误不会报错，只会变慢**——这正是它危险的地方：任务"能跑"，Profiling 里通信占比刺眼，根因却在启动脚本的一行参数里。
:::

## 6. Local Rank 与设备绑定

编号落位后，每个进程的开场三步：

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch_npu.npu.set_device(local_rank)     # ① 绑卡：本进程认领本机第 local_rank 张卡
x = x.npu()                              # ② 此后所有 .npu() 都落到这张卡
dist.init_process_group(backend="hccl")  # ③ 进组：领工号、连后端
```

两个容易踩的坑：

1. **可见性重映射**：`ASCEND_RT_VISIBLE_DEVICES=4,5` 时，进程里的"卡 0"是物理 4 号卡（回扣 [《昇腾与 HCCL》02-1](../../ascend/02-runtime-task-execution/01-host-device.md)）——排查"数据怎么跑到别的卡上了"先查这个；
2. **一卡多进程是反模式**：两个进程绑同一张卡 = 显存翻倍 + 调度互踩，通常立刻 OOM 或性能崩塌。

顺带把链路接完整：**进程（Host 单位）→ local_rank 绑卡（Device 单位）→ PG 分组（通信单位）→ 拓扑对齐（物理约束）**。HCCL 启动时做的拓扑探测与分级通信（[H01-5](../../ascend/hccl-source/05-comm-engines.md)），本质就是自动替你做本章这套对齐。

---

[继续单元 04-4 →](./04-launch-and-debug.md)
