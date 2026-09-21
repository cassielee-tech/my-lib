# 单元 04-4｜工程实操：跑起来与排障

> 所属章节：[第 4 章｜分布式基础：Rank、通信域与拓扑](../04-distributed-basics.md)

::: info 本单元目标
围绕 **最小分布式骨架与排障清单** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 7. 最小可跑骨架（torch_npu + HCCL）

十五行代码把本章全部概念走一遍——**绑卡、进组、报身份、做一次集合通信验证**：

```python
import os
import torch
import torch_npu
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(local_rank)          # ① 绑卡
    dist.init_process_group(backend="hccl")       # ② 进组（签到 + 建 HCCL 通信域）

    rank, world = dist.get_rank(), dist.get_world_size()
    print(f"rank {rank}/{world} on device {local_rank}")   # ③ 报身份

    t = torch.ones(4, device=f"npu:{local_rank}")
    dist.all_reduce(t)                            # ④ 全员验证：结果应为 world
    assert t[0].item() == world
    print(f"rank {rank}: all_reduce OK")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

启动（单机 8 卡）：

```bash
torchrun --nproc_per_node=8 demo.py
```

多机时每台机器各跑一条 `torchrun`，多出三个参数：`--nnodes`、`--node_rank`、`--master_addr/--master_port`——这正是 04-1 的编号表从纸面变成进程的过程。

## 8. 排障清单：按依赖顺序查

分布式故障的第一原则：**先查"人口登记"，再查物理资源，最后查日志**。

| 症状 | 排查点 | 命令/动作 |
| --- | --- | --- |
| 卡在 init 几分钟无输出 | rendezvous 没签上：端口占用/网络不通 | `ss -tlnp \| grep 29500`；ping master；换 MASTER_PORT |
| 启动即报卡数不匹配 | `nproc_per_node × nnodes` ≠ 可见卡数 | `npu-smi info`；核对 `ASCEND_RT_VISIBLE_DEVICES` |
| 各进程行为错乱 | 环境变量缺失或不一致 | `env \| grep -E "RANK\|WORLD\|MASTER"`（详见 [CANN 常用命令速查](../../ascend/cann常用命令.md)第 7 节） |
| OOM / 设备忙 | 一卡被多进程绑定 | `npu-smi info` 看每卡进程数；核对每进程 set_device |
| 报错栈驴唇不对马嘴 | 异步报错延迟浮出（回扣 [《昇腾与 HCCL》02-3](../../ascend/02-runtime-task-execution/03-stream-event-task.md)） | 开 HCCL 日志定位（速查第 8 节），看首次报错的 rank |

::: tip 本章收束
地基三件套就位：**编号**（rank/local_rank/world_size）、**分组**（Process Group）、**排布**（node-major 贴拓扑）。第 5 章起，每种并行策略的第一件事都是回答"怎么切张量"，第二件事就是"在本章的哪一组上发哪个集合通信"。
:::

---

[进入本章总结 →](./summary.md)
