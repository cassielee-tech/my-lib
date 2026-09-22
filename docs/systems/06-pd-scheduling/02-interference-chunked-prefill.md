# 单元 06-2｜混跑干扰与 Chunked Prefill

> 所属章节：[第 6 章｜Prefill/Decode 调度与推理指标](../06-pd-scheduling.md)

::: info 本单元目标
围绕 **"干扰的双向伤害与切块穿插"** 建立一条可以复述的因果链。读完后，尝试不看正文用一句话解释它。
:::

## 3. 干扰是双向的

连续批的混跑现场，两种负载互相踩脚：

**Prefill 伤害 TPOT**：一个 32K token 的 Prefill 要算几百毫秒——这段时间 Decode 的所有请求**一步都迈不出去**。用户视角：回复突然"卡顿"，正在流式输出的字停了半秒。TPOT 的毛刺就是这么来的（TPOT 平均值可能还行，**长尾**体验极差）。

**Decode 伤害 TTFT**：反过来，若调度偏向 Decode（小步快跑优先），新请求的 Prefill 一直排不上队——首字延迟飙升，用户对着"正在输入…"转圈。

```text
时间线（混跑）：
decode 步群: ██ ██ ██ ████████████████ ██ ██   ← 中间被 prefill 堵死
                ↑ 32K prefill（几百 ms）↑       TPOT 毛刺！
```

## 4. Chunked Prefill：把大单切小

**手段**：把长 Prefill 切成固定大小的块（如 512~2048 token/块），**逐块穿插**进 Decode 步群之间：

```text
切块后：
decode: ██ ██ ██ ████ ██ ██ ████ ██ ██    ← 节拍恢复
prefill:   ····[块1]····[块2]····[块3]    ← 分次喂入
```

- Decode 每两次步进之间最多插一个块——**TPOT 毛刺从"几百 ms"压到"一个块的时长"**（几十 ms 级）；
- Prefill 总计算量不变，只是**化整为零**——吞吐几乎不损失；
- 块大小是可调的旋钮：块大 → Prefill 效率高（大 GEMM）但毛刺大；块小 → 节拍顺但 Prefill 的算力利用率掉——**又一个熟悉的粒度权衡**（与 [《集合通信》05-3 的 Chunk](../../collective/05-topology-hierarchical-overlap/03-chunk-channel.md) 同一个思想：拆小换平滑）。

::: warning 缓解，不是根治
Chunked Prefill 削平了毛刺，但两种负载**仍共享同一份硬件与配置**——Decode 想要的 batch 形态、算子选择与 Prefill 想要的依然相反，"一山二虎"的矛盾只是被时间切片掩盖。根治 = 把两座山分开——下一单元的 PD 分离。
:::

---

[继续单元 06-3 →](./03-pd-disaggregation.md)
