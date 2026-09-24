# 课程总结｜HCOMM 源码阅读地图

> 所属课程：[HCOMM 源码学习](../hcomm-source.md)

七个单元走完，把结论压缩成地图，并给出与 HCCL 源码课程的联合视角与岗位能力对照。

## 一张图收束整个课程

```text
控制面（低频，建域时）                        数据面（高频，每次通信）
────────────────────────────                ──────────────────────────────
api_c_adpt 入口                              HcommLocalCopy/LocalReduce（本地）
HcclCommInit*（建域四变体）                   HcommThreadNotify Record/Wait（Thread 同步）
  → rank_info_detect（探测）                  HcommRead/Write(+Reduce) OnThread（Channel 通信）
  → rank_graph（Node/Edge/Link/层级建图）        Nbi / WithNotify 变体
  → resource_mgr local/remote（配资源）       HcommBatchMode / TaskCache（批量与缓存）
  → communicator（域成型）
                                              引擎侧：
查询：HcclRankGraphGet*（13 动词）             AICPU：Kernel 启动后动态编排
生命周期：Suspend/Resume/状态回调               AIV：编排静态化进 Kernel
team / dfx 五件套                             CCU：ccu:: 资源抽象 + 指令级执行
                                              （片上缓存把访存降一个数量级）
        HcclThread/Channel/Mem ←──────────→  Thread/Channel/Mem 的运行时
        （L2 域内资源）                         （L3 OnThread 系列的消费对象）
```

## 必须带走的概念清单

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| 三块源码 | base_comm / coll_communicator_mgr / legacy | 单元 0 |
| 头文件三族 | hccl/*（L2）/ hcomm_*（L3）/ ccu/*.hpp（CCU） | 单元 0 |
| 通信模型 | 通信内存 + Endpoint + Channel；网络/内存两种语义 | 单元 1 |
| 并发模型 | Thread 抽象；引擎决定落地（Stream+Notify 或 VectorCore+内存） | 单元 1 |
| 建域流水线 | 入口 → 探测 → 建图 → 配资源 → 域成型 | 单元 2 |
| 拓扑 13 动词 | HcclRankGraphGet*：算法选择的输入 | 单元 2 |
| 原语四家族 | 本地 / Thread 同步 / Channel 通信 / 批量缓存 | 单元 3 |
| 命名后缀 | OnThread=绑定执行；Nbi=非阻塞；WithNotify=搬运+通知 | 单元 3 |
| HCCL Buffer | 200MB 锁页中转内存，解决异步悬空指针 | 单元 3 |
| 七步流程 | 定义接口→查拓扑→选算法→建资源→下发→编排→完成同步 | 单元 4 |
| 编排时机 | AICPU 动态（下发后）/ AIV 静态 / CCU 硬件固化 | 单元 5 |
| CCU 三优势 | 访存降一个数量级 / 归约确定性 / 低时延不占核 | 单元 5 |
| KernelArg/TaskArg | 编排参数走入参；执行参数 LoadArg 动态加载 | 单元 5 |
| 样例两仓 | hcomm：建域三式+ACLGraph；hccl：custom_ops | 单元 6 |

## HCCL + HCOMM 联合地图

两个课程拼起来，才是一行 `dist.all_reduce` 的完整世界：

```text
dist.all_reduce（你的代码）
  ↓ torch_npu / ProcessGroupHCCL      ← 《昇腾与 HCCL》HCCL 第 0 章
HcclAllReduce（L1，hccl 仓）
  ↓ selector 选算法 → template → executor      ← HCCL 源码课程：内置算子怎么跑
HCOMM 控制面（建域时已备好名词）                ← 本课程：建域建图配资源
HCOMM 数据面（Write/Read/Notify 动词）          ← 本课程：原语四家族
  ↓ 引擎：AICPU 动态编排 / AIV 静态 / CCU 硬件   ← 本课程单元 4/5
HCCS / RoCE / UB / 物理链路
```

而当你**自己写扩展算子**时，路径从 `HcclEngineCtxGet` 直接进入 HCOMM——**绕过 HCCL 算子层**：查拓扑（L2-res）→ 拿 Thread/Channel（L2-res）→ 用原语编排（L3-prim）。这就是"通信平台与算子开发解耦"的准确含义。

## 岗位能力对照

学完 HCCL + HCOMM 两个课程，对照 HCCL 集合通信算子开发岗位的核心能力：

- [ ] **读懂内置算子**：selector/algorithm/executor 的调用链（[HCCL 源码 5/6](../hccl-source/05-coll-algorithms.md)）；
- [ ] **读懂底座**：建域流水线、拓扑建模、原语与资源（本课程单元 1/2/3）；
- [ ] **写出扩展算子**：七步流程 + 三引擎选型 + 样例起步（本课程单元 4/5/6）；
- [ ] **排障能力**：dfx 五件套入口 + 错误码文档（`docs/zh/error_codes`，EI0001-EI0020）+ [CANN 常用命令速查](../cann常用命令.md)；
- [ ] **跟进社区方向**：仓库 `docs/zh/rfcs/`（0001 topology-based-ccl-monitor、0002 host-nic-plugin）——开源社区的演进路标。

## 继续深入的三个方向

1. **动手**：按 [单元 6 五步路线图](06-examples-first-op.md) 跑通第一个自定义算子——纸上得来终觉浅；
2. **对照 NCCL**：把 HCOMM 的 Channel/QP/Notify 与 NCCL 的 Channel/Proxy 对读——两套实现同一套问题（[NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)）；
3. **回望课程**：带着源码问题回读《集合通信》——"selector 在哪一步用到了 α-β-γ"（本课程单元 4 步骤③ ↔ [算法选择](../../collective/03-tree-algorithms-selection.md#_03-3-选型地图-大小消息的分界线)）。

## 最终参考资料

- [HCOMM 通信基础库仓库](https://gitcode.com/cann/hcomm) 与 [HCCL 开源仓库](https://gitcode.com/cann/hccl)
- [通信算子开发指南（总目录）](https://gitcode.com/cann/hcomm/blob/master/docs/zh/comm_op_dev_guide/README.md)
- [通信算子开发 API 参考](https://gitcode.com/cann/hcomm/tree/master/docs/zh/api_ref/comm_opdev)
- [HCCL 源码学习课程](../hccl-source.md)

---

[返回课程导学 →](../hcomm-source.md) · [返回专栏目录 →](../index.md)
