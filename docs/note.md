

hccl仓一句解释："HCCL 是华为昇腾的集合通信库，解决大模型训练/推理里多块 NPU 之间的数据交换问题。它的结构分三层：最上是算子层（本仓 src/ops，每个算子按 selector→executor→template 三段式组织），最下是独立仓 hcomm 的通信底座，两层靠 dlsym 动态加载解耦。一次 AllReduce 执行时会先由 selector 根据数据量和拓扑挑选 Ring/Mesh 等算法，再由 executor 编排，最后落到 aicpu/aiv/ccu 某个通信引擎上实际搬运数据。它支持单算子和图模式，是 CANN 里承上启下（对上接框架、对下驱动网卡）的关键组件。"


```
src/ops/<算子名>/
├── selector/          # 第1步：选算法（用哪个算法跑）
└── algorithm/
    ├── executor/      # 第2步：执行器（编排执行流程）
    └── template/      # 第3步：引擎模板（aicpu/aiv/ccu 分别怎么干）
```
"三段式流水线"，全仓的万能骨架：selector 选算法 → executor 编排 → template 落地执行.


## 引擎
AICPI_TS
AIV
CCU

