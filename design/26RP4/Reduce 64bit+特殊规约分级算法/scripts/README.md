# Reduce 64bit/PROD 两级分级算法 — 实验环境一键执行脚本

配套：PR#3650（分支 `feature/reduce-64bit-prod-2level-hier`）+ STC《Reduce_64bit特殊规约两级分级算法_STC.xlsx》

## 脚本清单

| 脚本 | 用途 | 对应 STC 用例 |
| ---- | ---- | ---- |
| `run_reduce_64bit_ut.sh` | UT 一键构建+运行（含 core 崩溃调试模式） | UT_RD_2L_001~006 |
| `run_checker_case.sh` | 单条 Checker（L2 仿真）用例执行 | CHK_RD_*（23 条） |
| `run_reduce_st.sh` | ST 仓内仿真框架构建+运行 | ST_RD_2L_001~015（用例代码待随 PR 补充） |

## 快速上手（实验环境 Linux + 已装 CANN toolkit）

```bash
# 1. UT 全量（selector + core 两个目标）
bash run_reduce_64bit_ut.sh -r /path/to/hccl -c /usr/local/Ascend/ascend-toolkit/latest

# 2. UT 调试模式（当前 core_test 静态初始化段错误定位）：ASAN 构建 + gdb 栈回溯
bash run_reduce_64bit_ut.sh -r /path/to/hccl -c /usr/local/Ascend/ascend-toolkit/latest -t core -d

# 3. 单条 Checker 用例（148 拓扑 + int64 8M + sum，对应 CHK_RD_BASE_001）
bash run_checker_case.sh -r /path/to/hccl -m 148 -n 32 -d int64 -s 8M -o sum -r 0

# 4. Checker 后按关键字复核算法选中（可选）
bash run_checker_case.sh -r /path/to/hccl -m 148 -n 32 -d int64 -s 8M -o sum -r 0 \
     -k "ReduceAutoSelector" -k "OrchestrateLoop"
```

## 当前待办（脚本已就绪、等待执行）

1. **core_test 段错误定位**（最高优）：跑 `run_reduce_64bit_ut.sh -d`，把日志目录里
   `core_test.gdb.txt`（gdb 栈回溯）或 `core_test.asan.txt`（ASAN 报告）贴回——
   崩溃窗口已由 CI 探针锁定在"静态初始化收尾→首个用例前"（.init_array 未标优先级段，
   嫌疑为算法注册宏链），gdb 一跑即知具体帧。
2. selector_test 预期 10 用例全过（CI 已验证），可作环境自检对照组。
3. ST 用例代码尚未随 PR 提交，`run_reduce_st.sh` 已就绪待用。
