#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. Do not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Reduce 64bit/PROD 两级分级算法（Mesh+NHR）Checker L2 仿真用例平铺脚本
# 对应 STC 的 CHK_RD_* 28 条（echo 编号与 STC 一致，跑完直接回填执行情况列）
# 与 run_reduce_npu_test.sh / run_reduce_st.sh 构成 NPU/ST/CHK 三件套
# 须在 CheckerL2 环境会话内执行（hccl-vm 可用；CheckerL2 main + hcomm master + CANN 含 PR#3650@d5479596 已装）
# 不想跑的用例整段用 # 注释掉即可
# 执行前 4 个确认动作：
#   1) hccl-vm 可用 mock 拓扑清单 → 填下方 5 个特殊 yaml 变量（为空的用例自动跳过）
#   2) reduce_test --help → 确认对称内存/图模式/inplace 参数 → 填下方 3 个开关变量
#   3) STC 预置条件的 CheckerL2/CANN/hcomm 版本占位符回填
#   4) 集群 yaml 为 normal 形态（server topo 无 position=设备侧链路）
# -----------------------------------------------------------------------------------------------------------

# ============================== 环境准备 ==============================
command -v hccl-vm >/dev/null 2>&1 || { echo "错误: 找不到 hccl-vm 命令（须在 CheckerL2 环境会话内执行）"; exit 1; }

# 引擎模式（STC 预置条件要求）
export HCCL_OP_EXPANSION_MODE=AI_CPU

# reduce_test 二进制定位（优先用环境变量 REDUCE_TEST 覆盖）
REDUCE_TEST="${REDUCE_TEST:-$(find /home/workspace/Ascend -type f -name reduce_test -path "*hccl_test*" 2>/dev/null | sort -V | tail -1)}"
[[ -z "${REDUCE_TEST}" || ! -x "${REDUCE_TEST}" ]] && { echo "错误: 未找到 reduce_test（export REDUCE_TEST=<路径> 指定）"; exit 1; }
echo "reduce_test: ${REDUCE_TEST}"

# 特殊拓扑 yaml（执行前确认后填入；为空的用例自动跳过）
CLOS_YAML=""       # TODO: 两级 CLOS 形态 yaml 全名（CHK_RD_TOPO_001）
PCIEMIX_YAML=""    # TODO: 两级 pcie-mix 形态 yaml 全名（CHK_RD_TOPO_002）
M1DCLOS_YAML=""    # TODO: 两级 MESH_1D_CLOS 规整形态（level0PcieMix=false）yaml 全名（CHK_RD_TOPO_003）
ASYM_YAML=""       # TODO: L0 不对称两级 yaml 全名（如 8+4 卡两框，CHK_RD_ASYM_001）
S64_YAML=""        # TODO: 64rank 两级 yaml 全名（如 1pod×8srv×8rank，CHK_RD_SCALE_001）

# 特殊开关参数（执行前 reduce_test --help 确认后填入）
SYM_OPT=""         # TODO: 对称内存开启参数（CHK_RD_SYM_001；为空则按默认协商状态跑，回填时标注）
GRAPH_OPT=""       # TODO: 图模式提交参数（CHK_RD_GRAPH_001；为空则以 @checker 大图校验为判据）
INPLACE_OPT=""     # TODO: inplace 用法参数（CHK_RD_INPLACE_001；为空则跳过本用例）

# 日志根目录（时间戳隔离，每个用例单独子目录，多次运行不互相覆盖）
LOG_ROOT=./checker_logs/$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_ROOT}"

# ############################## 批次1｜冒烟（先跑通环境与门禁，2 条） ##############################

# ============================== CHK_RD_BASE_001 ==============================
# STC CHK_RD_BASE_001 | 两级基线：148 int64 8M sum | H
# 预期：自适应 cost 排序（8M 倾向新算法，不预判，以 [SelectorEngine] selected algo 为准）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_BASE_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_BASE_001 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_REG_001 ==============================
# STC CHK_RD_REG_001 | 常规场景路由不变：148 fp32 8M sum | H
# 预期：selected ≠ AicpuReduceSequenceNHRAicpuReduceNHR（新算法被 attrs op.opCustomCheck 门禁过滤）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_REG_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_REG_001 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d fp32 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ############################## 批次2｜148 参数矩阵（15 条） ##############################

# ============================== CHK_RD_SIZE_001 ==============================
# STC CHK_RD_SIZE_001 | 小消息：148 int64 4K sum | M
# 预期：门禁与数据量无关均进候选池；4K 小消息 SoleMesh 胜出为大概率合法结果（cost 最小者为准，不预判）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_SIZE_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_SIZE_001 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 4K -e 4K -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_SIZE_002 ==============================
# STC CHK_RD_SIZE_002 | 多loop：148 int64 64M sum | H
# 预期：loop>1 分轮正确，断批同步仅 RS-L1 NHRPcie 段；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_SIZE_002; mkdir -p $LOG_DIR
echo "===== CHK_RD_SIZE_002 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 64M -e 64M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_SIZE_003 ==============================
# STC CHK_RD_SIZE_003 | GB级压力：148 int64 1G sum | H
# 预期：loop 数量级暴增下分轮断批同步正确、无 CUSTOM_TIMEOUT 误报（风险3缓解验证）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_SIZE_003; mkdir -p $LOG_DIR
echo "===== CHK_RD_SIZE_003 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 1G -e 1G -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_TYPE_001 ==============================
# STC CHK_RD_TYPE_001 | 类型：148 uint64 8M sum | H
# 预期：若选中新算法：uint64 走 AicpuReduceTemplate<uint64_t> 分支执行正确；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_TYPE_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_TYPE_001 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d uint64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_TYPE_002 ==============================
# STC CHK_RD_TYPE_002 | 类型：148 fp64 8M sum | H
# 预期：若选中新算法：fp64 走 AicpuReduceTemplate<double> 分支执行正确；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_TYPE_002; mkdir -p $LOG_DIR
echo "===== CHK_RD_TYPE_002 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d fp64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_TYPE_003 ==============================
# STC CHK_RD_TYPE_003 | 类型×规约：148 uint64 8M prod | M
# 预期：uint64 PROD 走 AicpuReduceTemplate<uint64_t> 分支；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_TYPE_003; mkdir -p $LOG_DIR
echo "===== CHK_RD_TYPE_003 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d uint64 -b 8M -e 8M -o prod -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_TYPE_004 ==============================
# STC CHK_RD_TYPE_004 | 类型×规约：148 fp64 8M max | M
# 预期：fp64 MAX 走 AicpuReduceTemplate<double> 分支；已知限制：输入不含 NaN/Inf；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_TYPE_004; mkdir -p $LOG_DIR
echo "===== CHK_RD_TYPE_004 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d fp64 -b 8M -e 8M -o max -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_TYPE_005 ==============================
# STC CHK_RD_TYPE_005 | 类型×规约：148 uint64 8M min | M（12 组合补齐之 UINT64×MIN）
# 预期：uint64 MIN 走 AicpuReduceTemplate<uint64_t> 分支执行正确；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_TYPE_005; mkdir -p $LOG_DIR
echo "===== CHK_RD_TYPE_005 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d uint64 -b 8M -e 8M -o min -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_TYPE_006 ==============================
# STC CHK_RD_TYPE_006 | 类型×规约：148 fp64 8M prod | M（12 组合补齐之 FP64×PROD）
# 预期：fp64 PROD 走 AicpuReduceTemplate<double> 分支，累乘结果与参考一致（输入不含 NaN/Inf）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_TYPE_006; mkdir -p $LOG_DIR
echo "===== CHK_RD_TYPE_006 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d fp64 -b 8M -e 8M -o prod -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_OP_001 ==============================
# STC CHK_RD_OP_001 | 规约：148 int64 8M prod | H
# 预期：PROD 逐元素乘积与参考实现一致；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_OP_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_OP_001 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o prod -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_OP_002 ==============================
# STC CHK_RD_OP_002 | 规约：148 int64 8M max | H
# 预期：MAX 结果正确；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_OP_002; mkdir -p $LOG_DIR
echo "===== CHK_RD_OP_002 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o max -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_OP_003 ==============================
# STC CHK_RD_OP_003 | 规约：148 int64 8M min | H
# 预期：MIN 结果正确；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_OP_003; mkdir -p $LOG_DIR
echo "===== CHK_RD_OP_003 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o min -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_OP_004 ==============================
# STC CHK_RD_OP_004 | 规约：148 int32 8M prod | H（PROD@非64bit dtype 回绕语义）
# 预期：int32 PROD 走 uint32 无符号回绕分支，溢出回绕结果与参考一致；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_OP_004; mkdir -p $LOG_DIR
echo "===== CHK_RD_OP_004 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int32 -b 8M -e 8M -o prod -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_OP_005 ==============================
# STC CHK_RD_OP_005 | 规约：148 int8 8M prod | M（补齐 srs&sd 验收矩阵 PROD@int8 项）
# 预期：int8 PROD 走 uint8 无符号回绕分支，溢出回绕结果与参考一致；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_OP_005; mkdir -p $LOG_DIR
echo "===== CHK_RD_OP_005 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int8 -b 8M -e 8M -o prod -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_ROOT_001 ==============================
# STC CHK_RD_ROOT_001 | root 非 0：148 int64 8M sum root=16 | M
# 预期：仅 root=16 输出归约结果（AG-L0 root 语义，非 root 不被写由 ST 语义校验器机器断言佐证）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_ROOT_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_ROOT_001 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 16 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ############################## 批次3｜回归拓扑（2 条） ##############################

# ============================== CHK_RD_REG_002 ==============================
# STC CHK_RD_REG_002 | 三级维持 Sole：228 三级 int64 8M sum | H
# 预期：selected = AicpuReduceSoleNHRAicpuReduce（三级维持 Sole，新算法 attrs 两级门禁被拒）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_REG_002; mkdir -p $LOG_DIR
echo "===== CHK_RD_REG_002 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 228 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_REG_003 ==============================
# STC CHK_RD_REG_003 | 单机路由不变：18 单机 int64 8M sum | M
# 预期：selected = AicpuReduceSoleMesh（单机常规，新算法两级门禁不参选）；Checker All Success
LOG_DIR=$LOG_ROOT/CHK_RD_REG_003; mkdir -p $LOG_DIR
echo "===== CHK_RD_REG_003 ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 18 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 8 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ############################## 批次4｜特殊开关（4 条） ##############################

# ============================== CHK_RD_ALGO_001 ==============================
# STC CHK_RD_ALGO_001 | HCCL_ALGO 按名强制指定新算法：148 fp32 8M sum | M
# 预期：preferred 跳过软策略（cost_table.cc:244 needSoftCheck），fp32+sum 指定仍可选；
#       selected = AicpuReduceSequenceNHRAicpuReduceNHR（Phase3 强制收敛 count=1）；Checker All Success
export HCCL_ALGO="reduce:sequence{nhraicpureduce,nhr}"
LOG_DIR=$LOG_ROOT/CHK_RD_ALGO_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_ALGO_001 (指定新算法) ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d fp32 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log
unset HCCL_ALGO

# ============================== CHK_RD_SYM_001 ==============================
# STC CHK_RD_SYM_001 | 对称内存开启：148 int64 8M sum | H（srs&sd 验收矩阵项，对应风险5）
# 预期：新算法 symMem 路径与既有两级算法行为一致；Checker All Success
# SYM_OPT 为空时按默认协商状态跑（plog 复核协商日志，回填时标注默认状态）
[ -z "$SYM_OPT" ] && echo "提示: SYM_OPT 未确认，按默认对称内存协商状态执行（回填时标注）"
LOG_DIR=$LOG_ROOT/CHK_RD_SYM_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_SYM_001 (对称内存) ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 $SYM_OPT < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ============================== CHK_RD_INPLACE_001 ==============================
# STC CHK_RD_INPLACE_001 | inplace 拒绝路径：148 int64 8M sum inplace | H（srs&sd 验收矩阵项）
# 预期：新算法被 attrs isSupportInplace=false 过滤不参选，selected ≠ 新算法名
#       （路由到支持 inplace 的算法且结果正确，或返回 HCCL_E_NOT_SUPPORT——两者皆合法），不静默算错
if [ -z "$INPLACE_OPT" ]; then
    echo "跳过 CHK_RD_INPLACE_001（INPLACE_OPT 未确认：reduce_test --help 查 inplace 用法后填入）"
else
    LOG_DIR=$LOG_ROOT/CHK_RD_INPLACE_001; mkdir -p $LOG_DIR
    echo "===== CHK_RD_INPLACE_001 (inplace 模式) ====="
    hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
    hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
    mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 $INPLACE_OPT < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
    hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
    hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log
fi

# ============================== CHK_RD_GRAPH_001 ==============================
# STC CHK_RD_GRAPH_001 | 图模式（OFFLOAD 大图）：148 int64 8M sum | M
# 预期：图模式路由同样生效（路由位于公共 Selector 阶段，OPBASE/OFFLOAD 汇聚于 ReduceOutPlaceCommon）；Checker All Success（大图）
# GRAPH_OPT 为空时以 @checker 大图校验为图模式判据（hccl-vm plugin run @checker 自带单算子+大图双校验）
[ -z "$GRAPH_OPT" ] && echo "提示: GRAPH_OPT 未确认，以 @checker 大图校验为图模式判据"
LOG_DIR=$LOG_ROOT/CHK_RD_GRAPH_001; mkdir -p $LOG_DIR
echo "===== CHK_RD_GRAPH_001 (图模式) ====="
hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
hccl-vm mock-comm 148 2>&1 | tee $LOG_DIR/mock.log
mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 $GRAPH_OPT < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log

# ############################## 批次5｜特殊拓扑（5 条，yaml 未确认自动跳过） ##############################

# ============================== CHK_RD_TOPO_001 ==============================
# STC CHK_RD_TOPO_001 | 两级 CLOS 形态特判：int64 8M sum | H（attrs 三形态声明的 CLOS 验证）
# 预期：L0=CLOS 两级走 ExecuteSelector→SelectAicpuAlgo 特判（op_common.cc:184 分流），无条件选中：
#       selected = AicpuReduceSequenceNHRAicpuReduceNHR，[ReduceAutoSelector] Algo match 命中；Checker All Success
if [ -z "$CLOS_YAML" ]; then
    echo "跳过 CHK_RD_TOPO_001（CLOS_YAML 未确认：hccl-vm 拓扑清单确认后填入）"
else
    LOG_DIR=$LOG_ROOT/CHK_RD_TOPO_001; mkdir -p $LOG_DIR
    echo "===== CHK_RD_TOPO_001 (两级CLOS) ====="
    hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
    hccl-vm mock-comm $CLOS_YAML 2>&1 | tee $LOG_DIR/mock.log
    mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
    hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
    hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log
fi

# ============================== CHK_RD_TOPO_002 ==============================
# STC CHK_RD_TOPO_002 | 两级 pcie-mix 过滤：int64 8M sum | M
# 预期：新算法被 attrs isSupportLevel0PcieMix=false 过滤（cost_model.cc:225）不进候选池，回退常规/Sole
# 执行重点：特判分支（reduce_auto_selector.cc:282）本身不检查 level0PcieMix——若实测新算法被绕过过滤直接选中，
#          本条预期改写并升级为功能问题（本用例核心价值）
if [ -z "$PCIEMIX_YAML" ]; then
    echo "跳过 CHK_RD_TOPO_002（PCIEMIX_YAML 未确认）"
else
    LOG_DIR=$LOG_ROOT/CHK_RD_TOPO_002; mkdir -p $LOG_DIR
    echo "===== CHK_RD_TOPO_002 (两级pcie-mix) ====="
    hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
    hccl-vm mock-comm $PCIEMIX_YAML 2>&1 | tee $LOG_DIR/mock.log
    mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
    hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
    hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log
fi

# ============================== CHK_RD_TOPO_003 ==============================
# STC CHK_RD_TOPO_003 | 两级 MESH_1D_CLOS 规整形态特判：int64 8M sum | M（attrs 三形态第三形态验证）
# 预期：level0PcieMix=false 不触发 TOPO_002 过滤分支；MESH_1D_CLOS≠MESH_1D 走 else 分支特判无条件选中；
#       selected = AicpuReduceSequenceNHRAicpuReduceNHR；Checker All Success
if [ -z "$M1DCLOS_YAML" ]; then
    echo "跳过 CHK_RD_TOPO_003（M1DCLOS_YAML 未确认）"
else
    LOG_DIR=$LOG_ROOT/CHK_RD_TOPO_003; mkdir -p $LOG_DIR
    echo "===== CHK_RD_TOPO_003 (两级MESH_1D_CLOS) ====="
    hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
    hccl-vm mock-comm $M1DCLOS_YAML 2>&1 | tee $LOG_DIR/mock.log
    mpirun --allow-run-as-root --oversubscribe -np 32 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
    hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
    hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log
fi

# ============================== CHK_RD_ASYM_001 ==============================
# STC CHK_RD_ASYM_001 | L0 不对称两级（GCD 降维，探索性）：np=12 int64 8M sum | M
# 预期：探索性——选中或回退 Sole 均为合法结果；重点断言不崩溃且 Checker 数据校验通过；结果记录并同步设计评审
if [ -z "$ASYM_YAML" ]; then
    echo "跳过 CHK_RD_ASYM_001（ASYM_YAML 未确认）"
else
    LOG_DIR=$LOG_ROOT/CHK_RD_ASYM_001; mkdir -p $LOG_DIR
    echo "===== CHK_RD_ASYM_001 (非对称两级 np=12) ====="
    hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
    hccl-vm mock-comm $ASYM_YAML 2>&1 | tee $LOG_DIR/mock.log
    mpirun --allow-run-as-root --oversubscribe -np 12 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
    hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
    hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log
fi

# ============================== CHK_RD_SCALE_001 ==============================
# STC CHK_RD_SCALE_001 | 64rank 规模梯度：int64 8M sum | M
# 预期：reduceCopies∝rankSize 使大消息更倾向新算法（不预判）；关注大 rankSize 下 scratch multiplier 相乘后的
#       loop 数与断批同步次数；Checker All Success
if [ -z "$S64_YAML" ]; then
    echo "跳过 CHK_RD_SCALE_001（S64_YAML 未确认）"
else
    LOG_DIR=$LOG_ROOT/CHK_RD_SCALE_001; mkdir -p $LOG_DIR
    echo "===== CHK_RD_SCALE_001 (64rank两级 np=64) ====="
    hccl-vm start ascend950_cluster_32_server_normal.yaml 2>&1 | tee $LOG_DIR/vm.log
    hccl-vm mock-comm $S64_YAML 2>&1 | tee $LOG_DIR/mock.log
    mpirun --allow-run-as-root --oversubscribe -np 64 $REDUCE_TEST -d int64 -b 8M -e 8M -o sum -r 0 -f 2 -n 1 -w 0 -c 0 < /dev/null 2>&1 | tee $LOG_DIR/mpirun.log
    hccl-vm plugin run @checker 2>&1 | tee $LOG_DIR/checker.log
    hccl-vm exit 2>&1 | tee -a $LOG_DIR/vm.log
fi

# ============================== 收尾：选路日志关键字复核 ==============================
echo ""
echo "全部日志目录: $LOG_ROOT"
echo "复核提示（plog 归档 grep 选路关键行）："
echo "  - auto 用例按 cost 排序不预判，以 [SelectorEngine] selected algo 为准"
echo "  - ALGO_001 / TOPO_001 / TOPO_003 必须命中 AicpuReduceSequenceNHRAicpuReduceNHR"
echo "  - REG_001 / INPLACE_001 必须 ≠ 新算法名；TOPO_002 若命中新算法名 → attrs 过滤被特判绕过，预期改写并升级功能问题"
echo "  - 各 checker.log 应有 CHECKER_RUN_SUMMARY / Checker All Success"
