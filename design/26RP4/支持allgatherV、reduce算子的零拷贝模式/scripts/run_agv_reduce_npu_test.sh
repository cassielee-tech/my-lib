#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# AllGatherV / Reduce 零拷贝 NPU 真机测试
#
# 前置：
#   1) 已安装当前分支构建的 hccl 包到 CANN_DIR；
#   2) CANN_DIR/set_env.sh 可用；
#   3) MPI（默认 mpich）与 hostfile/rank table 已按机器配置。
#
# 用法：
#   ASCEND_DIR=/usr/local/Ascend CANN_DIR=/usr/local/Ascend/cann \
#   HOSTFILE=/path/to/hostfile HCCL_SOCKET_IFNAME=eth0 \
#   bash run_agv_reduce_npu_test.sh
#
# 说明：
#   - hccl_test 的 -m 1 会分配对称内存并调用 HcclCommSymWinRegister；
#   - 不要同时使用 -z 1（zero-copy 模式）；
#   - 真机不要设置 HCCL_ST_SYM_WIN/HCCL_ST_HCOMM_VERSION（仅 ST 仿真使用）；
#   - hccl_test 的 AGV 用例只生成全 rank 均匀 counts/displs；displs 空洞/非法场景需 ST 或自定义 harness。
#   - A3 若不支持 -a aicpu_ts，改用 export HCCL_OP_EXPANSION_MODE=AI_CPU 并去掉各用例的 -a aicpu_ts。
# -----------------------------------------------------------------------------------------------------------
set -u

ASCEND_DIR=${ASCEND_DIR:-/usr/local/Ascend}
CANN_DIR=${CANN_DIR:-${ASCEND_DIR}/cann}
HCCL_TEST_DIR=${HCCL_TEST_DIR:-${CANN_DIR}/tools/hccl_test}
MPI_HOME=${MPI_HOME:-/usr/local/mpi/mpich}
HOSTFILE=${HOSTFILE:-/home/l00841484/hostfile}
RANK_TABLE_FILE=${RANK_TABLE_FILE:-/mnt/hccl/ranktable.json}
HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp35s0f2,eth5}

if [[ ! -f "${CANN_DIR}/set_env.sh" ]]; then
    echo "CANN_DIR/set_env.sh not found: ${CANN_DIR}/set_env.sh" >&2
    exit 1
fi

source "${CANN_DIR}/set_env.sh"
export PATH="${MPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:${CANN_DIR}/lib64:${LD_LIBRARY_PATH:-}"
export RANK_TABLE_FILE
export ASCEND_GLOBAL_LOG_LEVEL=1
export HCCL_SOCKET_IFNAME
# 真机不要带 ST 仿真注入；如 shell 里已有 LD_PRELOAD（例如 libst_symwin.so），请手工清理。
unset HCCL_ST_SYM_WIN HCCL_ST_HCOMM_VERSION
unset HCCL_ALGO

LOG_ROOT=${LOG_ROOT:-${HOME}/npu_logs/$(date +%Y%m%d_%H%M%S)}
SUMMARY="${LOG_ROOT}/summary.txt"
mkdir -p "${LOG_ROOT}"

cd "${HCCL_TEST_DIR}" || exit 1
if ! make MPI_HOME="${MPI_HOME}" ASCEND_DIR="${CANN_DIR}" > "${LOG_ROOT}/build.log" 2>&1; then
    echo "hccl_test build failed, see ${LOG_ROOT}/build.log" >&2
    exit 1
fi

run_case() {
    local name="$1"
    shift
    local dir="${LOG_ROOT}/${name}"
    mkdir -p "${dir}"
    export ASCEND_PROCESS_LOG_PATH="${dir}"
    echo "=== ${name} ==="
    timeout 900 "$@" 2>&1 | tee "${dir}/mpirun.log"
    local ret=${PIPESTATUS[0]}
    echo "${name} ret=${ret}" | tee -a "${SUMMARY}"
}

# ============================== AGV（A3 仅单 Server） ==============================
# 基线 / 对称对照；-m 1 注册 send+recv 对称窗口
run_case NPU_AGV_SYM_001_base_8r_64m \
    mpirun -f "${HOSTFILE}" -n 8 ./bin/all_gatherv_test \
    -b 64m -e 64m -f 2 -d fp32 -m 0 -a aicpu_ts -p 8 -c 1 -n 5
run_case NPU_AGV_SYM_001_sym_8r_64m \
    mpirun -f "${HOSTFILE}" -n 8 ./bin/all_gatherv_test \
    -b 64m -e 64m -f 2 -d fp32 -m 1 -a aicpu_ts -p 8 -c 1 -n 5

# 4 rank + FP16
run_case NPU_AGV_SYM_001_sym_4r_16m_fp16 \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/all_gatherv_test \
    -b 16m -e 16m -f 2 -d fp16 -m 1 -a aicpu_ts -p 4 -c 1 -n 5

# 大数据（>256MB 单 loop）
run_case NPU_AGV_SYM_001_sym_8r_512m \
    mpirun -f "${HOSTFILE}" -n 8 ./bin/all_gatherv_test \
    -b 512m -e 512m -f 2 -d fp32 -m 1 -a aicpu_ts -p 8 -c 1 -n 3

# ============================== Reduce SoleNHR ==============================
run_case NPU_RED_SYM_001_sole_base_4r_1m \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/reduce_test \
    -b 1m -e 1m -f 2 -d fp32 -o sum -r 0 -m 0 -a aicpu_ts -p 4 -c 1 -n 5
run_case NPU_RED_SYM_001_sole_sym_4r_1m \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/reduce_test \
    -b 1m -e 1m -f 2 -d fp32 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 5

# 非 0 root
run_case NPU_RED_SYM_001_sole_sym_4r_root3 \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/reduce_test \
    -b 1m -e 1m -f 2 -d fp32 -o sum -r 3 -m 1 -a aicpu_ts -p 4 -c 1 -n 5

# ============================== Reduce ParallelMeshNHR ==============================
# 注意：当前 Parallel 零拷贝路径未启用（enableRemoteMemAccess=false），预期 CCL 回退；
# 本组只验证正确性，不要按零拷贝带宽收益评估。
run_case NPU_RED_SYM_001_parallel_base_16r_16m \
    mpirun -f "${HOSTFILE}" -n 16 ./bin/reduce_test \
    -b 16m -e 16m -f 2 -d fp32 -o max -r 0 -m 0 -a aicpu_ts -p 8 -c 1 -n 5
run_case NPU_RED_SYM_001_parallel_sym_16r_16m \
    mpirun -f "${HOSTFILE}" -n 16 ./bin/reduce_test \
    -b 16m -e 16m -f 2 -d fp32 -o max -r 0 -m 1 -a aicpu_ts -p 8 -c 1 -n 5

# ============================== 大包多轮 / 确定性 ==============================
run_case NPU_RED_SYM_002_sym_4r_512m_sum \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/reduce_test \
    -b 512m -e 512m -f 2 -d fp32 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 10

# 3 级 32rank：hostfile 支持 32 进程时再放开
# run_case NPU_RED_SYM_002_sym_32r_1m_max \
#     mpirun -f "${HOSTFILE}" -n 32 ./bin/reduce_test \
#     -b 1m -e 1m -f 2 -d fp32 -o max -r 0 -m 1 -a aicpu_ts -p 8 -c 1 -n 5

# ============================== 回退/负向 ==============================
# 64bit / FP64 / PROD 应回退 CCL
run_case NPU_RED_SYM_004_sym_4r_int64_sum \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/reduce_test \
    -b 1m -e 1m -f 2 -d int64 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3
run_case NPU_RED_SYM_004_sym_4r_fp64_sum \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/reduce_test \
    -b 1m -e 1m -f 2 -d fp64 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3
run_case NPU_RED_SYM_004_sym_4r_fp32_prod \
    mpirun -f "${HOSTFILE}" -n 4 ./bin/reduce_test \
    -b 1m -e 1m -f 2 -d fp32 -o prod -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3

echo "日志目录: ${LOG_ROOT}"
echo "汇总: ${SUMMARY}"
echo "判读示例:"
echo "  grep -R \"supportSymmetricMemory\\|rootUsesOutputWorkspace\\|RefreshSymmetricMemory\" ${LOG_ROOT}"
echo "  grep -R -E \"check.*fail|FAILED|Error|check_err\" ${LOG_ROOT}/*/mpirun.log"
echo "  grep -R \"bandwidth\" ${LOG_ROOT}/*/mpirun.log | head -30"
