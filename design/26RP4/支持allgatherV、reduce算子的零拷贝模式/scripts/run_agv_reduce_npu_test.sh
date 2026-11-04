#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# AllGatherV / Reduce 零拷贝 NPU 真机测试
# 对应 STC 的 NPU_AGV_SYM_001 / NPU_RED_SYM_001 / NPU_RED_SYM_002
# 不想跑的用例整段用 # 注释掉即可
#
# 用例选择：
#   CASES="NPU_AGV_SYM_001_sym_8r_64m NPU_RED_SYM_001_sole_sym_4r_1m" bash run_agv_reduce_npu_test.sh
#   不设置 CASES = 跑全部
# 只重跑失败用例：
#   CASES="$(tr '\n' ' ' < /path/to/old_logs/failed_cases.txt)" bash run_agv_reduce_npu_test.sh
# -----------------------------------------------------------------------------------------------------------

CASES=${CASES:-}

# ============================== 环境准备 ==============================
Ascend_dir="/home/l00841484/Ascend"
CANN_DIR=${Ascend_dir}/cann
HOSTFILE=/home/l00841484/hostfile

# CANN 环境加载
source ${CANN_DIR}/set_env.sh

# mpich 可执行与动态库、CANN lib64 优先加载
export PATH=/usr/local/mpi/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpi/mpich/lib/:${CANN_DIR}/lib64:$LD_LIBRARY_PATH

# 进入 hccl_test 工具目录并编译（增量编译，重复执行幂等）
cd ${CANN_DIR}/tools/hccl_test/
make MPI_HOME=/usr/local/mpi/mpich/ ASCEND_DIR=${CANN_DIR}/

# 集群 rank 表（16 进程 = 2 节点×8；4/8 进程按实际 hostfile 调整）
export RANK_TABLE_FILE=/mnt/hccl/ranktable.json

# INFO 级日志，便于确认 RefreshSymmetricMemory/选路
export ASCEND_GLOBAL_LOG_LEVEL=1

# 建链用网卡
export HCCL_SOCKET_IFNAME=enp35s0f2,eth5

# 真机不要带 ST 仿真注入
unset HCCL_ST_SYM_WIN
unset HCCL_ST_HCOMM_VERSION
unset HCCL_ALGO

# 日志根目录
LOG_ROOT=/home/l00841484/npu_logs/$(date +%Y%m%d_%H%M%S)
mkdir -p ${LOG_ROOT}

# ============================== AGV：8p 单机 / A3 仅单 Server ==============================

# NPU_AGV_SYM_001_base_8r_64m | 基线
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_AGV_SYM_001_base_8r_64m "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_AGV_SYM_001_base_8r_64m
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_base_8r_64m
echo "NPU_AGV_SYM_001_base_8r_64m"
mpirun -f ${HOSTFILE} -n 8 ./bin/all_gatherv_test -b 64m -e 64m -f 2 -d fp32 -m 0 -a aicpu_ts -p 8 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_AGV_SYM_001_base_8r_64m/stdout.log
fi

# NPU_AGV_SYM_001_sym_8r_64m | 对称内存
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_AGV_SYM_001_sym_8r_64m "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_64m
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_64m
echo "NPU_AGV_SYM_001_sym_8r_64m"
mpirun -f ${HOSTFILE} -n 8 ./bin/all_gatherv_test -b 64m -e 64m -f 2 -d fp32 -m 1 -a aicpu_ts -p 8 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_64m/stdout.log
fi

# NPU_AGV_SYM_001_sym_4r_16m_fp16 | 4 rank + FP16
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_AGV_SYM_001_sym_4r_16m_fp16 "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_AGV_SYM_001_sym_4r_16m_fp16
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_sym_4r_16m_fp16
echo "NPU_AGV_SYM_001_sym_4r_16m_fp16"
mpirun -f ${HOSTFILE} -n 4 ./bin/all_gatherv_test -b 16m -e 16m -f 2 -d fp16 -m 1 -a aicpu_ts -p 4 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_AGV_SYM_001_sym_4r_16m_fp16/stdout.log
fi

# NPU_AGV_SYM_001_base_8r_512m | 512m 基线
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_AGV_SYM_001_base_8r_512m "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_AGV_SYM_001_base_8r_512m
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_base_8r_512m
echo "NPU_AGV_SYM_001_base_8r_512m"
mpirun -f ${HOSTFILE} -n 8 ./bin/all_gatherv_test -b 512m -e 512m -f 2 -d fp32 -m 0 -a aicpu_ts -p 8 -c 1 -n 3 2>&1 | tee ${LOG_ROOT}/NPU_AGV_SYM_001_base_8r_512m/stdout.log
fi

# NPU_AGV_SYM_001_sym_8r_512m | >256MB 单 loop
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_AGV_SYM_001_sym_8r_512m "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_512m
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_512m
echo "NPU_AGV_SYM_001_sym_8r_512m"
mpirun -f ${HOSTFILE} -n 8 ./bin/all_gatherv_test -b 512m -e 512m -f 2 -d fp32 -m 1 -a aicpu_ts -p 8 -c 1 -n 3 2>&1 | tee ${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_512m/stdout.log
fi

# ============================== Reduce SoleNHR：4p 单机 ==============================

# NPU_RED_SYM_001_sole_base_4r_1m | 基线
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_001_sole_base_4r_1m "* ]]; then
export HCCL_ALGO="reduce:sole{nhr}"
mkdir -p ${LOG_ROOT}/NPU_RED_SYM_001_sole_base_4r_1m
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_sole_base_4r_1m
echo "NPU_RED_SYM_001_sole_base_4r_1m"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o sum -r 0 -m 0 -a aicpu_ts -p 4 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_001_sole_base_4r_1m/stdout.log
unset HCCL_ALGO
fi

# NPU_RED_SYM_001_sole_sym_4r_1m | 对称内存 root=0
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_001_sole_sym_4r_1m "* ]]; then
export HCCL_ALGO="reduce:sole{nhr}"
mkdir -p ${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_1m
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_1m
echo "NPU_RED_SYM_001_sole_sym_4r_1m"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_1m/stdout.log
unset HCCL_ALGO
fi

# NPU_RED_SYM_001_sole_sym_4r_root3 | 非 0 root
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_001_sole_sym_4r_root3 "* ]]; then
export HCCL_ALGO="reduce:sole{nhr}"
mkdir -p ${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_root3
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_root3
echo "NPU_RED_SYM_001_sole_sym_4r_root3"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o sum -r 3 -m 1 -a aicpu_ts -p 4 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_root3/stdout.log
unset HCCL_ALGO
fi

# ============================== Reduce ParallelMeshNHR：16p（2 节点×8） ==============================
# 注意：当前 Parallel 零拷贝未启用（enableRemoteMemAccess=false），预期 CCL 回退；
# 本组只验证正确性，不按零拷贝收益评估。

# 8p 环境默认注释；16p（2 节点×8）时取消下面两段的注释。
# NPU_RED_SYM_001_parallel_base_16r_16m | 基线
# if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_001_parallel_base_16r_16m "* ]]; then
# export HCCL_ALGO="reduce:parallel{mesh,nhr}"
# mkdir -p ${LOG_ROOT}/NPU_RED_SYM_001_parallel_base_16r_16m
# export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_parallel_base_16r_16m
# echo "NPU_RED_SYM_001_parallel_base_16r_16m"
# mpirun -f ${HOSTFILE} -n 16 ./bin/reduce_test -b 16m -e 16m -f 2 -d fp32 -o max -r 0 -m 0 -a aicpu_ts -p 8 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_001_parallel_base_16r_16m/stdout.log
# unset HCCL_ALGO
# fi

# NPU_RED_SYM_001_parallel_sym_16r_16m | 对称内存
# if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_001_parallel_sym_16r_16m "* ]]; then
# export HCCL_ALGO="reduce:parallel{mesh,nhr}"
# mkdir -p ${LOG_ROOT}/NPU_RED_SYM_001_parallel_sym_16r_16m
# export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_parallel_sym_16r_16m
# echo "NPU_RED_SYM_001_parallel_sym_16r_16m"
# mpirun -f ${HOSTFILE} -n 16 ./bin/reduce_test -b 16m -e 16m -f 2 -d fp32 -o max -r 0 -m 1 -a aicpu_ts -p 8 -c 1 -n 5 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_001_parallel_sym_16r_16m/stdout.log
# unset HCCL_ALGO
# fi

# ============================== 大包多轮 / 确定性 ==============================

# NPU_RED_SYM_002_sym_4r_512m_sum | 512m 多轮
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_002_sym_4r_512m_sum "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_RED_SYM_002_sym_4r_512m_sum
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_002_sym_4r_512m_sum
echo "NPU_RED_SYM_002_sym_4r_512m_sum"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 512m -e 512m -f 2 -d fp32 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 10 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_002_sym_4r_512m_sum/stdout.log
fi

# 3 级 32rank：hostfile 支持 32 进程时再放开
# export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_002_sym_32r_1m_max
# echo "NPU_RED_SYM_002_sym_32r_1m_max"
# mpirun -f ${HOSTFILE} -n 32 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o max -r 0 -m 1 -a aicpu_ts -p 8 -c 1 -n 5

# ============================== 回退/负向 ==============================

# NPU_RED_SYM_004_int64_sum | 64bit 应回退
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_004_int64_sum "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_RED_SYM_004_int64_sum
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_004_int64_sum
echo "NPU_RED_SYM_004_int64_sum"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d int64 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_004_int64_sum/stdout.log
fi

# NPU_RED_SYM_004_fp64_sum | FP64 应回退
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_004_fp64_sum "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_RED_SYM_004_fp64_sum
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_004_fp64_sum
echo "NPU_RED_SYM_004_fp64_sum"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp64 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_004_fp64_sum/stdout.log
fi

# NPU_RED_SYM_004_fp32_prod | PROD 应回退
if [[ -z "${CASES}" || " ${CASES} " == *" NPU_RED_SYM_004_fp32_prod "* ]]; then
mkdir -p ${LOG_ROOT}/NPU_RED_SYM_004_fp32_prod
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_004_fp32_prod
echo "NPU_RED_SYM_004_fp32_prod"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o prod -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3 2>&1 | tee ${LOG_ROOT}/NPU_RED_SYM_004_fp32_prod/stdout.log
fi

# ============================== 日志汇总（第一轮） ==============================
FAILED_FILE=${LOG_ROOT}/failed_cases.txt
: > ${FAILED_FILE}
echo
echo "================ 第一轮日志汇总 ================"
for dir in $(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'NPU_*' | sort); do
    case_name=$(basename "${dir}")
    stdout_files=$(find "${dir}" -maxdepth 1 -type f \( -name 'stdout.log' -o -name 'mpirun.log' \) 2>/dev/null)
    pass=0
    fail=0
    result=NO_STDOUT
    last_perf=""
    if [ -n "${stdout_files}" ]; then
        pass=$(cat ${stdout_files} 2>/dev/null | grep -cE '\| success[[:space:]]*$' || true)
        fail=$(cat ${stdout_files} 2>/dev/null | grep -cE '\| failed[[:space:]]*$|check result failed' || true)
        if [ "${fail}" -gt 0 ]; then
            result=FAIL
        elif [ "${pass}" -gt 0 ]; then
            result=PASS
        else
            result=UNKNOWN
        fi
        last_perf=$(cat ${stdout_files} 2>/dev/null | grep -E '\| (success|failed)[[:space:]]*$' | tail -1 || true)
    fi
    algo=$(grep -rhoIE 'selected algo type is [^ ,]+' "${dir}" 2>/dev/null | sed 's/.*is //' | sort -u | paste -sd, - || true)
    sym=$(grep -rhoIE 'supportSymmetricMemory[^ ,;]*' "${dir}" 2>/dev/null | sort -u | head -3 | paste -sd, - || true)
    symlog=$(grep -rhoIE 'HcclCommSymWinRegister|remoteOutputGraphMode|symmetric|RefreshSymmetricMemory' "${dir}" 2>/dev/null | sort -u | head -3 | paste -sd, - || true)
    symroot=$(grep -rhoIE 'symmetric root [a-z ]+' "${dir}" 2>/dev/null | sort -u | head -2 | paste -sd, - || true)
    err_count=$(grep -rhIcE '\[ERROR\]|HCCL_E_[A-Z_]+|check result failed' "${dir}" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    err_samples=$(grep -rhIE '\[ERROR\]|HCCL_E_[A-Z_]+|check result failed' "${dir}" 2>/dev/null | sed 's/^[[:space:]]*//' | sort -u | head -3 || true)
    echo "=== ${case_name} ==="
    echo "  result : ${result} (success=${pass}, failed=${fail})"
    echo "  algo   : ${algo:-<none>}"
    echo "  sym    : ${sym:-<none>}"
    if [ -n "${symlog}" ]; then
        echo "  symlog : ${symlog}"
    fi
    echo "  root   : ${symroot:-<none>}"
    echo "  errors : ${err_count}"
    if [ -n "${err_samples}" ]; then
        echo "${err_samples}" | sed 's/^/           /'
    fi
    if [ -n "${last_perf}" ]; then
        echo "  perf   : ${last_perf}"
    fi
    if [ "${result}" = FAIL ] || [ "${result}" = UNKNOWN ] || [ "${result}" = NO_STDOUT ]; then
        echo "${case_name}" >> "${FAILED_FILE}"
    fi
done
echo
if [ -s "${FAILED_FILE}" ]; then
    echo "失败/未知用例："
    cat "${FAILED_FILE}"
    echo "只重跑失败用例："
    echo "CASES=\"$(tr '\n' ' ' < ${FAILED_FILE})\" bash $0"
else
    echo "全部用例结果为 PASS 或未产生失败清单。"
fi
echo "全部日志目录: ${LOG_ROOT}"
