#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# AllGatherV / Reduce 零拷贝 NPU 真机测试
# 对应 STC 的 NPU_AGV_SYM_001 / NPU_RED_SYM_001 / NPU_RED_SYM_002
# 不想跑的用例整段用 # 注释掉即可
# -----------------------------------------------------------------------------------------------------------

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
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_base_8r_64m
echo "NPU_AGV_SYM_001_base_8r_64m"
mpirun -f ${HOSTFILE} -n 8 ./bin/all_gatherv_test -b 64m -e 64m -f 2 -d fp32 -m 0 -a aicpu_ts -p 8 -c 1 -n 5

# NPU_AGV_SYM_001_sym_8r_64m | 对称内存
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_64m
echo "NPU_AGV_SYM_001_sym_8r_64m"
mpirun -f ${HOSTFILE} -n 8 ./bin/all_gatherv_test -b 64m -e 64m -f 2 -d fp32 -m 1 -a aicpu_ts -p 8 -c 1 -n 5

# NPU_AGV_SYM_001_sym_4r_16m_fp16 | 4 rank + FP16
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_sym_4r_16m_fp16
echo "NPU_AGV_SYM_001_sym_4r_16m_fp16"
mpirun -f ${HOSTFILE} -n 4 ./bin/all_gatherv_test -b 16m -e 16m -f 2 -d fp16 -m 1 -a aicpu_ts -p 4 -c 1 -n 5

# NPU_AGV_SYM_001_sym_8r_512m | >256MB 单 loop
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_AGV_SYM_001_sym_8r_512m
echo "NPU_AGV_SYM_001_sym_8r_512m"
mpirun -f ${HOSTFILE} -n 8 ./bin/all_gatherv_test -b 512m -e 512m -f 2 -d fp32 -m 1 -a aicpu_ts -p 8 -c 1 -n 3

# ============================== Reduce SoleNHR：4p 单机 ==============================

# NPU_RED_SYM_001_sole_base_4r_1m | 基线
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_sole_base_4r_1m
echo "NPU_RED_SYM_001_sole_base_4r_1m"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o sum -r 0 -m 0 -a aicpu_ts -p 4 -c 1 -n 5

# NPU_RED_SYM_001_sole_sym_4r_1m | 对称内存 root=0
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_1m
echo "NPU_RED_SYM_001_sole_sym_4r_1m"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 5

# NPU_RED_SYM_001_sole_sym_4r_root3 | 非 0 root
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_sole_sym_4r_root3
echo "NPU_RED_SYM_001_sole_sym_4r_root3"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o sum -r 3 -m 1 -a aicpu_ts -p 4 -c 1 -n 5

# ============================== Reduce ParallelMeshNHR：16p（2 节点×8） ==============================
# 注意：当前 Parallel 零拷贝未启用（enableRemoteMemAccess=false），预期 CCL 回退；
# 本组只验证正确性，不按零拷贝收益评估。

# NPU_RED_SYM_001_parallel_base_16r_16m | 基线
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_parallel_base_16r_16m
echo "NPU_RED_SYM_001_parallel_base_16r_16m"
mpirun -f ${HOSTFILE} -n 16 ./bin/reduce_test -b 16m -e 16m -f 2 -d fp32 -o max -r 0 -m 0 -a aicpu_ts -p 8 -c 1 -n 5

# NPU_RED_SYM_001_parallel_sym_16r_16m | 对称内存
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_001_parallel_sym_16r_16m
echo "NPU_RED_SYM_001_parallel_sym_16r_16m"
mpirun -f ${HOSTFILE} -n 16 ./bin/reduce_test -b 16m -e 16m -f 2 -d fp32 -o max -r 0 -m 1 -a aicpu_ts -p 8 -c 1 -n 5

# ============================== 大包多轮 / 确定性 ==============================

# NPU_RED_SYM_002_sym_4r_512m_sum | 512m 多轮
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_002_sym_4r_512m_sum
echo "NPU_RED_SYM_002_sym_4r_512m_sum"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 512m -e 512m -f 2 -d fp32 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 10

# 3 级 32rank：hostfile 支持 32 进程时再放开
# export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_002_sym_32r_1m_max
# echo "NPU_RED_SYM_002_sym_32r_1m_max"
# mpirun -f ${HOSTFILE} -n 32 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o max -r 0 -m 1 -a aicpu_ts -p 8 -c 1 -n 5

# ============================== 回退/负向 ==============================

# NPU_RED_SYM_004_int64_sum | 64bit 应回退
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_004_int64_sum
echo "NPU_RED_SYM_004_int64_sum"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d int64 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3

# NPU_RED_SYM_004_fp64_sum | FP64 应回退
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_004_fp64_sum
echo "NPU_RED_SYM_004_fp64_sum"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp64 -o sum -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3

# NPU_RED_SYM_004_fp32_prod | PROD 应回退
export ASCEND_PROCESS_LOG_PATH=${LOG_ROOT}/NPU_RED_SYM_004_fp32_prod
echo "NPU_RED_SYM_004_fp32_prod"
mpirun -f ${HOSTFILE} -n 4 ./bin/reduce_test -b 1m -e 1m -f 2 -d fp32 -o prod -r 0 -m 1 -a aicpu_ts -p 4 -c 1 -n 3

echo "全部日志目录: ${LOG_ROOT}"
