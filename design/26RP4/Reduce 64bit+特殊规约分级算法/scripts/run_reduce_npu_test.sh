#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
# Reduce 64bit/PROD 两级分级算法（Mesh+NHR）NPU 真机用例脚本
# 对应 STC 的 NPU_RD_001~011（echo 编号与 STC 一致，跑完直接回填执行情况列）
# 不想跑的用例整段用 # 注释掉即可
# -----------------------------------------------------------------------------------------------------------

# ============================== 环境准备 ==============================
Ascend_dir="/home/l00841484/Ascend"

# CANN 环境加载（PATH/头文件/库路径等）
source $Ascend_dir/cann/set_env.sh

# mpich 可执行与动态库、CANN lib64 优先加载
export PATH=/usr/local/mpi/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpi/mpich/lib/:$Ascend_dir/cann/lib64:$LD_LIBRARY_PATH

# 进入 hccl_test 工具目录并编译（增量编译，重复执行幂等）
cd $Ascend_dir/cann/tools/hccl_test/
make MPI_HOME=/usr/local/mpi/mpich/ ASCEND_DIR=$Ascend_dir/cann/

# 集群 rank 表（16 进程 = 2 节点×8）
export RANK_TABLE_FILE=/mnt/hccl/ranktable.json

# INFO 级日志：plog 含 [SelectorEngine] selected algo type 选路日志，便于确认选中算法
export ASCEND_GLOBAL_LOG_LEVEL=1

# 建链用网卡
export HCCL_SOCKET_IFNAME=enp35s0f2,eth5

# 可选：batch 一致性（需要确定性场景时放开）
#export HCCL_DETERMINISTIC=STRICT

# 可选：性能采集（需要 prof 数据时放开）
#export HCCL_TEST_PROFILING=1
#export HCCL_TEST_PROFILING_PATH=/home/l00841484/prof/

# 日志根目录（时间戳隔离，每个用例单独子目录，多次运行不互相覆盖）
LOG_ROOT=/home/l00841484/npu_logs/$(date +%Y%m%d_%H%M%S)

# ============================== 用例区 ==============================

# NPU_RD_001 | STC NPU_RD_001 | 基线：int64 sum 64m
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_001
echo "NPU_RD_001"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_002 | STC NPU_RD_002 | fp64 sum 64m（-d 参数生效性复核）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_002
echo "NPU_RD_002"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d fp64 -o sum -p 8 -a aicpu_ts

# NPU_RD_003 | STC NPU_RD_003 | uint64 max 64m
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_003
echo "NPU_RD_003"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d uint64 -o max -p 8 -a aicpu_ts

# NPU_RD_004 | STC NPU_RD_004 | fp64 min 64m
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_004
echo "NPU_RD_004"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d fp64 -o min -p 8 -a aicpu_ts

# NPU_RD_005 | STC NPU_RD_005 | int32 prod 64m（PROD 回绕语义）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_005
echo "NPU_RD_005"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d int32 -o prod -p 8 -a aicpu_ts

# NPU_RD_006 | STC NPU_RD_006 | 小消息边界 8B（count=1）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_006
echo "NPU_RD_006"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 8 -e 8 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_007 | STC NPU_RD_007 | 小消息边界 24B（3 元素非对齐）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_007
echo "NPU_RD_007"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 24 -e 24 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_008 | STC NPU_RD_008 | 4G 大包多 loop 压力（风险3缓解验证）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_008
echo "NPU_RD_008"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 4g -e 4g -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_009a | STC NPU_RD_009 | 指定新算法（token 以 alg_parse 实际解析为准）
export HCCL_ALGO="reduce:sequence{mesh,nhraicpureduce}"
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_009a
echo "NPU_RD_009a"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_009b | STC NPU_RD_009 | Sole 对照（同参数耗时对比）
export HCCL_ALGO="reduce:sole{nhraicpureduce}"
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_009b
echo "NPU_RD_009b"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_010 | STC NPU_RD_010 | 自适应（unset 指定，验证自动选中）
unset HCCL_ALGO
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_010
echo "NPU_RD_010"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_011a | STC NPU_RD_011 | 门禁回归：int32 min 常规 dtype 不进入新算法
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_011a
echo "NPU_RD_011a"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 64m -e 64m -f 2 -d int32 -o min -p 8 -a aicpu_ts

# NPU_RD_011b | STC NPU_RD_011 | 单机 8 进程路由回归
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_011b
echo "NPU_RD_011b"
mpirun -f /home/l00841484/hostfile -n 8 ./bin/reduce_test -b 64m -e 64m -f 2 -d int64 -o sum -p 8 -a aicpu_ts

echo "全部日志目录: $LOG_ROOT"
