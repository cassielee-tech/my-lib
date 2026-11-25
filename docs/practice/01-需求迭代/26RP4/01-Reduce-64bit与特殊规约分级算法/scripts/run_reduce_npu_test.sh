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

# NPU_RD_006a | STC NPU_RD_006 | 8B 小消息 | 自适应基线（v2 cost 当前选中 SoleMesh）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_006a
echo "NPU_RD_006a (自适应)"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 8 -e 8 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_006b | 8B 小消息 | 强制新算法（token 生效性以 plog selected 为准）
export HCCL_ALGO="reduce:sequence{nhraicpureduce,nhr}"
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_006b
echo "NPU_RD_006b (新算法)"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 8 -e 8 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_006c | 8B 小消息 | 强制 Sole
export HCCL_ALGO="reduce:sole{nhraicpureduce}"
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_006c
echo "NPU_RD_006c (Sole)"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 8 -e 8 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

unset HCCL_ALGO

# NPU_RD_007a | STC NPU_RD_007 | 24B 小消息 | 自适应基线（v2 cost 当前选中 SoleMesh）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_007a
echo "NPU_RD_007a (自适应)"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 24 -e 24 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_007b | 24B 小消息 | 强制新算法（token 生效性以 plog selected 为准）
export HCCL_ALGO="reduce:sequence{nhraicpureduce,nhr}"
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_007b
echo "NPU_RD_007b (新算法)"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 24 -e 24 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_007c | 24B 小消息 | 强制 Sole
export HCCL_ALGO="reduce:sole{nhraicpureduce}"
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_007c
echo "NPU_RD_007c (Sole)"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 24 -e 24 -f 2 -d int64 -o sum -p 8 -a aicpu_ts

unset HCCL_ALGO

# NPU_RD_008 | STC NPU_RD_008 | 4G 大包多 loop 压力（风险3缓解验证）
export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/NPU_RD_008
echo "NPU_RD_008"
mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b 4g -e 4g -f 2 -d int64 -o sum -p 8 -a aicpu_ts

# NPU_RD_009a | STC NPU_RD_009 | 指定新算法（token 按算法名段匹配，实测验证）
export HCCL_ALGO="reduce:sequence{nhraicpureduce,nhr}"
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

# ============================== 临界点验证实验区（NPU_RD_TH_*） ==============================
# 背景: ST 仿真(2026-10-28, aarch64 实验服务器)测得两级 16rank × INT64+SUM 场景 SelectorEngine
#       cost 排序的 SoleMesh→新算法翻转临界点: count*=13549(INT64 元素)=每 rank 108392B≈105.85KB;
#       count<=13548 选 SoleMesh, >=13549 选新算法(黑盒二分+解析互验, 偏差<0.1%)。
# 本区实验(对应 STC 待补的临界点验证行, 编号 NPU_RD_TH_*):
#   实验一 NPU_RD_TH_S_* : 自适应选路(auto)邻域扫描 → 实测真机【选路翻转点】, 对照 ST 的 108392B
#   实验二 NPU_RD_TH_P_* : 强制双算法计时 → 实测真机【性能交叉点】, 对照选路翻转点(评估 cost 模型校准)
# 判读: TH_S 中 winner 由 SoleMesh 变为 Sequence 的首个点位=真机选路临界; TH_P 中同点位
#       P_M(SoleMesh) 耗时 > P_N(新算法) 的最小点位=真机性能交叉。真机 cost 系数来自真实带宽查询,
#       与 ST 桩系数不同, 临界点相对 108392B 偏移属正常, 重点看量级; 若性能交叉与选路翻转偏差大,
#       说明 cost 系数需校准(记录两个点位反馈调参)。
# 仅跑本区: 将上方用例区 NPU_RD_001~011b 各段用 # 注释掉即可。

# 实验一: 自适应选路邻域扫描(8=远端SoleMesh校准, 67108864=64m远端新算法校准, 其余为临界邻域)
TH_SIZES_AUTO=(8 65536 98304 106496 108384 108392 108400 110592 131072 163840 196608 262144 67108864)

# 实验二: 双算法强制计时点位(每点跑 SoleMesh 与新算法各一次)
TH_SIZES_PERF=(65536 98304 108392 131072 262144 1048576 16777216)

th_run() { # $1=用例名 $2=每rank字节数 $3=HCCL_ALGO(空=自适应)
    mkdir -p $LOG_ROOT/$1
    export ASCEND_PROCESS_LOG_PATH=$LOG_ROOT/$1
    echo "$1 (bytesPerRank=$2, HCCL_ALGO=${3:-auto})"
    if [ -n "$3" ]; then
        export HCCL_ALGO="$3"
    else
        unset HCCL_ALGO
    fi
    mpirun -f /home/l00841484/hostfile -n 16 ./bin/reduce_test -b $2 -e $2 -f 2 -d int64 -o sum -p 8 -a aicpu_ts \
        2>&1 | tee $LOG_ROOT/$1/stdout.log | tail -3
    unset HCCL_ALGO
}

for sz in "${TH_SIZES_AUTO[@]}"; do
    th_run "NPU_RD_TH_S_${sz}" "$sz" ""
done

for sz in "${TH_SIZES_PERF[@]}"; do
    th_run "NPU_RD_TH_P_M_${sz}" "$sz" "reduce:sole{mesh}"
    th_run "NPU_RD_TH_P_N_${sz}" "$sz" "reduce:sequence{nhraicpureduce,nhr}"
done

# 汇总: 每点选中算法(取 plog 首条 selected 日志) + 耗时(stdout 尾部提取, best-effort 需人工复核)
SUMMARY=$LOG_ROOT/threshold_summary.tsv
echo -e "case\tbytesPerRank\tselectedAlgo\telapsedHint" > $SUMMARY
for d in $LOG_ROOT/NPU_RD_TH_*; do
    [ -d "$d" ] || continue
    name=$(basename $d)
    sz=${name##*_}
    alg=$(grep -rh "selected algo type is" $d 2>/dev/null | head -1 | awk '{print $NF}')
    el=$(grep -oE "[0-9]+\.[0-9]+(us|ms|s)\b" $d/stdout.log 2>/dev/null | tail -1)
    echo -e "$name\t$sz\t$alg\t$el" >> $SUMMARY
done
echo "===== 临界点实验汇总 ====="
column -t $SUMMARY 2>/dev/null || cat $SUMMARY
echo "汇总文件: $SUMMARY"

echo "全部日志目录: $LOG_ROOT"
