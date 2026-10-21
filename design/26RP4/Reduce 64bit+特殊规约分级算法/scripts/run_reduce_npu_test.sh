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
# Reduce 64bit/PROD 两级分级算法（Mesh+NHR）NPU 真机用例一键执行
# 对应 STC 的 NPU_RD_001~011（参考 RS 侧实测脚本结构：每组用例独立 plog 目录 + echo 用例编号）
#
# 用法：
#   bash run_reduce_npu_test.sh [-c <CANN路径>] [-f <hostfile>] [-r <ranktable>] [-i <网卡>] [-l <log根目录>] [-g <用例组>] [-p] [-m]
# 选项：
#   -c, --cann     CANN 路径（默认 /home/l00841484/Ascend/cann）
#   -f, --hostfile hostfile 路径（默认 /home/l00841484/hostfile，16 进程 = 2 节点×8）
#   -r, --ranktable RANK_TABLE_FILE 路径（默认 /mnt/hccl/ranktable.json）
#   -i, --ifname   HCCL_SOCKET_IFNAME（默认 enp35s0f2,eth5）
#   -l, --logdir   plog 根目录（默认 ./npu_logs/<时间戳>）
#   -g, --group    用例组：all|base|dtype|boundary|algo|regress（默认 all）
#                    base=001,002  dtype=003~005  boundary=006~008  algo=009,010  regress=011
#   -p, --prof     开启 HCCL_TEST_PROFILING 采集（输出到 <logdir>/prof/）
#   -m, --make     强制重新编译 hccl_test 工具
# 输出：控制台 + <logdir>/summary.log
# -----------------------------------------------------------------------------------------------------------
set -uo pipefail

CANN="${CANN:-/home/l00841484/Ascend/cann}"
HOSTFILE="${HOSTFILE:-/home/l00841484/hostfile}"
RANKTABLE="${RANKTABLE:-/mnt/hccl/ranktable.json}"
IFNAME="${IFNAME:-enp35s0f2,eth5}"
MPIRUN=/usr/local/mpi/mpich/bin/mpirun
GROUP="all" PROF=0 REMAKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cann) CANN="$2"; shift 2 ;;
        -f|--hostfile) HOSTFILE="$2"; shift 2 ;;
        -r|--ranktable) RANKTABLE="$2"; shift 2 ;;
        -i|--ifname) IFNAME="$2"; shift 2 ;;
        -l|--logdir) LOGROOT="$2"; shift 2 ;;
        -g|--group) GROUP="$2"; shift 2 ;;
        -p|--prof) PROF=1; shift ;;
        -m|--make) REMAKE=1; shift ;;
        -h|--help) grep -E "^#   " "$0" | sed 's/^#//'; exit 0 ;;
        *) echo "未知参数: $1（-h 查看用法）"; exit 1 ;;
    esac
done

[[ -d "$CANN" ]] || { echo "错误: CANN 路径不存在: $CANN"; exit 1; }
[[ -f "$HOSTFILE" ]] || { echo "错误: hostfile 不存在: $HOSTFILE"; exit 1; }
[[ -f "$RANKTABLE" ]] || { echo "错误: ranktable 不存在: $RANKTABLE"; exit 1; }

LOGROOT="${LOGROOT:-$(pwd)/npu_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

# ---------- 1. 环境准备 ----------
# shellcheck disable=SC1091
source "$CANN/set_env.sh" >/dev/null 2>&1
export PATH=/usr/local/mpi/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpi/mpich/lib/:"$CANN"/lib64:${LD_LIBRARY_PATH:-}
export RANK_TABLE_FILE="$RANKTABLE"
export ASCEND_GLOBAL_LOG_LEVEL=1
export HCCL_SOCKET_IFNAME="$IFNAME"
if [[ "$PROF" == 1 ]]; then
    export HCCL_TEST_PROFILING=1
    export HCCL_TEST_PROFILING_PATH="$LOGROOT/prof/"
    mkdir -p "$LOGROOT/prof"
fi
log "CANN: $CANN | hostfile: $HOSTFILE | group: $GROUP | prof: $PROF"

# ---------- 2. 编译 hccl_test（幂等，-m 强制） ----------
TEST_DIR="$CANN/tools/hccl_test"
cd "$TEST_DIR" || { echo "错误: hccl_test 目录不存在: $TEST_DIR"; exit 1; }
if [[ "$REMAKE" == 1 || ! -x ./bin/reduce_test ]]; then
    log "编译 hccl_test ..."
    make MPI_HOME=/usr/local/mpi/mpich/ ASCEND_DIR="$CANN/" >/dev/null 2>&1 \
        || { echo "错误: hccl_test 编译失败"; exit 1; }
fi
[[ -x ./bin/reduce_test ]] || { echo "错误: ./bin/reduce_test 不存在"; exit 1; }

# ---------- 3. 用例表：编号|组|HCCL_ALGO(空=自适应)|np|数据量|dtype|规约 ----------
CASES=(
    "NPU_RD_001|base||16|64m|int64|sum"
    "NPU_RD_002|base||16|64m|fp64|sum"
    "NPU_RD_003|dtype||16|64m|uint64|max"
    "NPU_RD_004|dtype||16|64m|fp64|min"
    "NPU_RD_005|dtype||16|64m|int32|prod"
    "NPU_RD_006|boundary||16|8|int64|sum"
    "NPU_RD_007|boundary||16|24|int64|sum"
    "NPU_RD_008|boundary||16|4g|int64|sum"
    "NPU_RD_009a|algo|reduce:sequence{mesh,nhraicpureduce}|16|64m|int64|sum"
    "NPU_RD_009b|algo|reduce:sole{nhraicpureduce}|16|64m|int64|sum"
    "NPU_RD_010|algo||16|64m|int64|sum"
    "NPU_RD_011a|regress||16|64m|int32|min"
    "NPU_RD_011b|regress||8|64m|int64|sum"
)

# ---------- 4. 执行循环 ----------
run_case() {
    local id="$1" grp="$2" algo="$3" np="$4" sz="$5" dt="$6" op="$7"
    local logdir="$LOGROOT/${id}"
    mkdir -p "$logdir"
    if [[ -n "$algo" ]]; then
        export HCCL_ALGO="$algo"
    else
        unset HCCL_ALGO
    fi
    export ASCEND_PROCESS_LOG_PATH="$logdir"
    log "==== $id START (np=$np -b $sz -d $dt -o $op${algo:+ HCCL_ALGO=$algo})"
    "$MPIRUN" -f "$HOSTFILE" -n "$np" ./bin/reduce_test \
        -b "$sz" -e "$sz" -f 2 -d "$dt" -o "$op" -p 8 -a aicpu_ts \
        > "$logdir/console.log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]] && ! grep -qiE "error|failed" "$logdir/console.log"; then
        log "==== $id PASS (console: $logdir/console.log)"
        echo "$id PASS" >> "$SUMMARY.result"
    else
        log "==== $id FAIL rc=$rc（排查: $logdir/console.log 与 $logdir/plog/）"
        echo "$id FAIL rc=$rc" >> "$SUMMARY.result"
    fi
    unset HCCL_ALGO
}

: > "$SUMMARY.result"
for case in "${CASES[@]}"; do
    IFS='|' read -r id grp algo np sz dt op <<< "$case"
    [[ "$GROUP" != "all" && "$grp" != "$GROUP" ]] && continue
    run_case "$id" "$grp" "$algo" "$np" "$sz" "$dt" "$op"
done

# ---------- 5. 汇总 ----------
log "========== 汇总 =========="
PASS_N=$(grep -c " PASS" "$SUMMARY.result" || true)
FAIL_N=$(grep -c " FAIL" "$SUMMARY.result" || true)
log "PASS: $PASS_N  FAIL: $FAIL_N"
cat "$SUMMARY.result" | tee -a "$SUMMARY"
log "完整日志目录: $LOGROOT"
[[ "$FAIL_N" == 0 ]] || exit 1
