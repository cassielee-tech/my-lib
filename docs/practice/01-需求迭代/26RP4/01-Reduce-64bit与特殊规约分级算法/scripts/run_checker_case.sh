#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 单条 Checker（L2 仿真）用例一键执行：hccl-vm 起环境 → mock-comm → mpirun reduce_test → @checker → 关键字复核
# 配套 STC 的 CHK_RD_* 用例（28 条），每条 STC 行的参数按下表传入：
#   topo_meta→-m, np→-n, dataType→-d, 数据量→-s, 规约→-o, root→-r
#
# 用法：
#   bash run_checker_case.sh -r <hccl仓路径> -m 148 -n 32 -d int64 -s 8M -o sum -r 0 [选项]
# 选项：
#   -r, --repo     hccl 仓库路径（必填，用于 hccl-vm 配置定位与日志检索起点）
#   -m, --meta     mock-comm 拓扑名（如 148；特殊 yaml 用全名，如 228_asym_16_8）
#   -n, --np       mpirun 进程数
#   -d, --dtype    int64|uint64|fp64|int32|fp16|fp32|int8
#   -s, --size     单卡数据量（如 4K/8M/64M/1G）
#   -o, --op       sum|prod|max|min
#   -r, --root     root rank（默认 0）
#   -c, --cluster  hccl-vm 集群 yaml（默认 ascend950_cluster_32_server_normal.yaml，normal 集群）
#   -p, --bindir   hccl_test bin 目录（默认自动探测 /home/workspace/Ascend 下 reduce_test）
#   -k, --kw       运行后日志复核关键字（可多次；默认带算法选中与编排两组）
#       --no-checker  跳过 hccl-vm plugin run @checker（仅跑算子）
# 输出：控制台 + ./checker_logs/<时间戳>/
# -----------------------------------------------------------------------------------------------------------
set -uo pipefail

REPO="" META="" NP=32 DTYPE=int64 SIZE=8M OP=sum ROOT=0
CLUSTER="ascend950_cluster_32_server_normal.yaml" BIN_DIR="" RUN_CHECKER=1
KEYWORDS=("[ReduceAutoSelector]" "[ReduceSequenceExecutorAicpu3Level][OrchestrateLoop]")

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--repo) REPO="$2"; shift 2 ;;
        -m|--meta) META="$2"; shift 2 ;;
        -n|--np) NP="$2"; shift 2 ;;
        -d|--dtype) DTYPE="$2"; shift 2 ;;
        -s|--size) SIZE="$2"; shift 2 ;;
        -o|--op) OP="$2"; shift 2 ;;
        -R|--root) ROOT="$2"; shift 2 ;;
        -c|--cluster) CLUSTER="$2"; shift 2 ;;
        -p|--bindir) BIN_DIR="$2"; shift 2 ;;
        -k|--kw) KEYWORDS+=("$2"); shift 2 ;;
        --no-checker) RUN_CHECKER=0; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done
[[ -z "${REPO}" || -z "${META}" ]] && { echo "用法: bash $0 -r <hccl仓> -m <topo> -n <np> -d <dtype> -s <size> -o <op> [-R root]"; exit 1; }
REPO=$(cd "${REPO}" && pwd)

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$(pwd)/checker_logs/${META}_${DTYPE}_${SIZE}_${OP}_root${ROOT}_${TS}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
command -v hccl-vm >/dev/null 2>&1 || { echo "错误: 找不到 hccl-vm 命令（需在 CheckerL2 环境会话内执行）"; exit 1; }

# hccl_test 二进制定位
REDUCE_TEST=""
if [[ -n "${BIN_DIR}" ]]; then
    REDUCE_TEST="${BIN_DIR}/reduce_test"
else
    REDUCE_TEST=$(find /home/workspace/Ascend -type f -name reduce_test -path "*hccl_test*" 2>/dev/null | sort -V | tail -1)
fi
[[ -z "${REDUCE_TEST}" || ! -x "${REDUCE_TEST}" ]] && { echo "错误: 未找到 reduce_test（用 -p 指定 hccl_test bin 目录）"; exit 1; }
log "reduce_test: ${REDUCE_TEST}"

# ---------- 1. 起集群 ----------
log "hccl-vm start ${CLUSTER}"
hccl-vm start "${CLUSTER}" 2>&1 | tee "${LOG_DIR}/vm_start.log"

# ---------- 2. mock 拓扑 ----------
log "hccl-vm mock-comm ${META}"
hccl-vm mock-comm "${META}" 2>&1 | tee "${LOG_DIR}/mock_comm.log"

# ---------- 3. mpirun 执行算子 ----------
CMD="mpirun --allow-run-as-root --oversubscribe -np ${NP} ${REDUCE_TEST} -d ${DTYPE} -b ${SIZE} -e ${SIZE} -o ${OP} -r ${ROOT} -f 2 -n 1 -w 0 -c 0"
log "执行: ${CMD}"
${CMD} < /dev/null >"${LOG_DIR}/mpirun.log" 2>&1
MP_RET=$?
tail -5 "${LOG_DIR}/mpirun.log"
if [[ ${MP_RET} -ne 0 ]]; then
    log "reduce_test 退出码 ${MP_RET}，详见 ${LOG_DIR}/mpirun.log"
fi

# ---------- 4. Checker 校验 ----------
if [[ ${RUN_CHECKER} -eq 1 ]]; then
    log "hccl-vm plugin run @checker"
    hccl-vm plugin run @checker 2>&1 | tee "${LOG_DIR}/checker.log"
    grep -E "CHECKER_RUN_SUMMARY|Checker All|ErrorCode" "${LOG_DIR}/checker.log" | tail -5 || true
fi

# ---------- 5. 关键字复核 ----------
log "日志关键字复核: ${KEYWORDS[*]}"
ARCHIVE_DIR=$(ls -dt "${REPO}"/build*/plog* "${REPO}"/plog* ./plog* 2>/dev/null | head -1)
if [[ -n "${ARCHIVE_DIR}" ]]; then
    : >"${LOG_DIR}/kw_hits.txt"
    for kw in "${KEYWORDS[@]}"; do
        echo "== ${kw}" >>"${LOG_DIR}/kw_hits.txt"
        grep -rh "${kw}" "${ARCHIVE_DIR}" 2>/dev/null | sort -u | head -20 >>"${LOG_DIR}/kw_hits.txt"
    done
    log "命中摘要: ${LOG_DIR}/kw_hits.txt"
else
    log "未找到 plog 归档目录，跳过关键字复核（可手动 grep）"
fi

# ---------- 6. 收尾 ----------
hccl-vm exit 2>&1 | tee -a "${LOG_DIR}/vm_start.log" || true
log "完成。日志目录: ${LOG_DIR}"
