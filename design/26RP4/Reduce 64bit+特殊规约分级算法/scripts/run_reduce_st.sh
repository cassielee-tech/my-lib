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
# Reduce 64bit/PROD 两级分级 ST（仓内仿真）一键构建+运行
# 配套 STC 的 ST_RD_2L_001~015（注意：ST 用例代码尚未随 PR 提交，本脚本先就绪；
# 用例合入后 gtest 套件名为 ST_REDUCE_2LEVEL_HIER_TEST，二进制 st_reduce_aicpu_test）
#
# 用法：
#   bash run_reduce_st.sh -r <hccl仓路径> [-c <CANN路径>] [选项]
# 选项：
#   -r, --repo    hccl 仓库路径（必填）
#   -c, --cann    CANN toolkit 路径（默认 $ASCEND_HOME_PATH 或 /usr/local/Ascend/ascend-toolkit/latest）
#   -f, --filter  gtest_filter（默认跑全部；示例：-f 'ST_REDUCE_2LEVEL_HIER_TEST.two_level_int64_sum_baseline'）
#   -j, --jobs    并行编译数（默认 nproc）
#       --clean   清理 ST 构建目录重建
# 输出：控制台 + ./st_logs/<时间戳>/
# -----------------------------------------------------------------------------------------------------------
set -uo pipefail

REPO="" CANN="" FILTER="*" JOBS=$(nproc) CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--repo) REPO="$2"; shift 2 ;;
        -c|--cann) CANN="$2"; shift 2 ;;
        -f|--filter) FILTER="$2"; shift 2 ;;
        -j|--jobs) JOBS="$2"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done
[[ -z "${REPO}" ]] && { echo "错误: 必须用 -r 指定 hccl 仓库路径"; exit 1; }
REPO=$(cd "${REPO}" 2>/dev/null && pwd) || { echo "错误: 仓库路径不存在"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$(pwd)/st_logs/${TS}"
mkdir -p "${LOG_DIR}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "错误: $*"; exit 1; }

if [[ -z "${CANN}" ]]; then
    CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
fi
[[ -d "${CANN}" ]] || fail "CANN 路径不存在: ${CANN}"
# shellcheck disable=SC1091
source "${CANN}/set_env.sh" >/dev/null 2>&1
[[ -n "${ASCEND_HOME_PATH:-}" ]] || fail "source set_env.sh 后 ASCEND_HOME_PATH 为空"
log "CANN 就绪: ${ASCEND_HOME_PATH}"

cd "${REPO}"
log "代码基线: $(git rev-parse --short HEAD) @ $(git rev-parse --abbrev-ref HEAD)"

# ---------- 1. 构建 ST 框架（BUILD_OPEN_PROJECT + ST_TASKS=reduce） ----------
ST_DIR="${REPO}/test/st/algorithm"
[[ ${CLEAN} -eq 1 ]] && rm -rf "${ST_DIR}/build"
log "构建 ST 框架（ST_TASKS=reduce，首次较慢）"
(cd "${ST_DIR}" && ST_TASKS=reduce bash build.sh) >"${LOG_DIR}/st_build.log" 2>&1 \
    || { tail -30 "${LOG_DIR}/st_build.log"; fail "ST 构建失败，详见 ${LOG_DIR}/st_build.log"; }

# ---------- 2. 定位 reduce ST 二进制 ----------
ST_BIN=$(find "${ST_DIR}/build" -type f -name "st_reduce*test" -executable 2>/dev/null | head -1)
[[ -z "${ST_BIN}" ]] && { find "${ST_DIR}/build" -type f -executable | head -20; fail "未找到 st_reduce*test 二进制（ST 用例是否已合入？）"; }
log "ST 二进制: ${ST_BIN}"

# ---------- 3. 运行（LD_LIBRARY_PATH 对齐 ST 五段） ----------
export LD_LIBRARY_PATH="${ST_DIR}/utils/src:${ST_DIR}/utils/hccl_verifier:${ST_DIR}/utils/hccl_depends_stub:${ST_DIR}/utils/aicpu:${ASCEND_HOME_PATH}/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
log "运行 gtest_filter='${FILTER}'"
"${ST_BIN}" --gtest_filter="${FILTER}" >"${LOG_DIR}/st_run.txt" 2>&1
RET=$?
grep -E "\[==========\]|\[  PASSED  \]|\[  FAILED  \]" "${LOG_DIR}/st_run.txt" | tail -6
log "退出码 ${RET}，完整输出: ${LOG_DIR}/st_run.txt"

# ---------- 4. 算法选中复核 ----------
grep -E "Algo match|OrchestrateLoop" "${LOG_DIR}/st_run.txt" | sort -u | head -10 >"${LOG_DIR}/kw_hits.txt" || true
log "关键字命中: ${LOG_DIR}/kw_hits.txt"
log "完成。日志目录: ${LOG_DIR}"
exit ${RET}
