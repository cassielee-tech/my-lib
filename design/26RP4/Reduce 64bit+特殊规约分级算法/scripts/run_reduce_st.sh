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
# 配套 STC 的 ST_RD_2L_001~015（用例已随 PR#3650 提交 817dc2d2；
# gtest 套件名 ST_REDUCE_AICPU_TEST，二进制 st_reduce_aicpu_test，auto/spec 双模式共 22 条）
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

# ---------- 0. 安装目录 libhccl.so 部署自检（ST 链接安装目录的库，官方 run 包不含新算法） ----------
INSTALLED_LIB="${ASCEND_HOME_PATH}/x86_64-linux/lib64/libhccl.so"
if [[ -f "${INSTALLED_LIB}" ]] && strings "${INSTALLED_LIB}" | grep -q "AicpuReduceSequenceNHRAicpuReduceNHR"; then
    log "libhccl.so 部署自检通过（含新算法注册: ${INSTALLED_LIB}）"
else
    fail "安装目录 libhccl.so 未包含新算法（${INSTALLED_LIB}）。请先在仓内执行:
  source ${CANN}/set_env.sh && bash build.sh --pkg
  然后将产物（find build_out -name libhccl.so）备份并替换 ${INSTALLED_LIB} 后重跑本脚本"
fi

# ---------- 1. 构建 ST 框架（BUILD_OPEN_PROJECT + ST_TASKS=reduce_aicpu） ----------
ST_DIR="${REPO}/test/st/algorithm"
[[ ${CLEAN} -eq 1 ]] && rm -rf "${ST_DIR}/build"
log "构建 ST 框架（ST_TASKS=reduce_aicpu，首次较慢）"
(cd "${ST_DIR}" && ST_TASKS=reduce_aicpu bash build.sh) >"${LOG_DIR}/st_build.log" 2>&1 \
    || { tail -30 "${LOG_DIR}/st_build.log"; fail "ST 构建失败，详见 ${LOG_DIR}/st_build.log"; }

# ---------- 2. 定位 reduce_aicpu ST 二进制 ----------
ST_BIN=$(find "${ST_DIR}/build" -type f -name "st_reduce_aicpu_test" -executable 2>/dev/null | head -1)
[[ -z "${ST_BIN}" ]] && { find "${ST_DIR}/build" -type f -executable | head -20; fail "未找到 st_reduce_aicpu_test 二进制（817dc2d2 是否已合入该分支？）"; }
log "ST 二进制: ${ST_BIN}"

# ---------- 3. 运行（LD_LIBRARY_PATH 对齐主 build.sh run_st 的桩库四段 + 安装目录 lib64） ----------
export LD_LIBRARY_PATH="${ST_DIR}/build/utils/src:${ST_DIR}/build/utils/src/hccl_verifier:${ST_DIR}/build/utils/src/hccl_depends_stub:${ST_DIR}/build/utils/src/aicpu:${ASCEND_HOME_PATH}/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_GLOBAL_LOG_LEVEL=1
log "运行 gtest_filter='${FILTER}'"
"${ST_BIN}" --gtest_filter="${FILTER}" >"${LOG_DIR}/st_run.txt" 2>&1
RET=$?
grep -E "\[==========\]|\[  PASSED  \]|\[  FAILED  \]" "${LOG_DIR}/st_run.txt" | tail -6
log "退出码 ${RET}，完整输出: ${LOG_DIR}/st_run.txt"

# ---------- 4. 选路结果复核（SelectorEngine cost 排序口径） ----------
grep -E "selected algo type|OrchestrateLoop|SelectMinCost" "${LOG_DIR}/st_run.txt" | sort -u | head -12 >"${LOG_DIR}/kw_hits.txt" || true
log "关键字命中: ${LOG_DIR}/kw_hits.txt"
log "完成。日志目录: ${LOG_DIR}"
exit ${RET}
