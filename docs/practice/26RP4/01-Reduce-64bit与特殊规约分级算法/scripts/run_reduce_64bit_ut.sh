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
# Reduce 64bit/PROD 两级分级算法 UT 一键构建+运行（实验环境 Linux + CANN toolkit）
# 配套：PR#3650 分支 feature/reduce-64bit-prod-2level-hier，STC 用例 UT_RD_2L_001~006
#
# 用法：
#   bash run_reduce_64bit_ut.sh -r <hccl仓路径> [-c <CANN安装路径>] [选项]
# 选项：
#   -r, --repo     hccl 仓库路径（必填）
#   -c, --cann     CANN toolkit 安装路径（默认 $ASCEND_HOME_PATH 或 /usr/local/Ascend/ascend-toolkit/latest）
#   -b, --branch   期望分支（默认 feature/reduce-64bit-prod-2level-hier，仅校验不强制切换）
#   -t, --target   all|selector|core（默认 all）
#   -d, --debug    调试模式：ENABLE_ASAN 构建（对齐 CI），崩溃时自动 gdb 栈回溯
#   -j, --jobs     并行编译数（默认 nproc）
#   -F, --fetch    先 git fetch origin <branch> 并 checkout（默认不拉取，直接用现场代码）
#       --clean    删除构建目录全新配置
# 输出：
#   控制台实时输出 + 日志落盘 ./ut_logs/<时间戳>/（selector_test.txt / core_test.txt /
#   core_test.asan.txt / core_test.gdb.txt / summary.txt）
# -----------------------------------------------------------------------------------------------------------
set -uo pipefail

BRANCH_DEFAULT="feature/reduce-64bit-prod-2level-hier"
REPO="" CANN="" BRANCH="${BRANCH_DEFAULT}" TARGET="all" DEBUG=0 JOBS=$(nproc) FETCH=0 CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--repo) REPO="$2"; shift 2 ;;
        -c|--cann) CANN="$2"; shift 2 ;;
        -b|--branch) BRANCH="$2"; shift 2 ;;
        -t|--target) TARGET="$2"; shift 2 ;;
        -d|--debug) DEBUG=1; shift ;;
        -j|--jobs) JOBS="$2"; shift 2 ;;
        -F|--fetch) FETCH=1; shift ;;
        --clean) CLEAN=1; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done
[[ -z "${REPO}" ]] && { echo "错误: 必须用 -r 指定 hccl 仓库路径"; exit 1; }
REPO=$(cd "${REPO}" 2>/dev/null && pwd) || { echo "错误: 仓库路径不存在 ${REPO}"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$(pwd)/ut_logs/${TS}"
mkdir -p "${LOG_DIR}"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "错误: $*"; exit 1; }

# ---------- 1. 环境与分支 ----------
if [[ -z "${CANN}" ]]; then
    CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
fi
[[ -d "${CANN}" ]] || fail "CANN 路径不存在: ${CANN}（用 -c 指定）"
# shellcheck disable=SC1091
source "${CANN}/set_env.sh" >/dev/null 2>&1
[[ -n "${ASCEND_HOME_PATH:-}" ]] || fail "source set_env.sh 后 ASCEND_HOME_PATH 仍为空，CANN 环境异常: ${CANN}"
log "CANN 就绪: ${ASCEND_HOME_PATH}"

cd "${REPO}"
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "${CUR_BRANCH}" != "${BRANCH}" ]]; then
    if [[ ${FETCH} -eq 1 ]]; then
        log "拉取并切换到 ${BRANCH}"
        git fetch origin "${BRANCH}" || fail "git fetch origin ${BRANCH} 失败"
        git checkout "${BRANCH}" || fail "git checkout ${BRANCH} 失败"
    else
        log "警告: 当前分支 ${CUR_BRANCH} != 期望 ${BRANCH}（继续执行；如需切换加 -F）"
    fi
fi
log "代码基线: $(git rev-parse --short HEAD) @ $(git rev-parse --abbrev-ref HEAD)"

# ---------- 2. 构建配置（对齐 build.sh 的 UT 参数 + 独立 build_ut 目录） ----------
BUILD_DIR="${REPO}/build_ut"
[[ ${CLEAN} -eq 1 ]] && { log "清理 ${BUILD_DIR}"; rm -rf "${BUILD_DIR}"; }
mkdir -p "${BUILD_DIR}"

CMAKE_ARGS="-DPRODUCT_SIDE=host \
-DCMAKE_BUILD_TYPE=Debug \
-DASCEND_INSTALL_PATH=${ASCEND_HOME_PATH} \
-DCANN_3RD_LIB_PATH=${REPO}/third_party \
-DENABLE_TEST=ON \
-DENABLE_UT=ON \
-DLLT_KILL_TIME=1200"
[[ ${DEBUG} -eq 1 ]] && CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]] || [[ ${CLEAN} -eq 1 ]]; then
    log "cmake 配置: ${CMAKE_ARGS}"
    cmake -S "${REPO}" -B "${BUILD_DIR}" ${CMAKE_ARGS} >"${LOG_DIR}/cmake.log" 2>&1 \
        || { tail -30 "${LOG_DIR}/cmake.log"; fail "cmake 配置失败，详见 ${LOG_DIR}/cmake.log"; }
fi

TARGETS=()
[[ "${TARGET}" == "all" || "${TARGET}" == "selector" ]] && TARGETS+=(reduce_64bit_hier_selector_test)
[[ "${TARGET}" == "all" || "${TARGET}" == "core" ]] && TARGETS+=(reduce_64bit_hier_core_test)
[[ ${#TARGETS[@]} -eq 0 ]] && fail "-t 仅支持 all|selector|core"

log "增量编译: ${TARGETS[*]} (-j${JOBS})$( [[ ${DEBUG} -eq 1 ]] && echo ' [ASAN]' )"
cmake --build "${BUILD_DIR}" --target ${TARGETS[*]} -j"${JOBS}" >"${LOG_DIR}/make.log" 2>&1
BUILD_RET=$?
if [[ ${BUILD_RET} -ne 0 ]]; then
    grep -E "error:|undefined reference" "${LOG_DIR}/make.log" | head -30
    fail "编译失败，详见 ${LOG_DIR}/make.log"
fi

# ---------- 3. 运行 ----------
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/x86_64-linux/lib64:${BUILD_DIR}/src/common/hcomm_dlsym:${LD_LIBRARY_PATH:-}"
UT_DIR="${BUILD_DIR}/test/ut/reduce_64bit_hier"
run_one() {
    local bin="$1" logf="$2"
    log "运行 $(basename "${bin}") ..."
    "${bin}" >"${logf}" 2>&1
    local ret=$?
    if [[ ${ret} -eq 0 ]]; then
        log "  => PASS"
    elif [[ ${ret} -gt 128 ]]; then
        log "  => 崩溃（信号 $((ret-128))），详见 ${logf}"
    else
        log "  => 失败（退出码 ${ret}），详见 ${logf}"
    fi
    return ${ret}
}

SUMMARY=("${LOG_DIR}/summary.txt")
{
    echo "时间: ${TS}"
    echo "代码: $(git rev-parse --short HEAD) @ ${CUR_BRANCH}"
    echo "CANN: ${ASCEND_HOME_PATH}"
    echo "模式: $([[ ${DEBUG} -eq 1 ]] && echo 'ASAN 调试' || echo '常规')"
} >"${SUMMARY}"

for t in "${TARGETS[@]}"; do
    BIN="${UT_DIR}/${t}"
    [[ -x "${BIN}" ]] || fail "找不到测试二进制: ${BIN}"
    run_one "${BIN}" "${LOG_DIR}/${t}.txt"
    RET=$?
    echo "${t}: $([[ ${RET} -eq 0 ]] && echo PASS || echo "RET=${RET}")" >>"${SUMMARY}"

    # 崩溃时取证：ASAN 报告重跑 + gdb 栈回溯（静态初始化期崩溃的定位关键）
    if [[ ${RET} -gt 128 ]]; then
        log "取证: ASAN 重跑 $(basename "${BIN}")"
        ASAN_OPTIONS="detect_leaks=0:abort_on_error=0:handle_segv=1:print_stacktrace=1" \
            "${BIN}" >"${LOG_DIR}/${t}.asan.txt" 2>&1 || true
        if command -v gdb >/dev/null 2>&1; then
            log "取证: gdb 栈回溯 $(basename "${BIN}")"
            gdb -batch \
                -ex "set pagination off" -ex "set confirm off" \
                -ex run -ex "bt full" -ex "info threads" \
                --args "${BIN}" >"${LOG_DIR}/${t}.gdb.txt" 2>&1 || true
            log "  gdb 输出: ${LOG_DIR}/${t}.gdb.txt（静态初始化崩溃会直接给出出错的构造函数帧）"
        else
            log "  环境无 gdb，跳过栈回溯（建议安装 gdb）"
        fi
    fi
done

log "全部完成。日志目录: ${LOG_DIR}"
cat "${SUMMARY}"
exit 0
