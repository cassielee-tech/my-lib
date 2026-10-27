#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
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
# Reduce 64bit/PROD 两级分级算法 ST（仓内仿真）用例脚本
# 对应 STC 的 ST_RD_2L_001~015（echo 编号与 STC 一致，a=自适应 / b=指定，跑完直接回填执行情况列）
# 用例代码已随 PR#3650 提交：test/st/algorithm/testcase/reduce_aicpu_testcase.cc
# 不想跑的用例整段用 # 注释掉即可（010 的 16M 用例仿真耗时最长，建议先注释最后单跑）
#
# CI 现状：run#4167 首条 auto 用例段错误（0.91s 无堆栈）——本脚本环境准备段含一键构建部署
# libhccl.so，跑首条时自动抓崩溃现场（INFO 日志尾部），用于定位崩溃点
# -----------------------------------------------------------------------------------------------------------

# ============================== 环境准备 ==============================
# 路径按实际环境修改
REPO=/home/l00841484/hccl_st
Ascend_dir="/home/l00841484/Ascend"

# CANN 环境加载（ST 的 CMake 读 ASCEND_HOME_PATH 定位头文件与 libhccl.so）
source $Ascend_dir/cann/set_env.sh

# libhccl.so 部署自检：ST 用例链接的是安装目录的库，官方 run 包不含新算法，必须先部署 PR 分支版本
INSTALLED_LIB=$(find $Ascend_dir -name libhccl.so -path "*lib64*" | head -1)
if strings $INSTALLED_LIB | grep -q AicpuReduceSequenceNHRAicpuReduceNHR; then
    echo "[自检] libhccl.so 已含新算法: $INSTALLED_LIB"
else
    echo "[自检] libhccl.so 未含新算法，开始构建部署（约 17-25 分钟，仅首次）"
    cd $REPO
    bash build.sh --pkg
    NEW_LIB=$(find $REPO/build_out -name libhccl.so | head -1)
    cp $INSTALLED_LIB ${INSTALLED_LIB}.bak
    cp $NEW_LIB $INSTALLED_LIB
    echo "[部署] $NEW_LIB -> $INSTALLED_LIB（原库备份为 .bak）"
fi

# 进入 ST 目录构建（ST_TASKS=reduce_aicpu 只编本需求用例目标，首次较慢，重复执行幂等）
cd $REPO/test/st/algorithm
ST_TASKS=reduce_aicpu bash build.sh

# ST 二进制与桩库路径（对齐主 build.sh run_st 的桩库四段，均指 build 目录）
ST_BIN=$REPO/test/st/algorithm/build/testcase/st_reduce_aicpu_test
export LD_LIBRARY_PATH=$REPO/test/st/algorithm/build/utils/src:$REPO/test/st/algorithm/build/utils/src/hccl_verifier:$REPO/test/st/algorithm/build/utils/src/hccl_depends_stub:$REPO/test/st/algorithm/build/utils/src/aicpu:$LD_LIBRARY_PATH

# INFO 级日志：stdout 含 [SelectorEngine] selected algo type 选路日志 + 各模板执行日志（定位崩溃点用）
export ASCEND_GLOBAL_LOG_LEVEL=1

# 日志根目录（时间戳隔离，每个用例单独日志文件，多次运行不互相覆盖）
LOG_ROOT=$(pwd)/st_logs/$(date +%Y%m%d_%H%M%S)
mkdir -p $LOG_ROOT

# ============================== 崩溃现场抓取（CI 复现定位，确认后可删此段） ==============================
echo "========== 首条 auto 用例单独跑（CI run#4167 段错误复现） =========="
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_auto_test' 2>&1 | tee $LOG_ROOT/CI_crash_repro.txt
CRASH_RET=$?
echo "[首条退出码: $CRASH_RET]"
if [ $CRASH_RET -ne 0 ]; then
    echo "========== 崩溃现场：最后 60 行日志 =========="
    tail -60 $LOG_ROOT/CI_crash_repro.txt
    echo "========== 选路关键行 =========="
    grep -E "SelectorEngine|SelectMinCost|selected algo|OrchestrateLoop|ReduceMesh1D|ReduceSoleExecutor|CalcCostCoeff" $LOG_ROOT/CI_crash_repro.txt | tail -30
    echo "========== 完整日志: $LOG_ROOT/CI_crash_repro.txt —— 请把以上输出发给定位 =========="
    exit $CRASH_RET
fi

# ============================== 用例区 ==============================
# 说明：spec（指定）模式由用例内 setenv HCCL_ALGO="reduce:sequence{nhraicpureduce,nhr}" 实现，
#       auto（自适应）模式选路按 cost 排序，SoleMesh 胜出为合法结果（STC 已按此口径修订）

# ST_RD_2L_001a | 自适应：int64+sum 基线（候选池含 SoleMesh/Sole/新算法，选 cost 最小者）
echo "ST_RD_2L_001a (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_001a.txt

# ST_RD_2L_001b | 指定新算法：int64+sum 执行正确性（代表组合1/3，selected 应为新算法）
echo "ST_RD_2L_001b (指定)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_spec_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_001b.txt

# ST_RD_2L_002 | 自适应：uint64+sum
echo "ST_RD_2L_002 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_uint64_sum_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_002.txt

# ST_RD_2L_003 | 自适应：fp64+sum
echo "ST_RD_2L_003 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_fp64_sum_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_003.txt

# ST_RD_2L_004 | 自适应：int64+prod（PROD 在线规约）
echo "ST_RD_2L_004 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_prod_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_004.txt

# ST_RD_2L_005a | 自适应：int64+max
echo "ST_RD_2L_005a (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_max_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_005a.txt

# ST_RD_2L_005b | 指定新算法：int64+max（代表组合2/3，64bit×非SUM 规约）
echo "ST_RD_2L_005b (指定)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_max_spec_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_005b.txt

# ST_RD_2L_006 | 自适应：int64+min（与 MAX 同族）
echo "ST_RD_2L_006 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_min_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_006.txt

# ST_RD_2L_007a | 自适应：fp16+prod（PROD 分支×非64bit dtype）
echo "ST_RD_2L_007a (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_fp16_prod_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_007a.txt

# ST_RD_2L_007b | 指定新算法：fp16+prod（代表组合3/3）
echo "ST_RD_2L_007b (指定)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_fp16_prod_spec_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_007b.txt

# ST_RD_2L_008 | 自适应：fp32+prod
echo "ST_RD_2L_008 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_fp32_prod_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_008.txt

# ST_RD_2L_009a | 自适应：12rank+recvCount=243 非整除尾块
echo "ST_RD_2L_009a (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_12rank_fp64_sum_tail_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_009a.txt

# ST_RD_2L_009b | 指定新算法：非整除尾块两级切分（tailSize 语义）不越界
echo "ST_RD_2L_009b (指定)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_12rank_fp64_sum_tail_spec_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_009b.txt

# ST_RD_2L_010a | 自适应：16M 多 loop 分轮（仿真耗时最长，可先注释最后单跑）
echo "ST_RD_2L_010a (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_16m_multiloop_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_010a.txt

# ST_RD_2L_010b | 指定新算法：多 loop 分轮与逐 loop 断批同步（仅 RS-L1 NHRPcie 段）
echo "ST_RD_2L_010b (指定)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_16m_multiloop_spec_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_010b.txt

# ST_RD_2L_011a | 自适应回归：root=13（非0）
echo "ST_RD_2L_011a (自适应回归)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_root13_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_011a.txt

# ST_RD_2L_011b | 指定新算法（主）：root=13 时 AG-L0 仅 root 写回（isRootRank_ 语义）
echo "ST_RD_2L_011b (指定,主)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_root13_spec_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_011b.txt

# ST_RD_2L_012 | 自适应：三级 32rank 维持选 Sole（新算法 attrs 两级门禁被拒）
echo "ST_RD_2L_012 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_3level_32rank_int64_sum_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_012.txt

# ST_RD_2L_013 | 自适应：单机 8rank 维持单机算法（8KB<8MB 阈值选 SoleMesh）
echo "ST_RD_2L_013 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_1level_8rank_int64_sum_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_013.txt

# ST_RD_2L_014 | 自适应：fp32+max 门禁负向（非64bit非PROD，新算法不进候选池）
echo "ST_RD_2L_014 (自适应)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_fp32_max_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_014.txt

# ST_RD_2L_015a | 自适应回归：count=1 单元素（8B 小消息，SoleMesh 胜出合法）
echo "ST_RD_2L_015a (自适应回归)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_count1_auto_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_015a.txt

# ST_RD_2L_015b | 指定新算法（主）：count=1 切片退化（sliceSize/tailSize 最小粒度）不越界
echo "ST_RD_2L_015b (指定,主)"
$ST_BIN --gtest_filter='st_reduce_aicpu_2level_16rank_int64_sum_count1_spec_test' 2>&1 | tee $LOG_ROOT/ST_RD_2L_015b.txt

# ============================== 结果汇总 ==============================
grep -H "\[  PASSED  \]\|\[  FAILED  \]" $LOG_ROOT/*.txt | sort
echo "全部日志目录: $LOG_ROOT"
