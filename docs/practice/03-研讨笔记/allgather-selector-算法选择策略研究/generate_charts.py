# -*- coding: utf-8 -*-
"""AllGather Selector 研讨图表生成脚本（cann/hccl, HEAD 7b0fa2d2）。

说明：
- 代价模型按 src/ops/op_common/selector/cost_table.cc + cost_model.cc 复刻；
- 只用于可视化研讨，不等同于真实上板测量；
- 默认假设：非 POD、单端口向量、两级拓扑 L0=8 卡/服务器。
"""
import bisect
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.sans-serif": ["Hiragino Sans GB", "Songti SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#9AA3AE",
    "axes.labelcolor": "#1F2937",
    "text.color": "#1F2937",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8,
})

C = {
    "mesh": "#2E5EAA",
    "nhr": "#E07B39",
    "par": "#7C3AED",
    "ok": "#2A9D8F",
    "warn": "#C0392B",
    "gray": "#6B7280",
    "ink": "#1F2937",
    "bg": "#F5F7FA",
    "lightblue": "#DCE7F5",
    "lightorange": "#FBE3D1",
    "lightpurple": "#EDE4FB",
    "lightgray": "#ECEFF3",
}

BW = 56e9
LOCAL = 1400e9
MIB = 1024 * 1024

CLOS_TAB = [(int(0.125 * MIB), 0.10388), (int(0.25 * MIB), 0.10388), (int(0.5 * MIB), 0.10388),
            (1 * MIB, 0.1855), (2 * MIB, 0.3), (4 * MIB, 0.4288), (8 * MIB, 0.5302), (16 * MIB, 0.568),
            (32 * MIB, 0.6549), (64 * MIB, 0.7184), (128 * MIB, 0.7408), (256 * MIB, 0.7644)]
MESH_TAB = [(1 * MIB, 0.7135), (2 * MIB, 0.7758), (4 * MIB, 0.8112), (8 * MIB, 0.8301), (16 * MIB, 0.84),
            (32 * MIB, 0.8449), (64 * MIB, 0.8475), (128 * MIB, 0.8487), (256 * MIB, 0.8494)]


def ub_util(net, per_transfer, op="AG"):
    tab = CLOS_TAB if net == "CLOS" else MESH_TAB
    ds = per_transfer
    if op == "AG" and net == "CLOS" and ds < MIB:
        ds = MIB
    idx = bisect.bisect_right([b for b, _ in tab], ds)
    return tab[-1][1] if idx == len(tab) else tab[idx][1]


def nhr_a(rank, p):
    return (rank - 1) / (p * BW)


def mesh_a(n, net, rank, p):
    return n / BW if net == "MESH" else n * (rank - 1) / (p * BW)


def nhr_params(rank, net, p, dr=1.0):
    log2r = int(math.floor(math.log2(rank)))
    kernel = max(8 * log2r - 7, 10)
    reff = max(int(5 * log2r / 3), 2)
    task = 5 * (reff - 1) + 2 * (reff - 1) * 2
    return dict(A=nhr_a(rank, p), B=1.0 / LOCAL, C=2e-6 * kernel,
                D=1.2e-6 * task, dr=dr, net=net, rs=rank, at="NHR")


def mesh_params(rank, net, p, n, in_to_out=True):
    task = 5 * (rank - 1) + 2 * (rank - 1) * 2
    b = 1.0 / LOCAL + (1.0 / LOCAL if in_to_out else 0.0)
    return dict(A=mesh_a(n, net, rank, p), B=b, C=2e-6 * 8,
                D=0.6e-6 * task, dr=n, net=net, rs=rank, at="MESH")


def quantize(ratio):
    idx = math.floor(ratio / (1.0 / 8.0) + 0.5)
    return max(1, min(7, idx)) / 8.0


def alpha_of(l0, l1, p1):
    return quantize(l1 / (l1 + l0 * (l1 - 1) / p1))


def seg_cost(seg, ds, op="AG"):
    dr = seg["dr"]
    if seg["at"] == "NHR":
        dr *= seg["rs"] / 2.0
    per = int(np.float32(ds) * np.float32(dr))
    if per == 0:
        per = ds
    u = ub_util(seg["net"], per, op)
    return (seg["A"] / u + seg["B"]) * ds + seg["C"], u


def alg_cost(segs, groups, intra_max, ds, op="AG"):
    total_d = sum(s["D"] for s in segs)
    idx, total = 0, 0.0
    for g in groups:
        vals = []
        for _ in range(g):
            c, _ = seg_cost(segs[idx], ds, op)
            vals.append(c)
            idx += 1
        total += max(vals) if intra_max else sum(vals)
    return max(total, total_d), total_d


def sole_nhr(rank, p):
    return [nhr_params(rank, "CLOS", p)], [1], False


def parallel(rank, p1, l0=8):
    l1 = rank // l0
    a = alpha_of(l0, l1, p1)
    b = 1 - a
    segs = [mesh_params(l0, "MESH", 1, a, True),
            nhr_params(l1, "CLOS", p1, b),
            mesh_params(l0, "MESH", 1, b * l1, False),
            nhr_params(l1, "CLOS", p1, a * l0)]
    b_const = 23e-6
    d_add = 1.5e-6 * (9 * rank + 32) / 32.0
    for s in segs:
        s["C"] += b_const
        if s["D"] > 0:
            s["D"] += d_add
    return segs, [2, 2], True


def costs(rank, p, ds):
    cn, _ = alg_cost(*sole_nhr(rank, p), ds)
    cp, _ = alg_cost(*parallel(rank, p), ds)
    return cn, cp


# --------------------------------------------------------------------------- #
# 通用绘图工具
# --------------------------------------------------------------------------- #
def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("saved", path)


def box(ax, x, y, w, h, text, fc=C["lightgray"], ec="#9AA3AE", fs=10, bold=False, color=None, rad=0.02):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0.004,rounding_size={rad}",
                       linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", color=color or C["ink"], linespacing=1.35)


def arrow(ax, p1, p2, color="#6B7280", style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, linewidth=lw,
                                 color=color, linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


def diamond(ax, x, y, hw, hh, text, fs=10.5, fc="white", ec="#6B7280"):
    ax.add_patch(mpatches.Polygon([[x - hw, y], [x, y + hh], [x + hw, y], [x, y - hh]],
                                  closed=True, fc=fc, ec=ec, lw=1.5, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.25)


def plain_axes(figsize=(14, 8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- #
# 01 选路总览
# --------------------------------------------------------------------------- #
def chart01():
    fig, ax = plain_axes((15, 9))
    ax.text(0.5, 0.965, "AllGather 算法选路总览：两条路径并存", ha="center", va="center",
            fontsize=20, fontweight="bold")
    ax.text(0.5, 0.925, "入口 op_common.cc:225    ReSelector:957", ha="center", va="center",
            fontsize=11, color=C["gray"])

    box(ax, 0.5, 0.885, 0.30, 0.05, "OpParam + TopoInfo", fc=C["lightblue"], fs=13, bold=True)
    arrow(ax, (0.5, 0.858), (0.5, 0.833))
    diamond(ax, 0.5, 0.795, 0.105, 0.042, "useNdaSelector ?", fs=10.5)

    # 是 → NDA
    arrow(ax, (0.395, 0.795), (0.275, 0.795))
    ax.text(0.335, 0.817, "是", fontsize=10, color=C["gray"])
    box(ax, 0.155, 0.795, 0.22, 0.05, "老路径 NDA 分支\nSelectNdaAlgo", fc=C["lightorange"], fs=10)

    # 否 → 第二层判断
    arrow(ax, (0.5, 0.753), (0.5, 0.727))
    ax.text(0.518, 0.740, "否", fontsize=10, color=C["gray"])
    diamond(ax, 0.5, 0.685, 0.175, 0.045, "level0Topo == MESH_1D  或  IsDevType960() ?", fs=10)

    arrow(ax, (0.325, 0.685), (0.205, 0.605))
    arrow(ax, (0.675, 0.685), (0.795, 0.605))
    ax.text(0.305, 0.708, "否", fontsize=10, color=C["gray"])
    ax.text(0.688, 0.708, "是", fontsize=10, color=C["gray"])

    # 老路径
    box(ax, 0.20, 0.575, 0.30, 0.055, "老路径  ExecuteSelector\n手写阈值决策树", fc=C["lightorange"],
        fs=12, bold=True)
    engines = ["CCU_MS", "CCU_SCHED", "AICPU", "AIV", "DPU"]
    algos = ["SoleMesh/2Die", "SoleMesh/Parallel/\nSequence/SoleNHR", "SoleMesh/NHR/\nParallel/Pipeline",
             "SoleMesh", "Sequence/Pipeline"]
    for i, (e, alg) in enumerate(zip(engines, algos)):
        x = 0.055 + i * 0.058
        box(ax, x, 0.47, 0.055, 0.035, e, fc="white", fs=7.5, bold=True)
        ax.text(x, 0.416, alg, ha="center", va="center", fontsize=6.2, color=C["gray"])
        arrow(ax, (0.20, 0.548), (x, 0.488), lw=1.0)
    ax.text(0.20, 0.355, "命中即返回，决策边界直接写在 if/else 里", ha="center",
            fontsize=9, color=C["gray"])

    # 新路径
    box(ax, 0.80, 0.575, 0.30, 0.055, "新路径  SelectorEngine::Run\n按引擎分层的 min-cost", fc=C["lightpurple"],
        fs=12, bold=True)
    ys = [0.49, 0.415, 0.34, 0.265]
    labels = ["① costModel 五步\n引擎/topo/op/HCCL_ALGO 过滤",
              "② costTable 三阶段\n逐候选算法算 cost",
              "③ tuner 可选改写 cost",
              "④ 最高优先级层内取 min cost\n平票取注册序第一"]
    for i, (y, t) in enumerate(zip(ys, labels)):
        box(ax, 0.80, y, 0.30, 0.052, t, fc="white", fs=9.5)
        if i:
            arrow(ax, (0.80, ys[i - 1] - 0.026), (0.80, y + 0.026))
    arrow(ax, (0.80, 0.548), (0.80, 0.516))

    ax.text(0.80, 0.175, "显式 if/else 退化为\nopCustomCheck / topoCustomCheck /\n"
                         "topoPriorityCheck / opPriorityCheck\n等“补丁叶子”",
            ha="center", va="center", fontsize=9, color=C["warn"],
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDF2F2", ec=C["warn"], lw=1.0))

    ax.text(0.5, 0.06, "共同约束：topoPrecondCheck 硬前提始终执行；preferred / tuner 加载时软策略让位",
            ha="center", fontsize=9, color=C["gray"])
    save(fig, "01_选路总览.png")


# --------------------------------------------------------------------------- #
# 02 老路径分支矩阵
# --------------------------------------------------------------------------- #
def chart02():
    rows = ["CCU_MS", "CCU_SCHED", "AICPU", "AIV", "DPU", "NDA"]
    cols = ["单级\nMESH_1D", "两级\nMESH_1D", "UBX\n(MESH_1D_CLOS\n非 PCIE)", "PCIE 混合", "CLOS"]
    cells = [
        ["SoleMesh2Die\n/ SoleMesh", "—\n(仅 2P≥4MB 回退)", "SoleMesh ≤512KB\n否则不匹配", "全连通→Mesh\n否则不匹配", "不支持"],
        ["SoleMesh\nSoleMeshConcur\n(960,>16MB)", "SoleMesh / Parallel /\nSequence / SoleNHR\n按量切换",
         "Concur / Parallel /\nPipeline / SoleNHR", "全连通→Mesh1D\n否则不匹配", "SoleNHR"],
        ["SoleMesh /\nSoleMeshConcur", "PipeLine / Parallel /\nSequence / SoleNHR",
         "Concur / Parallel /\nPipeline / SoleNHR", "SoleMesh / Parallel /\nPipeline", "SoleNHRMultiLink"],
        ["SoleMesh", "SoleMesh", "SoleMesh\n(rank∈(4,8] 大数据\n被排除)", "SoleMesh", "SoleMesh"],
        ["—", "SequenceMeshNHR", "PipelineMeshNHRNHR", "SequenceMeshNHR", "—"],
        ["—", "SequenceNHRNHRNDA", "Sequence 或\nPipeline(>4MB)", "SequenceNHRNHRNDA", "—"],
    ]
    fig, ax = plt.subplots(figsize=(15, 7.6))
    fig.subplots_adjust(top=0.80, bottom=0.14)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -1.15)
    ax.axis("off")
    ax.set_title("老路径 AllGatherAutoSelector：引擎 × 拓扑 分支矩阵", fontsize=18, pad=26,
                 fontweight="bold")
    fig.text(0.5, 0.055, "单元格为简化后的主导算法；精确条件见决策树文档第 4 节",
             ha="center", fontsize=9.5, color=C["gray"])

    def color(t):
        if t.startswith("—") or t == "不支持":
            return C["lightgray"]
        if "NHR" in t and ("Parallel" in t or "Pipeline" in t or "Sequence" in t or "Concur" in t):
            return C["lightpurple"]
        if "NHR" in t:
            return C["lightorange"]
        if "Mesh" in t:
            return C["lightblue"]
        return "white"

    for i, r in enumerate(rows):
        ax.text(-0.62, i, r, ha="right", va="center", fontsize=11, fontweight="bold")
        for j, c in enumerate(cols):
            t = cells[i][j]
            ax.add_patch(Rectangle((j - 0.48, i - 0.42), 0.96, 0.84, facecolor=color(t),
                                   edgecolor="white", linewidth=2.5))
            ax.text(j, i, t, ha="center", va="center", fontsize=7.6,
                    color=C["ink"], linespacing=1.3)
    for j, c in enumerate(cols):
        ax.text(j, -0.80, c, ha="center", va="center", fontsize=9.5, fontweight="bold")
    handles = [mpatches.Patch(fc=C["lightblue"], label="Mesh 族"),
               mpatches.Patch(fc=C["lightorange"], label="NHR 族"),
               mpatches.Patch(fc=C["lightpurple"], label="多级 / 混合族"),
               mpatches.Patch(fc=C["lightgray"], label="不支持 / 不匹配")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False,
              fontsize=9)
    save(fig, "02_老路径分支矩阵.png")


# --------------------------------------------------------------------------- #
# 03 新路径管线
# --------------------------------------------------------------------------- #
def chart03():
    fig, ax = plain_axes((15, 8.5))
    ax.text(0.5, 0.965, "新路径 SelectorEngine：从 cost model 到 min-cost 的管线", ha="center",
            fontsize=19, fontweight="bold")
    ax.text(0.5, 0.925, "selector_engine.cc:413 Run", ha="center", fontsize=10, color=C["gray"])

    x = 0.11
    steps = [
        ("costModel\n五步", "cost_model.cc:314", C["lightblue"],
         "① InitModel\n② FilterByEngine\n③ HCCL_ALGO 打标\n④ topo 过滤+标定\n⑤ topoPriority 排他"),
        ("costTable\n三阶段", "cost_table.cc:163", C["lightpurple"],
         "Phase1 op 过滤+算 cost\nPhase2 opPriority 排他\nPhase3 HCCL_ALGO 收敛"),
        ("tuner\n可选", "TunerEnrichCostTable", C["lightgray"],
         "改写 cost\ncost<0 视为无效条目"),
        ("引擎\n分层", "GetEnginePriority:74", C["lightorange"],
         "URMA 档：URMA(1) 高于\n其余引擎(0)"),
        ("min-cost\n选择", "SelectMinCost:597", "#D5F0EA",
         "最高优先级层内\n取 cost 最小\n平票取首个+WARNING"),
    ]
    for i, (title, sub, fc, detail) in enumerate(steps):
        cx = x + i * 0.195
        box(ax, cx, 0.70, 0.155, 0.10, title, fc=fc, fs=13, bold=True)
        ax.text(cx, 0.632, sub, ha="center", fontsize=8, color=C["gray"])
        ax.text(cx, 0.47, detail, ha="center", va="center", fontsize=8.3, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", lw=0.8))
        ax.plot([cx, cx], [0.646, 0.525], color="#C9CED6", ls="--", lw=1.0, zorder=0)
        if i:
            arrow(ax, (cx - 0.215, 0.70), (cx - 0.077, 0.70), lw=2.0)

    ax.text(0.5, 0.27, "cost = max( Σ_group [ Σ 或 max_seg ( A/util + B + Breduce ) × dataSize + C ],  Σ D )",
            ha="center", fontsize=12.5,
            bbox=dict(boxstyle="round,pad=0.5", fc=C["bg"], ec="#9AA3AE", lw=1.0))
    ax.text(0.5, 0.175, "util 只作用于 A 项；NHR 类算法查表时 perTransfer 还要 ×(rankSize/2)",
            ha="center", fontsize=10, color=C["gray"])
    ax.text(0.5, 0.095, "大量显式阈值并未消失，而是变成 opCustomCheck / topoCustomCheck / priority 等补丁叶子",
            ha="center", fontsize=9.5, color=C["warn"])
    save(fig, "03_新路径管线.png")


# --------------------------------------------------------------------------- #
# 04 引擎优先级
# --------------------------------------------------------------------------- #
def chart04():
    prio = [
        ("CCU_MS", [("CCU_MS", 0), ("CCU_SCHED", 0), ("AICPU_TS", 0), ("HOSTCPU", 0)]),
        ("CCU_SCHED", [("CCU_SCHED", 0), ("AICPU_TS", 0), ("HOSTCPU", 0)]),
        ("CCU_MS_ONLY", [("CCU_MS", 0)]),
        ("CCU_SCHED_ONLY", [("CCU_SCHED", 0)]),
        ("AIV", [("AIV", 0), ("AIV_URMA", 0), ("AICPU_TS", 0), ("HOSTCPU", 0)]),
        ("AIV_URMA", [("AIV_URMA", 1), ("AICPU_TS", 0), ("HOSTCPU", 0)]),
        ("AIV_ONLY", [("AIV", 0)]),
        ("AICPU_TS", [("AICPU_TS", 0), ("HOSTCPU", 0)]),
        ("HOSTCPU", [("HOSTCPU", 0)]),
    ]
    ec = {"CCU_MS": "#8E44AD", "CCU_SCHED": "#B07CC6", "AICPU_TS": C["mesh"],
          "HOSTCPU": C["gray"], "AIV": C["ok"], "AIV_URMA": "#1D7A70"}
    fig, ax = plt.subplots(figsize=(13, 6.8))
    fig.subplots_adjust(left=0.19, bottom=0.16)
    ax.set_title("引擎候选与优先级分层（GetEnginePriority:74）", fontsize=18, fontweight="bold", pad=14)
    ax.set_yticks(range(len(prio)))
    ax.set_yticklabels([p[0] for p in prio], fontsize=11, fontweight="bold")
    ax.set_xlim(0, 4.2)
    ax.set_xlabel("同一配置内的候选顺序（同层取 min cost；优先级数值大者优先）", fontsize=10)
    ax.set_ylim(len(prio) - 0.45, -0.65)
    ax.grid(axis="x", color="#E5E7EB")
    for i, (_, items) in enumerate(prio):
        for j, (e, p) in enumerate(items):
            x = j + 0.5
            ax.add_patch(FancyBboxPatch((x - 0.42, i - 0.30), 0.84, 0.60,
                                        boxstyle="round,pad=0.02,rounding_size=0.06",
                                        fc=ec.get(e, C["gray"]), ec="white", lw=2, alpha=0.92))
            lab = e if p == 0 else f"{e}\nprio={p}"
            ax.text(x, i, lab, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    fig.text(0.5, 0.015, "URMA 存在则必选；仅当被功能过滤（算子未实现/拓扑不支持/CANN 版本不满足）才落到平级层",
             ha="center", fontsize=8.8, color=C["warn"])
    ax.set_xticks([])
    save(fig, "04_引擎优先级.png")


# --------------------------------------------------------------------------- #
# 05 util 阶梯与 AG 钳制
# --------------------------------------------------------------------------- #
def chart05():
    xs = np.logspace(math.log10(16 * 1024), math.log10(512 * MIB), 2000)
    raw_clos = [ub_util("CLOS", int(x), op="RAW") for x in xs]
    eff_clos = [ub_util("CLOS", int(x), op="AG") for x in xs]
    mesh = [ub_util("MESH", int(x)) for x in xs]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("util(dataSize)：分段常数阶跃函数，以及 AllGather-CLOS 的 1MB 钳制", fontsize=17,
                 fontweight="bold", pad=14)
    ax.plot(xs, mesh, color=C["mesh"], lw=2.6, label="MESH 表（Mesh 段）")
    ax.plot(xs, raw_clos, color=C["nhr"], lw=1.6, ls="--", alpha=0.65, label="CLOS 原始表")
    ax.plot(xs, eff_clos, color=C["nhr"], lw=2.8, label="CLOS 生效值（AllGather 钳制后）")
    ax.axvspan(16 * 1024, 2 * MIB, color=C["nhr"], alpha=0.07)
    ax.axvline(2 * MIB, color=C["warn"], ls=":", lw=1.6)
    ax.annotate("钳制区：perTransfer < 1MB → 1MB\n按查表语义命中 2MB 档 util=0.3",
                xy=(256 * 1024, 0.30), xytext=(24 * 1024, 0.55), fontsize=9.5, color=C["warn"],
                arrowprops=dict(arrowstyle="->", color=C["warn"], lw=1.2))
    ax.annotate("2MB 断点：0.3 → 0.4288\ncost 在此处不连续",
                xy=(2 * MIB, 0.36), xytext=(5 * MIB, 0.16), fontsize=9.5, color=C["warn"],
                arrowprops=dict(arrowstyle="->", color=C["warn"], lw=1.2))
    for b, u in [(1, 0.1855), (2, 0.3), (4, 0.4288), (8, 0.5302), (16, 0.568),
                 (32, 0.6549), (64, 0.7184), (128, 0.7408), (256, 0.7644)]:
        ax.plot([b * MIB], [u], "o", ms=4.5, color=C["nhr"], zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(16 * 1024, 512 * MIB)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("perTransfer dataSize（字节，对数轴）", fontsize=11)
    ax.set_ylabel("UB utilization", fontsize=11)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.text(0.01, 0.02, "注：CLOS 表在 0.125/0.25/0.5MB 还有三档 0.10388，图中按生效值被钳制覆盖",
            transform=ax.transAxes, fontsize=8.5, color=C["gray"])
    save(fig, "05_util阶梯与AG钳制.png")


# --------------------------------------------------------------------------- #
# 06 16P 代价曲线
# --------------------------------------------------------------------------- #
def chart06():
    rank, p = 16, 1
    ds = np.logspace(math.log10(4 * 1024), math.log10(128 * MIB), 1600).astype(np.int64)
    cn = np.array([alg_cost(*sole_nhr(rank, p), int(d))[0] for d in ds])
    cp = np.array([alg_cost(*parallel(rank, p), int(d))[0] for d in ds])
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("16P 两级 MESH_1D+CLOS：SoleNHR 与 ParallelMeshNHR 的 cost 曲线", fontsize=17,
                 fontweight="bold", pad=14)
    ax.plot(ds, cn * 1e6, color=C["nhr"], lw=2.6, label="AicpuAllGatherSoleNHR")
    ax.plot(ds, cp * 1e6, color=C["par"], lw=2.6, label="AicpuAllGatherParallelMeshNHR")
    win_n = cn <= cp
    ax.fill_between(ds, cn * 1e6, cp * 1e6, where=win_n, color=C["nhr"], alpha=0.12)
    ax.fill_between(ds, cn * 1e6, cp * 1e6, where=~win_n, color=C["par"], alpha=0.12)
    for b in [1, 2, 4, 8, 16, 32, 64]:
        ax.axvline(b * MIB, color="#D1D5DB", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("per-rank dataSize（字节，对数轴）", fontsize=11)
    ax.set_ylabel("cost（μs）", fontsize=11)
    ax.legend(fontsize=10.5, loc="upper left")
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.02, "假设：非 POD、CLOS 端口 P=1、L0=8、α 量化到 1/8；"
                        "采用同一套 cost model 复刻，非上板实测", ha="center", fontsize=9, color=C["gray"])
    save(fig, "06_16P代价曲线.png")


# --------------------------------------------------------------------------- #
# 07 64P 64KB 翻转
# --------------------------------------------------------------------------- #
def chart07():
    rank, p = 64, 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("Corner Case：64P、P=1，per-rank 数据量 64KB 处的选路翻转", fontsize=18, fontweight="bold")

    ds = np.logspace(math.log10(8 * 1024), math.log10(16 * MIB), 1400).astype(np.int64)
    cn = np.array([alg_cost(*sole_nhr(rank, p), int(d))[0] for d in ds]) * 1e6
    cp = np.array([alg_cost(*parallel(rank, p), int(d))[0] for d in ds]) * 1e6
    ax = axes[0]
    ax.plot(ds, cn, color=C["nhr"], lw=2.5, label="SoleNHR")
    ax.plot(ds, cp, color=C["par"], lw=2.5, label="ParallelMeshNHR")
    ax.fill_between(ds, cn, cp, where=cn <= cp, color=C["nhr"], alpha=0.13)
    ax.fill_between(ds, cn, cp, where=cn > cp, color=C["par"], alpha=0.13)
    ax.axvline(64 * 1024, color=C["warn"], ls="--", lw=1.8)
    ax.text(64 * 1024 * 1.12, ax.get_ylim()[1] * 0.82, "64KB\n断点", color=C["warn"], fontsize=10,
            fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("per-rank dataSize（对数轴）", fontsize=10.5)
    ax.set_ylabel("cost（μs）", fontsize=10.5)
    ax.legend(fontsize=10)
    ax.set_title("(a) 全局视图", fontsize=12)

    lo, hi = 64 * 1024 - 4096, 64 * 1024 + 4096
    zs = np.arange(lo, hi + 1, 16)
    zn = np.array([alg_cost(*sole_nhr(rank, p), int(d))[0] for d in zs]) * 1e6
    zp = np.array([alg_cost(*parallel(rank, p), int(d))[0] for d in zs]) * 1e6
    ax = axes[1]
    ax.plot(zs / 1024, zn, color=C["nhr"], lw=2.6, label="SoleNHR")
    ax.plot(zs / 1024, zp, color=C["par"], lw=2.6, label="ParallelMeshNHR")
    ax.axvline(64, color=C["warn"], ls="--", lw=1.6)
    ax.axhline((zn[-1] + zp[-1]) / 2, color="#D1D5DB", lw=1.0, ls=":")
    x0, x1 = 60, 68
    n_lo = alg_cost(*sole_nhr(rank, p), 65535)[0] * 1e6
    p_lo = alg_cost(*parallel(rank, p), 65535)[0] * 1e6
    n_hi = alg_cost(*sole_nhr(rank, p), 65536)[0] * 1e6
    p_hi = alg_cost(*parallel(rank, p), 65536)[0] * 1e6
    ax.annotate(f"65535B：Parallel 赢 ≈{n_lo - p_lo:.1f}μs", xy=(63.95, p_lo), xytext=(60.3, 305),
                fontsize=9.2, color=C["par"], arrowprops=dict(arrowstyle="->", color=C["par"], lw=1.2))
    ax.annotate(f"65536B：SoleNHR 赢 ≈{p_hi - n_hi:.1f}μs\n"
                f"util 0.3→0.4288（跳降≈{n_lo - n_hi:.1f}μs）",
                xy=(64.05, n_hi), xytext=(64.35, 242), fontsize=8.8, color=C["nhr"], va="bottom",
                arrowprops=dict(arrowstyle="->", color=C["nhr"], lw=1.2))
    ax.set_xlim(60, 68)
    ax.set_ylim(238, 336)
    ax.set_xlabel("per-rank dataSize（KB）", fontsize=10.5)
    ax.set_ylabel("cost（μs）", fontsize=10.5)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("(b) 64KB 附近放大", fontsize=12)
    fig.subplots_adjust(bottom=0.17)
    fig.text(0.5, 0.015, "perTransfer = ds × (rank/2) = 64KB × 32 = 2MB，恰好跨过 CLOS util 的 2MB 断点",
             ha="center", fontsize=9, color=C["gray"])
    save(fig, "07_64P_64KB翻转.png")


# --------------------------------------------------------------------------- #
# 08 代价构成
# --------------------------------------------------------------------------- #
def chart08():
    rank, p = 64, 1
    sizes = [16 * 1024, 64 * 1024, 256 * 1024, MIB, 4 * MIB, 16 * MIB]
    labels = ["16KB", "64KB", "256KB", "1MB", "4MB", "16MB"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    fig.suptitle("代价构成：A/util 项 vs 固定开销（64P，P=1）", fontsize=17, fontweight="bold")

    def breakdown(segs, groups, intra_max, ds):
        ab, fixed, total_d = 0.0, 0.0, sum(s["D"] for s in segs)
        idx = 0
        for g in groups:
            vals = []
            for _ in range(g):
                s = segs[idx]
                dr = s["dr"] * (s["rs"] / 2 if s["at"] == "NHR" else 1)
                per = max(int(np.float32(ds) * np.float32(dr)), 1)
                u = ub_util(s["net"], per)
                vals.append(((s["A"] / u + s["B"]) * ds, s["C"]))
                idx += 1
            if intra_max:
                pick = max(vals, key=lambda v: sum(v))
                ab += pick[0]
                fixed += pick[1]
            else:
                ab += sum(v[0] for v in vals)
                fixed += sum(v[1] for v in vals)
        return ab, fixed, total_d

    for ax, (title, maker, color) in zip(
            axes,
            [("SoleNHR", sole_nhr(rank, p), C["nhr"]),
             ("ParallelMeshNHR", parallel(rank, p), C["par"])]):
        ab, fx, td = [], [], []
        for s in sizes:
            a, f, d = breakdown(*maker, s)
            ab.append(a * 1e6)
            fx.append(f * 1e6)
            td.append(d * 1e6)
        x = np.arange(len(sizes))
        ax.bar(x, ab, width=0.62, color=color, alpha=0.85, label="(A/util + B)·dataSize")
        ax.bar(x, fx, width=0.62, bottom=ab, color="#C9CED6", label="C（latency 常数）")
        ax.plot(x, td, "o--", color=C["ink"], lw=1.6, ms=5, label="ΣD（下界）")
        for i in range(len(sizes)):
            total = max(ab[i] + fx[i], td[i])
            ax.text(i, total * 1.03, f"{total:.0f}", ha="center", fontsize=8.5, color=C["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("per-rank dataSize", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("cost 分量（μs）", fontsize=10.5)
    fig.text(0.5, 0.015, "最终 cost = max(分量和, ΣD)；柱顶数字为最终 cost。小数据由 D 主导，大数据由 A/util 主导。",
             ha="center", fontsize=9, color=C["gray"])
    save(fig, "08_代价构成分解.png")


# --------------------------------------------------------------------------- #
# 09 决策区域热力图
# --------------------------------------------------------------------------- #
def chart09():
    ranks = [16, 32, 64, 128, 256, 512]
    ds = np.logspace(math.log10(8 * 1024), math.log10(64 * MIB), 260).astype(np.int64)
    mat = np.zeros((len(ranks), len(ds)))
    flips = []
    for i, r in enumerate(ranks):
        prev = None
        for j, d in enumerate(ds):
            cn, cp = costs(r, 1, int(d))
            # 16P：SoleNHR 在 total 32~64MB 被 opCustomCheck 硬排除
            filtered = (r == 16 and 32 * MIB <= d * r <= 64 * MIB)
            mat[i, j] = 2 if filtered else (1 if cn <= cp else 0)
            cur = mat[i, j]
            if prev is not None and {cur, prev} == {0, 1}:
                flips.append((i, j))
            prev = cur
    fig, ax = plt.subplots(figsize=(14, 6.4))
    cmap = matplotlib.colors.ListedColormap([C["par"], C["nhr"], "#F2C4C4"])
    ax.pcolormesh(ds, np.arange(len(ranks)), mat, cmap=cmap,
                  shading="nearest", vmin=0, vmax=2, alpha=0.9)
    ax.set_yticks(np.arange(len(ranks)) + 0.5)
    ax.set_yticklabels([f"{r}P" for r in ranks], fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("per-rank dataSize（对数轴）", fontsize=11)
    ax.set_title("决策区域：SoleNHR vs ParallelMeshNHR（L0=8，P=1，α 量化）", fontsize=17,
                 fontweight="bold", pad=14)
    handles = [mpatches.Patch(fc=C["par"], label="ParallelMeshNHR 胜"),
               mpatches.Patch(fc=C["nhr"], label="SoleNHR 胜"),
               mpatches.Patch(fc="#F2C4C4", label="SoleNHR 被 opCustomCheck 排除（16P 32~64MB）"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=C["ink"], markersize=6,
                          label="相邻采样点之间发生胜负翻转")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False,
              fontsize=9)
    for i, j in flips:
        ax.plot(ds[j], i + 0.5, "o", ms=4, mfc="white", mec=C["ink"], mew=0.8)
    save(fig, "09_决策区域热力图.png")


# --------------------------------------------------------------------------- #
# 10 flip-flop 扫描
# --------------------------------------------------------------------------- #
def chart10():
    cases = [(64, 1), (32, 2), (16, 4), (128, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("flip-flop 扫描：Δcost = cost(SoleNHR) - cost(ParallelMeshNHR)", fontsize=18,
                 fontweight="bold")
    for ax, (rank, p) in zip(axes.flat, cases):
        ds = np.logspace(math.log10(8 * 1024), math.log10(64 * MIB), 900).astype(np.int64)
        delta = np.array([costs(rank, p, int(d))[0] - costs(rank, p, int(d))[1] for d in ds]) * 1e6
        ax.plot(ds, delta, color=C["ink"], lw=2.0)
        ax.axhline(0, color=C["warn"], lw=1.4, ls="--")
        ax.fill_between(ds, delta, 0, where=delta < 0, color=C["nhr"], alpha=0.18, label="SoleNHR 胜")
        ax.fill_between(ds, delta, 0, where=delta >= 0, color=C["par"], alpha=0.13, label="Parallel 胜")
        boundaries = [b for b, _ in CLOS_TAB if b >= 2 * MIB][:4]
        for b in boundaries:
            xb = b / (rank / 2.0)
            if ds[0] < xb < ds[-1]:
                ax.axvline(xb, color="#D1D5DB", ls=":", lw=1.0)
        sign = np.sign(delta)
        changes = np.where(sign[1:] != sign[:-1])[0]
        if len(changes):
            xf = ds[changes[0] + 1]
            ax.plot([xf], [0], "o", ms=7, mfc="white", mec=C["warn"], mew=1.8, zorder=5)
            ax.annotate(f"首次翻转\n{xf / 1024:.0f}KB", xy=(xf, 0), xytext=(xf * 0.85, 1500),
                        fontsize=8.2, color=C["warn"], ha="right",
                        arrowprops=dict(arrowstyle="->", color=C["warn"], lw=1.0))
        ax.set_yscale("symlog", linthresh=10, linscale=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
        ax.set_xscale("log")
        ax.set_title(f"rank={rank}P, CLOS P={p}", fontsize=12)
        ax.set_xlabel("per-rank dataSize（对数轴）", fontsize=9)
        ax.set_ylabel("Δcost（μs，symlog）", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.98, hspace=0.36, wspace=0.16)
    fig.text(0.5, 0.01, "竖虚线：SoleNHR 的 CLOS util 断点（2/4/8/16/32/64/128/256MB 折算到 per-rank）；"
                        "每次跨断点 cost 跳变，可能穿越 0 线。", ha="center", fontsize=9, color=C["gray"])
    save(fig, "10_flipflop扫描.png")


# --------------------------------------------------------------------------- #
# 11 阈值标尺
# --------------------------------------------------------------------------- #
def chart11():
    rows = [
        ("CCU_MS 2P detour", 4 * MIB, ">=", "AG_2P_DETOUR_DATA_SIZE"),
        ("CCU_MS UBX 小数据", 512 * 1024, ">", "SMALL_COUNT_512KB"),
        ("CCU_SCHED 8P 小数据", 1 * MIB, "<=", "AG_CCU_CLOS_SMALL_DATA_SIZE"),
        ("CCU_SCHED R>8 小数据", 128 * 1024, "<", "AG_FLATTEN_MAX_DATA_SIZE"),
        ("CCU_SCHED 序列上限", 4 * MIB, "<", "AG_CCU_SEQUENCE_MAX_DATA_SIZE"),
        ("CCU_SCHED 960 大数据", 16 * MIB, ">", "SMALL_COUNT_16M"),
        ("AICPU 小数据", 1 * MIB, ">", "AG_AICPU_SMALL_DATA_SIZE"),
        ("AICPU 两级总量", 1 * 1024 * MIB, ">", "AG_AICPU_1D_TWO_LEVEL_..."),
        ("AICPU 序列总量", 4 * 1024 * MIB, ">", "AG_AICPU_SEQUENCE_DATA_SIZE"),
        ("AIV 单卡上限", 8 * MIB, ">=", "AIV_MAX_PER_RANK_DATA_SIZE"),
    ]
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    ax.set_title("阈值标尺与比较符审计（数据量单位：字节，对数轴）", fontsize=17, fontweight="bold", pad=14)
    opcolor = {">": C["par"], ">=": "#9B59B6", "<": C["ok"], "<=": "#16A085"}
    for i, (name, v, op, const) in enumerate(rows):
        ax.plot([v], [i], "o", ms=11, color=opcolor[op], zorder=4)
        if v > 1 * 1024 * MIB:
            ax.text(v / 1.18, i, f"{op}  {const}", va="center", ha="right", fontsize=8.6, color=C["ink"])
        else:
            ax.text(v * 1.18, i, f"{op}  {const}", va="center", fontsize=8.6, color=C["ink"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(64 * 1024, 16 * 1024 * MIB)
    ax.invert_yaxis()
    ax.set_xlabel("dataSize（对数轴）", fontsize=10.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=opcolor[o], markersize=9, label=f"运算符 {o}")
               for o in [">", ">=", "<", "<="]]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True)
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.045, "同一 4MB 阈值：CCU_MS/AICPU 用 >=，NDA 用 >；同一 16MB：CCU_SCHED 用 >，"
                         "AICPU/NDA 用 <", ha="center", fontsize=8.8, color=C["warn"])
    fig.text(0.5, 0.012, "rank 阈值（不在数据量标尺上）：MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4（<=）、"
                         "MAX_RANK_NUM_FOR_SEQ_ALGO = 8（<=）、AG_CCU_MAX_RANK_SIZE = 64（>）",
             ha="center", fontsize=8.5, color=C["gray"])
    save(fig, "11_阈值标尺.png")


# --------------------------------------------------------------------------- #
# 12 两级拓扑与 4 段数据比
# --------------------------------------------------------------------------- #
def chart12():
    rank, p, l0 = 64, 1, 8
    l1 = rank // l0
    a = alpha_of(l0, l1, p)
    b = 1 - a
    fig = plt.figure(figsize=(15, 6.6))
    ax1 = fig.add_axes([0.04, 0.08, 0.40, 0.82])
    ax1.set_xlim(-0.6, l0 - 0.4)
    ax1.set_ylim(-0.6, l1 - 0.4)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(f"两级拓扑：{rank}P = L0({l0}) × L1({l1})", fontsize=13, pad=8)
    for s in range(l1):
        for c in range(l0):
            fc = C["lightblue"] if s % 2 == 0 else C["lightpurple"]
            ax1.add_patch(FancyBboxPatch((c - 0.42, s - 0.42), 0.84, 0.84,
                                         boxstyle="round,pad=0.02,rounding_size=0.08",
                                         fc=fc, ec="white", lw=1.5))
            ax1.text(c, s, f"{s * l0 + c}", ha="center", va="center", fontsize=7.5, color=C["ink"])
        ax1.add_patch(Rectangle((-0.5, s - 0.5), l0, 1.0, fill=False, ec=C["par"], lw=2.2))
    fig.text(0.045, 0.042, "L0：机内 MESH（8 卡）", fontsize=9.5, color=C["mesh"])
    fig.text(0.045, 0.010, "L1：服务器间 CLOS/NHR", fontsize=9.5, color=C["nhr"])
    ax2 = fig.add_axes([0.52, 0.14, 0.45, 0.76])
    segs = ["seg0\nMesh L0\nα", "seg1\nNHR L1\n1-α", "seg2\nMesh L0\n(1-α)·L1", "seg3\nNHR L1\nα·L0"]
    vals = [a, b, b * l1, a * l0]
    colors = [C["mesh"], C["nhr"], C["mesh"], C["nhr"]]
    bars = ax2.bar(range(4), vals, color=colors, alpha=0.9, width=0.62)
    for i, (v, bar) in enumerate(zip(vals, bars)):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.06, f"{v:.3f}", ha="center", fontsize=10,
                 fontweight="bold")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(segs, fontsize=9)
    ax2.set_ylabel("dataRatio（相对 per-rank dataSize）", fontsize=10)
    ax2.set_title("ParallelMeshNHR 的 4 段数据比与聚合方式", fontsize=13, pad=8)
    ax2.text(0.5, 0.95, f"α = quantize( L1 / (L1 + L0·(L1-1)/P) ) = {a:.3f}\n"
                        f"cost = max(seg0,seg1) + max(seg2,seg3)，再与 ΣD 取 max",
             transform=ax2.transAxes, ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.35", fc=C["bg"], ec="#D1D5DB"))
    ax2.set_ylim(0, max(vals) * 1.25)
    save(fig, "12_两级拓扑与4段数据比.png")


# --------------------------------------------------------------------------- #
# 13 代价公式映射
# --------------------------------------------------------------------------- #
def chart13():
    fig, ax = plain_axes((15, 8))
    ax.text(0.5, 0.955, "代价公式与代码映射", ha="center", fontsize=19, fontweight="bold")
    ax.text(0.5, 0.905, "cost_table.cc:360 CalcAlgCost", ha="center", fontsize=10, color=C["gray"])

    box(ax, 0.14, 0.72, 0.19, 0.09, "perTransfer\n= dataSize × dataRatio\nNHR 再 ×(rankSize/2)",
        fc=C["lightorange"], fs=9)
    box(ax, 0.37, 0.72, 0.19, 0.09, "util\n分档查表\n(AG-CLOS 有 1MB 钳制)", fc=C["lightblue"], fs=9)
    box(ax, 0.61, 0.72, 0.22, 0.09, "segCost\n= (A/util + B + Breduce)\n× dataSize + C", fc=C["lightpurple"], fs=9)
    box(ax, 0.85, 0.72, 0.20, 0.09, "聚合\ngroup: Σ 或 max\ninter: Σ\ncost = max(·, ΣD)",
        fc="#D5F0EA", fs=9)
    for x1, x2 in [(0.235, 0.275), (0.465, 0.50), (0.72, 0.75)]:
        arrow(ax, (x1, 0.72), (x2, 0.72), lw=2)

    box(ax, 0.14, 0.47, 0.22, 0.10, "A：跨卡传输斜率\nMesh: n/BW\nNHR: n·(rs-1)/(P·BW)",
        fc="white", ec=C["mesh"], fs=9)
    box(ax, 0.40, 0.47, 0.22, 0.10, "B：本地拷贝斜率\n1/1400GB/s\n（64bit/PROD 另加 Breduce）",
        fc="white", ec=C["ok"], fs=9)
    box(ax, 0.66, 0.47, 0.22, 0.10, "C / D：固定开销\nC = 2μs × kernelNum\nD = 1.2μs/0.6μs × taskNum",
        fc="white", ec=C["warn"], fs=9)
    arrow(ax, (0.61, 0.675), (0.14, 0.525), color="#9AA3AE", lw=1.2, ls="--")
    arrow(ax, (0.61, 0.675), (0.40, 0.525), color="#9AA3AE", lw=1.2, ls="--")
    arrow(ax, (0.66, 0.675), (0.66, 0.525), color="#9AA3AE", lw=1.2, ls="--")

    ax.text(0.5, 0.28, "带宽常量（cost_model.cc:114）：跨片 56 GB/s、本地拷贝 1400 GB/s、"
                       "本地 reduce 900 GB/s、ROCE 25 GB/s",
            ha="center", fontsize=10, color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.4", fc=C["bg"], ec="#D1D5DB"))
    ax.text(0.5, 0.17, "为什么会出现“数据量 +1 字节，算法翻转”：util 只影响 A 项，但它是 dataSize 的函数；"
                       "断点处 util 整体跳档，cost 本身不连续。",
            ha="center", fontsize=10.5, color=C["warn"])
    ax.text(0.5, 0.08, "注意：perTransfer 用于查 util，dataSize 用于乘 cost，两者不是同一个量。",
            ha="center", fontsize=9.5, color=C["gray"])
    save(fig, "13_代价公式映射.png")


# --------------------------------------------------------------------------- #
# 扫描角落用例并落盘
# --------------------------------------------------------------------------- #
def scan_and_dump():
    rows = []
    for rank in [16, 32, 64, 128, 256, 512]:
        for p in [1, 2, 4, 8]:
            sn = sole_nhr(rank, p)
            sp = parallel(rank, p)
            cands = set()
            for seg in sn[0] + sp[0]:
                scale = seg["dr"] * (seg["rs"] / 2 if seg["at"] == "NHR" else 1)
                if scale <= 0:
                    continue
                tab = CLOS_TAB if seg["net"] == "CLOS" else MESH_TAB
                for b, _ in tab:
                    if seg["net"] == "CLOS" and b < 2 * MIB:
                        continue
                    cands.add(int(round(b / scale)))
            for ds in sorted(c for c in cands if 2 < c < 4 * 1024 * MIB):
                d0 = costs(rank, p, ds - 1)
                d1 = costs(rank, p, ds)
                delta0 = d0[0] - d0[1]
                delta1 = d1[0] - d1[1]
                if (delta0 > 0) != (delta1 > 0) and abs(delta0 - delta1) > 1e-6:
                    rows.append([rank, p, ds, ds / MIB, delta0 * 1e6, delta1 * 1e6,
                                 d0[0] * 1e6, d0[1] * 1e6, d1[0] * 1e6, d1[1] * 1e6])
    path = os.path.join(OUT, "corner_case_scan.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "clos_port_P", "ds_bytes", "ds_MiB", "delta_before_us", "delta_after_us",
                    "nhr_before_us", "par_before_us", "nhr_after_us", "par_after_us"])
        w.writerows(rows)
    print("flip cases:", len(rows), "->", path)
    for r in rows:
        print(f"  rank={r[0]:3d} P={r[1]} ds={r[2]:9d} ({r[3]:7.3f}MiB) "
              f"Δ {r[4]:+8.2f} -> {r[5]:+8.2f} us")
    return rows



# --------------------------------------------------------------------------- #
# 14 35 个唯一算法名：引擎 × 编排 × 链路
# --------------------------------------------------------------------------- #
# 名单来源：src/ops/all_gather/algorithm/executor/*.cc 中 REGISTER_EXEC*/REGISTER_EXECUTOR*
# 宏的第一实参（HcclCMDType::HCCL_CMD_ALLGATHER 之后），去重后共 35 个。
AG_ALGO_NAMES = [
    "AicpuAllGatherConcurMeshNHR",
    "AicpuAllGatherParallelMeshNHR",
    "AicpuAllGatherParallelMeshNHRMultiJetty",
    "AicpuAllGatherParallelNHRNHR",
    "AicpuAllGatherPipeLineMeshNHR",
    "AicpuAllGatherPipeLineMeshNHRNHR",
    "AicpuAllGatherSequenceMeshConcurNHR",
    "AicpuAllGatherSequenceMeshConcurNHRNHR",
    "AicpuAllGatherSequenceMeshConcurRHD",
    "AicpuAllGatherSequenceMeshConcurRHDRHD",
    "AicpuAllGatherSoleMesh",
    "AicpuAllGatherSoleMeshConcur",
    "AicpuAllGatherSoleNHR",
    "AicpuAllGatherSoleNHRMultiLink",
    "AivAllGatherSoleMesh",
    "AivUrmaAllGatherSequenceMeshConcurMesh",
    "CcuMSAllGatherSoleMesh",
    "CcuMSAllGatherSoleMesh2Die",
    "CcuSchedAllGatherConcurMeshNHRMultiLink",
    "CcuSchedAllGatherParallelMeshNHR",
    "CcuSchedAllGatherParallelMeshNHRMultiLink",
    "CcuSchedAllGatherPipeLineMeshNHR",
    "CcuSchedAllGatherSequenceMeshMesh",
    "CcuSchedAllGatherSoleMesh",
    "CcuSchedAllGatherSoleMesh2Die",
    "CcuSchedAllGatherSoleMeshConcur",
    "CcuSchedAllGatherSoleNHR",
    "CcuSchedAllGatherSoleNHRMultiLink",
    "DpuAllGatherPipeLineMeshNHRNHR",
    "DpuAllGatherPipeLineMeshNHRNHRNDA",
    "DpuAllGatherSequenceMeshConcurNHRNHR",
    "DpuAllGatherSequenceMeshConcurRHDRHD",
    "DpuAllGatherSequenceMeshNHR",
    "DpuAllGatherSequenceMeshRHD",
    "DpuAllGatherSequenceNHRNHRNDA",
]

ENGINE_LABEL = {
    "Aicpu": "AICPU", "Aiv": "AIV", "AivUrma": "AIV_URMA",
    "CcuMS": "CCU_MS", "CcuSched": "CCU_SCHED", "Dpu": "DPU",
}
FAMILIES = ["Sole", "Parallel", "PipeLine", "Sequence", "Concur"]
FAM_COLOR = {
    "Sole": ("#DCE7F5", "#2E5EAA"),
    "Parallel": ("#EDE4FB", "#7C3AED"),
    "PipeLine": ("#FBE3D1", "#E07B39"),
    "Sequence": ("#DDF0EC", "#2A9D8F"),
    "Concur": ("#FBE0DC", "#C0392B"),
}


def _split_algo(name):
    engine_raw, rest = name.split("AllGather", 1)
    family = next(f for f in FAMILIES if rest.startswith(f))
    return ENGINE_LABEL[engine_raw], family, rest[len(family):]


def chart14():
    grid, engine_tot, fam_tot = {}, {}, {}
    for n in AG_ALGO_NAMES:
        eng, fam, suffix = _split_algo(n)
        grid.setdefault(eng, {}).setdefault(fam, []).append(suffix)
        engine_tot[eng] = engine_tot.get(eng, 0) + 1
        fam_tot[fam] = fam_tot.get(fam, 0) + 1

    engines = sorted(engine_tot, key=lambda e: -engine_tot[e])
    fams = sorted(fam_tot, key=lambda f: -fam_tot[f])

    fig, ax = plain_axes((16, 9))
    ax.text(0.5, 0.955, "35 个唯一算法名 = 引擎 × 编排 × 链路的合法组合",
            ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0.5, 0.915, "同一套编排换个引擎就是另一个名字；实际在册 35 个，老路径只会直接给出其中 27 个",
            ha="center", va="center", fontsize=11, color=C["gray"])

    x0, x1 = 0.185, 0.865
    top, bottom = 0.745, 0.170
    cw = (x1 - x0) / len(fams)
    ch = (top - bottom) / len(engines)

    # 列头
    for j, fam in enumerate(fams):
        cx = x0 + (j + 0.5) * cw
        ax.text(cx, 0.786, fam, ha="center", va="center", fontsize=12, fontweight="bold",
                color=FAM_COLOR[fam][1])
        ax.text(cx, 0.762, f"{fam_tot[fam]} 个", ha="center", va="center", fontsize=9, color=C["gray"])

    # 行标签
    for i, eng in enumerate(engines):
        cy = top - (i + 0.5) * ch
        ax.text(0.172, cy + 0.014, eng, ha="right", va="center", fontsize=11.5, fontweight="bold")
        ax.text(0.172, cy - 0.016, f"{engine_tot[eng]} 个", ha="right", va="center",
                fontsize=9, color=C["gray"])

    # 单元格
    for i, eng in enumerate(engines):
        for j, fam in enumerate(fams):
            cell_x = x0 + j * cw
            cell_y = top - (i + 1) * ch
            items = grid.get(eng, {}).get(fam, [])
            if not items:
                ax.add_patch(Rectangle((cell_x + 0.004, cell_y + 0.004), cw - 0.008, ch - 0.008,
                                       fc="#F8F9FB", ec="#EDEEF2", lw=0.8, zorder=1))
                continue
            fc, ec = FAM_COLOR[fam]
            ax.add_patch(FancyBboxPatch((cell_x + 0.004, cell_y + 0.004), cw - 0.008, ch - 0.008,
                                        boxstyle="round,pad=0.002,rounding_size=0.006",
                                        fc=fc, ec=ec, lw=1.0, zorder=1))
            ax.text(cell_x + 0.016, cell_y + ch - 0.020, str(len(items)), ha="left", va="center",
                    fontsize=13, fontweight="bold", color=ec, zorder=3)
            ax.text(cell_x + cw / 2 + 0.008, cell_y + ch / 2 - 0.008, "\n".join(items),
                    ha="center", va="center", fontsize=6.6, color=C["ink"], linespacing=1.42, zorder=3)

    # 汇总
    ax.text(0.5, 0.132,
            "  +  ".join(f"{fam} {fam_tot[fam]}" for fam in fams) + f"  =  {len(AG_ALGO_NAMES)} 个算法名",
            ha="center", va="center", fontsize=12, fontweight="bold", color=C["ink"])
    ax.text(0.5, 0.096,
            "名字读法：<引擎><算子><编排><链路/模板>   例：CcuSched · AllGather · Sole · NHRMultiLink",
            ha="center", va="center", fontsize=10, color=C["gray"])
    ax.text(0.5, 0.058,
            "来源：src/ops/all_gather/algorithm/executor/*.cc 的 REGISTER_EXEC* 宏，去重后 35 个；"
            "部分注册在 CANN_VERSION / AICPU_COMPILE 宏内，实际编入集合随构建而定",
            ha="center", va="center", fontsize=8.6, color="#9AA3AE")
    save(fig, "14_35个算法名.png")



# --------------------------------------------------------------------------- #
# 15 算法全景：注册 35 = 老路径 27 + 新路径独占 8
# --------------------------------------------------------------------------- #
# 老路径可达集合：src/ops/all_gather/selector/*.cc 中 selectAlgName = "..." 的字面量（27 个，全部在册）
AG_OLD_PATH_NAMES = {
    "AicpuAllGatherConcurMeshNHR",
    "AicpuAllGatherParallelMeshNHR",
    "AicpuAllGatherParallelNHRNHR",
    "AicpuAllGatherPipeLineMeshNHR",
    "AicpuAllGatherPipeLineMeshNHRNHR",
    "AicpuAllGatherSequenceMeshConcurNHRNHR",
    "AicpuAllGatherSoleMesh",
    "AicpuAllGatherSoleMeshConcur",
    "AicpuAllGatherSoleNHR",
    "AicpuAllGatherSoleNHRMultiLink",
    "AivAllGatherSoleMesh",
    "CcuMSAllGatherSoleMesh",
    "CcuMSAllGatherSoleMesh2Die",
    "CcuSchedAllGatherConcurMeshNHRMultiLink",
    "CcuSchedAllGatherParallelMeshNHR",
    "CcuSchedAllGatherParallelMeshNHRMultiLink",
    "CcuSchedAllGatherPipeLineMeshNHR",
    "CcuSchedAllGatherSequenceMeshMesh",
    "CcuSchedAllGatherSoleMesh",
    "CcuSchedAllGatherSoleMesh2Die",
    "CcuSchedAllGatherSoleMeshConcur",
    "CcuSchedAllGatherSoleNHR",
    "CcuSchedAllGatherSoleNHRMultiLink",
    "DpuAllGatherPipeLineMeshNHRNHR",
    "DpuAllGatherPipeLineMeshNHRNHRNDA",
    "DpuAllGatherSequenceMeshNHR",
    "DpuAllGatherSequenceNHRNHRNDA",
}

# 老路径分支矩阵（附录 02）里，各引擎在 5 种拓扑下的覆盖
ENGINE_TOPO = {
    "AICPU": ("5/5", "单级 · 两级 · UBX · PCIE · CLOS"),
    "CCU_SCHED": ("5/5", "单级 · 两级 · UBX · PCIE · CLOS"),
    "DPU": ("3/5", "两级 · UBX · PCIE（NDA 走这一支）"),
    "CCU_MS": ("3/5", "单级 · UBX · PCIE（两级仅 2P≥4MB 回退）"),
    "AIV": ("5/5", "单级 · 两级 · UBX · PCIE · CLOS"),
    "AIV_URMA": ("0/5", "无老路径分支，只有新路径能选"),
}
NEW_ONLY_COLOR = "#C0392B"


def chart15():
    grid, engine_tot = {}, {}
    old_cnt = 0
    for n in AG_ALGO_NAMES:
        eng, fam, suffix = _split_algo(n)
        is_old = n in AG_OLD_PATH_NAMES
        old_cnt += int(is_old)
        grid.setdefault(eng, {}).setdefault(fam, []).append((suffix, is_old))
        engine_tot[eng] = engine_tot.get(eng, 0) + 1

    engines = sorted(engine_tot, key=lambda e: -engine_tot[e])
    fams = sorted({f for e in grid for f in grid[e]},
                  key=lambda f: -sum(len(v) for e in grid for ff, v in grid[e].items() if ff == f))

    fig, ax = plain_axes((16, 9))
    ax.text(0.5, 0.955, "算法全景：注册 35 个，老路径只覆盖其中 27 个",
            ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0.5, 0.915, "同一套编排换个引擎就是另一个名字；带 ○ 的 8 个只有新路径（cost model）能选到",
            ha="center", va="center", fontsize=11, color=C["gray"])

    x0, x1 = 0.130, 0.722
    top, bottom = 0.690, 0.170
    cw = (x1 - x0) / len(fams)
    ch = (top - bottom) / len(engines)

    # 列头
    for j, fam in enumerate(fams):
        cx = x0 + (j + 0.5) * cw
        ax.text(cx, 0.728, fam, ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=FAM_COLOR[fam][1])
    ax.text(0.855, 0.728, "老路径的拓扑覆盖", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=C["ink"])

    # 行标签
    for i, eng in enumerate(engines):
        cy = top - (i + 0.5) * ch
        ax.text(0.122, cy + 0.012, eng, ha="right", va="center", fontsize=10.5, fontweight="bold")
        ax.text(0.122, cy - 0.016, f"{engine_tot[eng]} 个", ha="right", va="center",
                fontsize=8.5, color=C["gray"])

    lh = 0.0122
    for i, eng in enumerate(engines):
        for j, fam in enumerate(fams):
            cell_x = x0 + j * cw
            cell_y = top - (i + 1) * ch
            items = grid.get(eng, {}).get(fam, [])
            if not items:
                ax.add_patch(Rectangle((cell_x + 0.003, cell_y + 0.004), cw - 0.006, ch - 0.008,
                                       fc="#F8F9FB", ec="#EDEEF2", lw=0.8, zorder=1))
                continue
            fc, ec = FAM_COLOR[fam]
            ax.add_patch(FancyBboxPatch((cell_x + 0.003, cell_y + 0.004), cw - 0.006, ch - 0.008,
                                        boxstyle="round,pad=0.002,rounding_size=0.006",
                                        fc=fc, ec=ec, lw=1.0, zorder=1))
            ax.text(cell_x + 0.013, cell_y + ch - 0.017, str(len(items)), ha="left", va="center",
                    fontsize=11.5, fontweight="bold", color=ec, zorder=3)
            start_y = cell_y + ch / 2 - 0.006 + (len(items) - 1) * lh / 2
            for k, (suffix, is_old) in enumerate(items):
                label = suffix if is_old else f"○ {suffix}"
                ax.text(cell_x + cw / 2 + 0.006, start_y - k * lh, label, ha="center", va="center",
                        fontsize=6.3, color=C["ink"] if is_old else NEW_ONLY_COLOR, zorder=3)

        # 右侧：老路径拓扑覆盖
        cy = top - (i + 0.5) * ch
        cover, detail = ENGINE_TOPO.get(eng, ("—", ""))
        ax.text(0.742, cy + 0.013, f"老路径 {cover}", ha="left", va="center", fontsize=9,
                fontweight="bold", color=C["ink"] if cover != "0/5" else NEW_ONLY_COLOR)
        ax.text(0.742, cy - 0.015, detail, ha="left", va="center", fontsize=7.2, color=C["gray"])

    # 图例与汇总
    ax.text(0.135, 0.128, "○ 仅新路径可选", ha="left", va="center", fontsize=9.5, color=NEW_ONLY_COLOR)
    ax.text(0.300, 0.128, "其余为老路径也会直接选（27 个）", ha="left", va="center", fontsize=9.5,
            color=C["gray"])
    ax.text(0.5, 0.082, "注册 35 = 老路径可达 27 + 新路径独占 8",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C["ink"])
    ax.text(0.5, 0.040,
            "新路径独占的 8 个集中在 RHD / MultiJetty / URMA 等较新变体：老阈值树只为老算法加过分支",
            ha="center", va="center", fontsize=9, color=C["gray"])
    save(fig, "15_算法全景矩阵.png")


if __name__ == "__main__":
    chart01()
    chart02()
    chart03()
    chart04()
    chart05()
    chart06()
    chart07()
    chart08()
    chart09()
    chart10()
    chart11()
    chart12()
    chart13()
    chart14()
    chart15()
    scan_and_dump()
    print("done")
