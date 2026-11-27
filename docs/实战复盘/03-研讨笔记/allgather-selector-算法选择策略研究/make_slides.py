# -*- coding: utf-8 -*-
"""生成 10 分钟分享用的 12 页幻灯片（16:9 PNG）。

复用 generate_charts.py 的代价模型复刻函数。
字体使用 Hiragino Sans GB（简体字形），避免 PingFang HK 把“径”渲染成“徑”。
"""
import importlib.util
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "slides")
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location("gc", os.path.join(HERE, "generate_charts.py"))
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)

plt.rcParams.update({
    "font.sans-serif": ["Hiragino Sans GB", "Songti SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
})

INK = "#1F2937"
GRAY = "#6B7280"
LIGHT = "#F3F5F9"
BLUE = "#2E5EAA"
PURPLE = "#7C3AED"
ORANGE = "#E07B39"
RED = "#C0392B"
GREEN = "#2A9D8F"
MIB = 1024 * 1024
TOTAL = 10


def slide_frame(title, subtitle=None, page=None):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.add_patch(Rectangle((0.055, 0.895), 0.05, 0.010, color=BLUE))
        ax.text(0.055, 0.955, title, fontsize=29, fontweight="bold", va="top", color=INK)
    if subtitle:
        ax.text(0.055, 0.872, subtitle, fontsize=14, va="top", color=GRAY)
    ax.text(0.945, 0.032, f"AllGather Selector 研讨  ·  {page}/{TOTAL}" if page else "AllGather Selector 研讨",
            ha="right", fontsize=10, color="#9AA3AE")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc=LIGHT, ec="#9AA3AE", fs=13, bold=False, color=INK, rad=0.02, lw=1.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.4, zorder=3)


def arr(ax, p1, p2, color=GRAY, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                                 lw=lw, color=color, linestyle=ls, shrinkA=2, shrinkB=2, zorder=1))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# --------------------------------------------------------------------------- #
# S1 封面
# --------------------------------------------------------------------------- #
def s1():
    fig, ax = slide_frame("")
    ax.text(0.055, 0.79, "AllGather 选算法：", fontsize=39, fontweight="bold", color=INK)
    ax.text(0.055, 0.68, "从阈值树到代价模型", fontsize=39, fontweight="bold", color=BLUE)
    ax.text(0.055, 0.575, "机制 · 模型 · 边界", fontsize=18, color=GRAY)

    stats = [
        ("2 条", "选路路径并存", BLUE),
        ("5 + 3", "costModel 五步 / costTable 三阶段", PURPLE),
        ("1 B", "触发胜负翻转", RED),
    ]
    for i, (num, label, c) in enumerate(stats):
        x = 0.22 + i * 0.28
        ax.text(x, 0.40, num, ha="center", fontsize=30, fontweight="bold", color=c)
        ax.text(x, 0.335, label, ha="center", fontsize=11.5, color=INK)

    ax.text(0.5, 0.23, "结论：选路是可行集上的比价；util 阶梯让边界呈“跳变带”。",
            ha="center", fontsize=17, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#F3F5F9", ec="#D1D5DB", lw=1.0))
    ax.text(0.5, 0.115, "① 全景   ② 管线与代价   ③ util 阶梯   ④ corner case   ⑤ 跳变带",
            ha="center", fontsize=12, color=GRAY)
    save(fig, "S1_封面.png")


# --------------------------------------------------------------------------- #
# S2 全景
# --------------------------------------------------------------------------- #
def s2():
    fig, ax = slide_frame("算法选路总览：两条路径并存",
                          "入口 op_common.cc:225；ReSelector:957 复用同一二选一", page=2)

    def dbox(cx, cy, hw, hh, text, fs=11):
        ax.add_patch(mpatches.Polygon([[cx, cy + hh], [cx + hw, cy], [cx, cy - hh], [cx - hw, cy]],
                                      closed=True, fc="white", ec=GRAY, lw=1.5, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
                linespacing=1.3, zorder=3)

    # 入口
    rbox(ax, 0.50, 0.815, 0.30, 0.048, "OpParam + TopoInfo", fc="#EEF2F8", ec=BLUE, fs=12.5, bold=True)
    arr(ax, (0.50, 0.791), (0.50, 0.770), lw=1.6)

    # 判定 1：NDA
    dbox(0.50, 0.742, 0.105, 0.036, "useNdaSelector ?", fs=10.5)
    arr(ax, (0.395, 0.742), (0.302, 0.742), lw=1.6)
    ax.text(0.345, 0.762, "是", ha="center", fontsize=9.5, color=GRAY)
    rbox(ax, 0.180, 0.742, 0.24, 0.050, "", fc="#FDF0E6", ec=ORANGE, lw=1.6)
    ax.text(0.180, 0.753, "老路径 NDA 分支  SelectNdaAlgo", ha="center", fontsize=10.5,
            fontweight="bold", color=ORANGE)
    ax.text(0.180, 0.732, "未命中 → SelectDPUAlgo", ha="center", fontsize=8.5, color=GRAY)

    # 否 → 判定 2
    arr(ax, (0.50, 0.706), (0.50, 0.686), lw=1.6)
    ax.text(0.518, 0.696, "否", ha="left", fontsize=9.5, color=GRAY)
    dbox(0.50, 0.658, 0.175, 0.040, "level0Topo == MESH_1D  或  IsDevType960() ?", fs=9.5)

    arr(ax, (0.325, 0.658), (0.215, 0.580), lw=1.6)
    arr(ax, (0.675, 0.658), (0.785, 0.580), lw=1.6)
    ax.text(0.305, 0.676, "否", ha="right", fontsize=9.5, color=GRAY)
    ax.text(0.695, 0.676, "是", ha="left", fontsize=9.5, color=GRAY)

    # 老路径
    rbox(ax, 0.20, 0.552, 0.30, 0.052, "", fc="#FDF0E6", ec=ORANGE, lw=1.6)
    ax.text(0.20, 0.564, "老路径  ExecuteSelector", ha="center", fontsize=11.5,
            fontweight="bold", color=ORANGE)
    ax.text(0.20, 0.539, "手写阈值决策树", ha="center", fontsize=9.5, color=GRAY)

    engines = [
        ("CCU_MS", "SoleMesh / 2Die"),
        ("CCU_SCHED", "SoleMesh / Parallel\nSequence / SoleNHR"),
        ("AICPU", "SoleMesh / NHR\nParallel / Pipeline"),
        ("AIV", "SoleMesh"),
        ("DPU", "Sequence / Pipeline"),
    ]
    for i, (name, alg) in enumerate(engines):
        x = 0.20 + (i - 2) * 0.078
        rbox(ax, x, 0.448, 0.072, 0.034, name, fc="white", ec=ORANGE, fs=8.2, bold=True, lw=1.2)
        ax.text(x, 0.402, alg, ha="center", va="center", fontsize=7.2, color=GRAY, linespacing=1.4)
        arr(ax, (0.20, 0.526), (x, 0.466), lw=1.0, color="#C9A38A")
    ax.text(0.20, 0.342, "命中即返回，决策边界直接写在 if/else 里", ha="center",
            fontsize=9.2, color=GRAY)
    ax.text(0.20, 0.310, "DPU 分支仅在 NDA 机型未命中时进入", ha="center",
            fontsize=8.2, color="#B08968")

    # 新路径
    rbox(ax, 0.780, 0.552, 0.305, 0.052, "", fc="#F1EAFC", ec=PURPLE, lw=1.6)
    ax.text(0.780, 0.564, "新路径  SelectorEngine::Run", ha="center", fontsize=11.5,
            fontweight="bold", color=PURPLE)
    ax.text(0.780, 0.539, "按引擎分层的 min-cost", ha="center", fontsize=9.5, color=GRAY)
    arr(ax, (0.780, 0.526), (0.780, 0.490), lw=1.4)

    labels = [
        ("① costModel 五步\n引擎 / topo / op / HCCL_ALGO 过滤", 0.462),
        ("② costTable 三阶段\n逐候选算法算 cost", 0.386),
        ("③ tuner 可选改写 cost", 0.310),
        ("④ 最高优先级层内取 min cost\n平票取注册序第一", 0.234),
    ]
    for i, (text, y) in enumerate(labels):
        rbox(ax, 0.780, y, 0.305, 0.050, text, fc="white", ec=PURPLE, fs=9.2, lw=1.2)
        if i:
            arr(ax, (0.780, y + 0.036), (0.780, y + 0.025), lw=1.2)

    ax.text(0.780, 0.125, "显式 if/else 退化为\nopCustomCheck / topoCustomCheck /\n"
                          "topoPriorityCheck / opPriorityCheck\n等“补丁叶子”",
            ha="center", va="center", fontsize=8.6, color=RED, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDF2F2", ec=RED, lw=1.0))

    # 三层抽象（补左下空白）
    ax.text(0.315, 0.228, "三层抽象：cost model 评价的是 Executor × Template 组合",
            ha="center", fontsize=9.2, color=GRAY)
    layers = [(0.115, "Selector", "选算法名", BLUE),
              (0.315, "Executor", "编排执行", PURPLE),
              (0.515, "Template", "引擎模板", GREEN)]
    for x, name, desc, c in layers:
        rbox(ax, x, 0.165, 0.165, 0.072, "", fc="white", ec=c, lw=1.4)
        ax.text(x, 0.180, name, ha="center", fontsize=10, fontweight="bold", color=c)
        ax.text(x, 0.148, desc, ha="center", fontsize=8.2, color=GRAY)
    arr(ax, (0.205, 0.165), (0.228, 0.165), lw=1.2)
    arr(ax, (0.405, 0.165), (0.428, 0.165), lw=1.2)

    ax.text(0.5, 0.050, "共同约束：topoPrecondCheck 硬前提始终执行；customCheck 在 preferred / tuner 加载时让位",
            ha="center", fontsize=9.5, color=GRAY)

    save(fig, "S2_全景.png")


# --------------------------------------------------------------------------- #
# S3 新路径管线
# --------------------------------------------------------------------------- #
def s3():
    fig, ax = slide_frame("新路径管线：过滤 → 定价 → 分层 → 择优",
                          "SelectorEngine::Run:413；硬前提与软策略分离", page=4)
    cols = [
        ("costModel 五步", "cost_model.cc:314", BLUE,
         ["① InitModel", "② 引擎过滤", "③ HCCL_ALGO 打标", "④ topo 过滤 + 标定", "⑤ topoPriority 排他"]),
        ("costTable 三阶段", "cost_table.cc:163", PURPLE,
         ["Phase1 op 过滤 + 算 cost", "Phase2 opPriority 排他", "Phase3 HCCL_ALGO 收敛"]),
        ("tuner", "可选", GREEN,
         ["改写 cost", "cost < 0 视为无效", "未加载则跳过"]),
        ("分层选择", "SelectMinCost:597", ORANGE,
         ["引擎优先级分层", "层内取 min cost", "平票取注册序第一"]),
    ]
    for i, (title, sub, c, items) in enumerate(cols):
        x = 0.145 + i * 0.235
        rbox(ax, x, 0.72, 0.20, 0.085, title, fc="white", ec=c, fs=14, bold=True)
        ax.text(x, 0.655, sub, ha="center", fontsize=9, color=GRAY)
        ax.text(x, 0.56, "\n".join(items), ha="center", va="center", fontsize=9.8,
                color=INK, linespacing=1.9,
                bbox=dict(boxstyle="round,pad=0.35", fc=LIGHT, ec="#D1D5DB", lw=0.8))
        if i:
            arr(ax, (x - 0.135, 0.72), (x - 0.10, 0.72), lw=2.0)

    ax.text(0.5, 0.40, "硬：topoPrecondCheck 始终执行，不受 preferred / tuner 影响",
            ha="center", fontsize=12, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="#E8F5F3", ec=GREEN, lw=1.0))
    ax.text(0.5, 0.31, "软：preferred / tuner 加载时 opCustomCheck、topoCustomCheck 让位；priority 只豁免偏好项",
            ha="center", fontsize=12, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDF2F2", ec=RED, lw=1.0))
    ax.text(0.5, 0.20, "所以阈值没有消失：它变成了“补丁叶子”，只是不再直接决定最终算法。",
            ha="center", fontsize=12.5, color=GRAY)
    ax.text(0.5, 0.10, "例：16P、P=1、per-rank 1MB → SoleNHR 545μs  vs  Parallel 194μs → 选 Parallel",
            ha="center", fontsize=12, color=BLUE, fontweight="bold")
    save(fig, "S3_管线.png")


# --------------------------------------------------------------------------- #
# S4 代价函数
# --------------------------------------------------------------------------- #
def s4():
    fig, ax = slide_frame("代价函数：α-β 模型的工程化",
                          "cost_table.cc:360 CalcAlgCost", page=5)
    ax.text(0.5, 0.80, "cost = max( Σ_group [ Σ 或 max_seg ( A/util + B ) × dataSize + C ],  ΣD )",
            ha="center", fontsize=17, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc=LIGHT, ec="#9AA3AE", lw=1.1))

    items = [
        ("A", "跨卡传输斜率", "Mesh: n / BW\nNHR: n·(rs-1) / (P·BW)", BLUE),
        ("B", "本地拷贝斜率", "1 / 1400GB/s\n64bit/PROD 另加 Breduce", GREEN),
        ("C", "固定时延", "AICPU: 2μs × kernelNum\nAIV: 1μs × kernelNum", ORANGE),
        ("D", "下发/同步", "NHR: 1.2μs × taskNum\nMesh: 0.6μs × taskNum", RED),
    ]
    for i, (sym, name, detail, c) in enumerate(items):
        x = 0.135 + i * 0.225
        ax.add_patch(FancyBboxPatch((x - 0.095, 0.40), 0.19, 0.20,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="white", ec=c, lw=2.0))
        ax.text(x, 0.555, sym, ha="center", fontsize=24, fontweight="bold", color=c)
        ax.text(x, 0.505, name, ha="center", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x, 0.445, detail, ha="center", fontsize=8.8, color=GRAY, linespacing=1.5)

    # perTransfer 机制：两把尺子各管一段
    flow = [
        (0.095, 0.13, "dataSize", "per-rank 实际字节数", "#9AA3AE", LIGHT, INK),
        (0.360, 0.30, "perTransfer = dataSize × dr",
         "dr 是该段占比；NHR 再 × rankSize/2", PURPLE, "#F1EAFC", INK),
        (0.645, 0.17, "查 util 表", "按“单次消息量”定档", ORANGE, "#FDF0E6", INK),
        (0.875, 0.19, "A / util × dataSize", "util 只除 A，不除 B", BLUE, "#EEF2F8", INK),
    ]
    for cx, w, line1, line2, ec, fc, tc in flow:
        ax.add_patch(FancyBboxPatch((cx - w / 2, 0.215), w, 0.105,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc=fc, ec=ec, lw=1.4, zorder=2))
        ax.text(cx, 0.288, line1, ha="center", fontsize=10.0, fontweight="bold", color=tc, zorder=3)
        ax.text(cx, 0.242, line2, ha="center", fontsize=8.4, color=GRAY, zorder=3)
    for x1, x2 in ((0.160, 0.208), (0.510, 0.558), (0.730, 0.778)):
        arr(ax, (x1, 0.267), (x2, 0.267), lw=1.5)

    ax.text(0.5, 0.165, "两把尺子：perTransfer 用来查表，dataSize 用来乘出时间。",
            ha="center", fontsize=11.5, color=GRAY)
    ax.text(0.5, 0.085, "util 是唯一随 dataSize 跳变的项，所以它决定了边界的“形状”。",
            ha="center", fontsize=13, color=BLUE, fontweight="bold")
    save(fig, "S4_代价函数.png")


# --------------------------------------------------------------------------- #
# S5 Parallel 4 段
# --------------------------------------------------------------------------- #
def s5():
    fig, ax = slide_frame("多段算法：Parallel 的 4 段与 α 配平",
                          "ins_v2_all_gather_parallel_executor.cc:52", page=6)
    rank, p, l0 = 64, 1, 8
    l1 = rank // l0
    a = gc.alpha_of(l0, l1, p)

    # 拓扑小图
    axg = fig.add_axes([0.035, 0.16, 0.24, 0.52])
    axg.set_xlim(-0.6, l0 - 0.4)
    axg.set_ylim(-0.6, l1 - 0.4)
    axg.set_aspect("equal")
    axg.axis("off")
    for s in range(l1):
        for c in range(l0):
            fc = "#DCE7F5" if s % 2 == 0 else "#EDE4FB"
            axg.add_patch(FancyBboxPatch((c - 0.4, s - 0.4), 0.8, 0.8,
                                         boxstyle="round,pad=0.02,rounding_size=0.08",
                                         fc=fc, ec="white", lw=1.0))
        axg.add_patch(Rectangle((-0.5, s - 0.5), l0, 1.0, fill=False, ec=PURPLE, lw=1.8))
    ax.text(0.155, 0.105, "L0：机内 Mesh（8 卡/服务器）\nL1：服务器间 NHR（8 台）",
            ha="center", fontsize=9, color=GRAY, linespacing=1.5)

    # 4 段 dataRatio
    axb = fig.add_axes([0.36, 0.20, 0.33, 0.42])
    vals = [a, 1 - a, (1 - a) * l1, a * l0]
    labels = ["seg0\nMesh L0\nα", "seg1\nNHR L1\n1-α", "seg2\nMesh L0\n(1-α)L1", "seg3\nNHR L1\nαL0"]
    bars = axb.bar(range(4), vals, color=[BLUE, ORANGE, BLUE, ORANGE], width=0.6)
    notes = ["", "", f"= {1 - a:.3f} × {l1}", f"= {a:.3f} × {l0}"]
    for b, v, note in zip(bars, vals, notes):
        txt = f"{v:.3f}\n{note}" if note else f"{v:.3f}"
        axb.text(b.get_x() + b.get_width() / 2, v + 0.12, txt, ha="center",
                 fontsize=9.5, fontweight="bold", linespacing=1.4)
    axb.set_xticks(range(4))
    axb.set_xticklabels(labels, fontsize=8.5)
    axb.set_ylabel("dataRatio", fontsize=9)
    axb.set_ylim(0, max(vals) * 1.22)
    axb.tick_params(labelsize=8)
    axb.set_title("64P、P=1 的 4 段配比", fontsize=10.5)

    ax.text(0.82, 0.60, "α = quantize(\n  L1 / (L1 + L0·(L1-1)/P)\n, 1/8)",
            ha="center", fontsize=12, color=INK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="#EDE4FB", ec=PURPLE, lw=1.2))
    ax.text(0.82, 0.42, "cost = max(seg0, seg1)\n       + max(seg2, seg3)\n再与 ΣD 取 max",
            ha="center", fontsize=12, color=INK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="#9AA3AE", lw=1.0))
    ax.text(0.5, 0.108, "半 A（α）：Mesh → NHR，第二跳数据量 ×L0　　　半 B（1-α）：NHR → Mesh，第二跳数据量 ×L1",
            ha="center", fontsize=11, color=INK)
    ax.text(0.5, 0.062, "α 按两条路的第二跳耗时配平；64P、P=1 得 0.125，量化到 1/8 便于切分",
            ha="center", fontsize=10, color=GRAY)
    save(fig, "S5_parallel四段.png")


# --------------------------------------------------------------------------- #
# S6 util 阶梯
# --------------------------------------------------------------------------- #
def s6():
    fig, ax = slide_frame("util：分档查表 + 两个特殊处理",
                          "QueryUbUtil：按「单次消息量」查档位，边界值落到下一档", page=6)
    axp = fig.add_axes([0.055, 0.20, 0.52, 0.55])
    xs = np.logspace(math.log10(16 * 1024), math.log10(256 * MIB), 1600)
    raw = [gc.ub_util("CLOS", int(x), op="RAW") for x in xs]
    eff = [gc.ub_util("CLOS", int(x), op="AG") for x in xs]
    mesh = [gc.ub_util("MESH", int(x)) for x in xs]
    axp.plot(xs, mesh, color=BLUE, lw=2.2, label="MESH 表")
    axp.plot(xs, raw, color=ORANGE, lw=1.2, ls="--", alpha=0.6, label="CLOS 原始表")
    axp.plot(xs, eff, color=ORANGE, lw=2.6, label="CLOS 生效值（AG 钳制）")
    axp.axvline(2 * MIB, color=RED, ls=":", lw=1.6)
    axp.annotate("2MB: 0.3 → 0.4288", xy=(2 * MIB, 0.36), xytext=(5 * MIB, 0.12),
                 fontsize=10.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axp.annotate("perTransfer < 2MB\n统一 0.3", xy=(256 * 1024, 0.3), xytext=(22 * 1024, 0.53),
                 fontsize=10.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axp.set_xscale("log")
    axp.set_xlim(16 * 1024, 256 * MIB)
    axp.set_ylim(0, 1.0)
    axp.set_xlabel("perTransfer dataSize（对数轴）", fontsize=10)
    axp.set_ylabel("utilization", fontsize=10)
    axp.legend(loc="lower right", fontsize=8.5, frameon=True)
    axp.tick_params(labelsize=8.5)

    # util 表本体：直接看数
    ax.text(0.780, 0.792, "util 表（CLOS，节选）", ha="center", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(0.618, 0.752, "上界", fontsize=9, color=GRAY, fontweight="bold")
    ax.text(0.706, 0.752, "覆盖区间", fontsize=9, color=GRAY, fontweight="bold")
    ax.text(0.952, 0.752, "util", fontsize=9, color=GRAY, fontweight="bold", ha="right")
    ax.plot([0.607, 0.955], [0.740, 0.740], color="#D1D5DB", lw=0.9)

    rows = [
        ("512K", "< 512K", "0.1039", False),
        ("1M", "512K ~ 1M", "0.1855", False),
        ("2M", "1M ~ 2M", "0.3000", True),
        ("4M", "2M ~ 4M", "0.4288", True),
        ("8M", "4M ~ 8M", "0.5302", False),
        ("16M", "8M ~ 16M", "0.5680", False),
        ("32M", "16M ~ 32M", "0.6549", False),
        ("64M", "32M ~ 64M", "0.7184", False),
    ]
    for i, (ub, span, u, hi) in enumerate(rows):
        y = 0.712 - i * 0.040
        if hi:
            ax.add_patch(Rectangle((0.607, y - 0.017), 0.348, 0.034, fc="#FDF2F2",
                                   ec="none", zorder=0))
        ax.text(0.618, y, ub, fontsize=9.5, color=RED if hi else INK,
                fontweight="bold" if hi else "normal")
        ax.text(0.706, y, span, fontsize=9.2, color=INK)
        ax.text(0.952, y, u, fontsize=9.5, color=RED if hi else INK, ha="right",
                fontweight="bold" if hi else "normal")

    ax.text(0.607, 0.352, "查询值落在哪个区间，就用哪一行的 util；\n正好等于上界，算下一行。",
            fontsize=9.6, color=GRAY, linespacing=1.6, va="top")
    ax.text(0.607, 0.268, "AG 钳制：CLOS + AllGather 且 <1MB 时按 1MB 查，\n落进 [1M, 2M)，得到 0.3。",
            fontsize=9.6, color=INK, linespacing=1.6, va="top")
    ax.text(0.607, 0.192, "NHR 放大：perTransfer = ds × rankSize/2，\n阶梯整体被缩放到 per-rank 数据量。",
            fontsize=9.6, color=INK, linespacing=1.6, va="top")

    save(fig, "S6_util阶梯.png")


# --------------------------------------------------------------------------- #
# S7 方法
# --------------------------------------------------------------------------- #
def s7():
    fig, ax = slide_frame("方法：把边界当成“可枚举的断点”",
                          "不靠直觉找 corner case，而是对断点做参数化扫描", page=7)
    steps = [
        ("1", "每段求 scale", "s = dataRatio\n（NHR 再 × rs/2）", BLUE),
        ("2", "断点反解", "util 断点 B\n→ per-rank ds = B / s", PURPLE),
        ("3", "两侧取点", "在 ds-1 与 ds\n分别算 Δcost", GREEN),
        ("4", "符号翻转", "Δ 变号 →\n候选 corner case", RED),
    ]
    for i, (n, h, t, c) in enumerate(steps):
        x = 0.145 + i * 0.235
        ax.scatter([x], [0.700], s=850, color=c, zorder=3, edgecolors="none")
        ax.text(x, 0.700, n, ha="center", va="center", fontsize=13, color="white",
                fontweight="bold", zorder=4)
        ax.text(x, 0.625, h, ha="center", fontsize=13, fontweight="bold", color=c)
        ax.text(x, 0.545, t, ha="center", fontsize=10, color=INK, linespacing=1.6)
        if i:
            arr(ax, (x - 0.20, 0.700), (x - 0.065, 0.700), lw=1.5)

    # 走一遍：64P、P=1、2MB 断点
    ax.text(0.5, 0.492, "走一遍：64P、P=1，util 的 2MB 断点", ha="center",
            fontsize=10, color=GRAY)
    demo = [
        (0.20, "s = 1 × 32 = 32", "dataRatio 1，NHR 再 × rankSize/2", BLUE),
        (0.50, "ds = 2MB ÷ 32 = 64KB", "断点 B 反解到 per-rank", PURPLE),
        (0.80, "Δ: +51.8 → -22.0", "65535 vs 65536 字节：变号 = 翻转", RED),
    ]
    for x, l1, l2, c in demo:
        ax.add_patch(FancyBboxPatch((x - 0.14, 0.398), 0.28, 0.072,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc="white", ec=c, lw=1.3, zorder=2))
        ax.text(x, 0.445, l1, ha="center", fontsize=10.5, fontweight="bold", color=c, zorder=3)
        ax.text(x, 0.414, l2, ha="center", fontsize=8.4, color=GRAY, zorder=3)
    ax.text(0.5, 0.318, "扫描空间：rank ∈ {16, 32, 64, 128, 256, 512}   ×   CLOS 端口 P ∈ {1, 2, 4, 8}",
            ha="center", fontsize=13, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="#9AA3AE", lw=1.0))
    ax.text(0.5, 0.225, "结果：7 组翻转；翻转点集中在 NHR 的 util 断点折算到 per-rank 的位置。",
            ha="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(0.5, 0.140, "可复现：corner_case_scan.csv；可复用：换 util 表 / 算法集，方法不变。",
            ha="center", fontsize=11.5, color=GRAY)
    ax.text(0.5, 0.075, "这一步把“找边界”从经验问题变成了可重复的枚举问题。",
            ha="center", fontsize=12.5, color=BLUE, fontweight="bold")
    save(fig, "S7_方法.png")


# --------------------------------------------------------------------------- #
# S8 corner case 定量
# --------------------------------------------------------------------------- #
def s8():
    fig, ax = slide_frame("Corner case：64P、P=1、64KB",
                          "64P 两级拓扑（L0=8，L1=8），CLOS 端口 P=1", page=8)
    ax.text(0.025, 0.79, "算法", fontsize=11, color=GRAY)
    ax.text(0.27, 0.79, "65,535 B", ha="center", fontsize=13, color=GRAY)
    ax.text(0.46, 0.79, "65,536 B", ha="center", fontsize=13, color=GRAY)
    rows = [("ParallelMeshNHR", "276.0μs", "276.0μs"),
            ("SoleNHR", "327.8μs", "254.0μs")]
    for i, (name, a, b) in enumerate(rows):
        y = 0.70 - i * 0.10
        ax.text(0.025, y, name, fontsize=11, color=INK)
        ax.text(0.27, y, a, ha="center", fontsize=14, fontweight="bold",
                color=PURPLE if i == 0 else ORANGE)
        ax.text(0.46, y, b, ha="center", fontsize=14, fontweight="bold",
                color=PURPLE if i == 0 else ORANGE)
    ax.text(0.025, 0.50, "胜者", fontsize=11, color=GRAY)
    ax.text(0.27, 0.50, "Parallel", ha="center", fontsize=13, fontweight="bold", color=PURPLE)
    ax.text(0.46, 0.50, "SoleNHR", ha="center", fontsize=13, fontweight="bold", color=ORANGE)

    ax.text(0.25, 0.375, "NHR 代价跳降 73.8μs\n（util 0.3 → 0.4288）",
            ha="center", fontsize=13, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDF2F2", ec=RED, lw=1.2))
    ax.text(0.25, 0.24, "perTransfer = 64KB × 32 = 2MB\n恰好跨过 CLOS util 断点",
            ha="center", fontsize=10.5, color=GRAY, linespacing=1.6)
    ax.text(0.25, 0.13, "翻转前差距 51.8μs\n翻转后反向差距 22.0μs",
            ha="center", fontsize=10.5, color=INK, linespacing=1.6)

    ax2 = fig.add_axes([0.55, 0.18, 0.41, 0.64])
    lo, hi = 60 * 1024, 68 * 1024
    zs = np.arange(lo, hi + 1, 16)
    zn = np.array([gc.alg_cost(*gc.sole_nhr(64, 1), int(d))[0] for d in zs]) * 1e6
    zp = np.array([gc.alg_cost(*gc.parallel(64, 1), int(d))[0] for d in zs]) * 1e6
    ax2.plot(zs / 1024, zn, color=ORANGE, lw=2.5, label="SoleNHR")
    ax2.plot(zs / 1024, zp, color=PURPLE, lw=2.5, label="ParallelMeshNHR")
    ax2.axvline(64, color=RED, ls="--", lw=1.5)
    ax2.annotate("64KB", xy=(64, 300), xytext=(65.0, 300), fontsize=11, color=RED,
                 fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax2.set_xlim(60, 68)
    ax2.set_ylim(238, 336)
    ax2.set_xlabel("per-rank dataSize（KB）", fontsize=10)
    ax2.set_ylabel("cost（μs）", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.tick_params(labelsize=8.5)
    ax2.set_title("64KB 附近放大", fontsize=11)

    ax.text(0.5, 0.07, "不是 bug，是分段常数 util 的固有边界；模型复刻，非上板实测。",
            ha="center", fontsize=13, color=INK, fontweight="bold")
    save(fig, "S8_corner_case.png")


# --------------------------------------------------------------------------- #
# S9 不是孤例
# --------------------------------------------------------------------------- #
def s9():
    fig, ax = slide_frame("不是孤例：7 组翻转 + 3 处模型/工程债",
                          "corner_case_scan.csv 的参数化扫描结果", page=9)
    table = [
        ("16P", "4", "512KB", "15.7μs"),
        ("32P", "2", "128KB", "36.3μs"),
        ("32P", "8", "1MB", "17.0μs"),
        ("64P", "1", "64KB", "73.8μs"),
        ("128P", "1", "64KB", "66.3μs"),
        ("256P", "1", "64KB", "37.4μs"),
        ("512P", "1", "64KB", "139.7μs"),
    ]
    ax.text(0.04, 0.78, "rank", fontsize=10, color=GRAY)
    ax.text(0.13, 0.78, "P", fontsize=10, color=GRAY)
    ax.text(0.19, 0.78, "per-rank 断点", fontsize=10, color=GRAY)
    ax.text(0.36, 0.78, "NHR 跳降", fontsize=10, color=GRAY)
    for i, (r, p, b, d) in enumerate(table):
        y = 0.70 - i * 0.075
        ax.text(0.04, y, r, fontsize=10.5, color=INK)
        ax.text(0.13, y, p, fontsize=10.5, color=INK)
        ax.text(0.19, y, b, fontsize=10.5, color=INK)
        ax.text(0.36, y, d, fontsize=10.5, fontweight="bold",
                color=RED if "73.8" in d else ORANGE)
    ax.text(0.04, 0.12, "方向均为 Parallel → SoleNHR", fontsize=9.5, color=GRAY)

    debts = [
        ("模型失真区", "16P、32~64MB：cost model 低估 SoleNHR，\n已用 opCustomCheck 硬排除（临时补丁）", RED),
        ("阈值一致性", "同一 4MB：2P detour/CCU_MS 用 >=，NDA 用 >\n同一 16MB：CCU_SCHED 960 用 >，AICPU/UBX 用 <", ORANGE),
        ("语义待确认", "AG-CLOS 钳制注释写“用 1MB 的 util”，\n实际命中 2MB 档 0.3（不是 0.1855）", PURPLE),
    ]
    for i, (h, t, c) in enumerate(debts):
        y = 0.665 - i * 0.165
        ax.add_patch(FancyBboxPatch((0.50, y - 0.062), 0.45, 0.124,
                                    boxstyle="round,pad=0.01,rounding_size=0.015",
                                    fc="white", ec=c, lw=1.8))
        ax.text(0.525, y + 0.030, h, fontsize=12, fontweight="bold", color=c)
        ax.text(0.525, y - 0.012, t, fontsize=9.2, color=INK, linespacing=1.6, va="center")

    ax.text(0.5, 0.045, "这些不是实现 bug，而是“模型近似 + 工程补丁”的真实边界。",
            ha="center", fontsize=12.5, color=INK, fontweight="bold")
    save(fig, "S9_不是孤例.png")


# --------------------------------------------------------------------------- #
# S10 跳变带
# --------------------------------------------------------------------------- #
def s10():
    fig, ax = slide_frame("规律：边界不是阈值，是“跳变带”",
                          "同一组参数下，Δcost 会在窄区间内多次穿越 0", page=10)
    axp = fig.add_axes([0.055, 0.15, 0.55, 0.62])
    ds = np.logspace(math.log10(8 * 1024), math.log10(64 * MIB), 900).astype(np.int64)
    delta = np.array([gc.costs(64, 1, int(d))[0] - gc.costs(64, 1, int(d))[1] for d in ds]) * 1e6
    axp.plot(ds, delta, color=INK, lw=2.0)
    axp.axhline(0, color=RED, lw=1.3, ls="--")
    axp.fill_between(ds, delta, 0, where=delta < 0, color=ORANGE, alpha=0.18, label="SoleNHR 胜")
    axp.fill_between(ds, delta, 0, where=delta >= 0, color=PURPLE, alpha=0.12, label="Parallel 胜")
    for b, _ in gc.CLOS_TAB:
        if b < 2 * MIB:
            continue
        xb = b / 32.0
        if ds[0] < xb < ds[-1]:
            axp.axvline(xb, color="#D1D5DB", ls=":", lw=1.0)
    sign = np.sign(delta)
    changes = np.where(sign[1:] != sign[:-1])[0]
    if len(changes):
        xf = ds[changes[0] + 1]
        axp.plot([xf], [0], "o", ms=7, mfc="white", mec=RED, mew=1.8, zorder=5)
    axp.set_yscale("symlog", linthresh=10, linscale=0.7)
    axp.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    axp.set_xscale("log")
    axp.set_xlabel("per-rank dataSize（对数轴）", fontsize=10)
    axp.set_ylabel("Δcost（μs，symlog）", fontsize=10)
    axp.legend(fontsize=8.5, loc="upper left")
    axp.tick_params(labelsize=8.5)
    axp.set_title("rank=64P，P=1", fontsize=11)

    notes = [
        ("多次穿越", "同一参数下 Δcost 在窄区间内\n多次变号，不能用一个阈值描述", RED),
        ("参数相关", "首次翻转点随 rank/P 变化：\n51、57、120、502KB 各不相同", BLUE),
        ("测试含义", "回归要覆盖断点 ±1 字节，\n而不是只测连续区间", GREEN),
    ]
    for i, (h, t, c) in enumerate(notes):
        y = 0.64 - i * 0.18
        ax.text(0.65, y, "● " + h, fontsize=12, fontweight="bold", color=c)
        ax.text(0.67, y - 0.045, t, fontsize=9.5, color=INK, linespacing=1.5, va="top")
    ax.text(0.5, 0.075, "这正是“决策树”在 cost model 下变成“曲线族 + 断点”的原因。",
            ha="center", fontsize=12.5, color=BLUE, fontweight="bold")
    save(fig, "S10_跳变带.png")


# --------------------------------------------------------------------------- #
# S11 工程含义
# --------------------------------------------------------------------------- #
def s11():
    fig, ax = slide_frame("工程含义：三类动作", "把边界从“现象”变成“可回归的约束”", page=12)
    cols = [
        ("回归", BLUE, [
            "断点 ±1 字节纳入 UT/ST",
            "覆盖 rank × P 的组合",
            "固定 dtype / in-place / cclBuffer",
        ]),
        ("模型", PURPLE, [
            "复核 util 表标定来源",
            "决定是否强制连续/单调",
            "修正 16P 区段系数",
        ]),
        ("治理", ORANGE, [
            "补丁登记 owner 与移除条件",
            "日志：CalcAlgCost / SelectMinCost",
            "DFX_CostTableDump 留痕",
        ]),
    ]
    for i, (h, c, items) in enumerate(cols):
        x = 0.18 + i * 0.31
        ax.add_patch(FancyBboxPatch((x - 0.13, 0.24), 0.26, 0.47,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="white", ec=c, lw=2.2))
        ax.text(x, 0.645, h, ha="center", fontsize=18, fontweight="bold", color=c)
        sub = {"回归": "把边界变成用例", "模型": "把系数变成假设", "治理": "把补丁变成债"}[h]
        ax.text(x, 0.595, sub, ha="center", fontsize=10.5, color=GRAY)
        for j, t in enumerate(items):
            ax.text(x - 0.105, 0.505 - j * 0.10, "•  " + t, fontsize=10.2, color=INK, va="top")
    ax.text(0.5, 0.14, "目标：让选路在边界上“可解释、可回归、可追踪”。",
            ha="center", fontsize=15, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="#9AA3AE", lw=1.0))
    save(fig, "S11_工程含义.png")


# --------------------------------------------------------------------------- #
# S12 结论
# --------------------------------------------------------------------------- #
def s12():
    fig, ax = slide_frame("结论与开放问题", page=13)
    ax.text(0.06, 0.78, "带走 3 件事", fontsize=16, fontweight="bold", color=INK)
    take = [
        ("选路不是一棵树", "是可行集 + 比价；阈值以补丁叶子存在", BLUE),
        ("边界会跳变", "util 阶梯让 cost 在断点处不连续", PURPLE),
        ("模型仍有失真区", "16P 硬补丁说明系数仍需标定", RED),
    ]
    for i, (h, t, c) in enumerate(take):
        y = 0.66 - i * 0.14
        ax.text(0.06, y, f"{i + 1}. {h}", fontsize=13, fontweight="bold", color=c)
        ax.text(0.08, y - 0.05, t, fontsize=10, color=GRAY)

    ax.text(0.52, 0.78, "开放问题", fontsize=16, fontweight="bold", color=INK)
    qs = [
        "AG-CLOS 钳制的预期：1MB 档 0.1855 还是 2MB 档 0.3？",
        "16P 32~64MB 补丁：何时移除、谁负责？",
        "CLOS / UBX 是否迁到 cost model，还是长期双轨？",
    ]
    for i, q in enumerate(qs):
        ax.text(0.52, 0.67 - i * 0.10, f"{i + 1}. {q}", fontsize=10.5, color=INK)
    ax.text(0.52, 0.31, "下一步：把 64KB / 128KB / 256KB / 512KB / 1MB\n等 util 断点加入选路回归用例。",
            fontsize=11.5, color=INK, linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="#9AA3AE", lw=1.0))
    ax.text(0.5, 0.12, "一句话：选路是可行集上的比价；边界在 util 断点上，必须按断点测。",
            ha="center", fontsize=13.5, color=BLUE, fontweight="bold")
    save(fig, "S12_结论.png")


if __name__ == "__main__":
    s1()
    s2()
    s3()
    s4()
    # s5() 已从分享版移除（多段算法举例太绕），函数保留备用
    # s5()
    s6()
    s7()
    s8()
    s9()
    s10()
    # s11() / s12() 已从分享版移除（工程含义、结论），函数保留备用
    # s11()
    # s12()
    print("done")
