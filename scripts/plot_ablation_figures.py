# -*- coding: utf-8 -*-
"""
plot_ablation_figures.py —— 消融与泛化结果可视化（论文图表）

三张图：
    图1 消融森林图    : 8 配置 ΔF1 均值±std（CSIC 5-seed），显著性星号
                        -> ablation/figures/ablation_forest.png
    图2 跨域对比图    : 7 个 LOO 配置的 CSIC vs ATRDF ΔF1 并排
                        -> ablation/figures/cross_dataset_loo.png
    图3 泛化阶梯图    : 从同域上界到零样本崩塌的 F1 递减叙事
                        -> results/atrdf/generalization_ladder.png

数据来源（均为已落盘实验结果，本脚本只做可视化、不做任何计算实验）：
    ablation/data/ablation_multiseed_raw.csv       5 seeds × 8 配置逐次结果
    ablation/data/ablation_multiseed_summary.csv   Δ 均值与 p 值
    ablation/data/atrdf_leave_one_out.csv          ATRDF 同协议 LOO
    results/atrdf/atrdf_cross_subset.csv           跨子集检验（RF 行）
    results/atrdf/atrdf_metrics.csv                零样本迁移

运行方式（项目根目录）：
    python scripts/plot_ablation_figures.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import RESULTS_DIR

ABLATION_DATA = PROJECT_ROOT / "ablation" / "data"
ABLATION_FIGS = PROJECT_ROOT / "ablation" / "figures"
ATRDF_DIR = RESULTS_DIR / "atrdf"

# ---------- 全局样式：论文级、中文可显示、300dpi ----------
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

C_MAIN = "#2b6cb0"     # 主色（蓝）
C_NEUTRAL = "#a0aec0"  # 不显著（灰）
C_CSIC = "#2b6cb0"     # CSIC
C_ATRDF = "#dd6b20"    # ATRDF（橙）
C_SAME = "#2f855a"     # 同域（绿）
C_ZERO = "#c53030"     # 零样本（红）

BONFERRONI_ALPHA = 0.05 / 7   # 每指标 7 次比较的校正阈值


def sig_mark(p: float) -> str:
    if p < BONFERRONI_ALPHA:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ============================================================
# 图1 消融森林图（CSIC 5-seed：ΔF1 均值 ± std + 显著性）
# ============================================================
def fig1_forest() -> None:
    raw = pd.read_csv(ABLATION_DATA / "ablation_multiseed_raw.csv")
    summ = pd.read_csv(ABLATION_DATA / "ablation_multiseed_summary.csv")
    base = raw[raw["配置"] == "完整方案(基线)"].set_index("seed")["F1_攻击"]
    order = ["LOO-2_去规则特征", "LOO-1_去TF-IDF", "LOO-6_去编码熵族",
             "LOO-5_去特殊字符族", "LOO-4_去关键词族", "LOO-3_单轮解码",
             "LOO-7_去类权重平衡"]
    deltas, errs, marks, colors = [], [], [], []
    for name in order:
        g = raw[raw["配置"] == name].set_index("seed")["F1_攻击"]
        d = (g - base) * 100                      # 同 seed 配对差（百分点）
        deltas.append(d.mean())
        errs.append(d.std(ddof=1))
        p = float(summ[summ["配置"] == name]["p值_F1_攻击"].iloc[0])
        m = sig_mark(p)
        marks.append(m)
        colors.append(C_MAIN if m == "**" else (C_NEUTRAL if m == "n.s." else "#63b3ed"))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(order))
    ax.barh(y, deltas, xerr=errs, color=colors, height=0.62,
            error_kw=dict(ecolor="#4a5568", lw=1.2, capsize=3))
    for yi, (d, e, m) in enumerate(zip(deltas, errs, marks)):
        ax.text(d + e + 0.35, yi, f"{d:+.2f} {m}", va="center", fontsize=9,
                color="#2d3748")
    ax.axvline(0, color="#2d3748", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", "\n", 1) for n in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("移除该组件后攻击类 F1 的变化（百分点，5 seeds 均值 ± std）")
    ax.set_title("细粒度消融：逐项移除组件对 F1 的影响（CSIC 2010，随机森林）",
                 fontsize=11, pad=10)
    ax.text(0.99, 0.02, "** Bonferroni 校正后显著(p<0.0071)   * p<0.05   n.s. 不显著",
            transform=ax.transAxes, ha="right", fontsize=8, color="#718096")
    fig.savefig(ABLATION_FIGS / "ablation_forest.png")
    plt.close(fig)
    print("√ 图1 已保存: ablation/figures/ablation_forest.png")


# ============================================================
# 图2 CSIC vs ATRDF 跨域对比（LOO ΔF1 并排）
# ============================================================
def fig2_cross_domain() -> None:
    summ = pd.read_csv(ABLATION_DATA / "ablation_multiseed_summary.csv")
    atrdf = pd.read_csv(ABLATION_DATA / "atrdf_leave_one_out.csv")
    order = ["LOO-1_去TF-IDF", "LOO-2_去规则特征", "LOO-3_单轮解码",
             "LOO-4_去关键词族", "LOO-5_去特殊字符族", "LOO-6_去编码熵族",
             "LOO-7_去类权重平衡"]
    short = ["去TF-IDF", "去规则特征", "单轮解码", "去关键词族",
             "去特殊字符族", "去编码熵族", "去类权重"]
    csic = [float(summ[summ["配置"] == n]["ΔF1_攻击_pp"].iloc[0]) for n in order]
    atrdf_map = atrdf.set_index("配置")["ΔF1_攻击_pp"]
    atr = [float(atrdf_map[n]) for n in order]
    inconsistent = {"LOO-1_去TF-IDF", "LOO-7_去类权重平衡"}

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    b1 = ax.bar(x - w / 2, csic, w, color=C_CSIC, label="CSIC 2010（5-seed 均值）")
    b2 = ax.bar(x + w / 2, atr, w, color=C_ATRDF, label="ATRDF ds3（单 seed 点估计）")
    for xi, n in zip(x, order):
        if n in inconsistent:                    # 高亮跨域不一致项
            ax.axvspan(xi - 0.5, xi + 0.5, color="#fefcbf", alpha=0.55, zorder=0)
    for bars in (b1, b2):
        for r in bars:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2,
                    v + (0.35 if v >= 0 else -0.95), f"{v:+.2f}",
                    ha="center", fontsize=8, color="#2d3748")
    ax.axhline(0, color="#2d3748", lw=1)
    ax.set_ylim(-16.5, 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("移除组件后 ΔF1（百分点）")
    ax.set_title("组件贡献的跨数据集对比：两处不一致（黄底）均有明确机制解释",
                 fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.text(0.99, 0.42, "黄底 ① TF-IDF 增益仅出现在正常流量复杂的 CSIC\n"
            "        ② 类权重作用方向由类别分布决定",
            transform=ax.transAxes, ha="right", fontsize=8, color="#744210")
    fig.savefig(ABLATION_FIGS / "cross_dataset_loo.png")
    plt.close(fig)
    print("√ 图2 已保存: ablation/figures/cross_dataset_loo.png")


# ============================================================
# 图3 泛化阶梯图（同域上界 -> 无泄漏跨子集 -> 零样本崩塌）
# ============================================================
def fig3_ladder() -> None:
    # 全部为已落盘实验结果中的攻击类 F1（出处见注释）
    items = [
        ("ATRDF 同子集\n重训（上界）", 0.9840, C_SAME,   "验证集, ablation LOO 基线"),
        ("CSIC 2010\n5-seed 均值",    0.8688, C_SAME,   "验证集, multiseed 基线"),
        ("跨子集\nds3→ds1",           0.9358, C_MAIN,   "测试集, RF"),
        ("跨子集\nds3→ds2",           0.9282, C_MAIN,   "测试集, RF"),
        ("跨子集\nds3→ds4",           0.4859, C_MAIN,   "测试集, RF（正常风格差异大）"),
        ("零样本迁移\nCSIC→ATRDF",    0.2224, C_ZERO,   "测试集, 不重训"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(items))
    bars = ax.bar(x, [v for _, v, _, _ in items],
                  color=[c for _, _, c, _ in items], width=0.6)
    for xi, (label, v, _, note) in zip(x, items):
        ax.text(xi, v + 0.012, f"{v:.4f}", ha="center", fontsize=9,
                color="#2d3748", fontweight="bold")
    ax.axhline(1.0, color="#cbd5e0", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, *_ in items], fontsize=8.6)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("攻击类 F1")
    ax.set_title("泛化成绩阶梯：同域上界 → 无泄漏跨子集 → 零样本崩塌", fontsize=11, pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_SAME),
               plt.Rectangle((0, 0), 1, 1, color=C_MAIN),
               plt.Rectangle((0, 0), 1, 1, color=C_ZERO)]
    ax.legend(handles, ["同域（重训）", "无泄漏跨子集（重训）", "零样本（不重训）"],
              frameon=False, fontsize=9, loc="upper right")
    fig.savefig(ATRDF_DIR / "generalization_ladder.png")
    plt.close(fig)
    print("√ 图3 已保存: results/atrdf/generalization_ladder.png")


def main() -> None:
    ABLATION_FIGS.mkdir(parents=True, exist_ok=True)
    ATRDF_DIR.mkdir(parents=True, exist_ok=True)
    fig1_forest()
    fig2_cross_domain()
    fig3_ladder()
    print("\n全部图表生成完毕（300dpi）。")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
