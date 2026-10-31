# -*- coding: utf-8 -*-
"""
run_ablation_multiseed.py —— 论文级多 seed 重复消融（统计支撑版）

动机：
    run_leave_one_out.py 的结论基于单次划分（seed=42），"Δ<1pp 视为噪声"
    的判定本身缺少统计支撑。本脚本用多个随机种子重复完整消融流程，
    为每个组件的贡献给出 均值±标准差 与配对 t 检验 p 值。

设计：
    - 5 个划分种子 seeds = [42, 1337, 2024, 7, 99]；
      每个种子独立做 64/16/20 三段划分（验证集评估，测试集不触碰），
      并把随机森林的 random_state 同步设为该种子（随机性全部由 seed 控制）。
    - 8 个配置 = 完整方案 + LOO-1~7（与 run_leave_one_out.py 完全同口径）；
      LOO-4/5/6 复用同一种子下完整方案的特征矩阵做列置零（词表一致），
      LOO-1/2/3 各自重新 fit 管线，LOO-7 仅关类权重。
    - 统计：每配置 mean±std；Δ = 配置均值 - 完整方案均值（同一 seed 内配对）；
      scipy.stats.ttest_rel 配对 t 检验（n=5）；|Δ| 与 p 联合判读。

运行方式（项目根目录）：
    python scripts/run_ablation_multiseed.py    # 约需 1 小时

产出（写入 ablation/data/）：
    ablation_multiseed_raw.csv       每个 seed × 配置的逐次结果
    ablation_multiseed_summary.csv   均值±标准差 + Δ + p 值汇总表
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.features import FeaturePipeline
from src.models import build_models
from scripts.run_leave_one_out import RULE_GROUPS, evaluate, zero_out_columns

ABLATION_DATA_DIR = PROJECT_ROOT / "ablation" / "data"
SEEDS = [42, 1337, 2024, 7, 99]

# 与 run_leave_one_out.py 完全一致的配置清单
SWITCH_CONFIGS = [
    ("LOO-1_去TF-IDF", dict(use_rules=True, use_tfidf=False, decode_rounds=3)),
    ("LOO-2_去规则特征", dict(use_rules=False, use_tfidf=True, decode_rounds=3)),
    ("LOO-3_单轮解码", dict(use_rules=True, use_tfidf=True, decode_rounds=1)),
]


def one_seed(seed: int) -> list:
    """跑一个种子的全部 8 个配置，返回逐行结果。"""
    df = load_dataset(rebuild=False)
    train_df, _test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=seed)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    del df
    del _test

    rf_balanced = build_models(y_train)["随机森林"]
    rf_plain = clone(rf_balanced).set_params(class_weight=None,
                                             random_state=seed)

    rows = []

    def run(name, Xt, Xv, model):
        model.fit(Xt, y_train)
        r = evaluate(y_val, model.predict_proba(Xv)[:, 1])
        rows.append({"seed": seed, "配置": name, "特征维度": Xv.shape[1],
                     **{k: v for k, v in r.items() if not k.startswith("_")}})
        print(f"    seed={seed} {name}: recall={r['召回率_攻击']:.4f} "
              f"F1={r['F1_攻击']:.4f} FPR={r['FPR_误报率']:.4f}", flush=True)

    # 完整方案（基线）
    pipe_full = FeaturePipeline(use_rules=True, use_tfidf=True, decode_rounds=3)
    X_train = pipe_full.fit_transform(train_df)
    X_val = pipe_full.transform(val_df)
    rf = clone(rf_balanced).set_params(random_state=seed)
    run("完整方案(基线)", X_train, X_val, rf)

    # LOO-1/2/3：重新 fit 管线
    for name, kwargs in SWITCH_CONFIGS:
        pipe = FeaturePipeline(**kwargs)
        run(name, pipe.fit_transform(train_df), pipe.transform(val_df),
            clone(rf_balanced).set_params(random_state=seed))
        del pipe

    # LOO-4/5/6：列置零复用基线特征矩阵
    rule_names = pipe_full.rule_feature_names_
    for name, (group, removed) in zip(
            ("LOO-4_去关键词族", "LOO-5_去特殊字符族", "LOO-6_去编码熵族"),
            RULE_GROUPS.items()):
        col_idx = [rule_names.index(n) for n in removed]
        run(name, zero_out_columns(X_train, col_idx),
            zero_out_columns(X_val, col_idx),
            clone(rf_balanced).set_params(random_state=seed))

    # LOO-7：去类权重
    run("LOO-7_去类权重平衡", X_train, X_val, rf_plain)
    return rows


def main() -> None:
    print("#" * 70)
    print("#  多 seed 重复消融：5 seeds × 8 配置（随机森林 / 验证集评估）")
    print("#" * 70, flush=True)

    ABLATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n===== seed {seed}（{i}/{len(SEEDS)}）=====", flush=True)
        t0 = time.time()
        all_rows += one_seed(seed)
        print(f"  —— seed {seed} 完成，耗时 {time.time() - t0:.0f}s", flush=True)

    raw = pd.DataFrame(all_rows)
    p_raw = ABLATION_DATA_DIR / "ablation_multiseed_raw.csv"
    raw.to_csv(p_raw, index=False, encoding="utf-8-sig")
    print(f"\n√ 逐次结果已保存: {p_raw}")

    # ---------- 汇总：mean±std + Δ + 配对 t 检验 ----------
    base = raw[raw["配置"] == "完整方案(基线)"].set_index("seed")
    metrics = ["召回率_攻击", "精确率_攻击", "F1_攻击", "FPR_误报率", "准确率"]
    rows = []
    for name, g in raw.groupby("配置"):
        g = g.set_index("seed").loc[SEEDS]
        row = {"配置": name}
        for m in metrics:
            row[f"{m}_均值"] = g[m].mean()
            row[f"{m}_标准差"] = g[m].std(ddof=1)
        # 与完整方案同 seed 配对差值 → Δ 均值 + 配对 t 检验
        if name != "完整方案(基线)":
            for m in ["F1_攻击", "召回率_攻击", "FPR_误报率"]:
                diff = g[m] - base.loc[SEEDS, m]
                row[f"Δ{m}_pp"] = diff.mean() * 100
                t, p = stats.ttest_rel(g[m], base.loc[SEEDS, m])
                row[f"p值_{m}"] = p
        rows.append(row)
    summary = pd.DataFrame(rows)

    p_sum = ABLATION_DATA_DIR / "ablation_multiseed_summary.csv"
    summary.to_csv(p_sum, index=False, encoding="utf-8-sig")
    print(f"√ 汇总已保存: {p_sum}\n")
    cols = (["配置"] + [f"{m}_均值" for m in metrics[:4]]
            + ["ΔF1_攻击_pp", "p值_F1_攻击"])
    print("===== 汇总（mean±std 见 CSV；此处展示核心列）=====")
    view = summary[cols].copy()
    for c in view.columns[1:]:
        view[c] = view[c].map(lambda v: f"{v:.4f}")
    print(view.to_string(index=False))
    print("\n判读规则：|ΔF1| > 2×基线间标准差 且 p<0.05 → 组件贡献显著；")
    print("          p≥0.05 → 如实报告为'未观察到显著差异'（冗余/噪声）。")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
