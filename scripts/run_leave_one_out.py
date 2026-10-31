# -*- coding: utf-8 -*-
"""
run_leave_one_out.py —— 细粒度消融实验（Leave-One-Out，评审缺口 2 深化）

背景：
    run_ablation.py 已回答"规则特征 / TF-IDF / 多轮解码"三大组件的贡献。
    本脚本进一步下沉到组件内部，逐个移除更细粒度的设计项，
    回答"40 维规则特征里哪几族真正有贡献、类权重平衡是否必要"。

七个 LOO 配置（统一随机森林，其余设置与 run.py 完全一致）：
    LOO-1 去 TF-IDF        : use_rules=True,  use_tfidf=False（= 消融 A）
    LOO-2 去规则特征       : use_rules=False, use_tfidf=True （= 消融 B）
    LOO-3 单轮解码         : decode_rounds=1                 （= 消融 D）
    LOO-4 去关键词命中族   : 置零 n_*_keywords 共 5 列（A6）
    LOO-5 去特殊字符族     : 置零 count_* + special_char_density 共 15 列（A3）
    LOO-6 去编码与熵族     : 置零编码特征 + url_entropy 共 6 列（A4+A5）
    LOO-7 去类权重平衡     : 随机森林 class_weight=None，其余同完整方案

    其中 LOO-4/5/6 采用"列置零掩码"实现：完整方案只 fit 一次特征流水线，
    将被移除特征族的列置 0 后重训随机森林（常数为 0 的列不会产生任何分裂，
    统计上等价于删除该族特征），保证所有配置使用同一份 TF-IDF 词表，
    对照更严格。LOO-1/2/3 因特征矩阵本身不同，需各自重新 fit。

诚实性约定（写入 README）：
    - 与完整方案差值 < 1 个百分点视为噪声；
    - 移除后性能无下降的组件如实报告为"在该数据集上冗余"。

划分与 run.py 相同（seed=42 的 64/16/20 三段划分），在验证集上评估。

运行方式（项目根目录）：
    python scripts/run_leave_one_out.py

产出：
    ablation/data/ablation_leave_one_out.csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset, RESULTS_DIR
from src.features import FeaturePipeline
from src.models import build_models

# 40 维规则特征的三族划分（列名与 src/features.py 提取顺序一致）：
#   A3 特殊字符计数与密度（15 维）
#   A4 编码特征 + A5 URL 信息熵（6 维）
#   A6 攻击关键词命中（5 维）
# （A1 方法 one-hot、A2 长度结构为基础结构性信息，不做 LOO）
RULE_GROUPS = {
    "关键词命中族(A6)": [
        "n_sqli_keywords", "n_xss_keywords",
        "n_traversal_keywords", "n_cmd_keywords", "n_total_keywords",
    ],
    "特殊字符族(A3)": [
        "count_quote_s", "count_quote_d", "count_lt", "count_gt",
        "count_semicolon", "count_lparen", "count_rparen",
        "count_backtick", "count_pipe", "count_amp", "count_dollar",
        "count_backslash", "count_double_dash", "count_slash_star",
        "special_char_density",
    ],
    "编码与熵族(A4+A5)": [
        "n_encoded", "encoded_ratio", "double_encoded",
        "n_consecutive_slashes", "n_dotdot_slash", "url_entropy",
    ],
}


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """与 run_ablation.py 完全相同的口径：恶意类为正类，阈值 0.5。"""
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "准确率": (tp + tn) / (tp + tn + fp + fn),
        "召回率_攻击": tp / max(tp + fn, 1),
        "精确率_攻击": tp / max(tp + fp, 1),
        "F1_攻击": 2 * tp / max(2 * tp + fp + fn, 1),
        "FPR_误报率": fp / max(fp + tn, 1),
        "FNR_漏报率": fn / max(fn + tp, 1),
    }


def zero_out_columns(X, col_idx: list):
    """把稀疏矩阵的指定列置 0（等价于删除该族特征后重训）。"""
    Xm = X.tocsc()
    Xm[:, col_idx] = 0
    Xm = Xm.tocsr()
    Xm.eliminate_zeros()
    return Xm


def main() -> None:
    print("#" * 70)
    print("#  细粒度消融（Leave-One-Out）：随机森林 / 验证集评估")
    print("#" * 70)

    # ---------- 数据：与 run.py 完全一致的三段划分（保证可比） ----------
    print("\n[1/3] 加载数据并做 64/16/20 三段划分 ...")
    df = load_dataset(rebuild=False)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    print(f"  训练 {len(train_df)} / 验证 {len(val_df)}（测试集本次不使用）")

    # 统一用随机森林（当前最优模型）；LOO-7 用同一超参但关闭类权重
    rf_balanced = build_models(y_train)["随机森林"]
    rf_plain = clone(rf_balanced).set_params(class_weight=None)

    rows = []
    print("\n[2/3] 逐配置训练 + 验证集评估 ...")

    # ---------- 完整方案（基线，供计算 Δ） ----------
    print("\n  —— 完整方案（基线）")
    pipe_full = FeaturePipeline(use_rules=True, use_tfidf=True, decode_rounds=3)
    X_train = pipe_full.fit_transform(train_df)
    X_val = pipe_full.transform(val_df)
    m = rf_balanced
    m.fit(X_train, y_train)
    base = evaluate(y_val, m.predict_proba(X_val)[:, 1])
    rows.append({"配置": "完整方案(基线)", "移除组件": "无",
                 "特征维度": X_val.shape[1], **base})
    print(f"    维度={X_val.shape[1]}  F1={base['F1_攻击']:.4f}  "
          f"recall={base['召回率_攻击']:.4f}  FPR={base['FPR_误报率']:.4f}")

    # 规则特征列在拼接矩阵中的列号（rules 块在最前，顺序 = rule_feature_names_）
    rule_names = pipe_full.rule_feature_names_
    missing = [n for g in RULE_GROUPS.values() for n in g if n not in rule_names]
    if missing:
        raise RuntimeError(f"特征分族列名与实际不符: {missing}")

    # ---------- LOO-1/2/3：复用流水线开关（特征矩阵不同，重新 fit） ----------
    switch_configs = [
        ("LOO-1_去TF-IDF", "TF-IDF 文本特征(5000维)",
         dict(use_rules=True, use_tfidf=False, decode_rounds=3)),
        ("LOO-2_去规则特征", "40维规则/统计特征",
         dict(use_rules=False, use_tfidf=True, decode_rounds=3)),
        ("LOO-3_单轮解码", "递归解码(3轮->1轮)",
         dict(use_rules=True, use_tfidf=True, decode_rounds=1)),
    ]
    for name, removed, kwargs in switch_configs:
        print(f"\n  —— {name}: {kwargs}")
        pipe = FeaturePipeline(**kwargs)
        Xt = pipe.fit_transform(train_df)
        Xv = pipe.transform(val_df)
        m = rf_balanced
        m.fit(Xt, y_train)
        r = evaluate(y_val, m.predict_proba(Xv)[:, 1])
        rows.append({"配置": name, "移除组件": removed,
                     "特征维度": Xv.shape[1], **r})
        print(f"    维度={Xv.shape[1]}  F1={r['F1_攻击']:.4f}  "
              f"recall={r['召回率_攻击']:.4f}  FPR={r['FPR_误报率']:.4f}")

    # ---------- LOO-4/5/6：特征分族置零掩码（复用基线特征矩阵） ----------
    for name, (group, removed) in zip(
            ("LOO-4_去关键词族", "LOO-5_去特殊字符族", "LOO-6_去编码熵族"),
            RULE_GROUPS.items()):
        print(f"\n  —— {name}: 置零 {group}（{len(removed)} 列）")
        col_idx = [rule_names.index(n) for n in removed]
        Xt = zero_out_columns(X_train, col_idx)
        Xv = zero_out_columns(X_val, col_idx)
        m = rf_balanced
        m.fit(Xt, y_train)
        r = evaluate(y_val, m.predict_proba(Xv)[:, 1])
        rows.append({"配置": name, "移除组件": group,
                     "特征维度": Xv.shape[1], **r})
        print(f"    维度={Xv.shape[1]}  F1={r['F1_攻击']:.4f}  "
              f"recall={r['召回率_攻击']:.4f}  FPR={r['FPR_误报率']:.4f}")

    # ---------- LOO-7：去掉 class_weight="balanced"（复用基线特征矩阵） ----------
    print("\n  —— LOO-7_去类权重平衡: class_weight=None")
    m = rf_plain
    m.fit(X_train, y_train)
    r = evaluate(y_val, m.predict_proba(X_val)[:, 1])
    rows.append({"配置": "LOO-7_去类权重平衡", "移除组件": "class_weight=balanced",
                 "特征维度": X_val.shape[1], **r})
    print(f"    维度={X_val.shape[1]}  F1={r['F1_攻击']:.4f}  "
          f"recall={r['召回率_攻击']:.4f}  FPR={r['FPR_误报率']:.4f}")

    # ---------- 计算相对完整方案的 Δ（百分点） ----------
    print("\n[3/3] 保存结果 ...")
    out = pd.DataFrame(rows)
    for src, dst in (("F1_攻击", "ΔF1_攻击_pp"),
                     ("召回率_攻击", "Δ召回率_攻击_pp"),
                     ("FPR_误报率", "ΔFPR_误报率_pp")):
        out[dst] = (out[src] - base[src]) * 100.0

    save_path = PROJECT_ROOT / "ablation" / "data" / "ablation_leave_one_out.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  √ 已保存: {save_path}")

    print("\n===== 细粒度消融结果汇总（验证集，Δ 为相对完整方案的百分点差）=====")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n解读要点：")
    print("  · Δ 为负 = 移除该组件后性能下降（组件有正贡献）")
    print("  · |Δ| < 1pp 视为噪声；移除后无下降的组件如实报告为该数据集上冗余")
    print("  · LOO-7 若 FPR 上升明显，说明类权重平衡主要在压误报")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
