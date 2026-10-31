# -*- coding: utf-8 -*-
"""
run_ablation.py —— 消融实验：量化每个组件的独立贡献（评审缺口 2）

背景：
    完整方案 = 40 维规则特征 + n-gram(1~3) TF-IDF + 3 轮递归解码。
    只有"完整方案成绩"无法说明各组件分别带来了什么收益，
    本脚本用四组对照实验回答"拆掉某个组件性能掉多少"。

四组配置（统一使用随机森林，其余设置与 run.py 完全一致）：
    A 仅规则特征      : use_rules=True,  use_tfidf=False
    B 仅 TF-IDF       : use_rules=False, use_tfidf=True
    C 完整方案（基线）: use_rules=True,  use_tfidf=True, decode_rounds=3
    D 单轮解码对照    : 同 C 但 decode_rounds=1（验证递归解码的价值）

划分与 run.py 相同（seed=42 的 64/16/20 三段划分），在验证集上评估，
避免触碰测试集。

运行方式（项目根目录）：
    python scripts/run_ablation.py

产出：
    ablation/data/ablation_results.csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset, RESULTS_DIR
from src.features import FeaturePipeline
from src.models import build_models


def main() -> None:
    """四组消融配置：同一切分下训练随机森林并在验证集上评估。"""
    print("#" * 70)
    print("#  消融实验：各组件贡献量化（随机森林 / 验证集评估）")
    print("#" * 70)

    # ---------- 数据：与 run.py 完全一致的三段划分（保证可比） ----------
    print("\n[1/3] 加载数据并做 64/16/20 三段划分 ...")
    df = load_dataset(rebuild=False)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42)
    y_val = val_df["label"].to_numpy()
    print(f"  训练 {len(train_df)} / 验证 {len(val_df)}（测试集本次不使用）")

    # ---------- 四组消融配置 ----------
    configs = [
        ("A_仅规则特征",   dict(use_rules=True,  use_tfidf=False)),
        ("B_仅TF-IDF",     dict(use_rules=False, use_tfidf=True)),
        ("C_完整方案",     dict(use_rules=True,  use_tfidf=True)),
        ("D_单轮解码对照", dict(use_rules=True,  use_tfidf=True, decode_rounds=1)),
    ]

    print("\n[2/3] 逐组训练 + 验证集评估 ...")
    rows = []
    y_train = train_df["label"].to_numpy()
    rf = build_models(y_train)["随机森林"]  # 统一用随机森林做消融（当前最优模型）
    for name, kwargs in configs:
        print(f"\n  —— 配置 {name}: {kwargs}")
        pipe = FeaturePipeline(**kwargs)
        X_train = pipe.fit_transform(train_df)
        X_val = pipe.transform(val_df)
        m = rf
        m.fit(X_train, y_train)
        # 评估：与 evaluate.evaluate_model 相同口径（恶意类为正类）
        proba = m.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        rows.append({
            "配置": name,
            "特征维度": X_val.shape[1],
            "准确率": (tp + tn) / (tp + tn + fp + fn),
            "召回率_攻击": tp / max(tp + fn, 1),
            "精确率_攻击": tp / max(tp + fp, 1),
            "F1_攻击": 2 * tp / max(2 * tp + fp + fn, 1),
            "FPR_误报率": fp / max(fp + tn, 1),
            "FNR_漏报率": fn / max(fn + tp, 1),
        })
        print(f"    维度={X_val.shape[1]}  recall={rows[-1]['召回率_攻击']:.4f}  "
              f"acc={rows[-1]['准确率']:.4f}  FPR={rows[-1]['FPR_误报率']:.4f}")

    # ---------- 保存 ----------
    print("\n[3/3] 保存消融结果 ...")
    out = pd.DataFrame(rows)
    save_path = PROJECT_ROOT / "ablation" / "data" / "ablation_results.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  √ 已保存: {save_path}")
    print("\n===== 消融结果汇总（验证集）=====")
    print(out.to_string(index=False,
                        float_format=lambda v: f"{v:.4f}"))
    print("\n解读要点：")
    print("  · C 与 A/B 的差值 = TF-IDF / 规则特征各自的增益")
    print("  · C 与 D 的差值 = 多轮递归解码对编码变体的增益")
    print("  · 若 C 显著优于 D，说明抗编码绕过设计有效（论文级证据）")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
