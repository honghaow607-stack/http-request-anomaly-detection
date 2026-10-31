# -*- coding: utf-8 -*-
"""
run_leave_one_out_atrdf.py —— ATRDF 上的 LOO 消融（跨数据集重复验证）

动机：
    ablation/report.md 的 LOO 结论（规则特征=精度+稳定性支柱、TF-IDF 次之、
    关键词/特殊字符族冗余）目前只在 CSIC 2010 上成立。本脚本在 ATRDF
    dataset_3（REST API 风格、恶意占比 12.4%、含 4 类 CSIC 没有的攻击）
    上重复完全相同的 8 配置消融，验证"组件贡献结构"是否跨数据集成立。

协议与 CSIC 版 run_leave_one_out.py 完全一致：
    同 8 配置（完整方案 + LOO-1~7）、同 LOO-4/5/6 列置零掩码、
    同 seed=42 三段划分、验证集评估（测试集不触碰）、随机森林。
    唯一差异：数据源为 ATRDF ds3（去重后 129,674 条）。
    * 单 seed 点估计口径（ATRDF 数据量大，5-seed 版本成本约 5 倍，
      暂不重复；CSIC 版已有多 seed 统计支撑）。

运行方式（项目根目录）：
    python scripts/run_leave_one_out_atrdf.py    # 约 10~20 分钟

产出：
    ablation/data/atrdf_leave_one_out.csv
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.features import FeaturePipeline
from src.models import build_models
from scripts.run_atrdf_eval import ATRDF_MAIN, load_atrdf
from scripts.run_leave_one_out import RULE_GROUPS, evaluate, zero_out_columns

OUT_PATH = PROJECT_ROOT / "ablation" / "data" / "atrdf_leave_one_out.csv"


def main() -> None:
    print("#" * 70)
    print("#  ATRDF(ds3) 上的 LOO 消融：8 配置 / 随机森林 / 验证集评估")
    print("#" * 70, flush=True)

    # ---------- 数据：与 run_atrdf_eval.py 一致的解析 + 去重 + 三段划分 ----------
    df = load_atrdf(ATRDF_MAIN)
    print("\n[1/3] 64/16/20 三段划分（seed=42）...", flush=True)
    train_df, _test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    print(f"  训练 {len(train_df)} / 验证 {len(val_df)}（测试集不触碰）", flush=True)

    rf_balanced = build_models(y_train)["随机森林"]
    rf_plain = clone(rf_balanced).set_params(class_weight=None)

    rows = []
    print("\n[2/3] 逐配置训练 + 验证集评估 ...", flush=True)

    def run(name, removed, Xt, Xv, model):
        model.fit(Xt, y_train)
        r = evaluate(y_val, model.predict_proba(Xv)[:, 1])
        rows.append({"配置": name, "移除组件": removed,
                     "特征维度": Xv.shape[1], **r})
        print(f"    维度={Xv.shape[1]}  F1={r['F1_攻击']:.4f}  "
              f"recall={r['召回率_攻击']:.4f}  FPR={r['FPR_误报率']:.4f}",
              flush=True)

    # 完整方案（基线）
    t0 = time.time()
    print("\n  —— 完整方案（基线）", flush=True)
    pipe_full = FeaturePipeline(use_rules=True, use_tfidf=True, decode_rounds=3)
    X_train = pipe_full.fit_transform(train_df)
    X_val = pipe_full.transform(val_df)
    run("完整方案(基线)", "无", X_train, X_val, rf_balanced)
    base = rows[0]

    rule_names = pipe_full.rule_feature_names_
    missing = [n for g in RULE_GROUPS.values() for n in g if n not in rule_names]
    if missing:
        raise RuntimeError(f"特征分族列名与实际不符: {missing}")

    # LOO-1/2/3：管线开关（特征矩阵不同，重新 fit）
    switch_configs = [
        ("LOO-1_去TF-IDF", "TF-IDF 文本特征",
         dict(use_rules=True, use_tfidf=False, decode_rounds=3)),
        ("LOO-2_去规则特征", "40维规则/统计特征",
         dict(use_rules=False, use_tfidf=True, decode_rounds=3)),
        ("LOO-3_单轮解码", "递归解码(3轮->1轮)",
         dict(use_rules=True, use_tfidf=True, decode_rounds=1)),
    ]
    for name, removed, kwargs in switch_configs:
        print(f"\n  —— {name}: {kwargs}", flush=True)
        pipe = FeaturePipeline(**kwargs)
        run(name, removed, pipe.fit_transform(train_df),
            pipe.transform(val_df), rf_balanced)
        del pipe

    # LOO-4/5/6：列置零掩码（复用基线特征矩阵，同一 TF-IDF 词表）
    for name, (group, removed) in zip(
            ("LOO-4_去关键词族", "LOO-5_去特殊字符族", "LOO-6_去编码熵族"),
            RULE_GROUPS.items()):
        print(f"\n  —— {name}: 置零 {group}（{len(removed)} 列）", flush=True)
        col_idx = [rule_names.index(n) for n in removed]
        run(name, group, zero_out_columns(X_train, col_idx),
            zero_out_columns(X_val, col_idx), rf_balanced)

    # LOO-7：去类权重
    print("\n  —— LOO-7_去类权重平衡: class_weight=None", flush=True)
    run("LOO-7_去类权重平衡", "class_weight=balanced",
        X_train, X_val, rf_plain)

    # ---------- Δ 百分点 + 保存 ----------
    print(f"\n[3/3] 保存结果（总耗时 {time.time() - t0:.0f}s）...", flush=True)
    out = pd.DataFrame(rows)
    for src, dst in (("F1_攻击", "ΔF1_攻击_pp"),
                     ("召回率_攻击", "Δ召回率_攻击_pp"),
                     ("FPR_误报率", "ΔFPR_误报率_pp")):
        out[dst] = (out[src] - base[src]) * 100.0
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  √ 已保存: {OUT_PATH}")

    print("\n===== ATRDF LOO 消融汇总（验证集，Δ 为相对完整方案的百分点差）=====")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n解读要点：")
    print("  · 关注与 CSIC 版（ablation_leave_one_out.csv）的结论一致性：")
    print("    规则特征/TF-IDF 的支柱地位是否保持、族级冗余是否复现")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
