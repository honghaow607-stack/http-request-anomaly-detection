# -*- coding: utf-8 -*-
"""
e2e_test.py —— 端到端生产链路测试（评审缺口 3：提供可复现的召回率证据）

目的：
    1) 验证"原始报文文本 → model.pkl 预测"的生产链路与训练管线一致；
    2) 以固定随机种子从 data/raw 独立抽样并留下报告文件，
       为 README 中"生产召回率"提供可查证的记录；
    3) 输出阈值扫描表（0.3 / 0.5 / 0.7），展示召回-误报的运营权衡。

抽样说明（诚实性）：
    样本来自与训练相同的两个原始文件，约 80% 的报文模型在训练中见过
    （"开卷"成分），因此本结果高于测试集闭卷指标是预期现象；
    它验证的是链路正确性，而非泛化能力。

运行方式（项目根目录，需先运行 python run.py 生成 model.pkl）：
    python scripts/e2e_test.py

产出：
    results/e2e_report.csv   逐条预测明细
    results/e2e_thresholds.csv  阈值扫描汇总
"""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.data_loader import discover_raw_files, parse_request_file, RESULTS_DIR

SAMPLE_PER_CLASS = 100   # 每类抽样条数
SEED = 42                # 固定种子保证可复现


def sample_raw_records() -> pd.DataFrame:
    """从真实 CSIC 原始文件（排除合成数据）各随机抽取报文级记录。"""
    records = []
    for path, label, atype in discover_raw_files():
        if "synthetic" in path.name.lower():     # 合成增广文件不参与端到端抽样
            continue
        recs = parse_request_file(path, label, atype)
        rng = random.Random(SEED)
        picked = rng.sample(recs, min(SAMPLE_PER_CLASS, len(recs)))
        print(f"  - {path.name}: 共 {len(recs)} 条，抽样 {len(picked)} 条 (label={label})")
        records.extend(picked)
    return pd.DataFrame(records)


def main() -> None:
    print("#" * 70)
    print("#  端到端生产链路测试（原始报文 → model.pkl）")
    print("#" * 70)

    print("\n[1/4] 加载 model.pkl ...")
    bundle = joblib.load(PROJECT_ROOT / "model.pkl")
    pipeline, model = bundle["pipeline"], bundle["model"]
    print(f"  模型: {bundle['model_name']}")

    print("\n[2/4] 从原始文件抽样 ...")
    df = sample_raw_records()
    y_true = df["label"].to_numpy()

    print(f"\n[3/4] 生产链路预测（共 {len(df)} 条）...")
    proba = model.predict_proba(pipeline.transform(df))[:, 1]

    # ---------- 逐条明细落盘 ----------
    out = df[["label", "attack_type", "method", "url"]].copy()
    out["malicious_prob"] = proba.round(4)
    out["pred@0.5"] = (proba >= 0.5).astype(int)
    out["correct@0.5"] = (out["pred@0.5"] == out["label"]).astype(int)
    detail_path = RESULTS_DIR / "e2e" / "e2e_report.csv"
    out.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"  √ 明细已保存: {detail_path}")

    # ---------- 阈值扫描（召回-误报权衡） ----------
    print("\n[4/4] 阈值扫描 ...")
    rows = []
    for thr in (0.3, 0.5, 0.7):
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        rows.append({
            "阈值": thr,
            "召回率_攻击": round(tp / max(tp + fn, 1), 4),
            "精确率_攻击": round(tp / max(tp + fp, 1), 4),
            "FPR_误报率": round(fp / max(fp + tn, 1), 4),
            "FNR_漏报率": round(fn / max(fn + tp, 1), 4),
            "准确率": round((tp + tn) / len(y_true), 4),
        })
    thr_df = pd.DataFrame(rows)
    thr_path = RESULTS_DIR / "e2e" / "e2e_thresholds.csv"
    thr_df.to_csv(thr_path, index=False, encoding="utf-8-sig")
    print(thr_df.to_string(index=False))
    print(f"  √ 阈值表已保存: {thr_path}")

    # ---------- 漏检样例 ----------
    missed = out[(out["label"] == 1) & (out["pred@0.5"] == 0)]
    print(f"\n漏检攻击样例（阈值 0.5，共 {len(missed)} 条）：")
    for _, r in missed.head(5).iterrows():
        print(f"  [p={r['malicious_prob']:.4f}] {str(r['url'])[:90]}")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
