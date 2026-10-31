# -*- coding: utf-8 -*-
"""
run_atrdf_eval.py —— ATRDF 数据集泛化评估（外部数据集验证）

背景：
    主实验只在 CSIC 2010 上训练与评估，评审关心"换一个数据集还成立吗"。
    ATRDF（API Threat Response Dataset Framework）是较新的 HTTP 攻击数据集：
    REST API 风格流量、攻击占比仅 7.5~22%（CSIC 约 62% 恶意，分布几乎相反），
    且含 CSIC 没有的攻击类型（Cookie 注入 / Log Forging / RCE / LOG4J），
    是理想的泛化测试床。

数据说明（data/raw/atrdf/extracted/dataset_{1..4}_{train,val}/）：
    每条记录 = {request: {headers(dict), method, url(含主机), body},
                response: {...}}；request.Attack_Tag 缺失 => 正常请求。
    共 4 个子集（train 文件 4283 / 127502 / 144502 / 182770 条），
    攻击类型 7 类：SQL Injection / Cookie Injection / XSS / Log Forging /
    Directory Traversal / RCE / LOG4J。
    * 官方 val 文件全部为正常流量（无 Attack_Tag），无法作为带标签测试集，
      因此本实验只使用 train 文件，自行按项目协议三段划分。
    * 本脚本主用 dataset_3_train.json（14.4 万条，规模与多样性均衡）。

两组实验：
    ① 重训重评（泛化主实验）：ATRDF 上按项目协议 64/16/20 划分，
       训练 3 个模型并在 ATRDF 测试集评估 —— 验证特征体系/管线本身
       换数据集是否仍然有效；
    ② 零样本迁移（跨数据集）：CSIC 训练的 model.pkl（pipeline+模型）
       直接 transform/predict ATRDF 测试集，不做任何重训 ——
       度量"只学过 CSIC 分布"的模型在异域流量上的退化幅度。

运行方式（项目根目录）：
    python scripts/run_atrdf_eval.py

产出：
    results/atrdf_metrics.csv        各模型指标（重训 3 模型 + 零样本迁移）
    results/atrdf_attack_recall.csv  按攻击类型的召回率细分（含正常类误报率）
"""

import gc
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import RESULTS_DIR, _canonical_headers
from src.features import FeaturePipeline
from src.models import build_models

ATRDF_DIR = PROJECT_ROOT / "data" / "raw" / "atrdf" / "extracted"
ATRDF_MAIN = ATRDF_DIR / "dataset_3_train" / "dataset_3_train.json"
MODEL_PATH = PROJECT_ROOT / "model.pkl"   # CSIC 训练的最优模型（零样本迁移用）


def load_atrdf(path: Path) -> pd.DataFrame:
    """
    解析 ATRDF JSON 为与 dataset.csv 同构的 DataFrame。

    字段映射：
        request.url(含主机) -> url，并用 urlsplit 拆出 path / query
        request.headers(dict) -> "k: v" 逐行拼接文本（与 CSIC headers 同构）
        request.body -> 字符串（dict/list 则序列化为 JSON 文本）
        Attack_Tag 存在 -> label=1, attack_type=标签值；缺失 -> label=0, normal
    """
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[1/4] 加载 {path.name}（{size_mb:.0f} MB，大文件加载需约 1 分钟）...")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  原始记录数: {len(raw)}")

    recs = []
    for rec in raw:
        req = rec.get("request") or {}
        tag = req.get("Attack_Tag") or rec.get("Attack_Tag") or ""
        url = str(req.get("url") or "")
        if url and "://" not in url:          # 补协议头便于 urlsplit 拆 path/query
            url = "http://" + url
        sp = urlsplit(url)
        headers = req.get("headers") or {}
        if isinstance(headers, dict):
            headers_text = "\n".join(f"{k}: {v}" for k, v in headers.items())
        else:
            headers_text = str(headers)
        body = req.get("body") or ""
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        recs.append({
            "label": 1 if tag else 0,
            "attack_type": str(tag) if tag else "normal",
            "method": str(req.get("method") or "GET"),
            "url": url,
            "path": sp.path or "/",
            "query": sp.query,
            "headers": headers_text,
            "body": str(body),
        })
    del raw
    gc.collect()

    df = pd.DataFrame(recs)
    df = df[df["url"].str.len() > 0]
    # 与主实验完全一致的激进去重（防近似重复样本跨集合泄漏）
    df["_h_canon"] = df["headers"].map(_canonical_headers)
    n_before = len(df)
    df = df.drop_duplicates(subset=["method", "url", "_h_canon", "body", "label"])
    df = df.drop(columns="_h_canon").reset_index(drop=True)
    print(f"  √ 解析 + 去重: {n_before} -> {len(df)} 条（去除 {n_before - len(df)} 条重复）")
    print("  标签分布:", df["label"].value_counts().rename(
        {0: "正常", 1: "恶意"}).to_dict())
    print("  攻击类型分布:")
    for atype, cnt in df[df["label"] == 1]["attack_type"].value_counts().items():
        print(f"    {atype:24s} {cnt}")
    return df


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """与主实验相同口径：恶意类为正类，阈值 0.5。"""
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
        "_tp": tp, "_fp": fp, "_fn": fn, "_tn": tn,
    }


def main() -> None:
    print("#" * 70)
    print("#  ATRDF 泛化评估：① 重训重评  ② 零样本迁移（CSIC -> ATRDF）")
    print("#" * 70)

    df = load_atrdf(ATRDF_MAIN)

    # ---------- 与主实验一致的三段划分 ----------
    print("\n[2/4] 64/16/20 三段划分（seed=42）...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    print(f"  训练 {len(train_df)} / 验证 {len(val_df)} / 测试 {len(test_df)}")

    # ---------- 实验①：ATRDF 重训 3 模型 ----------
    print("\n[3/4] 实验①：ATRDF 重训三模型 ...")
    pipe = FeaturePipeline(use_rules=True, use_tfidf=True, decode_rounds=3)
    X_train = pipe.fit_transform(train_df)
    X_test = pipe.transform(test_df)
    print(f"  特征维度: {X_train.shape[1]}")

    models = build_models(y_train)
    metric_rows, detail = [], {}
    for name, model in models.items():
        print(f"  训练 {name} ...")
        model.fit(X_train, y_train)
        r = evaluate(y_test, model.predict_proba(X_test)[:, 1])
        detail[name] = r
        metric_rows.append({
            "模型": name, "训练数据": "ATRDF(ds3)", "测试数据": "ATRDF(ds3)-test",
            "样本数": len(test_df),
            **{k: v for k, v in r.items() if not k.startswith("_")},
        })
        print(f"    √ recall={r['召回率_攻击']:.4f}  F1={r['F1_攻击']:.4f}  "
              f"FPR={r['FPR_误报率']:.4f}")

    # ---------- 实验②：零样本迁移（CSIC 模型直接上 ATRDF 测试集） ----------
    print("\n[4/4] 实验②：零样本迁移（CSIC model.pkl -> ATRDF 测试集）...")
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        zs_name = f"零样本迁移({bundle.get('model_name', '?')}, CSIC训练)"
        print(f"  载入 {MODEL_PATH.name}（模型: {bundle.get('model_name')}），"
              f"用其特征流水线 transform ATRDF 测试集 ...")
        X_zs = bundle["pipeline"].transform(test_df)
        zs_model = bundle["model"]
        proba = (zs_model.predict_proba(X_zs)[:, 1]
                 if hasattr(zs_model, "predict_proba")
                 else zs_model.predict(X_zs).ravel())
        r = evaluate(y_test, np.asarray(proba, dtype=float))
        detail[zs_name] = r
        metric_rows.append({
            "模型": zs_name, "训练数据": "CSIC 2010", "测试数据": "ATRDF(ds3)-test",
            "样本数": len(test_df),
            **{k: v for k, v in r.items() if not k.startswith("_")},
        })
        print(f"    √ recall={r['召回率_攻击']:.4f}  F1={r['F1_攻击']:.4f}  "
              f"FPR={r['FPR_误报率']:.4f}")
    else:
        print(f"  ! 未找到 {MODEL_PATH}，跳过零样本迁移（请先运行 python run.py）")

    # ---------- 保存指标 ----------
    out = pd.DataFrame(metric_rows)
    p1 = RESULTS_DIR / "atrdf" / "atrdf_metrics.csv"
    out.to_csv(p1, index=False, encoding="utf-8-sig")
    print(f"\n  √ 已保存: {p1}")
    print("\n===== ATRDF 泛化评估汇总（测试集）=====")
    print(out.drop(columns=["样本数"]).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    # ---------- 按攻击类型的细分召回（含正常类的误报率行） ----------
    # 逐模型重建 0/1 预测，计算每类攻击的召回率
    all_preds = {}
    for name, model in models.items():
        all_preds[name] = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    if MODEL_PATH.exists() and zs_name in detail:
        all_preds[zs_name] = (np.asarray(proba) >= 0.5).astype(int)
    test_pos = test_df.reset_index(drop=True)
    pos_idx = test_pos.index[test_pos["label"] == 1]
    rows = []
    for atype, cnt in test_pos[test_pos["label"] == 1]["attack_type"].value_counts().items():
        mask = (test_pos.loc[pos_idx, "attack_type"] == atype).to_numpy()
        idx = pos_idx[mask]
        row = {"攻击类型": atype, "测试样本数": len(idx)}
        for name, pred in all_preds.items():
            row[f"召回率_{name}"] = pred[idx].mean()
        rows.append(row)
    norm_idx = test_pos.index[test_pos["label"] == 0]
    row = {"攻击类型": "normal(误报率)", "测试样本数": len(norm_idx)}
    for name, pred in all_preds.items():
        row[f"召回率_{name}"] = pred[norm_idx].mean()   # 正常类被误判为攻击的比例
    rows.append(row)
    out2 = pd.DataFrame(rows)
    p2 = RESULTS_DIR / "atrdf" / "atrdf_attack_recall.csv"
    out2.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n  √ 已保存: {p2}")
    print("\n===== 按攻击类型细分 =====")
    print(out2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n解读要点：")
    print("  · 重训三模型成绩高 => 特征体系/管线跨数据集成立（泛化主证据）")
    print("  · 零样本迁移明显退化 => 域差异大（REST API 正常流量 vs CSIC 电商），")
    print("    说明部署前必须用目标域数据重训，这是诚实且有价值的结论")
    print("  · 新攻击类型（Cookie 注入/Log Forging/RCE/LOG4J）召回率反映")
    print("    规则特征与 TF-IDF 对未见攻击族的迁移能力")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
