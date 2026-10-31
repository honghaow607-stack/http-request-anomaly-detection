# -*- coding: utf-8 -*-
"""
run_atrdf_cross_subset.py —— ATRDF 跨子集严格泛化检验（回应"召回率 1.0 是否虚高"）

动机：
    run_atrdf_eval.py 中 ds3 同子集三段划分下 6 类攻击召回率 = 1.0000。
    需要排除两类解释：
      ① 实现缺陷（标签错误 / 划分泄漏 / 指标口径错误）—— 本脚本先自检；
      ② 模板级复用（同一攻击模板的微小变体跨训练/测试段出现，
        "半开卷"效应）—— 用模板重叠率量化，并用跨子集实验隔离：
        ds3 训练的模型直接评估完全未见的 ds1/ds2/ds4 全量带标签集。

自检项：
    [a] 复现性：ds3 同子集测试成绩必须与 run_atrdf_eval.py 完全一致
        （同 seed/同划分/同管线），否则说明实现不稳定；
    [b] 硬泄漏检查：ds3 训练段与测试段之间 (method,url,body,label)
        完全相同的样本数必须为 0；
    [c] 模板重叠率：攻击模板键（url+body 小写、数字归并为 #）在 ds3
        训练段与各评估集之间的重叠比例 —— 定量解释召回率为什么高。

运行方式（项目根目录）：
    python scripts/run_atrdf_cross_subset.py

产出：
    results/atrdf_cross_subset.csv          各目标子集 × 模型的总体指标
    results/atrdf_cross_subset_recall.csv   各目标子集按攻击类型细分召回
    results/atrdf_template_overlap.csv      攻击模板重叠率量化
"""

import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import RESULTS_DIR
from src.features import FeaturePipeline
from src.models import build_models
from scripts.run_atrdf_eval import ATRDF_DIR, load_atrdf, evaluate

TARGETS = {
    "ds1": ATRDF_DIR / "dataset_1_train" / "dataset_1_train.json",
    "ds2": ATRDF_DIR / "dataset_2_train" / "dataset_2_train.json",
    "ds4": ATRDF_DIR / "dataset_4_train" / "dataset_4_train.json",
}


def template_keys(sub: pd.DataFrame) -> set:
    """攻击模板键：url+body 小写、数字归并为 #、截断 300 字符。"""
    s = (sub["url"] + " " + sub["body"]).str.lower()
    s = s.str.replace(r"\d+", "#", regex=True)
    return set(s.str[:300])


def per_type_rows(target_name: str, tgt: pd.DataFrame, preds: dict) -> list:
    """按攻击类型的细分召回行 + normal 误报率行。"""
    tgt = tgt.reset_index(drop=True)
    pos_idx = tgt.index[tgt["label"] == 1]
    norm_idx = tgt.index[tgt["label"] == 0]
    rows = []
    for atype in tgt[tgt["label"] == 1]["attack_type"].value_counts().index:
        mask = (tgt.loc[pos_idx, "attack_type"] == atype).to_numpy()
        idx = pos_idx[mask]
        row = {"目标数据集": target_name, "攻击类型": atype, "样本数": len(idx)}
        for name, pred in preds.items():
            row[f"召回率_{name}"] = float(pred[idx].mean())
        rows.append(row)
    row = {"目标数据集": target_name, "攻击类型": "normal(误报率)",
           "样本数": len(norm_idx)}
    for name, pred in preds.items():
        row[f"召回率_{name}"] = float(pred[norm_idx].mean())
    rows.append(row)
    return rows


def main() -> None:
    print("#" * 70)
    print("#  ATRDF 跨子集严格泛化检验：ds3 训练 -> ds1/ds2/ds4 全量评估")
    print("#" * 70)

    # ---------- [1/5] ds3 加载 + 与 run_atrdf_eval.py 完全一致的划分 ----------
    ds3 = load_atrdf(ATRDF_DIR / "dataset_3_train" / "dataset_3_train.json")
    train_df, test_df = train_test_split(
        ds3, test_size=0.2, stratify=ds3["label"], random_state=42)
    train_df, _val = train_test_split(          # 与原实验同口径：只用 64% 训练
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42)
    del ds3
    gc.collect()
    y_train = train_df["label"].to_numpy()
    print(f"\n[1/5] ds3 划分完成：训练 {len(train_df)} / 测试 {len(test_df)}"
          f"（同 run_atrdf_eval.py，可直接对比）")

    # ---------- [2/5] 自检 b：训练/测试硬重复必须为 0 ----------
    # 注意：merge 前必须先 drop_duplicates，否则同一键的多行会做 many-to-many
    # 连接产生笛卡尔积，把"同键出现多次"误计成海量"重复样本"。
    strict_cols = ["method", "url", "headers", "body", "label"]
    tr_keys = train_df[strict_cols].drop_duplicates()
    te_keys = test_df[strict_cols].drop_duplicates()
    n_dup = len(pd.merge(tr_keys, te_keys, how="inner"))
    print(f"[2/5] 自检b 硬重复：训练∩测试完全相同样本（含请求头全文）= "
          f"{n_dup} 条（应为 0）")

    # ---------- [3/5] 拟合管线 + 训练三模型 ----------
    print("\n[3/5] 拟合特征流水线并训练三模型 ...")
    pipe = FeaturePipeline(use_rules=True, use_tfidf=True, decode_rounds=3)
    X_train = pipe.fit_transform(train_df)
    print(f"  特征维度: {X_train.shape[1]}")
    models = build_models(y_train)
    for name, model in models.items():
        print(f"  训练 {name} ...")
        model.fit(X_train, y_train)
    del X_train
    gc.collect()

    # ---------- [4/5] 同子集复现自检 + 跨子集评估（逐个加载、用后即释） ----------
    train_keys = template_keys(train_df[train_df["label"] == 1])
    metric_rows, recall_rows, overlap_rows = [], [], []
    eval_plan = [("ds3-test(同子集)", None)] + list(TARGETS.items())

    for tname, tpath in eval_plan:
        tdf = test_df if tpath is None else load_atrdf(tpath)
        print(f"\n[4/5] 评估目标 {tname}（{len(tdf)} 条）...")
        Xt = pipe.transform(tdf)
        yt = tdf["label"].to_numpy()
        preds = {}
        for name, model in models.items():
            proba = model.predict_proba(Xt)[:, 1]
            preds[name] = (proba >= 0.5).astype(int)
            r = evaluate(yt, proba)
            metric_rows.append({
                "目标数据集": tname, "模型": name, "样本数": len(tdf),
                **{k: v for k, v in r.items() if not k.startswith("_")},
            })
            print(f"  {name}: recall={r['召回率_攻击']:.4f}  "
                  f"F1={r['F1_攻击']:.4f}  FPR={r['FPR_误报率']:.4f}")
        recall_rows += per_type_rows(tname, tdf, preds)

        # 模板重叠率（仅攻击样本，vs ds3 训练段）
        t_keys = template_keys(tdf[tdf["label"] == 1])
        ov = len(train_keys & t_keys) / max(len(t_keys), 1)
        overlap_rows.append({
            "目标数据集": tname,
            "攻击样本数": int((tdf["label"] == 1).sum()),
            "攻击模板键数": len(t_keys),
            "与ds3训练段模板重叠率": ov,
        })
        print(f"  模板重叠率(vs ds3训练段): {ov:.2%}")

        del Xt, preds
        if tpath is not None:
            del tdf
        gc.collect()

    # ---------- [5/5] 保存 + 复现性自检 [a] ----------
    print("\n[5/5] 保存结果 ...")
    out = pd.DataFrame(metric_rows)
    p1 = RESULTS_DIR / "atrdf" / "atrdf_cross_subset.csv"
    out.to_csv(p1, index=False, encoding="utf-8-sig")

    out2 = pd.DataFrame(recall_rows)
    p2 = RESULTS_DIR / "atrdf" / "atrdf_cross_subset_recall.csv"
    out2.to_csv(p2, index=False, encoding="utf-8-sig")

    out3 = pd.DataFrame(overlap_rows)
    p3 = RESULTS_DIR / "atrdf" / "atrdf_template_overlap.csv"
    out3.to_csv(p3, index=False, encoding="utf-8-sig")
    print(f"  √ 已保存: {p1}\n  √ 已保存: {p2}\n  √ 已保存: {p3}")

    # 复现性自检 [a]：与 run_atrdf_eval.py 的 ds3 同子集成绩逐项比对
    prev = RESULTS_DIR / "atrdf" / "atrdf_metrics.csv"
    if prev.exists():
        ref = pd.read_csv(prev, encoding="utf-8-sig")
        ref = ref[(ref["训练数据"] == "ATRDF(ds3)")
                  & (ref["测试数据"] == "ATRDF(ds3)-test")]
        cur = out[out["目标数据集"] == "ds3-test(同子集)"]
        ok = True
        for _, r1 in ref.iterrows():
            r2 = cur[cur["模型"] == r1["模型"]]
            if r2.empty or abs(float(r2["召回率_攻击"].iloc[0])
                               - float(r1["召回率_攻击"])) > 1e-9:
                ok = False
        print(f"\n自检a 复现性：ds3 同子集成绩与 run_atrdf_eval.py "
              f"{'完全一致 ✓' if ok else '不一致 ✗（需排查）'}")

    print("\n===== 总体指标（各目标子集 × 模型）=====")
    print(out.drop(columns=["样本数"]).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))
    print("\n===== 按攻击类型细分 =====")
    print(out2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n===== 攻击模板重叠率 =====")
    print(out3.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n解读要点：")
    print("  · 自检a/b 通过 => 1.0 召回不是实现 bug；")
    print("  · 同子集模板重叠率高 => 1.0 召回的'半开卷'解释成立；")
    print("  · 跨子集（ds1/ds2/ds4，模板完全未见）成绩才是无泄漏泛化的下限，")
    print("    若仍高则结论扎实，若下降则如实报告为模板依赖度。")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
