# -*- coding: utf-8 -*-
"""
adversarial_test.py —— 参数重编码对抗实验（回应评审核心问题）

评审问题：规范化（递归解码）与混合特征，是否让"编码变体"识别更稳定？

设计（v2，全部使用真实流量，无手工构造样本）：
    1) 从与训练无关的验证/测试段抽取真实 GET 请求（正常/攻击各 60 条）；
    2) 对每条请求的查询参数值做 0/1/2 轮"额外" URL 重编码：
       depth=1 原始报文（基线），depth=2/3 模拟攻击者拦截重放时的
       二次/三次编码绕过（%252e 类手法）。报文其余部分保持原样。
    3) 用与 run.py 一致的 64/16/20 划分（seed=42）在训练集上拟合 4 条
       管线（混合特征/仅 TF-IDF × decode_rounds=1/3），对比各深度下的
       召回率与误报率。

    预期逻辑：重编码同时改变了攻击与正常样本的原始特征分布——
    单轮管线无法还原（两类特征值都"漂移"），多轮管线解码后特征
    回到原始值。若三轮管线在高深度下保持基线水平的召回与误报、
    单轮管线漂移，即隔离出递归解码的独立贡献。

运行方式（项目根目录）：
    python scripts/adversarial_test.py

产出：
    results/adversarial_encoding.csv
"""

import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import RESULTS_DIR, load_dataset
from src.features import FeaturePipeline
from src.models import build_models

SEED = 42
THRESHOLD = 0.5
N_PER_CLASS = 60   # 每类抽取的真实请求条数

CONFIGS = [
    ("混合特征-单轮", dict(use_rules=True, use_tfidf=True, decode_rounds=1)),
    ("混合特征-三轮", dict(use_rules=True, use_tfidf=True, decode_rounds=3)),
    ("仅TF-IDF-单轮", dict(use_rules=False, use_tfidf=True, decode_rounds=1)),
    ("仅TF-IDF-三轮", dict(use_rules=False, use_tfidf=True, decode_rounds=3)),
]


def _reencode_url(url: str, extra_rounds: int) -> str:
    """把 URL 查询串中的每个参数值额外重编码 extra_rounds 轮（键名不动）。"""
    if extra_rounds <= 0:
        return url
    split = urlsplit(url)
    if not split.query:
        return url
    parts = []
    for part in split.query.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        for _ in range(extra_rounds):
            v = quote(v, safe="")
        parts.append(f"{k}={v}")
    return urlunsplit((split.scheme, split.netloc, split.path,
                       "&".join(parts), split.fragment))


def build_adversarial_frame(val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """从验证/测试段抽取真实 GET 请求并生成 depth=1/2/3 的重编码变体。"""
    pool = pd.concat([val_df, test_df], ignore_index=True)
    pool = pool[(pool["method"].str.upper() == "GET")
                & (pool["query"].str.len() > 0)]
    rows = []
    for label in (1, 0):
        sub = pool[pool["label"] == label]
        take = min(N_PER_CLASS, len(sub))
        picked = sub.sample(n=take, random_state=SEED)
        for _, r in picked.iterrows():
            for depth in (1, 2, 3):
                url = _reencode_url(r["url"], extra_rounds=depth - 1)
                rows.append({
                    "label": label,
                    "attack_type": r["attack_type"],
                    "method": r["method"],
                    "url": url,
                    "path": r["path"],
                    "query": urlsplit(url).query,
                    "headers": r["headers"],
                    "body": r["body"],
                    "encode_depth": depth,
                })
    return pd.DataFrame(rows)


def _metrics(y_true, y_pred) -> tuple:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return recall, fpr


def main() -> None:
    print("#" * 70)
    print("#  参数重编码对抗实验：单轮 vs 三轮递归解码（随机森林 / 阈值 0.5）")
    print("#" * 70)

    # ---------- [1/4] 与 run.py 一致的三段划分（只用训练集拟合） ----------
    print("\n[1/4] 加载数据并复现 64/16/20 三段划分 ...")
    df = load_dataset(rebuild=False)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=SEED)
    y_train = train_df["label"].to_numpy()
    print(f"  训练集 {len(train_df)} 条（对抗变体取自验证/测试段，训练时未见）")

    # ---------- [2/4] 从真实流量构造重编码变体 ----------
    print("\n[2/4] 从验证/测试段真实请求构造重编码变体 ...")
    adv = build_adversarial_frame(val_df, test_df)
    n_att = int((adv["label"] == 1).sum())
    n_norm = int((adv["label"] == 0).sum())
    print(f"  对抗集共 {len(adv)} 条：攻击 {n_att} + 正常 {n_norm}，"
          f"每条生成 depth=1/2/3 三个版本")
    print("  示例（某攻击请求的三种深度）:")
    sample_url = adv[(adv["label"] == 1) & (adv["encode_depth"] == 1)]["url"].iloc[0]
    path_only = sample_url.split("?")[0]
    for d in (1, 2, 3):
        u = adv[(adv["url"].str.split("?").str[0] == path_only)
                & (adv["encode_depth"] == d)]["url"]
        if len(u):
            print(f"    depth={d}: {u.iloc[0][:110]}")

    # ---------- [3/4] 四条管线在训练集上拟合（特征配置 x 解码轮数） ----------
    print("\n[3/4] 训练四条管线（特征配置 x 解码轮数）...")
    y_adv = adv["label"].to_numpy()
    depth_adv = adv["encode_depth"].to_numpy()
    rows_out = []
    for name, kwargs in CONFIGS:
        pipe = FeaturePipeline(**kwargs)
        X_train = pipe.fit_transform(train_df)
        rf = build_models(y_train)["随机森林"]
        rf.fit(X_train, y_train)
        pred = (rf.predict_proba(pipe.transform(adv))[:, 1] >= THRESHOLD).astype(int)
        print(f"  √ {name}: 训练完成并在对抗集上预测")

        for depth in (1, 2, 3):
            m = depth_adv == depth
            recall, fpr = _metrics(y_adv[m], pred[m])
            rows_out.append({
                "管线": name, "样本组": "攻击", "编码深度": depth,
                "样本数": int((m & (y_adv == 1)).sum()),
                "召回率": round(recall, 4), "误报率": "",
            })
            m_n = m & (y_adv == 0)
            _, fpr_n = _metrics(y_adv[m_n], pred[m_n])
            rows_out.append({
                "管线": name, "样本组": "正常", "编码深度": depth,
                "样本数": int(m_n.sum()),
                "召回率": "", "误报率": round(fpr_n, 4),
            })

    # ---------- [4/4] 汇总保存 ----------
    print("\n[4/4] 保存对比结果 ...")
    out = pd.DataFrame(rows_out)
    save_path = RESULTS_DIR / "adversarial" / "adversarial_encoding.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  √ 已保存: {save_path}")

    print("\n===== 攻击召回率（按重编码深度）=====")
    piv = out[out["样本组"] == "攻击"].pivot_table(
        index="管线", columns="编码深度", values="召回率", aggfunc="first")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n===== 正常样本误报率（按重编码深度）=====")
    piv_fpr = out[out["样本组"] == "正常"].pivot_table(
        index="管线", columns="编码深度", values="误报率", aggfunc="first")
    print(piv_fpr.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n解读要点：")
    print("  · depth=1 为原始报文基线；depth=2/3 为参数重编码变体；")
    print("  · 若单轮管线随深度增加召回/误报明显漂移，而三轮管线保持基线水平，")
    print("    即证明递归解码让编码变体识别更稳定（正面回答评审问题）；")
    print("  · 混合特征与仅 TF-IDF 两组对比可看出规则特征是否提供额外兜底。")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
