# -*- coding: utf-8 -*-
"""
evaluate.py —— 评估模块（第 3 阶段）

功能：
    1. evaluate_model(): 计算准确率 / 精确率 / 召回率 / F1 / ROC-AUC；
    2. print_classification_report(): 打印逐类别详细报告；
    3. plot_confusion_matrix(): 单模型混淆矩阵热力图 -> results/*.png；
    4. plot_roc_curves(): 多模型 ROC 曲线对比图 -> results/roc_curves.png；
    5. plot_feature_importance(): Top-N 特征重要性图 -> results/*.png；
    6. save_metrics_summary(): 指标汇总表 -> results/metrics_summary.csv。

指标含义（二分类：1=恶意为正类）：
    准确率 Accuracy : 全部预测对的比例 —— 不平衡时会被多数类"虚高"
    精确率 Precision: 预测为恶意的请求里，真恶意的比例（误报越少越高）
    召回率 Recall   : 真恶意请求里，被成功揪出的比例（漏报越少越高）
    F1              : 精确率与召回率的调和平均
    ROC-AUC         : 随机取一正一负样本，正样本得分更高的概率（阈值无关）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 后端设为 Agg：无显示环境（服务器/脚本）也能正常出图
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

# 中文字体（Windows 自带微软雅黑/黑体），负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 模型中文名 -> 图表文件名用的英文（避免文件名编码问题）
MODEL_EN_NAMES = {
    "逻辑回归": "logistic_regression",
    "随机森林": "random_forest",
    "梯度提升树": "xgboost",
}

CLASS_LABELS = ["正常(0)", "恶意(1)"]  # 0=正常为负类，1=恶意为正类


# ============================================================
# 一、指标计算
# ============================================================
def evaluate_model(model, X_test, y_test) -> dict:
    """
    在测试集上计算 5 项核心指标。

    参数:
        model: 已训练好的模型
        X_test / y_test: 测试集特征与标签
    返回:
        {"accuracy":, "precision":, "recall":, "f1":, "roc_auc":} 全部为 float
    """
    y_pred = model.predict(X_test)
    # predict_proba 输出每类的概率，取第 1 列 = P(恶意)，ROC-AUC 需要
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def print_classification_report(name: str, model, X_test, y_test) -> None:
    """打印单模型的逐类别详细报告（精确率/召回率/F1/样本数）。"""
    print(f"\n----- {name} 分类报告 -----")
    print(classification_report(
        y_test, model.predict(X_test),
        target_names=CLASS_LABELS, digits=4,
    ))


# ============================================================
# 二、可视化图表
# ============================================================
def plot_confusion_matrix(name: str, model, X_test, y_test) -> Path:
    """
    画单模型混淆矩阵热力图并保存。

    混淆矩阵四格含义（本任务关注右下与左上）：
        左上=真阴性 TN(正常判对)  右上=假阳性 FP(正常误报为恶意)
        左下=假阴性 FN(恶意漏报!) 右下=真阳性 TP(恶意判对)
    返回: 保存的图片路径
    """
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"混淆矩阵 —— {name}")
    fig.tight_layout()
    save_path = RESULTS_DIR / f"confusion_matrix_{MODEL_EN_NAMES[name]}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_roc_curves(models: dict, X_test, y_test) -> Path:
    """
    画多模型 ROC 曲线对比图并保存。

    ROC 曲线越贴近左上角、AUC 越接近 1，模型区分能力越强；
    对角虚线 = 随机猜测的基准线（AUC=0.5）。
    返回: 保存的图片路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测 (AUC = 0.5)")
    ax.set_xlabel("假阳性率 FPR（正常请求被误判为恶意的比例）")
    ax.set_ylabel("真阳性率 TPR（恶意请求被正确拦截的比例）")
    ax.set_title("各模型 ROC 曲线对比")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = RESULTS_DIR / "roc_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_feature_importance(name: str, model, feature_names: list,
                            top_n: int = 20) -> Path:
    """
    画 Top-N 特征重要性横向条形图并保存。

    不同模型的重要性来源不同：
        随机森林 / LightGBM -> feature_importances_（分裂增益贡献）
        逻辑回归            -> |coef_|（特征权重绝对值，越大影响越强）
    返回: 保存的图片路径
    """
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        note = "（基于分裂增益）"
    else:  # 逻辑回归没有 feature_importances_，用系数绝对值代替
        importances = np.abs(np.asarray(model.coef_[0]))
        note = "（基于|回归系数|）"

    order = np.argsort(importances)[::-1][:top_n]   # 取重要性最大的前 N 个下标
    top_names = [feature_names[i][:28] for i in order]  # 截断超长 TF-IDF 词名
    top_vals = importances[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    # 条形图默认从下往上画，反转 y 轴让最重要的排在最上面
    ax.barh(range(len(top_names)), top_vals[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("重要性得分")
    ax.set_title(f"Top {top_n} 特征重要性 —— {name} {note}")
    fig.tight_layout()
    save_path = RESULTS_DIR / f"feature_importance_{MODEL_EN_NAMES[name]}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_metrics_summary(metrics: dict) -> Path:
    """
    把所有模型的指标汇总保存为 CSV 并打印格式化表格。

    参数:
        metrics: {"模型名": {"accuracy":..., "recall":..., ...}, ...}
    返回: 保存的 CSV 路径
    """
    summary = pd.DataFrame(metrics).T          # 行=模型，列=指标
    summary.index.name = "模型"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULTS_DIR / "metrics_summary.csv"
    summary.to_csv(save_path, encoding="utf-8-sig", float_format="%.4f")
    print("\n" + "=" * 70)
    print("模型评估指标汇总（测试集）")
    print("=" * 70)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n√ 指标汇总已保存: {save_path}")
    return save_path
