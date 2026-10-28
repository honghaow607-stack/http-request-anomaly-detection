# -*- coding: utf-8 -*-
"""
models.py —— 模型定义与训练模块（第 3 阶段）

实现 3 个对比模型：
    1. 逻辑回归 (LogisticRegression)
       —— 线性模型，训练快、可解释性强；在稀疏 TF-IDF 特征上表现出色，
          是文本分类的经典基线。缺点：只能学线性决策边界。
    2. 随机森林 (RandomForestClassifier)
       —— Bagging 集成树模型，对噪声鲁棒、不易过拟合，
          能自动捕捉特征间的非线性交互。缺点：高维稀疏特征上不如线性模型高效。
    3. 梯度提升树 (XGBoost / XGBClassifier)
       —— Boosting 集成树模型，通常精度最高，自带特征重要性。
          缺点：比逻辑回归更容易过拟合，需控制树的深度。

    为什么用 XGBoost 而不是 LightGBM？
        两者都是梯度提升树的代表实现（技术要求二选一）。本机 Anaconda
        环境下 LightGBM 4.7 的二进制库存在 DLL 崩溃问题（access violation，
        连最小用例都无法运行），因此选用同样满足要求的 XGBoost。

类别不平衡处理：
    - 逻辑回归 / 随机森林: class_weight="balanced"
      少数类样本权重自动放大，等价于按类别频率倒数加权；
    - XGBoost: 没有 class_weight 参数，使用等效的 scale_pos_weight
      = 负类样本数 / 正类样本数，对正类（恶意）损失加权。
    （本数据集恶意占比约 58%，不平衡程度中等，上述加权已足够；
      若未来数据极度不平衡如 1:100，可改用 SMOTE 过采样，见 README。）

所有模型共享同一份特征矩阵（由 FeaturePipeline 生成），保证对比公平。
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 全局随机种子：保证每次运行结果可复现
RANDOM_STATE = 42


def build_models(y_train=None) -> dict:
    """
    构建待对比的模型字典（未训练）。

    参数:
        y_train: 训练集标签。传入后 XGBoost 会按类别比例自动设置
                 scale_pos_weight（实现与 class_weight="balanced" 等效的平衡）。
    返回:
        {"逻辑回归": LogisticRegression, "随机森林": RandomForest,
         "梯度提升树": XGBClassifier}
    """
    # ---------- 计算 XGBoost 的类别平衡权重 ----------
    # scale_pos_weight = 负类(正常)样本数 / 正类(恶意)样本数
    # 正类占比越高，该值越小（恶意是多数类时 < 1）
    scale_pos_weight = 1.0
    if y_train is not None:
        y_arr = np.asarray(y_train)
        n_neg = int((y_arr == 0).sum())
        n_pos = int((y_arr == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)  # 防除零

    models = {
        # ---------- 1. 逻辑回归 ----------
        # max_iter=5000: 真实 CSIC 数据（2 万+ 训练样本、1540 维特征）下
        #                lbfgs 需要更多迭代才能收敛，2000 会触发 ConvergenceWarning
        # C=1.0: 正则化强度的倒数（默认值），过大易过拟合、过小欠拟合
        # class_weight="balanced": 自动按类别频率的倒数加权，处理不平衡
        "逻辑回归": LogisticRegression(
            max_iter=5000, C=1.0, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # ---------- 2. 随机森林 ----------
        # n_estimators=200: 树的数量，越多越稳但越慢，200 在精度/速度间平衡
        # n_jobs=-1: 使用全部 CPU 核心并行训练
        "随机森林": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        # ---------- 3. 梯度提升树 (XGBoost) ----------
        # n_estimators=200: 提升轮数；learning_rate=0.1: 每棵树的贡献步长
        # max_depth=6: 单棵树最大深度（控制过拟合）
        # eval_metric="logloss": 二分类的标准评估/早停指标，避免版本警告
        "梯度提升树": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss",
        ),
    }
    return models


def train_all_models(X_train, y_train) -> dict:
    """
    训练全部对比模型。

    参数:
        X_train: 训练集特征矩阵（稀疏或稠密均可）
        y_train: 训练集标签（0=正常, 1=恶意）
    返回:
        {"模型中文名": 已训练好的模型对象, ...}
    """
    trained = {}
    for name, model in build_models(y_train).items():
        print(f"  正在训练: {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"    √ {name} 训练完成")
    return trained
