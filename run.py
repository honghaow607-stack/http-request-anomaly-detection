# -*- coding: utf-8 -*-
"""
run.py —— 一键跑通全流程（第 3 阶段入口脚本）

执行内容：加载数据 -> 划分训练/测试集 -> 特征工程 -> 训练 3 个模型
          -> 评估并出图 -> 选出最优模型 -> 保存 model.pkl

运行方式（在项目根目录）：
    python run.py

产出：
    model.pkl                     最优模型（含特征流水线，预测阶段直接复用）
    results/metrics_summary.csv   各模型指标汇总
    results/confusion_matrix_*.png        每个模型的混淆矩阵热力图
    results/roc_curves.png                各模型 ROC 曲线对比
    results/feature_importance_*.png      每个模型的 Top20 特征重要性

关于"数据泄漏"的说明（重要）：
    TF-IDF 词表必须在训练集上 fit，再 transform 测试集；
    如果在全部数据上先 fit 再划分，测试集的词频信息会"泄漏"进训练特征，
    导致评估分数虚高。本项目严格按"先划分、后拟合"的顺序执行。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import ensure_dirs, load_dataset
from src.features import FeaturePipeline
from src.models import train_all_models
from src import evaluate as ev

# 保存最优模型的路径（项目根目录下 model.pkl）
MODEL_PATH = PROJECT_ROOT / "model.pkl"
# 随机种子：保证每次运行划分结果一致，实验可复现
RANDOM_STATE = 42


def main() -> None:
    """一键跑通：数据 -> 特征 -> 训练 -> 评估 -> 保存最优模型。"""
    ensure_dirs()

    print("#" * 70)
    print("#  基于机器学习的恶意 HTTP 请求识别系统 —— 训练流水线")
    print("#" * 70)

    # ---------- [1/6] 加载数据 ----------
    print("\n[1/6] 加载数据集 ...")
    # rebuild=True 强制从 data/raw 重新解析（保证激进去重等数据层改动生效）
    df = load_dataset(rebuild=True)
    print(f"  共 {len(df)} 条请求，恶意占比 {df['label'].mean():.1%}")

    # ---------- [2/6] 三段划分 64/16/20（训练/验证/测试，全部分层抽样） ----------
    # 评审要求：模型选择必须用独立验证集，测试集只在最终评估时使用一次，
    # 避免"用测试集选模型"导致的评估乐观偏差
    print("\n[2/6] 三段划分（训练 64% / 验证 16% / 测试 20%，分层抽样）...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"],
        random_state=RANDOM_STATE,
    )
    print(f"  训练集: {len(train_df)} 条（恶意 {train_df['label'].mean():.1%}）")
    print(f"  验证集: {len(val_df)} 条（恶意 {val_df['label'].mean():.1%}）→ 用于选最优模型")
    print(f"  测试集: {len(test_df)} 条（恶意 {test_df['label'].mean():.1%}）→ 仅终评一次")

    # ---------- [3/6] 特征工程（只在训练集上拟合词表，防数据泄漏） ----------
    print("\n[3/6] 特征工程（规则特征 + TF-IDF，仅在训练集上拟合）...")
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train_df)   # 拟合词表 + 生成训练特征
    X_val = pipeline.transform(val_df)           # 复用同一词表生成验证特征
    X_test = pipeline.transform(test_df)         # 复用同一词表生成测试特征
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    print(f"  训练特征矩阵: {X_train.shape}")
    print(f"  验证特征矩阵: {X_val.shape}")
    print(f"  测试特征矩阵: {X_test.shape}")

    # ---------- [4/6] 训练三个模型 ----------
    print("\n[4/6] 训练模型（逻辑回归 / 随机森林 / 梯度提升树）...")
    models = train_all_models(X_train, y_train)

    # ---------- [5/6] 评估：验证集选模 → 测试集终评 ----------
    print("\n[5/6] 评估模型并生成图表 ...")
    # 5a. 验证集指标：仅用于选择最优模型（不写入最终报告）
    metrics_val = {name: ev.evaluate_model(m, X_val, y_val)
                   for name, m in models.items()}
    print("  —— 验证集指标（用于选模）——")
    for name, m in metrics_val.items():
        print(f"    {name}: recall={m['recall']:.4f} f1={m['f1']:.4f} "
              f"acc={m['accuracy']:.4f}")
    best_name = max(metrics_val,
                    key=lambda k: (metrics_val[k]["recall"], metrics_val[k]["f1"]))
    print(f"  √ 验证集选出的最优模型: {best_name}")

    # 5b. 测试集终评：所有模型的最终成绩（一次性使用，不反馈给训练/选模）
    metrics = {name: ev.evaluate_model(m, X_test, y_test)
               for name, m in models.items()}
    ev.save_metrics_summary(metrics)

    for name, model in models.items():
        # 逐类别详细报告（精确率/召回率/F1）
        ev.print_classification_report(name, model, X_test, y_test)
        # 混淆矩阵热力图
        p1 = ev.plot_confusion_matrix(name, model, X_test, y_test)
        print(f"  √ 混淆矩阵已保存: {p1.name}")
        # Top20 特征重要性图
        p2 = ev.plot_feature_importance(name, model, pipeline.get_feature_names_out())
        print(f"  √ 特征重要性已保存: {p2.name}")

    p3 = ev.plot_roc_curves(models, X_test, y_test)
    print(f"  √ ROC 对比图已保存: {p3.name}")

    # ---------- [6/6] 保存最优模型（测试集终评成绩写入 model.pkl） ----------
    print("\n[6/6] 保存最优模型 ...")
    # 选择依据来自验证集；此处打印其在测试集上的最终成绩
    print(f"  最优模型: {best_name}（验证集选出）")
    print(f"    测试集终评: recall={metrics[best_name]['recall']:.4f}  "
          f"f1={metrics[best_name]['f1']:.4f}  "
          f"roc_auc={metrics[best_name]['roc_auc']:.4f}")

    # model.pkl 同时保存：特征流水线（TF-IDF 词表在里面）+ 模型本身
    # 这样预测阶段对单条请求 transform 时，与训练阶段的特征完全一致
    joblib.dump(
        {
            "pipeline": pipeline,
            "model": models[best_name],
            "model_name": best_name,
            "metrics": metrics[best_name],
        },
        MODEL_PATH,
    )
    print(f"  √ 最优模型已保存: {MODEL_PATH}")

    print("\n" + "#" * 70)
    print("#  全流程跑通！接下来可以：")
    print(f"#    1) 查看图表  : results/ 目录下 7 个评估图表")
    print(f"#    2) 命令行预测: python src/predict.py --text \"GET /?id=1' OR 1=1 --\"")
    print(f"#    3) 打开页面  : streamlit run src/app.py")
    print("#" * 70)


if __name__ == "__main__":
    # Windows 控制台强制 UTF-8 输出，避免中文乱码
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
