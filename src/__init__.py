# -*- coding: utf-8 -*-
"""
src 包 —— 项目核心源代码模块。

包含以下模块：
    data_loader.py  数据读取、清洗、打标签
    preprocess.py   文本清洗、URL 解析、单条请求预处理
    features.py     规则特征 + 统计特征 + TF-IDF 特征
    models.py       模型定义与训练
    evaluate.py     评估指标、混淆矩阵、ROC 曲线
    predict.py      单条请求预测（命令行）
    app.py          Streamlit 交互页面
"""
