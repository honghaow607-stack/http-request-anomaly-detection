# -*- coding: utf-8 -*-
"""
features.py —— 特征工程模块（第 2 阶段）

特征体系（对每条 HTTP 请求提取）
--------------------------------
A. 规则特征 + 统计特征（共 40 维，稀疏可解释，是特征重要性的主要来源）：
   1) HTTP 方法 one-hot（GET/POST/PUT/DELETE/HEAD/OPTIONS/PATCH，7 维）
   2) 长度与结构：URL 长度、路径长度、查询串长度、Body 长度、
      请求头长度、Query 参数个数、路径深度（7 维）
   3) 特殊字符计数与密度：' " < > ; ( ) ` | & $ \\ -- /* 等
      （14 个计数 + 1 个总密度，15 维）
   4) 编码特征：%xx 编码次数、编码占比、双重编码迹象、
      连续斜杠、../ 出现次数（5 维）
   5) URL 信息熵：混淆后的 URL 字符分布更随机，熵更高（1 维）
   6) 攻击关键词命中：SQLi / XSS / 路径遍历 / 命令注入 四套词表
      在"解码后全文"上的命中计数（4+1 维）
B. TF-IDF 文本特征：对 URL 和 Body 分别向量化（训练时拟合词表，
   预测时复用同一词表，保证特征一致），默认最多 1000 + 500 维。

FeaturePipeline 类把 A+B 封装成 fit / transform 接口，
训练时 fit_transform，预测时 transform，最终随模型一起保存到 model.pkl。
"""

import re
import sys
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# 保证项目根目录在 sys.path 中，既支持"python src/features.py"直接运行，
# 也支持"from src.features import ..."包方式导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess_dataset  # noqa: E402

# ============================================================
# 常量定义：词表与特征配置
# ============================================================

# 1) 方法 one-hot 用的方法列表
METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]

# 2) 需要单独计数的特殊字符/字符串（攻击载荷的高频成分）
#    例如：' " 用于 SQL 注入闭合字符串；< > 是 XSS 标签；
#    ` | & $ 是命令注入常用符号；-- 是 SQL 注释；/* 是 SQL 块注释
COUNT_CHARS = [
    ("'", "count_quote_s"), ('"', "count_quote_d"),
    ("<", "count_lt"), (">", "count_gt"),
    (";", "count_semicolon"), ("(", "count_lparen"), (")", "count_rparen"),
    ("`", "count_backtick"), ("|", "count_pipe"), ("&", "count_amp"),
    ("$", "count_dollar"), ("\\", "count_backslash"),
    ("--", "count_double_dash"), ("/*", "count_slash_star"),
]

# 特殊字符总密度用的字符类（正则字符集，re.escape 后拼进 [...]）
SPECIAL_CHARS_CLASS = re.escape("'\"<>;()[]{}=?@#$&*,-_.!~`|%^\\/")

# 3) 攻击关键词词表（匹配"解码后的小写全文"，所以编码后攻击也能命中）
SQLI_KEYWORDS = [
    "union", "select", "insert into", "update set", "delete from",
    "drop table", "drop database", "exec", "xp_cmdshell", "sleep(",
    "waitfor delay", "benchmark(", "information_schema", "load_file",
    "into outfile", "or 1=1", "' or", '" or', "--", "#", "concat(",
    "group_concat", "hex(", "unhex(", "declare @", "cast(",
]
XSS_KEYWORDS = [
    "<script", "</script", "onerror", "onload", "onclick", "onmouseover",
    "onfocus", "javascript:", "alert(", "prompt(", "confirm(",
    "document.cookie", "document.location", "<iframe", "<svg", "<img",
    "<body", "<embed", "<object", "eval(", "fromcharcode", "string.fromCharCode",
    "innerHTML",
]
TRAVERSAL_KEYWORDS = [
    "../", "..\\", "..%2f", "%2e%2e", "....//", "etc/passwd", "etc%2fpasswd",
    "etc/shadow", "win.ini", "boot.ini", "system32", "/proc/self",
    "%c0%ae", "..\/", "/etc/", "c:\\windows",
]
CMD_KEYWORDS = [
    "; cat", "| ls", "&& ", "|| ", "$( ", "`", "whoami", "ping -",
    "nc -e", "rm -rf", "del ", "shutdown", "wget ", "curl http",
    "| type", "cmd.exe", "/bin/sh", "%0acat", "chmod ", "dir ",
]

# 4) TF-IDF 配置
URL_TFIDF_MAX_FEATURES = 1000   # URL 文本词表上限
BODY_TFIDF_MAX_FEATURES = 500   # Body 文本词表上限
TFIDF_TOKEN_PATTERN = r"(?u)[\w%\-./:=?&]+"  # 自定义分词：保留 % . / = ? 等符号
                                              # （默认分词会丢掉 %2e 这类关键 token）


# ============================================================
# 辅助函数
# ============================================================
def shannon_entropy(text: str) -> float:
    """
    计算字符串的香农信息熵（字符分布的随机程度，取值 0 ~ log2(字符种数)）。

    直觉：正常 URL（如 /index.html?id=5）字符分布有规律、熵较低；
    攻击 URL 编码多、字符杂乱，熵通常更高。
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _count_query_params(query: str) -> int:
    """统计查询串里的参数个数：'a=1&b=2' -> 2；空串 -> 0。"""
    query = str(query)
    return query.count("&") + 1 if query else 0


# ============================================================
# 规则特征提取（无状态函数，训练 / 预测共用）
# ============================================================
def extract_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 的每行（一条请求）提取 40 维规则特征 + 统计特征。

    参数:
        df: 必须包含 method/url/path/query/headers/body 列
            （可含任意其他列，会被忽略）
    返回:
        仅包含数值特征的 DataFrame（顺序固定，可复现）
    """
    # 先做文本预处理（得到解码文本列）；若已有 text_all 列则直接复用
    if "text_all" not in df.columns:
        df = preprocess_dataset(df)

    feats = pd.DataFrame(index=df.index)

    # ---------- A1. HTTP 方法 one-hot ----------
    method_upper = df["method"].str.upper()
    for m in METHODS:
        feats[f"method_{m}"] = (method_upper == m).astype(np.float32)

    # ---------- A2. 长度与结构特征 ----------
    feats["url_length"] = df["url"].str.len()
    feats["path_length"] = df["path"].str.len()
    feats["query_length"] = df["query"].str.len()
    feats["body_length"] = df["body"].str.len()
    feats["headers_length"] = df["headers"].str.len()
    feats["n_query_params"] = df["query"].apply(_count_query_params)
    feats["path_depth"] = df["path"].str.count("/")          # 路径层级数

    # ---------- A3. 特殊字符计数与密度（在原始 url+body 上统计） ----------
    full_raw = df["url"] + " " + df["body"]
    text_len = full_raw.str.len().replace(0, 1)              # 防除零
    for ch, name in COUNT_CHARS:
        feats[name] = full_raw.str.count(re.escape(ch)).astype(np.float32)
    # 特殊字符总密度 = 敏感字符总数 / 文本总长度
    feats["special_char_density"] = (
        full_raw.str.count("[" + SPECIAL_CHARS_CLASS + "]") / text_len
    ).astype(np.float32)

    # ---------- A4. 编码特征 ----------
    encoded_pat = r"%[0-9a-fA-F]{2}"                          # %xx 形态
    feats["n_encoded"] = full_raw.str.count(encoded_pat).astype(np.float32)
    feats["encoded_ratio"] = (feats["n_encoded"] / df["url"].str.len().replace(0, 1)).astype(np.float32)
    feats["double_encoded"] = full_raw.str.contains("%25", regex=False).astype(np.float32)  # %25=% 的编码 => 双重编码迹象
    feats["n_consecutive_slashes"] = full_raw.apply(
        lambda s: max((len(x) for x in re.findall(r"/{2,}", s)), default=0)
    ).astype(np.float32)                                     # 连续斜杠最大长度
    feats["n_dotdot_slash"] = df["text_all"].str.count(re.escape("../")).astype(np.float32)

    # ---------- A5. URL 信息熵 ----------
    feats["url_entropy"] = df["url"].apply(shannon_entropy).astype(np.float32)

    # ---------- A6. 攻击关键词命中计数（解码后小写全文） ----------
    text_all = df["text_all"]
    feats["n_sqli_keywords"] = text_all.apply(
        lambda s: sum(1 for kw in SQLI_KEYWORDS if kw in s)).astype(np.float32)
    feats["n_xss_keywords"] = text_all.apply(
        lambda s: sum(1 for kw in XSS_KEYWORDS if kw in s)).astype(np.float32)
    feats["n_traversal_keywords"] = text_all.apply(
        lambda s: sum(1 for kw in TRAVERSAL_KEYWORDS if kw in s)).astype(np.float32)
    feats["n_cmd_keywords"] = text_all.apply(
        lambda s: sum(1 for kw in CMD_KEYWORDS if kw in s)).astype(np.float32)
    feats["n_total_keywords"] = (
        feats["n_sqli_keywords"] + feats["n_xss_keywords"]
        + feats["n_traversal_keywords"] + feats["n_cmd_keywords"]
    )

    return feats


# ============================================================
# 特征流水线：规则特征 + TF-IDF（保证训练/预测特征一致）
# ============================================================
class FeaturePipeline:
    """
    完整特征流水线。

    用法:
        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(train_df)   # 训练：拟合 TF-IDF 词表
        X_test  = pipeline.transform(test_df)        # 预测：复用已拟合词表

    该对象会随模型一起保存进 model.pkl，保证预测时特征与训练一致。
    """

    def __init__(self):
        # URL 文本 TF-IDF：词级 1~2 元组，保留编码符号，子线性 TF 抑制长文本影响
        self.url_tfidf = TfidfVectorizer(
            max_features=URL_TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            token_pattern=TFIDF_TOKEN_PATTERN,
            sublinear_tf=True,
        )
        # Body 文本 TF-IDF
        self.body_tfidf = TfidfVectorizer(
            max_features=BODY_TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            token_pattern=TFIDF_TOKEN_PATTERN,
            sublinear_tf=True,
        )
        self.rule_feature_names_ = None   # 规则特征列名（fit 后确定）
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """训练阶段入口：拟合 TF-IDF 词表并输出完整特征矩阵。"""
        df = preprocess_dataset(df)
        rule_feats = extract_rule_features(df)      # 规则特征（无状态）
        self.rule_feature_names_ = rule_feats.columns.tolist()

        url_mat = self.url_tfidf.fit_transform(df["text_url"])    # 拟合 + 转换
        body_mat = self.body_tfidf.fit_transform(df["text_body"])
        self._fitted = True
        return self._stack(rule_feats, url_mat, body_mat)

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """预测阶段入口：只做转换，不重新拟合词表。"""
        if not self._fitted:
            raise RuntimeError("FeaturePipeline 尚未 fit，请先调用 fit_transform！")
        df = preprocess_dataset(df)
        rule_feats = extract_rule_features(df)
        url_mat = self.url_tfidf.transform(df["text_url"])        # 只转换
        body_mat = self.body_tfidf.transform(df["text_body"])
        return self._stack(rule_feats, url_mat, body_mat)

    def _stack(self, rule_feats: pd.DataFrame, url_mat, body_mat) -> sparse.csr_matrix:
        """把规则特征（稠密）与两块 TF-IDF（稀疏）横向拼接为一个稀疏矩阵。"""
        rule_sparse = sparse.csr_matrix(rule_feats.to_numpy(dtype=np.float32))
        return sparse.hstack([rule_sparse, url_mat, body_mat]).tocsr()

    def get_feature_names_out(self) -> list:
        """
        返回全部特征名（顺序与特征矩阵列一一对应）：
        规则特征名 + 'tfidf_url__词' + 'tfidf_body__词'。
        特征重要性图会用到。
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline 尚未 fit！")
        return (
            self.rule_feature_names_
            + [f"tfidf_url__{w}" for w in self.url_tfidf.get_feature_names_out()]
            + [f"tfidf_body__{w}" for w in self.body_tfidf.get_feature_names_out()]
        )


# ============================================================
# 命令行入口：输出特征矩阵并打印维度
# ============================================================
def main() -> None:
    """第 2 阶段验证入口（features 部分）：构建特征矩阵并打印维度。"""
    from src.data_loader import load_dataset

    df = load_dataset()
    print(f"数据集样本数: {len(df)}，恶意占比: {df['label'].mean():.1%}")

    print()
    print("=" * 70)
    print("第 1 步：提取规则特征 + 统计特征")
    print("=" * 70)
    rule_feats = extract_rule_features(df)
    print(f"规则特征共 {rule_feats.shape[1]} 维：")
    for i in range(0, len(rule_feats.columns), 5):
        print("  " + "  ".join(f"{c:22s}" for c in rule_feats.columns[i:i + 5]))

    print()
    print("=" * 70)
    print("第 2 步：构建完整特征矩阵（规则特征 + URL TF-IDF + Body TF-IDF）")
    print("=" * 70)
    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(df)
    n_url_words = len(pipeline.url_tfidf.vocabulary_)
    n_body_words = len(pipeline.body_tfidf.vocabulary_)
    print(f"特征矩阵维度      : {X.shape[0]} 行 x {X.shape[1]} 列")
    print(f"  - 规则/统计特征 : {len(pipeline.rule_feature_names_)} 维")
    print(f"  - URL  TF-IDF   : {n_url_words} 维（上限 {URL_TFIDF_MAX_FEATURES}）")
    print(f"  - Body TF-IDF   : {n_body_words} 维（上限 {BODY_TFIDF_MAX_FEATURES}）")
    print(f"稀疏度             : 非零元素占比 {X.nnz / (X.shape[0] * X.shape[1]):.2%}")

    print()
    print("=" * 70)
    print("第 3 步：单条样本特征示例（取 1 条 SQL 注入请求）")
    print("=" * 70)
    idx = df[df["attack_type"] == "sql_injection"].index[0]
    row = rule_feats.loc[idx]
    print(f"URL: {df.loc[idx, 'url'][:90]}")
    print("非零规则特征:")
    for name, val in row[row != 0].items():
        print(f"  {name:24s} = {val}")

    print()
    print("√ 第 2 阶段完成：特征矩阵构建流程跑通！")


if __name__ == "__main__":
    # Windows 控制台强制 UTF-8 输出，避免中文乱码
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
