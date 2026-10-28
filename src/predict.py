# -*- coding: utf-8 -*-
"""
predict.py —— 单条 HTTP 请求预测（命令行工具，第 4 阶段）

功能：
    加载 run.py 训练保存的 model.pkl（特征流水线 + 最优模型），
    对一条"原始 HTTP 请求文本"预测它是正常还是恶意，并解释预测依据
    （命中的攻击关键词）。

使用方式（在项目根目录运行）：
    1) 直接指定请求文本:
       python src/predict.py --text "GET /index.html HTTP/1.1"
       python src/predict.py --text "GET /?id=1' OR 1=1 -- HTTP/1.1"
    2) 从文件读取（适合多行完整报文）:
       python src/predict.py --file request.txt
    3) 交互式输入（不带参数）:
       python src/predict.py
       按提示粘贴多行报文，最后单独输入一行 END 结束。

预测流程与训练完全一致（特征一致性由 model.pkl 中的 FeaturePipeline 保证）：
    原始文本 -> parse_http_request() 解析 -> pipeline.transform() 特征
             -> model.predict_proba() 恶意概率 -> 阈值 0.5 判定
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import joblib

# 保证项目根目录在 sys.path 中（支持 python src/predict.py 直接运行）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import parse_http_request, recursive_url_decode  # noqa: E402
from src.features import (                                            # noqa: E402
    SQLI_KEYWORDS, XSS_KEYWORDS, TRAVERSAL_KEYWORDS, CMD_KEYWORDS,
)

MODEL_PATH = PROJECT_ROOT / "model.pkl"

# 四类攻击词表（与 features.py 保持一致），用于解释"为什么判为恶意"
ATTACK_KEYWORD_MAP = [
    ("SQL 注入", SQLI_KEYWORDS),
    ("XSS", XSS_KEYWORDS),
    ("路径遍历", TRAVERSAL_KEYWORDS),
    ("命令注入", CMD_KEYWORDS),
]


def load_model_bundle(path: Path = None) -> dict:
    """
    加载训练阶段保存的模型包。

    包内容: {"pipeline": FeaturePipeline, "model": 已训练模型,
             "model_name": 模型名, "metrics": 训练时指标}
    """
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"未找到模型文件 {path}，请先在项目根目录运行 python run.py 训练模型！")
    return joblib.load(path)


def find_hit_keywords(parsed: dict) -> list:
    """
    在请求文本中查找命中的攻击关键词（用于解释预测依据）。

    匹配对象 = URL + Query + Body 的多轮解码小写文本，
    与训练时关键词特征的口径一致。
    返回:
        [(攻击类型, [命中的关键词, ...]), ...]（只保留有命中的类型）
    """
    text = recursive_url_decode(
        parsed["url"] + " " + parsed["query"] + " " + parsed["body"]
    ).lower()
    hits = []
    for atype, keywords in ATTACK_KEYWORD_MAP:
        matched = [kw for kw in keywords if kw in text]
        if matched:
            hits.append((atype, matched))
    return hits


def predict_request(raw_text: str, bundle: dict = None) -> dict:
    """
    对单条 HTTP 请求文本做预测。

    参数:
        raw_text: 原始 HTTP 请求文本（完整报文 / 简化报文 / 纯 URL 均可）
        bundle:   已加载的模型包（不传则自动加载）
    返回:
        {
          "label":       0=正常 / 1=恶意,
          "label_text":  "正常请求" / "恶意请求",
          "malicious_prob": 恶意概率 (0~1),
          "parsed":      解析后的结构化字段,
          "hits":        命中的攻击关键词列表,
          "model_name":  使用的模型名,
        }
    """
    bundle = bundle or load_model_bundle()
    parsed = parse_http_request(raw_text)              # ① 解析报文

    one_df = pd.DataFrame([parsed])                    # ② 单行 DataFrame
    X = bundle["pipeline"].transform(one_df)           # ③ 与训练一致的特征
    proba = float(bundle["model"].predict_proba(X)[0, 1])  # ④ P(恶意)

    label = int(proba >= 0.5)                          # ⑤ 0.5 阈值判定
    return {
        "label": label,
        "label_text": "恶意请求" if label == 1 else "正常请求",
        "malicious_prob": proba,
        "parsed": parsed,
        "hits": find_hit_keywords(parsed),
        "model_name": bundle["model_name"],
    }


def print_prediction(result: dict) -> None:
    """把预测结果格式化打印到终端。"""
    p = result["parsed"]
    print()
    print("=" * 62)
    print("请求解析")
    print("=" * 62)
    print(f"  方法   : {p['method']}")
    print(f"  URL    : {p['url'][:90]}")
    if p["query"]:
        print(f"  Query  : {p['query'][:90]}")
    if p["body"]:
        print(f"  Body   : {p['body'][:90]}")

    print()
    print("=" * 62)
    print("预测结果")
    print("=" * 62)
    print(f"  判定       : {result['label_text']} (label={result['label']})")
    print(f"  恶意概率   : {result['malicious_prob']:.4f}（判定阈值 0.5）")
    print(f"  使用模型   : {result['model_name']}")
    if not str(p.get("headers", "")).strip():
        print("  [提示]     : 未检测到请求头（Host/User-Agent 等），")
        print("               训练数据的正常请求均带完整请求头，")
        print("               建议粘贴完整报文以获得更准确的判定。")
    if result["hits"]:
        print("  命中关键词 :")
        for atype, kws in result["hits"]:
            print(f"    [{atype}] {'、'.join(kws[:8])}")
    else:
        print("  命中关键词 : 无")
    print("=" * 62)


def read_interactive() -> str:
    """交互式读取多行请求文本：最后单独一行输入 END 结束。"""
    print("请粘贴 HTTP 请求（支持多行），最后单独输入一行 END 后回车：")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    """命令行入口：解析参数 -> 预测 -> 打印结果。"""
    parser = argparse.ArgumentParser(description="恶意 HTTP 请求识别（单条预测）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", "-t", help="待预测的 HTTP 请求文本")
    group.add_argument("--file", "-f", help="待预测的 HTTP 请求文本文件路径")
    args = parser.parse_args()

    if args.text:
        raw = args.text
    elif args.file:
        raw = Path(args.file).read_text(encoding="utf-8", errors="replace")
    else:
        raw = read_interactive()

    if not raw.strip():
        print("x 输入为空，无法预测！")
        sys.exit(1)

    bundle = load_model_bundle()
    result = predict_request(raw, bundle)
    print_prediction(result)


if __name__ == "__main__":
    # Windows 控制台强制 UTF-8 输出，避免中文乱码
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
