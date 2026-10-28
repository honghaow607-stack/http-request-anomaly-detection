# -*- coding: utf-8 -*-
"""
preprocess.py —— 文本清洗、URL 解析、单条请求预处理模块（第 2 阶段）

功能概述
--------
1. recursive_url_decode(): 多轮 URL 解码。
   攻击者常用 %252e%252e（即 .. 的双重编码）绕过检测，所以要循环解码
   直到文本不再变化（最多 3 轮），解码后的文本才方便做关键词匹配；
2. parse_http_request(): 把一条"原始 HTTP 报文"（用户粘贴的多行文本）
   解析成与 dataset.csv 一致的结构化字段，供 predict.py / app.py 使用；
3. preprocess_dataset(): 给 DataFrame 增加文本预处理列：
       decoded_url / decoded_body   解码后的 URL / Body
       text_url  / text_body        原文 + 解码文拼接（喂给 TF-IDF）
       text_all                     全文小写版（喂给攻击关键词匹配）
   这样规则特征与 TF-IDF 在训练和预测时使用完全相同的文本管线。

设计原则：所有函数都保持"无训练状态"，可以安全地对单条请求重复调用。
"""

import re
from urllib.parse import urlsplit, unquote

import pandas as pd

# 请求行正则：形如 "GET /xxx?a=1 HTTP/1.1"（方法 + 目标 + HTTP 版本）
# 目标部分用贪婪 (.+)：兼容教科书式载荷（如 GET /?id=1' OR 1=1 -- HTTP/1.1），
# 这类载荷含空格、不符合严格 HTTP 规范（真实攻击中会被 URL 编码），
# 但作为教学检测工具应完整保留载荷而不是按空格截断
REQUEST_LINE_RE = re.compile(
    r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+(.+?)\s+HTTP/[\d.]+\s*$",
    re.IGNORECASE,
)

# 合法 HTTP 方法集合（用于识别简化输入）
VALID_METHODS = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}


def recursive_url_decode(text, max_rounds: int = 3) -> str:
    """
    多轮 URL 解码：把 %2e%2e%2f、%252e 这类编码还原成明文。

    为什么需要"多轮"？
        单轮解码只能还原一层编码。攻击者为了绕过 WAF 规则，会把载荷
        编码两次（如 .. -> %2e%2e -> %252e%252e），单轮解码后仍是
        "%2e%2e"，检测不到攻击特征。循环解码直到结果不再变化为止。

    参数:
        text:       原始文本
        max_rounds: 最多解码轮数（防御性上限，防止死循环）
    返回:
        解码后的字符串（非字符串输入一律返回空串）
    """
    if not isinstance(text, str):
        return ""
    if "%" not in text:          # 不含编码字符，无需解码
        return text
    prev = text
    for _ in range(max_rounds):
        cur = unquote(prev)      # 标准百分号解码
        if cur == prev:          # 再解码也不变了 => 停止
            break
        prev = cur
    return prev


def parse_http_request(raw_text: str) -> dict:
    """
    解析用户粘贴的原始 HTTP 请求文本为结构化字典。

    支持三种输入格式（自动识别）：
        ① 完整报文: 请求行 + 请求头 + 空行 + 请求体
        ② 简化报文: "GET /path?a=1"（只有一行，省略 HTTP/1.1）
        ③ 纯 URL:   "/path?a=1" 或 "http://host/path?a=1"

    返回:
        dict，字段与 dataset.csv 保持一致：
        method / url / path / query / headers / body
    """
    text = str(raw_text).replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("HTTP 请求内容为空，请输入有效请求！")
    lines = text.split("\n")
    first = lines[0].strip()

    # ---------- 情形①：第一行是标准请求行 ----------
    m = REQUEST_LINE_RE.match(first)
    if m:
        method, target = m.group(1).upper(), m.group(2)
        header_lines, body_lines = [], []
        in_body = False
        for line in lines[1:]:
            s = line.strip()
            if s == "":              # 空行之后 => 请求体开始
                in_body = True
            elif in_body:
                body_lines.append(s)
            else:
                header_lines.append(s)
    # ---------- 情形②/③：简化输入 ----------
    else:
        parts = first.split()
        if len(parts) >= 2 and parts[0].upper() in VALID_METHODS:
            method = parts[0].upper()
            # 目标取"方法之后的全部内容"：兼容含空格的教科书式载荷
            # （合法请求行目标无空格，此写法不影响正常输入）
            target = " ".join(parts[1:])
        else:
            method, target = "GET", first
        header_lines, body_lines = [], []
        in_body = False
        for line in lines[1:]:
            s = line.strip()
            if s == "":
                in_body = True
                continue
            if in_body:
                body_lines.append(s)
            elif re.match(r"^[A-Za-z0-9\-_]+:\s*.+", s):  # 形如 "Key: value" => 请求头
                header_lines.append(s)
            else:                                          # 其余 => 请求体
                body_lines.append(s)

    # 拆分 target 为 path 和 query（无协议时补一个占位主机，便于统一解析）
    target = target if ("://" in target or target.startswith("/")) else "/" + target
    split = urlsplit(target if "://" in target else "http://localhost" + target)
    return {
        "method": method,
        "url": target,
        "path": split.path or "/",
        "query": split.query,
        "headers": "\n".join(header_lines),
        "body": " ".join(body_lines),
    }


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    为数据集 DataFrame 添加文本预处理列（训练与预测共用同一管线）。

    必需输入列: method / url / path / query / headers / body
                （缺失的列自动补空字符串，增强容错）

    新增列:
        decoded_url / decoded_body : 多轮解码后的 URL / Body
        text_url  / text_body      : 原文 + 解码文拼接，喂给 TF-IDF
                                     （编码形态与明文形态都有区分度，两者都保留）
        text_all                   : 全部文本拼接并小写，喂给攻击关键词匹配
    返回:
        带新列的 DataFrame 副本（不修改原对象）
    """
    df = df.copy()
    # ---------- 清洗：保证 6 个基础列都是非空字符串 ----------
    for col in ("method", "url", "path", "query", "headers", "body"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # ---------- 多轮解码 ----------
    df["decoded_url"] = df["url"].apply(recursive_url_decode)
    df["decoded_body"] = df["body"].apply(recursive_url_decode)

    # ---------- TF-IDF 输入文本：原文 + 解码文（两种形态都保留） ----------
    df["text_url"] = df["url"] + " " + df["decoded_url"]
    df["text_body"] = df["body"] + " " + df["decoded_body"]

    # ---------- 关键词匹配输入文本：全文小写 ----------
    df["text_all"] = (
        df["url"] + " " + df["query"] + " " + df["body"]
        + " " + df["decoded_url"] + " " + df["decoded_body"]
    ).str.lower()
    return df


# ============================================================
# 命令行入口：演示清洗与解析效果
# ============================================================
def main() -> None:
    """第 2 阶段验证入口（preprocess 部分）：演示解码与报文解析。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_loader import load_dataset

    df = load_dataset()
    print("=" * 70)
    print("演示 1：URL 多轮解码效果")
    print("=" * 70)
    demo_texts = [
        "%3Cscript%3Ealert(1)%3C%2Fscript%3E",      # 单层编码的 XSS
        "%252e%252e%252fetc%252fpasswd",            # 双重编码的路径遍历
        "/normal/page?id=123",                      # 没有编码的普通 URL
    ]
    for t in demo_texts:
        print(f"  原始: {t}")
        print(f"  解码: {recursive_url_decode(t)}\n")

    print("=" * 70)
    print("演示 2：数据集预处理（新增文本列）")
    print("=" * 70)
    processed = preprocess_dataset(df)
    sample = processed[processed["label"] == 1].iloc[0]
    print(f"  原始 URL      : {sample['url'][:80]}")
    print(f"  解码后 URL    : {sample['decoded_url'][:80]}")

    print()
    print("=" * 70)
    print("演示 3：解析用户粘贴的原始 HTTP 报文")
    print("=" * 70)
    raw_request = (
        "POST /tienda1/publico/login.jsp HTTP/1.1\n"
        "Host: localhost:8080\n"
        "Content-Type: application/x-www-form-urlencoded\n"
        "\n"
        "user=admin' OR 1=1 --&pass=123"
    )
    parsed = parse_http_request(raw_request)
    for k, v in parsed.items():
        print(f"  {k:8s}: {v}")

    print("\n√ preprocess.py 验证通过！")


if __name__ == "__main__":
    main()
