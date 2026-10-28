# -*- coding: utf-8 -*-
"""
app.py —— Streamlit 交互式预测页面（第 4 阶段）

功能：
    在浏览器中粘贴一条 HTTP 请求（完整报文 / 简化报文 / 纯 URL 均可），
    点击"开始预测"立即得到：
      - 判定结果（恶意 / 正常）与恶意概率进度条；
      - 命中的攻击关键词（预测依据解释）；
      - 请求解析详情（方法 / URL / 头部 / 请求体）。

运行方式（在项目根目录）：
    streamlit run src/app.py
    首次运行浏览器会自动打开 http://localhost:8501
    （若未自动打开，手动访问该地址即可；按 Ctrl+C 停止服务）

前置条件：先运行 python run.py 生成 model.pkl
"""

import sys
from pathlib import Path

# streamlit run 的脚本工作目录是 src/，需要手动把项目根加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.predict import load_model_bundle, predict_request

# ---------- 页面基础配置 ----------
st.set_page_config(page_title="恶意 HTTP 请求识别系统", page_icon="🛡️", layout="wide")
st.title("🛡️ 基于机器学习的恶意 HTTP 请求识别系统")
st.caption("粘贴一条 HTTP 请求，模型将判断它是正常请求还是攻击（SQL 注入 / XSS / 路径遍历 / 命令注入）")


# ---------- 模型加载（缓存：页面刷新不重复加载） ----------
@st.cache_resource(show_spinner="正在加载模型 ...")
def get_bundle() -> dict:
    """加载 model.pkl（只加载一次，之后所有预测复用）。"""
    return load_model_bundle()


bundle = get_bundle()
model_name = bundle["model_name"]
train_metrics = bundle.get("metrics", {})

# ---------- 侧边栏：模型信息 ----------
with st.sidebar:
    st.header("模型信息")
    st.metric("当前模型", model_name)
    for key, label in [("accuracy", "准确率"), ("precision", "精确率"),
                       ("recall", "召回率"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]:
        if key in train_metrics:
            st.metric(label, f"{train_metrics[key]:.4f}")
    st.divider()
    st.markdown(
        "**判定阈值**: 0.5\n\n"
        "**预测依据**: 解码后的请求文本会与四类攻击词表匹配，"
        "命中的关键词会展示在结果中。\n\n"
        "⚠️ 模型基于 CSIC 2010 场景训练，"
        "对训练分布之外的攻击模式泛化能力有限。"
    )

# ---------- 内置示例（一键填入，方便快速体验） ----------
EXAMPLES = {
    "正常 - 浏览首页": (
        "GET /index.html HTTP/1.1\n"
        "Host: localhost:8080\n"
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
    ),
    "正常 - 登录请求": (
        "POST /tienda1/publico/autenticar.jsp HTTP/1.1\n"
        "Host: localhost:8080\n"
        "Content-Type: application/x-www-form-urlencoded\n"
        "\n"
        "usr_login=alice&usr_passwd=pass123"
    ),
    "攻击 - SQL 注入": (
        # 注意：URL 中的空格必须编码为 %20（HTTP 请求行不允许出现空格）
        "GET /tienda1/publico/buscar.jsp?texto=1%27%20UNION%20SELECT%20username%2C%20password%20FROM%20users%20-- HTTP/1.1\n"
        "Host: localhost:8080"
    ),
    "攻击 - XSS": (
        "GET /tienda1/publico/articulo.jsp?id=<script>alert(document.cookie)</script> HTTP/1.1\n"
        "Host: localhost:8080"
    ),
    "攻击 - 路径遍历": (
        "GET /tienda1/publico/descarga.jsp?file=../../../../etc/passwd HTTP/1.1\n"
        "Host: localhost:8080"
    ),
    "攻击 - 命令注入": (
        "POST /tienda1/publico/ping.jsp HTTP/1.1\n"
        "Host: localhost:8080\n"
        "Content-Type: application/x-www-form-urlencoded\n"
        "\n"
        "host=127.0.0.1; cat /etc/passwd"
    ),
}

# ---------- 输入区 ----------
st.subheader("1. 输入 HTTP 请求")
# 示例按钮：点击后写入 session_state，输入框随之更新
btn_cols = st.columns(len(EXAMPLES))
for (name, text), col in zip(EXAMPLES.items(), btn_cols):
    if col.button(name, use_container_width=True):
        st.session_state["request_input"] = text

user_input = st.text_area(
    "支持：完整报文 / 简化报文（GET /path?a=1）/ 纯 URL",
    value=st.session_state.get("request_input", ""),
    key="request_input",
    height=180,
    placeholder="例如：\nGET /index.html HTTP/1.1\nHost: localhost:8080",
)

# ---------- 预测区 ----------
st.subheader("2. 开始预测")
if st.button("🚀 开始预测", type="primary", use_container_width=False):
    if not user_input.strip():
        st.warning("请先输入 HTTP 请求内容！")
    else:
        try:
            with st.spinner("正在预测 ..."):
                result = predict_request(user_input, bundle)  # 复用 CLI 的同一预测函数

            # ---- 大字判定结果：恶意红色 / 正常绿色 ----
            if result["label"] == 1:
                st.error(f"### ⛔ 判定：恶意请求（{result['model_name']}）")
            else:
                st.success(f"### ✅ 判定：正常请求（{result['model_name']}）")

            # ---- 恶意概率进度条 ----
            prob = result["malicious_prob"]
            st.write(f"**恶意概率**: {prob:.4f}（判定阈值 0.5）")
            st.progress(prob, text=f"P(恶意) = {prob:.2%}")

            # ---- 两列布局：命中关键词 + 解析详情 ----
            left, right = st.columns(2)
            with left:
                st.markdown("#### 🔍 命中的攻击关键词")
                if result["hits"]:
                    for atype, kws in result["hits"]:
                        st.markdown(
                            f"- **{atype}**: " + "、".join(f"`{kw}`" for kw in kws[:10])
                        )
                else:
                    st.write("未命中任何攻击词表关键词。")
            with right:
                st.markdown("#### 📄 请求解析详情")
                p = result["parsed"]
                st.write(f"**方法**: {p['method']}")
                st.code(p["url"][:300], language="http")
                if p["query"]:
                    st.write(f"**Query**: {p['query'][:200]}")
                if p["headers"]:
                    with st.expander("请求头"):
                        st.code(p["headers"], language="http")
                if p["body"]:
                    with st.expander("请求体"):
                        st.code(p["body"][:300], language="http")
        except Exception as e:
            st.error(f"预测失败：{type(e).__name__}: {e}")
else:
    st.info("输入请求后点击上方按钮，或先点击一个示例快速体验。")

# ---------- 页脚 ----------
st.divider()
st.caption("项目：基于机器学习的恶意 HTTP 请求识别系统 | 模型：逻辑回归 / 随机森林 / XGBoost 对比训练")
