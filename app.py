"""
Streamlit UI for the Multilingual RAG QA system.

This UI is intentionally query-driven:
- Retrieval, generation, and metrics run once per submitted query
- SHAP and counterfactual analysis are lazy and reuse cached results
- Visuals are derived from the current run, not hardcoded demo data
"""

import html
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pio.templates.default = "plotly_white"


def apply_light_plot_theme(fig, height=None):
    """Keep all Plotly charts readable on the app's light background."""
    text_color = "#0b1720"
    layout = {
        "template": "plotly_white",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"color": text_color, "size": 13},
        "title": {"font": {"color": text_color, "size": 18}},
        "legend": {"font": {"color": text_color, "size": 12}},
        "coloraxis": {
            "colorscale": [
                [0.0, "#93c5fd"],
                [0.35, "#38bdf8"],
                [0.7, "#0f766e"],
                [1.0, "#0f172a"],
            ],
            "colorbar": {
                "tickfont": {"color": text_color, "size": 12},
                "title": {"font": {"color": text_color, "size": 12}},
            },
        },
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": "rgba(16, 32, 39, 0.18)",
            "font": {"color": text_color},
        },
        "uniformtext": {"mode": "hide", "minsize": 10},
        "xaxis": {
            "color": text_color,
            "linecolor": "rgba(11, 23, 32, 0.42)",
            "title": {"font": {"color": text_color, "size": 13}},
            "tickfont": {"color": text_color, "size": 12},
            "gridcolor": "rgba(11, 23, 32, 0.14)",
            "zerolinecolor": "rgba(11, 23, 32, 0.24)",
        },
        "yaxis": {
            "color": text_color,
            "linecolor": "rgba(11, 23, 32, 0.42)",
            "title": {"font": {"color": text_color, "size": 13}},
            "tickfont": {"color": text_color, "size": 12},
            "gridcolor": "rgba(11, 23, 32, 0.14)",
            "zerolinecolor": "rgba(11, 23, 32, 0.24)",
        },
        "margin": {"l": 20, "r": 20, "t": 48, "b": 20},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_traces(
        marker_line_color="rgba(16, 32, 39, 0.22)",
        marker_line_width=1,
        selector=dict(type="bar"),
    )
    return fig


st.set_page_config(
    page_title="Multilingual QA RAG",
    page_icon="RAG",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.13), transparent 28%),
            radial-gradient(circle at 8% 16%, rgba(15, 118, 110, 0.14), transparent 30%),
            linear-gradient(180deg, #f7f4ef 0%, #eef6f4 42%, #f8fafc 100%);
        color: #102027;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 26px;
        padding-bottom: 42px;
    }
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none;
    }
    .main,
    .main p,
    .main span,
    .main div,
    .main label,
    .main li,
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main h5,
    .main h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #102027;
    }
    h1, h2, h3, h4, h5, h6,
    p, label, li, strong,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] p {
        color: #102027;
    }
    code,
    pre,
    [data-testid="stMarkdownContainer"] code {
        background: #e7f3f0;
        color: #102027;
        border: 1px solid rgba(15, 118, 110, 0.16);
        border-radius: 6px;
        padding: 2px 6px;
    }
    .stCaptionContainer,
    [data-testid="stCaptionContainer"] p,
    .caption {
        color: #475569;
    }
    textarea,
    input,
    div[data-baseweb="select"] * {
        color: #102027;
    }
    textarea,
    input,
    div[data-baseweb="select"] > div {
        background-color: #ffffff;
    }
    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        border: 1px solid rgba(15, 35, 43, 0.14);
        box-shadow: 0 8px 18px rgba(16, 32, 39, 0.08);
        color: #102027;
        background: #ffffff;
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stLinkButton > a:hover {
        transform: translateY(-1px);
        border-color: rgba(15, 118, 110, 0.45);
        box-shadow: 0 12px 24px rgba(16, 32, 39, 0.12);
    }
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        color: #ffffff;
        border: 0;
    }
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button span {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #334155;
    }
    .stTabs [aria-selected="true"] p {
        color: #0f766e;
        font-weight: 800;
    }
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid rgba(16, 32, 39, 0.10);
        border-radius: 8px;
        box-shadow: 0 10px 24px rgba(16, 32, 39, 0.05);
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] div[role="button"],
    [data-testid="stExpander"] div[data-testid="stExpanderToggleIcon"] {
        background: #ffffff;
        color: #102027;
    }
    [data-testid="stExpander"] * {
        color: #102027;
    }
    [data-testid="stExpander"] svg {
        color: #0f766e;
        fill: #0f766e;
    }
    [data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid rgba(16, 32, 39, 0.10);
    }
    [data-testid="stAlert"] * {
        color: #102027;
    }
    [data-testid="stDataFrame"] {
        background: #ffffff;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(16, 32, 39, 0.08);
        box-shadow: 0 10px 22px rgba(16, 32, 39, 0.05);
    }
    [data-testid="stDataFrame"] table,
    [data-testid="stTable"] table {
        background: #ffffff;
    }
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        background: #ffffff;
        color: #102027;
        border-color: rgba(16, 32, 39, 0.10);
    }
    [data-testid="stDataFrame"] th,
    [data-testid="stTable"] th {
        background: #eef6f4;
        color: #102027;
        font-weight: 800;
    }
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] *,
    .stDataFrame *,
    .stTable * {
        color: #102027;
    }
    [data-testid="stDataFrame"] div,
    [data-testid="stDataFrame"] span,
    [data-testid="stDataFrame"] canvas {
        background-color: #ffffff;
    }
    .js-plotly-plot {
        background: #ffffff;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 12px 28px rgba(16, 32, 39, 0.05);
    }
    .js-plotly-plot,
    .js-plotly-plot * {
        color: #102027;
    }
    .js-plotly-plot text,
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .gtitle,
    .js-plotly-plot .legendtext {
        fill: #0b1720 !important;
        color: #0b1720 !important;
        opacity: 1 !important;
        font-weight: 600;
    }
    hr {
        border-color: rgba(16, 32, 39, 0.10);
        margin: 26px 0 18px 0;
    }
    .hero-shell {
        background:
            linear-gradient(135deg, rgba(10, 24, 32, 0.88), rgba(15, 118, 110, 0.58)),
            url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1800&q=80");
        background-size: cover;
        background-position: center;
        color: white;
        border-radius: 8px;
        padding: 34px 36px;
        margin-bottom: 10px;
        min-height: 230px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        box-shadow: 0 22px 52px rgba(16, 32, 39, 0.22);
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -80px -120px auto;
        width: 300px;
        height: 300px;
        background: rgba(255, 255, 255, 0.14);
        border-radius: 999px;
    }
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 880px;
    }
    .hero-shell,
    .hero-shell * {
        color: #ffffff;
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 11px;
        font-weight: 700;
        opacity: 0.84;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 10px;
    }
    .hero-copy {
        font-size: 15px;
        max-width: 900px;
        opacity: 0.92;
        line-height: 1.7;
    }
    .hero-stats {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 22px;
    }
    .hero-stat {
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 800;
        backdrop-filter: blur(10px);
    }
    .query-shell {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(16, 32, 39, 0.08);
        border-radius: 8px;
        padding: 18px 18px 8px 18px;
        box-shadow: 0 18px 42px rgba(16, 32, 39, 0.08);
        margin-bottom: 16px;
    }
    .prompt-title {
        color: #102027;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 10px 0;
    }
    .demo-strip {
        background: rgba(15, 118, 110, 0.08);
        border: 1px solid rgba(15, 118, 110, 0.16);
        border-radius: 8px;
        padding: 12px 14px;
        margin: 12px 0 8px 0;
        color: #102027;
        font-size: 13px;
        font-weight: 700;
        line-height: 1.55;
    }
    .demo-actions {
        display: grid;
        grid-template-columns: minmax(180px, 260px) 1fr;
        gap: 12px;
        align-items: stretch;
        margin-top: 12px;
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.90);
        border: 1px solid rgba(16, 32, 39, 0.08);
        border-radius: 8px;
        padding: 18px 18px 20px 18px;
        box-shadow: 0 18px 42px rgba(16, 32, 39, 0.08);
    }
    .answer-box {
        background: #ffffff;
        color: #102027;
        padding: 24px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 16px;
        line-height: 1.8;
        border-left: 6px solid #0f766e;
        box-shadow: 0 14px 32px rgba(16, 32, 39, 0.08);
    }
    .answer-box,
    .answer-box * {
        color: #102027;
    }
    .soft-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(16, 32, 39, 0.08);
        border-radius: 8px;
        padding: 14px 16px;
        margin: 8px 0;
        box-shadow: 0 10px 22px rgba(16, 32, 39, 0.06);
    }
    .badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-high { background: #d8f3dc; color: #1b4332; }
    .badge-medium { background: #fff3bf; color: #7f5539; }
    .badge-low { background: #ffe3e3; color: #9d0208; }
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 10px 0;
    }
    .meta-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 7px 11px;
        background: #ffffff;
        border: 1px solid rgba(16, 32, 39, 0.10);
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        box-shadow: 0 6px 14px rgba(16, 32, 39, 0.05);
    }
    .section-title {
        color: #102027;
        font-size: 22px;
        font-weight: 800;
        margin: 18px 0 6px 0;
    }
    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .summary-card {
        background: #ffffff;
        border: 1px solid rgba(16, 32, 39, 0.08);
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 104px;
        box-shadow: 0 10px 24px rgba(16, 32, 39, 0.07);
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }
    .summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 32px rgba(16, 32, 39, 0.10);
    }
    .summary-label {
        color: #0f766e;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .summary-value {
        color: #0f2c3f;
        font-size: 30px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
    }
    .summary-note {
        color: #64748b;
        font-size: 13px;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 10px 0 4px 0;
    }
    .side-panel {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(16, 32, 39, 0.08);
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 18px 42px rgba(16, 32, 39, 0.08);
        min-height: 100%;
    }
    .side-panel-title {
        color: #102027;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .side-panel-copy {
        color: #475569;
        font-size: 14px;
        line-height: 1.65;
        margin-bottom: 12px;
    }
    .mini-list {
        display: grid;
        gap: 8px;
    }
    .mini-item {
        background: rgba(15, 118, 110, 0.08);
        border: 1px solid rgba(15, 118, 110, 0.13);
        border-radius: 8px;
        padding: 9px 11px;
        color: #102027;
        font-size: 13px;
        font-weight: 700;
    }
    .doc-meta-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        margin-bottom: 12px;
    }
    .doc-meta-pill {
        background: #eef6f4;
        border: 1px solid rgba(15, 118, 110, 0.16);
        border-radius: 8px;
        padding: 8px 10px;
        color: #102027;
        font-size: 13px;
        font-weight: 700;
    }
    .doc-preview {
        background: #ffffff;
        border: 1px solid rgba(16, 32, 39, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
        color: #102027;
        line-height: 1.7;
    }
    .query-hit {
        background: #fde68a;
        color: #102027;
        border-radius: 4px;
        padding: 1px 4px;
        font-weight: 800;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0f766e 0%, #2563eb 100%);
    }
    .empty-shell {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid rgba(16, 32, 39, 0.08);
        padding: 22px 24px;
        box-shadow: 0 12px 28px rgba(16, 32, 39, 0.06);
    }
    .empty-title {
        color: #102027;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .empty-copy {
        color: #475569;
        line-height: 1.7;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load heavy components once per app process."""
    from evaluation.metrics import RAGMetrics
    from generation.qa_generator import QAGenerator
    from retrieval.search import Retriever

    retriever = Retriever()
    generator = QAGenerator()
    evaluator = RAGMetrics()
    return retriever, generator, evaluator


def initialize_session_state():
    defaults = {
        "current_query": "",
        "last_results": None,
        "shap_results": None,
        "counterfactual_results": None,
        "whatif_result": None,
        "suggested_counterfactuals": None,
        "analysis_mode": "quick",
        "demo_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=30)
def load_recent_runs(limit=12, db_path="mlflow.db"):
    """Load recent MLflow runs directly from the local SQLite database."""
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = """
        SELECT
            r.run_uuid,
            r.start_time,
            COALESCE(MAX(CASE WHEN p.key = 'query' THEN p.value END), '') AS query,
            COALESCE(MAX(CASE WHEN p.key = 'detected_language' THEN p.value END), '') AS detected_language,
            COALESCE(MAX(CASE WHEN p.key = 'answer_language' THEN p.value END), '') AS answer_language,
            COALESCE(MAX(CASE WHEN p.key = 'role' THEN p.value END), '') AS role,
            COALESCE(MAX(CASE WHEN m.key = 'overall_score' THEN m.value END), NULL) AS overall_score,
            COALESCE(MAX(CASE WHEN m.key = 'retrieval_score' THEN m.value END), NULL) AS retrieval_score,
            COALESCE(MAX(CASE WHEN m.key = 'generation_score' THEN m.value END), NULL) AS generation_score,
            COALESCE(MAX(CASE WHEN m.key = 'faithfulness' THEN m.value END), NULL) AS faithfulness,
            COALESCE(MAX(CASE WHEN m.key = 'retrieval_time' THEN m.value END), NULL) AS retrieval_time,
            COALESCE(MAX(CASE WHEN m.key = 'generation_time' THEN m.value END), NULL) AS generation_time
        FROM runs r
        LEFT JOIN params p ON r.run_uuid = p.run_uuid
        LEFT JOIN latest_metrics m ON r.run_uuid = m.run_uuid
        GROUP BY r.run_uuid, r.start_time
        ORDER BY r.start_time DESC
        LIMIT ?
        """
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def reset_analysis_state():
    """Clear cached analysis for a new primary run."""
    st.session_state["shap_results"] = None
    st.session_state["counterfactual_results"] = None
    st.session_state["whatif_result"] = None
    st.session_state["suggested_counterfactuals"] = None
    st.session_state["demo_loaded"] = False


def get_demo_payload():
    """Static demo payload used when the user wants to preview the dashboard."""
    demo_results = {
        "query": "Who is Elon Musk?",
        "query_en": "Who is Elon Musk?",
        "query_lang": "en",
        "answer_lang": "en",
        "role": "student",
        "answer": (
            "Elon Musk is a technology entrepreneur best known for leading Tesla and "
            "SpaceX. He has also been involved with companies such as Neuralink and X."
        ),
        "context_en": (
            "[S1] Elon Musk is an entrepreneur and investor known for Tesla and SpaceX.\n\n"
            "[S2] He has served as CEO of Tesla and has played a major role in commercial spaceflight.\n\n"
            "[S3] Musk is frequently discussed in the context of electric vehicles, rockets, and technology innovation."
        ),
        "retrieved_docs": [
            {
                "title": "Elon Musk",
                "text": "Elon Musk is an entrepreneur and investor known for Tesla, SpaceX, and other technology ventures.",
                "source": "external",
                "language": "en",
                "retrieval_stage": "demo",
                "hybrid_score": 0.96,
                "url": "https://en.wikipedia.org/wiki/Elon_Musk",
            },
            {
                "title": "Tesla",
                "text": "Tesla is an electric vehicle and clean energy company with Elon Musk as a high-profile executive.",
                "source": "local",
                "language": "en",
                "retrieval_stage": "demo",
                "hybrid_score": 0.88,
            },
            {
                "title": "SpaceX",
                "text": "SpaceX is a private aerospace company founded by Elon Musk and focused on launch vehicles and spacecraft.",
                "source": "local",
                "language": "en",
                "retrieval_stage": "demo",
                "hybrid_score": 0.85,
            },
        ],
        "metrics": {
            "overall_score": 0.82,
            "retrieval_score": 0.85,
            "generation_score": 0.79,
            "precision_at_5": 0.88,
            "mrr": 0.90,
            "ndcg_at_5": 0.86,
            "retrieval_quality": 0.82,
            "completeness": 0.85,
            "relevance": 0.76,
            "context_use": 0.72,
            "fluency": 0.95,
            "length_ratio": 0.74,
            "faithfulness": 0.84,
            "confidence": "high",
        },
        "faithfulness_details": {
            "faithfulness_score": 0.84,
            "confidence": "high",
            "unigram_overlap": 0.79,
            "bigram_overlap": 0.58,
        },
        "timings": {
            "retrieval": 0.42,
            "translation_to_en": 0.0,
            "generation": 1.18,
            "evaluation": 0.31,
            "total": 1.91,
        },
        "diagnostics": {
            "source_type": "external",
            "top_score": 0.96,
            "num_results": 3,
            "normalized_query": "Who is Elon Musk?",
            "code_mixed": False,
            "fallback_triggered": False,
            "retrieval_stage": "demo",
            "query_variants": ["Who is Elon Musk?", "Elon Musk"],
        },
    }

    demo_shap = {
        "query_importance": {
            "Elon": 0.45,
            "Musk": 0.38,
            "Who": 0.12,
            "is": 0.05,
        },
        "context_terms": {
            "entrepreneur": 0.32,
            "Tesla": 0.28,
            "SpaceX": 0.25,
            "CEO": 0.22,
            "founder": 0.18,
            "technology": 0.15,
        },
        "summary": {
            "num_query_terms": 4,
            "top_query_term": "Elon",
            "analysis_mode": "demo",
        },
    }

    demo_counterfactual = {
        "baseline_score": 0.96,
        "analysis_mode": "demo",
        "summary": "Top impactful words: Elon and Musk strongly drive retrieval quality.",
        "counterfactuals": [
            {
                "removed_word": "Elon",
                "impact": 0.35,
                "counterfactual_score": 0.61,
                "interpretation": "'Elon' is CRITICAL - removing it drops retrieval score significantly",
            },
            {
                "removed_word": "Musk",
                "impact": 0.30,
                "counterfactual_score": 0.66,
                "interpretation": "'Musk' is CRITICAL - removing it drops retrieval score significantly",
            },
            {
                "removed_word": "Who",
                "impact": 0.05,
                "counterfactual_score": 0.91,
                "interpretation": "'Who' is HELPFUL - removing it slightly affects retrieval",
            },
            {
                "removed_word": "is",
                "impact": 0.02,
                "counterfactual_score": 0.94,
                "interpretation": "'is' is NEUTRAL - removing it has minimal effect",
            },
        ],
    }

    return demo_results, demo_shap, demo_counterfactual


def load_demo_state():
    demo_results, demo_shap, demo_counterfactual = get_demo_payload()
    st.session_state["current_query"] = demo_results["query"]
    st.session_state["last_results"] = demo_results
    st.session_state["shap_results"] = demo_shap
    st.session_state["counterfactual_results"] = demo_counterfactual
    st.session_state["whatif_result"] = None
    st.session_state["suggested_counterfactuals"] = None
    st.session_state["demo_loaded"] = True


def detect_query_language(query: str) -> str:
    from langdetect import detect

    try:
        detected = detect(query)
        lang_map = {
            "en": "en",
            "hi": "hi",
            "bn": "bn",
            "ar": "ar",
            "ur": "ar",
            "pa": "hi",
        }
        return lang_map.get(detected, "en")
    except Exception:
        return "en"


def build_context(retrieved_docs, max_context_chars=1200):
    """Build display/generation context and retain raw source snippets."""
    if not retrieved_docs:
        return "", []

    if retrieved_docs[0]["source"] == "external":
        text = retrieved_docs[0].get("text", "")[:max_context_chars]
        return text, [text]

    context_parts = []
    source_texts = []
    used_chars = 0

    for idx, doc in enumerate(retrieved_docs[:5], 1):
        raw_text = doc.get("text", "").strip()
        if not raw_text:
            continue

        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break

        snippet = raw_text[:remaining]
        title = doc.get("title") or "Untitled"
        context_parts.append(f"[S{idx}] Title: {title}\n{snippet}")
        source_texts.append(snippet)
        used_chars += len(snippet)

    return "\n\n".join(context_parts), source_texts


def build_source_info(retrieved_docs):
    source_info = {"source_type": retrieved_docs[0]["source"], "sources": []}

    if retrieved_docs[0]["source"] == "external":
        source_info["sources"] = [{
            "title": retrieved_docs[0].get("title", "Wikipedia Article"),
            "url": retrieved_docs[0].get("url", ""),
            "wiki_lang": retrieved_docs[0].get("wiki_lang", "en"),
        }]
    else:
        for doc in retrieved_docs[:5]:
            source_info["sources"].append({
                "title": doc.get("title", "Untitled"),
                "hybrid_score": doc.get("hybrid_score", 0) or 0,
                "language": doc.get("language", "unknown"),
            })

    return source_info


def run_core_pipeline(query, answer_lang, role, retriever, generator, evaluator):
    """
    Run retrieval, generation, translation, and evaluation once.
    Optional explainability is triggered later from cached outputs.
    """
    from evaluation.faithfulness import compute_faithfulness

    progress = st.progress(0)
    status = st.empty()

    results = {
        "query": query,
        "answer_lang": answer_lang,
        "role": role,
        "timings": {},
    }

    status.info("Detecting language...")
    query_lang = detect_query_language(query)
    results["query_lang"] = query_lang
    progress.progress(10)

    status.info("Retrieving documents...")
    retrieval_start = time.time()
    retrieved_docs = retriever.search(query, language=query_lang)
    results["timings"]["retrieval"] = time.time() - retrieval_start
    results["retrieved_docs"] = retrieved_docs
    progress.progress(32)

    if not retrieved_docs:
        raise ValueError("No documents retrieved for the given query.")

    context, source_texts = build_context(retrieved_docs)
    source_info = build_source_info(retrieved_docs)

    normalized_query = retrieved_docs[0].get("normalized_query", query)
    query_variants = retrieved_docs[0].get("query_variants", [query])
    code_mixed = bool(retrieved_docs[0].get("code_mixed", False))
    fallback_triggered = retrieved_docs[0].get("source") == "external" or str(
        retrieved_docs[0].get("retrieval_stage", "")
    ).startswith("fallback")

    results["context"] = context
    results["source_texts"] = source_texts
    results["source_info"] = source_info
    results["diagnostics"] = {
        "normalized_query": normalized_query,
        "query_variants": query_variants,
        "code_mixed": code_mixed,
        "fallback_triggered": fallback_triggered,
        "retrieval_stage": retrieved_docs[0].get("retrieval_stage", "unknown"),
        "source_type": retrieved_docs[0].get("source", "unknown"),
        "top_score": retrieved_docs[0].get("hybrid_score"),
        "num_results": len(retrieved_docs),
    }

    status.info("Preparing multilingual context...")
    translation_start = time.time()
    query_en = query if query_lang == "en" else retriever.translate(query, query_lang, "en")
    context_en = context if query_lang == "en" else retriever.translate(context[:2000], query_lang, "en")
    results["timings"]["translation_to_en"] = time.time() - translation_start
    results["query_en"] = query_en
    results["context_en"] = context_en
    progress.progress(48)

    status.info("Generating answer...")
    generation_start = time.time()
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        answer_en = generator.generate_answer(
            question=query_en,
            context=context_en,
            role=role,
            language="en",
            source_info=source_info,
        )

    answer = answer_en
    if answer_lang != "en":
        answer = retriever.translate(answer_en, "en", answer_lang)
    results["timings"]["generation"] = time.time() - generation_start
    results["answer"] = answer
    results["answer_en"] = answer_en
    progress.progress(76)

    status.info("Computing evaluation metrics...")
    evaluation_start = time.time()
    metrics = evaluator.compute_all_metrics(
        question=query_en,
        answer=answer_en,
        context=context_en,
        retrieved_docs=retrieved_docs,
    )
    faithfulness = compute_faithfulness(answer_en, source_texts)
    metrics["faithfulness"] = faithfulness["faithfulness_score"]
    metrics["confidence"] = faithfulness["confidence"]
    results["metrics"] = metrics
    results["faithfulness_details"] = faithfulness
    results["timings"]["evaluation"] = time.time() - evaluation_start
    results["timings"]["total"] = sum(results["timings"].values())
    progress.progress(100)
    status.success("Answer ready.")

    return results


def run_shap_analysis(results, retriever, generator, analysis_mode="quick"):
    from explainability.shap_explainer import RAGShapExplainer

    explainer = RAGShapExplainer(generator, retriever)
    num_samples = 8 if analysis_mode == "quick" else 20
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        query_importance = explainer.explain_query_importance(
            results["query_en"],
            language="en",
            num_samples=num_samples,
        )

    return {
        "query_importance": query_importance,
        "context_terms": derive_context_term_importance(results["query_en"], results["context_en"]),
        "summary": {
            "num_query_terms": len(query_importance),
            "top_query_term": max(query_importance, key=lambda k: abs(query_importance[k])) if query_importance else None,
            "analysis_mode": analysis_mode,
        },
    }


def run_counterfactual_analysis(results, retriever, generator, analysis_mode="quick"):
    from explainability.counterfactual_explainer import CounterfactualExplainer

    explainer = CounterfactualExplainer(generator, retriever)
    top_k = 3 if analysis_mode == "quick" else 5
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        query_cf = explainer.explain_query_words(
            results["query_en"],
            language="en",
            top_k=top_k,
        )

    query_cf["analysis_mode"] = analysis_mode
    return query_cf


def derive_context_term_importance(question, context, limit=12):
    """
    Lightweight dynamic context-term importance derived from question/context overlap.
    This keeps the UI responsive while still surfacing the most salient evidence terms.
    """
    import re
    from collections import Counter

    q_words = {
        token
        for token in re.findall(r"\w+", question.lower())
        if len(token) > 3
    }
    tokens = [
        token
        for token in re.findall(r"\w+", context.lower())
        if len(token) > 3
    ]

    counts = Counter(tokens)
    ranked = []
    for token, count in counts.most_common():
        overlap_bonus = 2 if token in q_words else 0
        score = count + overlap_bonus
        ranked.append((token, score))

    return dict(ranked[:limit])


def metric_badge(confidence):
    mapping = {
        "high": "badge-high",
        "medium": "badge-medium",
        "low": "badge-low",
    }
    badge_class = mapping.get(confidence, "badge-low")
    return f'<span class="badge {badge_class}">Confidence: {confidence.upper()}</span>'


def highlight_query_terms(text, query, max_chars=520):
    """Highlight query terms inside document previews."""
    if not text:
        return ""

    preview = text[:max_chars]
    words = {
        token.lower()
        for token in re.findall(r"\w+", query)
        if len(token) > 2
    }
    if not words:
        return html.escape(preview)

    escaped = html.escape(preview)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    highlighted = pattern.sub(r'<mark class="query-hit">\1</mark>', escaped)
    return highlighted


def build_timeline_entries(results):
    """Create human-readable stage timeline rows from runtime data."""
    timings = results.get("timings", {})
    order = [
        ("retrieval", "Retrieval"),
        ("translation_to_en", "Translate To EN"),
        ("generation", "Generation"),
        ("evaluation", "Evaluation"),
    ]

    rows = []
    elapsed = 0.0
    for key, label in order:
        if key not in timings:
            continue
        duration = float(timings[key])
        start_at = elapsed
        elapsed += duration
        rows.append({
            "Stage": label,
            "Start (s)": round(start_at, 2),
            "Duration (s)": round(duration, 2),
            "End (s)": round(elapsed, 2),
        })

    return rows


def render_hero():
    st.markdown(
        """
<div class="hero-shell">
    <div class="hero-content">
        <div class="hero-kicker">Multilingual QA Studio</div>
        <div class="hero-title">Ask once. See the answer, evidence, and confidence.</div>
        <div class="hero-copy">
            A modern workspace for multilingual retrieval, grounded generation, and optional explanation views across English, Hindi, Bengali, and Arabic.
        </div>
        <div class="hero-stats">
            <span class="hero-stat">4 Languages</span>
            <span class="hero-stat">Evidence-First</span>
            <span class="hero-stat">Live Metrics</span>
            <span class="hero-stat">Optional XAI</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def display_answer(results):
    lang_name = {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali",
        "ar": "Arabic",
    }.get(results["answer_lang"], results["answer_lang"])

    answer_html = html.escape(results["answer"]).replace("\n", "<br>")

    st.markdown('<div class="section-title">Generated Answer</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="meta-row">
    <span class="meta-pill">Language: {lang_name}</span>
    <span class="meta-pill">Style: {results['role'].capitalize()}</span>
    <span class="meta-pill">Source: {results['diagnostics']['source_type'].capitalize()}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="answer-box">{answer_html}</div>', unsafe_allow_html=True)
    st.markdown(metric_badge(results["metrics"]["confidence"]), unsafe_allow_html=True)


def display_summary_cards(results):
    metrics = results["metrics"]
    diag = results["diagnostics"]

    top_score = diag["top_score"] if diag["top_score"] is not None else 1.0
    cards = [
        ("Overall Score", f"{metrics['overall_score']:.3f}", "Combined performance of the current run"),
        ("Retrieval Score", f"{metrics['retrieval_score']:.3f}", "Quality of retrieved evidence"),
        ("Generation Score", f"{metrics['generation_score']:.3f}", "Answer quality for this response"),
        ("Faithfulness", f"{metrics['faithfulness']:.3f}", "Grounding to retrieved evidence"),
        ("Top Doc Score", f"{top_score:.3f}", f"{diag['num_results']} retrieved result(s)"),
    ]
    cards_html = "\n".join(
        f"""
<div class="summary-card">
    <div class="summary-label">{label}</div>
    <div class="summary-value">{value}</div>
    <div class="summary-note">{note}</div>
</div>
"""
        for label, value, note in cards
    )
    st.markdown(f'<div class="summary-grid">{cards_html}</div>', unsafe_allow_html=True)


def display_metrics_dashboard(results):
    metrics = results["metrics"]
    faithfulness = results["faithfulness_details"]
    timings = results["timings"]

    st.markdown("---")
    st.markdown('<div class="section-title">Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Scores are computed from the current answer, evidence, and runtime.</div>',
        unsafe_allow_html=True,
    )

    top_left, top_right, top_far = st.columns([1.25, 1.25, 1.0])

    with top_left:
        retrieval_df = pd.DataFrame({
            "Metric": ["Precision@5", "MRR", "NDCG@5", "Retrieval Quality"],
            "Score": [
                metrics["precision_at_5"],
                metrics["mrr"],
                metrics["ndcg_at_5"],
                metrics["retrieval_quality"],
            ],
        })
        fig = px.bar(
            retrieval_df,
            x="Metric",
            y="Score",
            color="Score",
            color_continuous_scale="GnBu",
            range_y=[0, 1],
            title="Retrieval Evidence Profile",
        )
        apply_light_plot_theme(fig, height=380)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with top_right:
        metric_df = pd.DataFrame({
            "Metric": [
                "Completeness",
                "Relevance",
                "Context Use",
                "Fluency",
                "Length Ratio",
            ],
            "Score": [
                metrics["completeness"],
                metrics["relevance"],
                metrics["context_use"],
                metrics["fluency"],
                metrics["length_ratio"],
            ],
        })
        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            color="Score",
            color_continuous_scale="Mint",
            range_y=[0, 1],
            title="Answer Quality Profile",
        )
        apply_light_plot_theme(fig, height=380)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with top_far:
        faith_df = pd.DataFrame({
            "Component": ["Unigram Overlap", "Bigram Overlap"],
            "Score": [
                faithfulness.get("unigram_overlap", 0.0),
                faithfulness.get("bigram_overlap", 0.0),
            ],
        })
        fig = px.bar(
            faith_df,
            x="Component",
            y="Score",
            color="Component",
            title="Grounding Profile",
            range_y=[0, 1],
            color_discrete_sequence=["#0f766e", "#f97316"],
        )
        apply_light_plot_theme(fig, height=380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    bottom_left, bottom_right = st.columns([1.5, 1.5])

    with bottom_left:
        core_df = pd.DataFrame({
            "Metric": ["Overall", "Retrieval", "Generation", "Faithfulness"],
            "Score": [
                metrics["overall_score"],
                metrics["retrieval_score"],
                metrics["generation_score"],
                metrics["faithfulness"],
            ],
        })
        fig = px.bar(
            core_df,
            x="Metric",
            y="Score",
            color="Score",
            color_continuous_scale="GnBu",
            range_y=[0, 1],
            title="Core Score Snapshot",
        )
        apply_light_plot_theme(fig, height=320)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with bottom_right:
        timing_df = pd.DataFrame({
            "Stage": list(timings.keys()),
            "Seconds": list(timings.values()),
        })
        fig = px.bar(
            timing_df,
            x="Stage",
            y="Seconds",
            color="Seconds",
            color_continuous_scale="Teal",
            title="Runtime by Stage",
        )
        apply_light_plot_theme(fig, height=320)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def display_retrieval_analysis(results):
    docs = results["retrieved_docs"]

    st.markdown("---")
    st.markdown('<div class="section-title">Evidence Retrieved</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Top documents and snippets used to ground the answer.</div>',
        unsafe_allow_html=True,
    )

    chart_rows = []
    for idx, doc in enumerate(docs[:5], 1):
        score = doc.get("hybrid_score")
        if score is None:
            score = 1.0 if doc.get("source") == "external" else 0.0
        chart_rows.append({
            "Rank": f"#{idx}",
            "Title": (doc.get("title") or "Untitled")[:50],
            "Score": float(score),
            "Source": "Wikipedia" if doc.get("source") == "external" else "Local",
        })

    col_chart, col_diag = st.columns([2.2, 1.3])

    with col_chart:
        df = pd.DataFrame(chart_rows)
        fig = px.bar(
            df,
            x="Rank",
            y="Score",
            color="Source",
            hover_data=["Title"],
            title="Top Retrieved Document Scores",
            range_y=[0, 1],
            barmode="group",
            color_discrete_map={"Local": "#0f766e", "Wikipedia": "#f97316"},
        )
        apply_light_plot_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_diag:
        diag = results["diagnostics"]
        st.markdown(
            f"""
<div class="soft-card">
    <div class="side-panel-title">Run Context</div>
    <div class="mini-list">
        <div class="mini-item">Detected language: {results['query_lang'].upper()}</div>
        <div class="mini-item">Evidence source: {diag['source_type'].capitalize()}</div>
        <div class="mini-item">Documents found: {diag['num_results']}</div>
        <div class="mini-item">Fallback used: {'Yes' if diag['fallback_triggered'] else 'No'}</div>
        <div class="mini-item">Query variants: {len(diag['query_variants'])}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    display_retrieved_docs(docs)


def display_retrieved_docs(docs):
    st.markdown("#### Retrieved Documents")

    active_query = st.session_state.get("last_results", {}).get("query", "")

    for idx, doc in enumerate(docs[:5], 1):
        score = doc.get("hybrid_score")
        if score is None:
            score = 1.0 if doc.get("source") == "external" else 0.0
        source_badge = "Wikipedia" if doc.get("source") == "external" else "Local Corpus"
        title = doc.get("title", "Untitled")
        preview = doc.get("text", "")[:500]

        with st.expander(f"[{idx}] {title} | Score: {score:.3f} | {source_badge}"):
            st.markdown(
                f"""
<div class="doc-meta-row">
    <div class="doc-meta-pill">Source: {html.escape(str(doc.get('source', 'unknown')))}</div>
    <div class="doc-meta-pill">Language: {html.escape(str(doc.get('language', 'unknown')))}</div>
    <div class="doc-meta-pill">Stage: {html.escape(str(doc.get('retrieval_stage', 'unknown')))}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if doc.get("url"):
                st.markdown(f"URL: [{doc['url']}]({doc['url']})")
            highlighted = highlight_query_terms(preview + ("..." if len(doc.get("text", "")) > 500 else ""), active_query)
            st.markdown(f'<div class="doc-preview">{highlighted}</div>', unsafe_allow_html=True)


def display_shap_analysis(results, shap_results):
    st.markdown("**SHAP / Query Importance**")
    st.caption(f"Analysis mode: {shap_results.get('summary', {}).get('analysis_mode', 'quick')}")

    query_importance = shap_results.get("query_importance", {})
    if not query_importance:
        st.info("No SHAP query-word importance was produced for this query.")
    else:
        df = pd.DataFrame({
            "Word": list(query_importance.keys()),
            "Importance": list(query_importance.values()),
        }).sort_values("Importance", ascending=True)
        fig = px.bar(
            df,
            x="Importance",
            y="Word",
            orientation="h",
            color="Importance",
            color_continuous_scale="RdYlGn",
            title="Query Word Importance",
        )
        apply_light_plot_theme(fig, height=max(320, len(df) * 42))
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Context Terms Most Relevant To This Run**")
    context_terms = shap_results.get("context_terms", {})
    if not context_terms:
        st.info("No context-term importance available.")
        return

    ctx_df = pd.DataFrame({
        "Term": list(context_terms.keys()),
        "Score": list(context_terms.values()),
    }).sort_values("Score", ascending=True)
    fig = px.bar(
        ctx_df,
        x="Score",
        y="Term",
        orientation="h",
        color="Score",
        color_continuous_scale="Blues",
        title="Context Term Salience",
    )
    apply_light_plot_theme(fig, height=max(320, len(ctx_df) * 34))
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def display_counterfactual_analysis(counterfactual_results):
    st.markdown("**Counterfactual / Query Sensitivity**")
    st.caption(f"Analysis mode: {counterfactual_results.get('analysis_mode', 'quick')}")

    counterfactuals = counterfactual_results.get("counterfactuals", [])
    if not counterfactuals:
        st.info(counterfactual_results.get("summary", "No counterfactual analysis available."))
        return

    df = pd.DataFrame([{
        "Removed Word": item.get("removed_word", "Unknown"),
        "Impact": item.get("impact", 0.0),
        "Counterfactual Score": item.get("counterfactual_score", 0.0),
        "Interpretation": item.get("interpretation", ""),
    } for item in counterfactuals])

    fig = px.bar(
        df,
        x="Removed Word",
        y="Impact",
        color="Impact",
        color_continuous_scale="RdYlGn_r",
        title="Impact Of Removing Each Query Word",
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    apply_light_plot_theme(fig, height=360)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.table(df[["Removed Word", "Impact", "Counterfactual Score", "Interpretation"]])


def display_manual_whatif(whatif_result):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Answer**")
        st.info(whatif_result["original_answer"])
    with col2:
        st.markdown("**Modified Answer**")
        st.success(whatif_result["modified_answer"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Answer Similarity", f"{whatif_result['answer_similarity']:.2%}")
    c2.metric("Retrieval Changed", "Yes" if whatif_result["retrieval_changed"] else "No")
    c3.metric("Docs Retrieved", f"{whatif_result['original_num_docs']} -> {whatif_result['modified_num_docs']}")


def display_auto_counterfactual_suggestions(suggestions):
    if not suggestions:
        st.info("No strong automatic counterfactual suggestions were found for this query.")
        return

    rows = pd.DataFrame([{
        "Type": s.get("type", "unknown"),
        "Modified Query": s.get("modified_query", ""),
        "Change": s.get("change_description", ""),
    } for s in suggestions])
    st.table(rows)

    for idx, suggestion in enumerate(suggestions, 1):
        with st.expander(f"Suggestion {idx}: {suggestion.get('change_description', 'View details')}"):
            st.write(f"Modified query: `{suggestion.get('modified_query', '')}`")
            st.write(suggestion.get("modified_answer", ""))


def display_runtime_timeline(results):
    st.markdown("---")
    st.markdown("### Execution Timeline")

    rows = build_timeline_entries(results)
    if not rows:
        st.info("No runtime timeline data available.")
        return

    timeline_df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=timeline_df["Stage"],
        x=timeline_df["Duration (s)"],
        base=timeline_df["Start (s)"],
        orientation="h",
        marker=dict(
            color=timeline_df["Duration (s)"],
            colorscale="Tealgrn",
            showscale=False,
        ),
        text=[f"{val:.2f}s" for val in timeline_df["Duration (s)"]],
        textposition="inside",
        hovertemplate=(
            "Stage=%{y}<br>"
            "Start=%{base:.2f}s<br>"
            "Duration=%{x:.2f}s<br>"
            "<extra></extra>"
        ),
    ))
    apply_light_plot_theme(fig, height=280)
    fig.update_layout(
        title="Stage-by-Stage Runtime Timeline",
        xaxis_title="Seconds from request start",
        yaxis_title="Stage",
        barmode="stack",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    timeline_col, table_col = st.columns([1.6, 1.0])
    with timeline_col:
        st.table(timeline_df)
    with table_col:
        st.markdown(
            f"""
<div class="soft-card">
    <div class="side-panel-title">Timeline Summary</div>
    <div class="mini-list">
        <div class="mini-item">Total runtime: {results['timings'].get('total', 0):.2f}s</div>
        <div class="mini-item">Retrieval: {results['timings'].get('retrieval', 0):.2f}s</div>
        <div class="mini-item">Generation: {results['timings'].get('generation', 0):.2f}s</div>
        <div class="mini-item">Evaluation: {results['timings'].get('evaluation', 0):.2f}s</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


def display_run_history():
    st.markdown("### MLflow Query History")
    st.caption("This section is loaded from your local `mlflow.db` and shows previously logged queries and scores.")
    runs = load_recent_runs()
    if not runs:
        st.info("No MLflow run history found in `mlflow.db` yet.")
        return

    rows = []
    for run in runs:
        rows.append({
            "Run": run["run_uuid"][:8],
            "Started": pd.to_datetime(run["start_time"], unit="ms").strftime("%d-%m %H:%M") if run.get("start_time") else "-",
            "Query": (run.get("query") or "")[:70],
            "Detected": run.get("detected_language") or "-",
            "Answer": run.get("answer_language") or "-",
            "Role": run.get("role") or "-",
            "Overall": round(run["overall_score"], 3) if run.get("overall_score") is not None else None,
            "Retrieval": round(run["retrieval_score"], 3) if run.get("retrieval_score") is not None else None,
            "Generation": round(run["generation_score"], 3) if run.get("generation_score") is not None else None,
            "Faithfulness": round(run["faithfulness"], 3) if run.get("faithfulness") is not None else None,
        })

    df = pd.DataFrame(rows)
    st.table(df)

    numeric_cols = [col for col in ["Overall", "Retrieval", "Generation", "Faithfulness"] if col in df.columns]
    melted = df[["Run"] + numeric_cols].melt(id_vars="Run", var_name="Metric", value_name="Score").dropna()
    if not melted.empty:
        fig = px.line(
            melted,
            x="Run",
            y="Score",
            color="Metric",
            markers=True,
            title="Recent MLflow Metric Trends",
        )
        apply_light_plot_theme(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)


def display_results(results, retriever, generator):
    if st.session_state.get("demo_loaded"):
        st.info("Demo data is currently loaded. Submit a new query any time to replace it with live results.")

    display_answer(results)
    st.markdown('<div class="section-title">Run Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Quick quality signals for this response.</div>',
        unsafe_allow_html=True,
    )
    display_summary_cards(results)

    display_metrics_dashboard(results)
    display_retrieval_analysis(results)

    st.markdown("---")
    st.markdown("### Explainability and What-If Analysis")
    tab1, tab2, tab3 = st.tabs(["SHAP", "Counterfactual", "What-If"])

    with tab1:
        if st.session_state["shap_results"] is None:
            st.caption("Run SHAP only when you want a deeper explanation.")
            if st.button("Run SHAP Analysis", key="run_shap_btn", use_container_width=True):
                with st.spinner("Running SHAP analysis..."):
                    st.session_state["shap_results"] = run_shap_analysis(
                        results,
                        retriever,
                        generator,
                        st.session_state.get("analysis_mode", "quick"),
                    )
        if st.session_state["shap_results"] is not None:
            display_shap_analysis(results, st.session_state["shap_results"])

    with tab2:
        if st.session_state["counterfactual_results"] is None:
            st.caption("Counterfactual analysis reuses the current run and checks query sensitivity.")
            if st.button("Run Counterfactual Analysis", key="run_cf_btn", use_container_width=True):
                with st.spinner("Running counterfactual analysis..."):
                    st.session_state["counterfactual_results"] = run_counterfactual_analysis(
                        results,
                        retriever,
                        generator,
                        st.session_state.get("analysis_mode", "quick"),
                    )
        if st.session_state["counterfactual_results"] is not None:
            display_counterfactual_analysis(st.session_state["counterfactual_results"])

    with tab3:
        st.markdown("**Manual What-If**")
        modified_query = st.text_area(
            "Modified Query",
            value=results["query"],
            placeholder="Edit the query to see how the answer changes...",
            height=110,
            key="modified_query_box",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Compare Answers", key="compare_answers_btn", use_container_width=True):
                if modified_query.strip() and modified_query.strip() != results["query"].strip():
                    from explainability.counterfactual_explainer import CounterfactualExplainer

                    with st.spinner("Generating what-if comparison..."):
                        cf_explainer = CounterfactualExplainer(generator, retriever)
                        st.session_state["whatif_result"] = cf_explainer.explain_manual_whatif(
                            original_query=results["query"],
                            modified_query=modified_query.strip(),
                            language=results["query_lang"],
                            target_lang=results["answer_lang"],
                        )
                else:
                    st.warning("Please change the query before comparing.")

        with c2:
            if st.button("Suggest Query Alternatives", key="suggest_cf_btn", use_container_width=True):
                from explainability.counterfactual_explainer import CounterfactualExplainer

                with st.spinner("Generating automatic counterfactual suggestions..."):
                    cf_explainer = CounterfactualExplainer(generator, retriever)
                    st.session_state["suggested_counterfactuals"] = cf_explainer.suggest_counterfactuals(
                        query=results["query"],
                        language=results["query_lang"],
                        target_lang=results["answer_lang"],
                        top_k_suggestions=3,
                    )

        if st.session_state["whatif_result"] is not None:
            display_manual_whatif(st.session_state["whatif_result"])

        if st.session_state["suggested_counterfactuals"] is not None:
            st.markdown("**Suggested Counterfactual Queries**")
            display_auto_counterfactual_suggestions(st.session_state["suggested_counterfactuals"])

    display_runtime_timeline(results)


def main():
    initialize_session_state()

    render_hero()

    try:
        with st.spinner("Loading models..."):
            retriever, generator, evaluator = load_models()
    except Exception as exc:
        st.error(f"Failed to load models: {exc}")
        st.stop()

    examples = [
        "Who founded Microsoft?",
        "भारत की राजधानी क्या है?",
        "বাংলাদেশের জাতীয় ফুল কী?",
        "ما هي الطاقة الشمسية؟",
    ]

    examples = [
        "Who founded Microsoft?",
        "भारत की राजधानी क्या है?",
        "বাংলাদেশের জাতীয় ফুল কী?",
        "ما هي الطاقة الشمسية؟",
    ]

    examples = [
        "Who founded Microsoft?",
        "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "\u09ac\u09be\u0982\u09b2\u09be\u09a6\u09c7\u09b6\u09c7\u09b0 \u099c\u09be\u09a4\u09c0\u09af\u09bc \u09ab\u09c1\u09b2 \u0995\u09c0?",
        "\u0645\u0627 \u0647\u064a \u0627\u0644\u0637\u0627\u0642\u0629 \u0627\u0644\u0634\u0645\u0633\u064a\u0629\u061f",
    ]

    st.markdown('<div class="prompt-title">Try a sample question</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(examples))
    for col, example in zip(ex_cols, examples):
        if col.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state["current_query"] = example

    st.markdown('<div class="demo-actions">', unsafe_allow_html=True)
    if st.button("Load Demo Data", key="load_demo_data_btn", use_container_width=True):
        load_demo_state()
        st.rerun()
    st.markdown(
        '<div class="demo-strip">Preview the full dashboard instantly with a built-in run, including evidence, metrics, SHAP, and counterfactual views.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    form_col, guide_col = st.columns([1.65, 1.0])
    with form_col:
        with st.form("query_form", clear_on_submit=False):
            query = st.text_area(
                "Your Question",
                value=st.session_state["current_query"],
                placeholder="Type your question in any supported language...",
                height=136,
            )
            st.session_state["current_query"] = query

            col1, col2 = st.columns(2)
            with col1:
                lang_options = {
                    "English": "en",
                    "Hindi": "hi",
                    "Bengali": "bn",
                    "Arabic": "ar",
                }
                answer_lang_display = st.selectbox("Answer Language", list(lang_options.keys()))
                answer_lang = lang_options[answer_lang_display]

            with col2:
                role = st.selectbox(
                    "Answer Style",
                    ["student", "beginner", "teacher"],
                    format_func=lambda value: value.capitalize(),
                )

            submit = st.form_submit_button("Get Answer", type="primary", use_container_width=True)

    with guide_col:
        if st.session_state["last_results"] is None:
            st.markdown(
                """
<div class="side-panel">
    <div class="side-panel-title">What appears after you ask?</div>
    <div class="side-panel-copy">
        The workspace fills with the answer, evidence, quality scores, runtime, and optional explanation tools.
    </div>
    <div class="mini-list">
        <div class="mini-item">Grounded answer with source type</div>
        <div class="mini-item">Retrieval and generation quality</div>
        <div class="mini-item">Evidence snippets and runtime view</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            timings = st.session_state["last_results"].get("timings", {})
            st.markdown(
                f"""
<div class="side-panel">
    <div class="side-panel-title">Latest Run</div>
    <div class="side-panel-copy">
        Query results are shown below. Use the form to run another question without losing the current layout.
    </div>
    <div class="mini-list">
        <div class="mini-item">Total: {timings.get('total', 0):.2f}s</div>
        <div class="mini-item">Retrieval: {timings.get('retrieval', 0):.2f}s</div>
        <div class="mini-item">Generation: {timings.get('generation', 0):.2f}s</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

    if submit:
        if not query or len(query.strip()) < 3:
            st.warning("Please enter a query with at least 3 characters.")
            return

        reset_analysis_state()
        try:
            with st.spinner("Running retrieval and generation..."):
                st.session_state["last_results"] = run_core_pipeline(
                    query.strip(),
                    answer_lang,
                    role,
                    retriever,
                    generator,
                    evaluator,
                )
        except Exception as exc:
            st.error(f"Error while processing query: {exc}")
            return

    if st.session_state["last_results"] is not None:
        top_tab, history_tab = st.tabs(["Current Query", "MLflow History"])
        with top_tab:
            display_results(st.session_state["last_results"], retriever, generator)
        with history_tab:
            display_run_history()


if __name__ == "__main__":
    main()
