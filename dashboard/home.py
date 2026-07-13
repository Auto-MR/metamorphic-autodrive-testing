import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import inspect
import base64

# ──────────────────────────────────────────────────────────────────────────────
#  PROJECT ROOT + DYNAMIC COUNTS
# ──────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from core.metamorphic_engine.mr_base import MetamorphicRelation
    from metamorphic_relations import steering_mrs, cnn_mrs, lstm_mrs, cross_model_mrs

    _mr_modules = [steering_mrs, cnn_mrs, lstm_mrs, cross_model_mrs]
    _seen_mrs = set()
    for _m in _mr_modules:
        _classes = {
            c for _, c in inspect.getmembers(_m, inspect.isclass)
            if issubclass(c, MetamorphicRelation)
            and c is not MetamorphicRelation
            and c.__module__ == _m.__name__
        }
        _seen_mrs |= _classes
    mr_count = len(_seen_mrs)
except Exception:
    mr_count = 20

MODEL_OPTIONS = ["DAVE-2 (CNN)", "GPR", "SVR", "Random Forest", "Linear Regression", "CNN Depth"]
TRANSFORM_OPTIONS = [
    "brightness", "contrast", "noise", "fog", "rain", "snow",
    "flip", "rotate_5", "scale", "crop",
    "translate", "translate_large",
    "composed_bright_noise", "composed_weather_blur", "composed_shift_bright",
]
model_count = len(MODEL_OPTIONS)
transform_count = len(TRANSFORM_OPTIONS)
mr_category_count = 4
engine_count = 2  # Standard AutoMR + HighPerformanceAutoMR (HPC)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoMR | Metamorphic Testing Framework",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS — design tokens: dark instrumentation panel + signal/scan accents
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-base: #0A0E17;
        --bg-panel: #121826;
        --bg-panel-alt: #182236;
        --accent-signal: #FF6B35;
        --accent-signal-soft: rgba(255, 107, 53, 0.16);
        --accent-scan: #19D3C5;
        --accent-scan-soft: rgba(25, 211, 197, 0.16);
        --accent-indigo: #6C7BFF;
        --accent-rose: #FF5C7A;
        --text-primary: #1D3354;
        --text-secondary: #EDF1F7;
        --text-muted: #8A93A6;
        --border-hairline: #232C42;
    }

    html { scroll-behavior: smooth; }
    * { margin: 0; padding: 0; box-sizing: border-box; }

    .main {
        font-family: 'Inter', sans-serif;
        background:
            radial-gradient(circle at 18% 12%, rgba(25,211,197,0.07), transparent 38%),
            radial-gradient(circle at 84% 6%, rgba(255,107,53,0.06), transparent 42%),
            linear-gradient(var(--bg-base), var(--bg-base));
        color: var(--text-primary);
    }

    /* faint engineering grid backdrop */
    .grid-bg {
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,0,0,0.9), transparent 75%);
    }

    a { color: inherit; }
    .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 500;
        letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent-scan);
        display: flex; align-items: center; gap: 0.5rem; justify-content: center; margin-bottom: 0.6rem;
    }
    .eyebrow::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent-scan); box-shadow: 0 0 8px var(--accent-scan); }
    .section-heading { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.1rem; text-align: center; color: var(--text-primary); margin-bottom: 0.5rem; }
    .section-sub { text-align: center; color: var(--text-muted); font-size: 1rem; max-width: 620px; margin: 0 auto 2.2rem; line-height: 1.55; }
    .section-wrap { max-width: 1100px; margin: 3.4rem auto 0; position: relative; z-index: 5; padding: 0 1rem; }

    /* ── Sticky Navbar ── */
    .navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;

    background: rgba(10,14,23,0.82);
    backdrop-filter: blur(14px);

    border-bottom: 1px solid var(--border-hairline);

    z-index: 1000;

    padding: 0.9rem 2.4rem;

    display: flex;
    justify-content: space-between;
    align-items: center;
    }
    .navbar-logo { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--text-secondary); display: flex; align-items: center; gap: 0.55rem; }
    .navbar-logo .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent-signal); box-shadow: 0 0 10px var(--accent-signal); }
    .navbar-links { display: flex; gap: 2rem; }
    .navbar-links a {
        text-decoration: none; color: var(--text-muted); font-weight: 500; font-size: 0.82rem;
        letter-spacing: 0.06em; text-transform: uppercase; transition: color 0.2s ease;
    }
    .navbar-links a:hover, .navbar-links a:focus-visible { color: var(--accent-scan); outline: none; }

    /* ── Hero ── */
    .hero-wrap { max-width: 1180px; margin: 0 auto; padding: 1rem 1.5rem 2rem; position: relative; z-index: 5; display: flex; align-items: center; gap: 3rem; flex-wrap: wrap; }
    .hero-content { flex: 1 1 420px; min-width: 320px; }
    .hero-tag { font-family: 'JetBrains Mono', monospace; font-size: 2rem; letter-spacing: 0.14em; color: var(--accent-signal); text-transform: uppercase; margin-bottom: 1rem; display: inline-block; padding: 0.35rem 0.9rem; border: 1px solid rgba(255,107,53,0.35); border-radius: 30px; background: var(--accent-signal-soft); }
    .hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 3.6rem; font-weight: 800; line-height: 1.05; color: var(--text-primary); letter-spacing: -0.01em; margin-bottom: 1rem; animation: fadeSlideUp 0.7s ease; }
    .hero h1 span { color: var(--accent-scan); }
    @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
    .hero-subtitle { font-size: 1.08rem; font-weight: 400; color: var(--text-muted); max-width: 480px; line-height: 1.65; margin-bottom: 1.8rem; }
    .hero-ctas { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
    .ghost-btn { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.9rem; color: var(--text-primary); border: 1px solid var(--border-hairline); padding: 0.8rem 1.6rem; border-radius: 10px; text-decoration: none; transition: all 0.2s ease; }
    .ghost-btn:hover, .ghost-btn:focus-visible { border-color: var(--accent-scan); color: var(--accent-scan); outline: none; }

    /* ── Radar / scan signature visual ── */
    .hero-visual { flex: 0 0 300px; display: flex; flex-direction: column; align-items: center; }
    .radar-panel { position: relative; width: 280px; height: 280px; }
    .radar-ring { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); border: 1px solid var(--border-hairline); border-radius: 50%; }
    .radar-r1 { width: 100%; height: 100%; }
    .radar-r2 { width: 68%; height: 68%; }
    .radar-r3 { width: 36%; height: 36%; }
    .radar-cross::before, .radar-cross::after { content: ""; position: absolute; background: var(--border-hairline); }
    .radar-cross::before { top: 0; left: 50%; width: 1px; height: 100%; }
    .radar-cross::after { top: 50%; left: 0; width: 100%; height: 1px; }
    .radar-sweep { position: absolute; inset: 0; border-radius: 50%; background: conic-gradient(from 0deg, var(--accent-scan-soft), transparent 32%); animation: radarSpin 4.2s linear infinite; }
    @keyframes radarSpin { to { transform: rotate(360deg); } }
    .radar-blip { position: absolute; width: 8px; height: 8px; border-radius: 50%; background: var(--accent-scan); animation: blipPulse 2.6s ease-in-out infinite; }
    .radar-blip.flagged { background: var(--accent-signal); animation-name: blipPulseFlag; }
    .b1 { top: 26%; left: 64%; animation-delay: 0s; }
    .b2 { top: 58%; left: 28%; animation-delay: 0.7s; }
    .b3 { top: 72%; left: 66%; animation-delay: 1.4s; }
    .b4 { top: 36%; left: 40%; animation-delay: 2.1s; }
    @keyframes blipPulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(25,211,197,0.45); } 50% { box-shadow: 0 0 0 9px rgba(25,211,197,0); } }
    @keyframes blipPulseFlag { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,107,53,0.45); } 50% { box-shadow: 0 0 0 9px rgba(255,107,53,0); } }
    .radar-caption { margin-top: 1.3rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; color: var(--text-muted); text-transform: uppercase; }

    /* ── Telemetry strip ── */
    .telemetry { background: var(--bg-panel); border: 1px solid var(--border-hairline); border-radius: 16px; display: flex; max-width: 1100px; margin: 1rem auto 0; position: relative; z-index: 5; overflow: hidden; }
    .telemetry-item { flex: 1; text-align: center; padding: 1.4rem 1rem; border-right: 1px solid var(--border-hairline); }
    .telemetry-item:last-child { border-right: none; }
    .telemetry-num { font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 600; color: var(--accent-scan); }
    .telemetry-label { font-size: 0.74rem; font-weight: 500; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

    /* ── Panels (overview / about) ── */
    .panel { background: var(--bg-panel); border: 1px solid var(--border-hairline); border-radius: 18px; padding: 2.4rem 2.6rem; max-width: 1000px; margin: 0 auto; position: relative; z-index: 5; }
    .panel p { font-size: 1.02rem; line-height: 1.75; color: var(--text-muted); margin-top: 0.9rem; }
    .panel p:first-of-type { margin-top: 0; }
    .panel strong { color: var(--text-secondary); }

    /* ── Pipeline (how it works) ── */
    .pipeline { display: flex; justify-content: center; align-items: stretch; gap: 0; flex-wrap: wrap; background: var(--bg-panel); border: 1px solid var(--border-hairline); border-radius: 18px; overflow: hidden; }
    .pipe-step { flex: 1; min-width: 200px; padding: 1.8rem 1.5rem; border-right: 1px solid var(--border-hairline); position: relative; }
    .pipe-step:last-child { border-right: none; }
    .pipe-num { font-family: 'JetBrains Mono', monospace; color: var(--accent-signal); font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; }
    .pipe-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--text-secondary); margin: 0.5rem 0 0.4rem; }
    .pipe-desc { font-size: 0.86rem; color: var(--text-muted); line-height: 1.5; }

    /* ── Capability / model cards ── */
    .feature-card { background: var(--bg-panel); border: 1px solid var(--border-hairline); border-top: 2px solid transparent; border-radius: 14px; padding: 1.5rem 1.3rem; height: 100%; transition: border-color 0.2s ease, transform 0.2s ease; margin-bottom: 1.1rem; }
    .feature-card:hover { border-top-color: var(--accent-scan); transform: translateY(-4px); }
    .feature-icon { font-size: 1.5rem; }
    .feature-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: var(--text-secondary); font-size: 1.02rem; margin-top: 0.6rem; }
    .feature-desc { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.45rem; line-height: 1.5; min-height: 58px; }

    .model-card { background: var(--bg-panel); border: 1px solid var(--border-hairline); border-radius: 14px; padding: 1.3rem 1rem; text-align: center; height: 100%; transition: transform 0.2s ease, border-color 0.2s ease; }
    .model-card:hover { transform: translateY(-4px); border-color: var(--accent-scan); }
    .model-card .feature-icon { font-size: 1.6rem; }
    .model-name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.92rem; color: var(--text-secondary); margin-top: 0.5rem; }
    .model-tag { display: inline-block; margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--accent-scan); background: var(--accent-scan-soft); padding: 0.22rem 0.6rem; border-radius: 30px; }

    .idx-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: var(--accent-signal);
    background: var(--accent-signal-soft);
    border: 1px solid rgba(255,107,53,0.28);
    padding: 0.28rem 0.6rem;
    border-radius: 6px;
    }
    .model-card .idx-tag { color: var(--accent-scan); background: var(--accent-scan-soft); border-color: rgba(25,211,197,0.28); }
    .link-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em; color: var(--accent-scan); margin-right: 0.45rem; }


    /* ── MR category spec cards ── */
    .cat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1.1rem; }
    .cat-card { background: var(--bg-panel); border: 1px solid var(--border-hairline); border-top: 3px solid var(--cat-color); border-radius: 14px; padding: 1.5rem; }
    .cat-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--cat-color); }
    .cat-card h4 { font-family: 'Space Grotesk', sans-serif; font-size: 1.12rem; color: var(--text-secondary); margin: 0.4rem 0 0.6rem; }
    .cat-card p { font-size: 0.86rem; color: var(--text-muted); line-height: 1.55; }
    .cat-formula { font-family: 'JetBrains Mono', monospace; background: var(--bg-panel-alt); border: 1px solid var(--border-hairline); color: var(--cat-color); padding: 0.45rem 0.7rem; border-radius: 8px; display: inline-block; margin-top: 0.9rem; font-size: 0.78rem; }

    /* ── Engine comparison cards ── */
    .engine-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.2rem; }
    .engine-card { background: var(--bg-panel); border: 1px solid var(--border-hairline); border-radius: 16px; padding: 1.8rem 1.8rem; }
    .engine-card.hpc { border-color: rgba(255,107,53,0.35); background: linear-gradient(160deg, var(--bg-panel), var(--bg-panel-alt)); }
    .engine-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent-scan); }
    .engine-card.hpc .engine-tag { color: var(--accent-signal); }
    .engine-card h4 { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; color: var(--text-secondary); margin: 0.4rem 0 0.8rem; }
    .engine-list { list-style: none; display: flex; flex-direction: column; gap: 0.55rem; }
    .engine-list li { font-size: 0.87rem; color: var(--text-muted); padding-left: 1.2rem; position: relative; line-height: 1.5; }
    .engine-list li::before { content: "▸"; position: absolute; left: 0; color: var(--accent-scan); }
    .engine-card.hpc .engine-list li::before { color: var(--accent-signal); }

    /* ── Tech stack badges ── */
    .stack-row { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.7rem; }
    .stack-pill { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--text-secondary); background: var(--bg-panel); border: 1px solid var(--border-hairline); padding: 0.6rem 1.2rem; border-radius: 30px; transition: border-color 0.2s ease, transform 0.2s ease; }
    .stack-pill:hover { border-color: var(--accent-signal); transform: translateY(-3px); }

    /* ── About / author row ── */
    .about-grid { display: flex; gap: 2.2rem; flex-wrap: wrap; margin-top: 1.6rem; }
    .about-item-label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent-scan); }
    .about-item-value { font-size: 0.98rem; color: var(--text-secondary); margin-top: 0.25rem; font-weight: 500; }
    .about-links { margin-top: 1.4rem; display: flex; gap: 1.2rem; }
    .about-links a { font-size: 0.9rem; font-weight: 600; color: var(--accent-signal); text-decoration: none; }
    .about-links a:hover, .about-links a:focus-visible { color: var(--accent-scan); outline: none; }

    /* ── CTA band ── */
    .cta-band { text-align: center; background: linear-gradient(135deg, var(--bg-panel), var(--bg-panel-alt)); border: 1px solid var(--border-hairline); border-radius: 20px; padding: 3rem 2rem; max-width: 1000px; margin: 0 auto; position: relative; z-index: 5; }
    .cta-band h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; color: var(--text-secondary); margin-bottom: 0.6rem; }
    .cta-band p { color: var(--text-muted); margin-bottom: 1.6rem; }

    /* ── Buttons (Streamlit) ── */
    .stButton > button {
        font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; padding: 0.95rem 2.4rem; border-radius: 10px;
        background: var(--accent-signal); color: #0A0E17; box-shadow: 0 10px 26px rgba(255,107,53,0.25);
        transition: all 0.2s ease; border: none;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(255,107,53,0.4); background: #ff7a4d; }
    .stButton > button:focus-visible { outline: 2px solid var(--accent-scan); outline-offset: 2px; }

    /* ── Footer ── */
    .block-container {
    padding-bottom: 0rem !important;
}

    footer {
    text-align: center;
    padding: 2.4rem 1rem;
    color: var(--text-muted);
    border-top: 1px solid var(--border-hairline);
    margin-top: 3.5rem;
    width: 100vw;
    margin-left: calc(50% - 50vw);
    margin-right: calc(50% - 50vw);
    position: relative;
    z-index: 5;
    font-size: 0.85rem;
    background: rgba(10,14,23,0.82);
    backdrop-filter: blur(14px);
}

footer .footer-links { margin-bottom: 0.8rem; }
footer .footer-links a,
footer a:link,
footer a:visited {
    color: var(--text-muted) !important;
    text-decoration: none !important;
    margin: 0 0.7rem;
    font-weight: 500;
}
footer a:hover, footer a:focus-visible { color: var(--accent-scan) !important; outline: none; }

    .anchor { position: relative; top: -80px; visibility: hidden; }

    @media (prefers-reduced-motion: reduce) {
        .radar-sweep, .radar-blip, .hero h1 { animation: none !important; }
    }
    @media (max-width: 768px) {
        .hero h1 { font-size: 2.5rem; }
        .telemetry { flex-wrap: wrap; }
        .telemetry-item { flex: 1 1 45%; border-right: none; border-bottom: 1px solid var(--border-hairline); }
    }
</style>
""", unsafe_allow_html=True)

# Layout fixes for Streamlit containers
st.markdown("""
<style>
    .block-container {
    position: relative;
    z-index: 5;
    padding-top: 70px !important;   /* was 0 — pushes content below fixed navbar */
    margin-top: 0 !important;
    background: transparent !important;}
    section[data-testid="stAppViewContainer"] { background: transparent !important; }
    section[data-testid="stMain"] { overflow-y: visible !important; overflow-x: hidden !important; }
    .main .block-container { max-width: 100% !important; display: flex; flex-direction: column; align-items: center; padding-bottom: 0; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
            
    /* keep Streamlit's top header from covering the fixed navbar ── */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="grid-bg"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  STICKY NAVBAR
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-logo"><span class="dot"></span> AutoMR</div>
    <div class="navbar-links">
        <a href="#overview">Overview</a>
        <a href="#how">How It Works</a>
        <a href="#capabilities">Capabilities</a>
        <a href="#engines">Engines</a>
        <a href="#categories">MR Categories</a>
        <a href="#models">Models</a>
        <a href="#about">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  HERO
# ──────────────────────────────────────────────────────────────────────────────
logo_b64 = None
logo_path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="width:46px;height:46px;border-radius:10px;margin-bottom:1rem;">'
    if logo_b64 else ""
)

st.markdown(
    f'''
    <div class="hero-wrap">
        <div class="hero-content">
            {logo_html}
            <span class="hero-tag">AutoMR</span>
            <h1 class="hero">Stress-test driving<br>models like a <span>sensor sweep</span>.</h1>
            <p class="hero-subtitle">AutoMR is a model-agnostic, generalized metamorphic testing framework for
            regression-based autonomous driving models - applying realistic transformations, checking whether model
            behavior stays consistent where it should, and scaling from a single sanity check to full HPC-accelerated
            sweeps with zero ground-truth labels.</p>
            <div class="hero-ctas">
                <a href="#categories" class="ghost-btn">View MR Categories</a>
                <a href="#engines" class="ghost-btn">View Execution Engines</a>
            </div>
        </div>
        <div class="hero-visual">
            <div class="radar-panel">
                <div class="radar-ring radar-r1"></div>
                <div class="radar-ring radar-r2"></div>
                <div class="radar-ring radar-r3"></div>
                <div class="radar-cross"></div>
                <div class="radar-sweep"></div>
                <span class="radar-blip b1"></span>
                <span class="radar-blip flagged b2"></span>
                <span class="radar-blip b3"></span>
                <span class="radar-blip flagged b4"></span>
            </div>
            <div class="radar-caption">Scanning for relation violations</div>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)
# # Primary CTA — a real Streamlit button (styled via .stButton CSS above) so st.switch_page works.
# # It sits left-aligned just below the hero, under the tagline.
# hcol1, hcol2 = st.columns([1, 2])
# with hcol1:
#     if st.button("Launch Test Console →", use_container_width=True, key="start_testing"):
#         st.switch_page("pages/App.py")

# ──────────────────────────────────────────────────────────────────────────────
#  TELEMETRY STRIP
# ──────────────────────────────────────────────────────────────────────────────
components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
    body {{ margin:0; font-family:'JetBrains Mono',monospace; background:transparent; }}
    .telemetry {{ background:#121826; border:1px solid #232C42; border-radius:16px; display:flex; max-width:1100px; margin:0 auto; overflow:hidden; }}
    .telemetry-item {{ flex:1; text-align:center; padding:1.4rem 1rem; border-right:1px solid #232C42; }}
    .telemetry-item:last-child {{ border-right:none; }}
    .telemetry-num {{ font-size:2.1rem; font-weight:600; color:#19D3C5; }}
    .telemetry-label {{ font-size:0.74rem; font-weight:500; color:#8A93A6; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.3rem; font-family:'Inter',sans-serif; }}
</style>
<div class="telemetry">
    <div class="telemetry-item"><div class="telemetry-num" data-target="{model_count}">0</div><div class="telemetry-label">Models Supported</div></div>
    <div class="telemetry-item"><div class="telemetry-num" data-target="{transform_count}">0</div><div class="telemetry-label">Transformations</div></div>
    <div class="telemetry-item"><div class="telemetry-num" data-target="{mr_count}">0</div><div class="telemetry-label">Metamorphic Relations</div></div>
    <div class="telemetry-item"><div class="telemetry-num" data-target="{mr_category_count}">0</div><div class="telemetry-label">MR Categories</div></div>
    <div class="telemetry-item"><div class="telemetry-num" data-target="{engine_count}">0</div><div class="telemetry-label">Execution Engines</div></div>
</div>
<script>
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    document.querySelectorAll('.telemetry-num').forEach(counter => {{
        const target = +counter.getAttribute('data-target');
        if (reduceMotion) {{ counter.innerText = target; return; }}
        let count = 0;
        const step = Math.max(1, Math.ceil(target / 40));
        const update = () => {{
            count += step;
            if (count < target) {{ counter.innerText = count; setTimeout(update, 30); }}
            else {{ counter.innerText = target; }}
        }};
        update();
    }});
</script>
""", height=110)

# ──────────────────────────────────────────────────────────────────────────────
#  OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="overview" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Why Metamorphic Testing</div>', unsafe_allow_html=True)
st.markdown(
    '''
    <div class="panel">
        <p>AutoMR tests the <strong>robustness and reliability</strong> of autonomous driving models using
        <strong>metamorphic testing</strong> - a technique that doesn't need ground-truth labels to catch bugs.</p>
        <p>It applies realistic transformations, like lighting shifts, weather, noise, and fog, to driving scenes
        and checks whether the model's output stays consistent with what a defined metamorphic relation expects.</p>
        <p>The framework is <strong>model-agnostic</strong> and <strong>backend-agnostic</strong>: any model exposing
        a <code>predict()</code> interface can be tested, execution can run on CPU or GPU with a single switch, and the
        same relations scale from a quick single-model check to a full parallel, cache-accelerated sweep across an
        entire dataset.</p>
        <p>This surfaces hidden failure modes that traditional accuracy-based testing typically misses, especially
        in regression models where there's no simple "correct answer" to test against.</p>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  HOW IT WORKS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="how" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">How It Works</div>', unsafe_allow_html=True)
st.markdown(
    '''
    <div class="pipeline">
        <div class="pipe-step">
            <div class="pipe-num">01</div>
            <div class="pipe-title">Pick a Model</div>
            <div class="pipe-desc">Choose a built-in model or upload your own .pkl, .h5, .pt or .onnx file.</div>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">02</div>
            <div class="pipe-title">Select an Engine</div>
            <div class="pipe-desc">Standard AutoMR for quick checks, or the HPC engine for parallel, batched, large-scale runs.</div>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">03</div>
            <div class="pipe-title">Apply a Transform</div>
            <div class="pipe-desc">Lighting, weather, noise, fog, geometric shifts, or composed effects - CPU or GPU.</div>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">04</div>
            <div class="pipe-title">Check Relations</div>
            <div class="pipe-desc">Verify whether the relevant metamorphic relations hold within a statistical tolerance.</div>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">05</div>
            <div class="pipe-title">Get a Score</div>
            <div class="pipe-desc">Review consistency, robustness, and epsilon-sensitivity results, broken down by category.</div>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  CAPABILITIES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="capabilities" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Capabilities</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">What AutoMR Can Do</div>', unsafe_allow_html=True)

features = [
    ("Single Model Test", "Test any supported model against all applicable metamorphic relations instantly."),
    ("Cross-Model Analysis", "Compare robustness across multiple architectures side-by-side."),
    ("Parametric Sweep", "Find the exact transformation intensity where a model starts to fail."),
    ("Upload Your Model", "Bring .pkl, .h5, .pt or .onnx — auto-detected, validated, and tested."),
    (f"{transform_count} Transformations", "Weather, lighting, noise, geometric, and composed effects."),
    ("Statistical Thresholds", "Dataset-driven ε tolerances instead of arbitrary pass/fail cutoffs."),
    ("Epsilon Sensitivity Analysis", "Automated threshold sweeps report first-failure, stabilization, and a recommended ε."),
    ("HPC Execution Engine", "A parallel, batched, cache-accelerated engine for large-scale metamorphic testing."),
    ("GPU Acceleration", "CUDA-backed transformations with automatic CPU/GPU backend switching."),
    ("Multi-Framework Support", "Works natively with TensorFlow, PyTorch, scikit-learn, XGBoost, ONNX Runtime, and remote APIs."),
    ("Batch Inference & Caching", "Baseline predictions are computed once and reused across every MR sweep."),
    ("Live Testing Dashboard", "Real-time webcam/video evaluation with configurable MRs and adjustable epsilon."),
    ("Failure & Severity Analysis", "Ranks violations by deviation magnitude and isolates unstable parameter ranges."),
    ("Exportable Reports", "Generates CSV, JSON, and text reports for every relation checked and its result."),
    ("Extensible MR Engine", "Define new metamorphic relations by extending a common base class."),
    ("Plugin Architecture", "Register custom transformations and relations at runtime, no core changes needed."),
]

fcols = st.columns(4)
for i, (title, desc) in enumerate(features):
    with fcols[i % 4]:
        st.markdown(f"""
        <div class="feature-card">
            <span class="idx-tag">F{i+1:02d}</span>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  EXECUTION ENGINES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="engines" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Execution</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Two Engines, One API</div>', unsafe_allow_html=True)
st.markdown(
    '''
    <div class="engine-grid">
        <div class="engine-card">
            <div class="engine-tag">Standard</div>
            <h4>AutoMR Engine</h4>
            <ul class="engine-list">
                <li>Sequential execution suited to small and medium datasets</li>
                <li>CPU or GPU backend, selectable with a single call</li>
                <li>Full metamorphic relation library and statistical thresholds</li>
                <li>Ideal for quick, single-model sanity checks</li>
            </ul>
        </div>
        <div class="engine-card hpc">
            <div class="engine-tag">High-Performance</div>
            <h4>HighPerformanceAutoMR (HPC)</h4>
            <ul class="engine-list">
                <li>Parallel dataset processing across configurable worker threads</li>
                <li>Batched model inference to cut per-sample overhead</li>
                <li>Shared baseline prediction cache reused across every MR sweep</li>
                <li>GPU-accelerated transformations via OpenCV CUDA</li>
                <li>Built for large-scale, latency-sensitive testing runs</li>
            </ul>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MR CATEGORIES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="categories" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">MR Categories</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Four Ways We Test Consistency</div>', unsafe_allow_html=True)

st.markdown("""
<div class="cat-grid">
    <div class="cat-card" style="--cat-color:#19D3C5;">
        <div class="cat-tag">Category A</div>
        <h4>Invariance</h4>
        <p>Output stays stable under lighting, noise, weather and fog changes that shouldn't affect the result.</p>
        <span class="cat-formula">|f(x) − f(T(x))| &lt; ε</span>
    </div>
    <div class="cat-card" style="--cat-color:#FF6B35;">
        <div class="cat-tag">Category B</div>
        <h4>Symmetry</h4>
        <p>Flipping the scene horizontally should flip the predicted steering angle correspondingly.</p>
        <span class="cat-formula">f(flip(x)) ≈ −f(x)</span>
    </div>
    <div class="cat-card" style="--cat-color:#6C7BFF;">
        <div class="cat-tag">Category C</div>
        <h4>Temporal</h4>
        <p>Consecutive frames in a driving sequence should yield smooth, consistent predictions.</p>
        <span class="cat-formula">|f(xₜ) − f(xₜ₊₁)| &lt; δ</span>
    </div>
    <div class="cat-card" style="--cat-color:#FF5C7A;">
        <div class="cat-tag">Category D</div>
        <h4>Composition</h4>
        <p>Stacking two transformations should still preserve the behavior expected of each on its own.</p>
        <span class="cat-formula">f(T₂(T₁(x)))</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  SUPPORTED MODELS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="models" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Compatibility</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Works Across Architectures</div>', unsafe_allow_html=True)

models = [
    ("DAVE-2 (CNN)", "Deep Learning"),
    ("GPR", "Probabilistic"),
    ("SVR", "Classical ML"),
    ("Random Forest", "Ensemble"),
    ("Linear Regression", "Baseline"),
    ("CNN Depth", "Deep Learning"),
]
mcols = st.columns(6)
for i, (name, arch) in enumerate(models):
    with mcols[i]:
        st.markdown(f"""
        <div class="model-card">
            <span class="idx-tag">M{i+1:02d}</span>
            <div class="model-name">{name}</div>
            <div class="model-tag">{arch}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    '''
    <p style="text-align:center; color:#8A93A6; font-size:0.86rem; margin-top:1.4rem;">
    Uploaded models are auto-detected and wrapped through the same pipeline — TensorFlow/Keras, PyTorch,
    scikit-learn, XGBoost, ONNX Runtime, and remote REST APIs are all supported out of the box.
    </p>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  TECH STACK - adjust this list if it doesn't match your actual dependencies
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="stack" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Built With</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Technology Stack</div>', unsafe_allow_html=True)
st.markdown("""
<div class="stack-row">
    <span class="stack-pill">Python</span>
    <span class="stack-pill">Streamlit</span>
    <span class="stack-pill">TensorFlow / Keras</span>
    <span class="stack-pill">PyTorch</span>
    <span class="stack-pill">scikit-learn</span>
    <span class="stack-pill">XGBoost</span>
    <span class="stack-pill">ONNX Runtime</span>
    <span class="stack-pill">NumPy</span>
    <span class="stack-pill">Pandas</span>
    <span class="stack-pill">OpenCV (CUDA)</span>
    <span class="stack-pill">Matplotlib</span>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  CTA BAND
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown(
    '''
    <div class="cta-band">
        <h3>Ready to put a model through its paces?</h3>
        <p>Launch the test console and run your first metamorphic relation in minutes.</p>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
cta1, cta2, cta3 = st.columns([1, 1, 1])
with cta2:
    if st.button("Start Testing Now", use_container_width=True, key="cta_start_testing"):
        st.switch_page("pages/App.py")
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  ABOUT / AUTHOR
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="about" class="anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">About</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">About This Project</div>', unsafe_allow_html=True)
st.markdown("""
<div class="panel">
    <p><strong>AutoMR</strong> implements "A Generalized Metamorphic Testing Platform for Regression Models
    in Autonomous Driving Systems" — an undergraduate research project addressing the test oracle problem
    in regression-based autonomous driving models (steering prediction, trajectory estimation, and lane
    keeping) through metamorphic testing.</p>
    <p>Rather than relying on exact expected outputs, the framework defines metamorphic relations — rules
    that should hold between a source input and a transformed follow-up input — and flags violations as
    potential faults, even when the "correct" output is unknown. Beyond the standard testing engine, the
    platform includes a High-Performance (HPC) execution mode with parallel processing, batched inference,
    and prediction caching for evaluating models at scale.</p>
    <div class="about-grid">
        <div>
            <div class="about-item-label">Team</div>
            <div class="about-item-value">Akurana B.N.T.M.<br>Pabasara P.M.G.T.<br>Peiris P.R.S.<br>Perera G.C.M.</div>
        </div>
        <div>
            <div class="about-item-label">Supervisor</div>
            <div class="about-item-value">Dr. Kaveen Liyanage</div>
        </div>
        <div>
            <div class="about-item-label">Co-Supervisor</div>
            <div class="about-item-value">Mrs. Sithara Mahagama</div>
        </div>
        <div>
            <div class="about-item-label">Institution</div>
            <div class="about-item-value">Dept. of Electrical and Information Engineering<br>Faculty of Engineering, University of Ruhuna</div>
        </div>
    </div>
    <div class="about-links">
        <a href="https://github.com/thurunu711/metamorphic-autodrive-testing" target="_blank">🔗 GitHub Repository</a>
        <a href="mailto:you@email.com"><span class="link-tag">@</span>Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<footer>
    <div class="footer-links">
        <a href="#overview">Overview</a>·
        <a href="#how">How It Works</a>·
        <a href="#capabilities">Capabilities</a>·
        <a href="#engines">Engines</a>·
        <a href="#categories">Categories</a>·
        <a href="#models">Models</a>·
        <a href="#about">About</a>
    </div>
    <div>AutoMR - Metamorphic Testing Framework for Autonomous Driving Models</div>
    <div style="opacity:0.7; margin-top:0.4rem;">© 2026 · Final Year Project, University of Ruhuna</div>
</footer>
""", unsafe_allow_html=True)