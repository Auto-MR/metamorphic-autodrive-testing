import streamlit as st
import sys
import os
import inspect

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

import streamlit as st

st.set_page_config(
    page_title="AutoMR | Metamorphic Testing Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    .main { font-family: 'Inter', sans-serif; background: linear-gradient(145deg, #b2e0fa 0%, #7ec8e0 100%); }
    .city-bg { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; overflow: hidden; pointer-events: none; }
    .sky { position: absolute; width: 100%; height: 100%; background: radial-gradient(circle at 30% 20%, #ffdd99, #6fc3df); }
    .sun { position: absolute; top: 8%; right: 8%; width: 100px; height: 100px; background: radial-gradient(circle, #ffec80, #ffb347); border-radius: 50%; filter: blur(3px); box-shadow: 0 0 50px rgba(255,200,100,0.8); animation: floatSun 10s infinite alternate ease-in-out; }
    @keyframes floatSun { 0% { transform: translateY(0px); } 100% { transform: translateY(20px); } }
    .cloud { position: absolute; background: rgba(255,255,245,0.85); border-radius: 80% 20% 75% 25% / 60% 55% 45% 40%; filter: blur(15px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); animation: floatCloud 20s infinite ease-in-out; }
    @keyframes floatCloud { 0%, 100% { transform: translateX(0); } 50% { transform: translateX(30px); } }
    .building { position: absolute; bottom: 180px; background: linear-gradient(135deg, #2c3e66, #1a2a4a); border-top-left-radius: 12px; border-top-right-radius: 12px; box-shadow: -5px 0 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.15); }
    .building::before { content: ""; position: absolute; top: 12%; left: 10%; width: 80%; height: 25%; background: repeating-linear-gradient(90deg, rgba(255,235,150,0.6) 0px, rgba(255,235,150,0.6) 2px, transparent 2px, transparent 12px); border-radius: 8px; }
    .road { position: fixed; bottom: 0; width: 100%; height: 240px; background: #2c2f36; z-index: 5; box-shadow: 0 -8px 25px rgba(0,0,0,0.4); border-top: 3px solid #ffb347; }
    .road-lane { position: absolute; width: 100%; top: 50%; height: 6px; background: repeating-linear-gradient(90deg, #FFE484, #FFE484 40px, transparent 40px, transparent 80px); transform: translateY(-50%); animation: roadMove 1s linear infinite; }
    @keyframes roadMove { from { background-position-x: 0; } to { background-position-x: -120px; } }
    .cars-layer { position: fixed; bottom: 110px; left: 0; width: 100%; height: 100px; z-index: 20; pointer-events: none; }
    .vehicle { position: absolute; width: 140px; height: 50px; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.3); }
    .vehicle::before, .vehicle::after { content: ""; position: absolute; bottom: -12px; width: 24px; height: 24px; background: #111; border: 4px solid #666; border-radius: 50%; }
    .vehicle::before { left: 18px; }
    .vehicle::after { right: 18px; }
    .car1 { background: linear-gradient(to right, #ff5f6d, #ff9966); animation: drive1 12s linear infinite; }
    .bus { width: 190px; height: 60px; background: linear-gradient(to right, #2563eb, #38bdf8); animation: drive2 18s linear infinite; }
    .truck { width: 210px; background: linear-gradient(to right, #10b981, #34d399); animation: drive3 20s linear infinite; }
    @keyframes drive1 { from { transform: translateX(-200px); } to { transform: translateX(120vw); } }
    @keyframes drive2 { from { transform: translateX(120vw); } to { transform: translateX(-300px); } }
    @keyframes drive3 { from { transform: translateX(-350px); } to { transform: translateX(120vw); } }
    .hero { text-align: center; padding: 1rem 1rem 1.5rem 1rem; position: relative; z-index: 15; }
    .hero h1 { font-size: 4.8rem; font-weight: 800; background: linear-gradient(135deg, #0f172a, #1e293b); -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: -0.02em; animation: fadeSlideUp 0.8s ease; }
    @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }
    .hero-subtitle { font-size: 1.3rem; font-weight: 500; color: #0a2540; max-width: 800px; margin: 1rem auto; background: rgba(255,255,245,0.4); backdrop-filter: blur(12px); padding: 0.6rem 1.8rem; border-radius: 60px; display: inline-block; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .overview { background: rgba(255,255,255,0.92); backdrop-filter: blur(14px); padding: 2rem 2.5rem; border-radius: 24px; margin: 1.5rem auto; max-width: 950px; box-shadow: 0 25px 45px -12px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,200,0.5); position: relative; z-index: 15; transition: transform 0.25s ease; }
    .overview:hover { transform: scale(1.01); }
    .stButton > button { font-size: 1.6rem; font-weight: 700; padding: 1.2rem 3.5rem; border-radius: 60px; background: linear-gradient(105deg, #ff8c42, #ff5e00); color: white; box-shadow: 0 18px 35px rgba(255,94,0,0.35); transition: all 0.25s ease; border: none; }
    .stButton > button:hover { transform: scale(1.06); box-shadow: 0 22px 40px rgba(255,94,0,0.55); background: linear-gradient(105deg, #ff9a55, #ff6a20); }
    footer { text-align: center; padding: 2rem; color: #ffffff; background: rgba(0,0,0,0.08); backdrop-filter: blur(5px); margin-top: 2rem; position: relative; z-index: 15; font-weight: 500; }

    /* Stats strip */
    .stats-strip { display: flex; justify-content: center; gap: 1.6rem; flex-wrap: wrap; margin: 1.2rem auto; max-width: 1100px; position: relative; z-index: 15; }
    .stat-pill { background: rgba(255,255,255,0.85); backdrop-filter: blur(10px); border-radius: 22px; padding: 1.4rem 2.2rem; text-align: center; min-width: 180px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.6); transition: transform 0.2s ease; }
    .stat-pill:hover { transform: translateY(-4px); }
    .stat-num { font-size: 2.2rem; font-weight: 800; color: #ff5e00; line-height: 1; }
    .stat-label { font-size: 1rem; font-weight: 600; color: #1f2a3f; margin-top: 0.3rem; letter-spacing: 0.02em; }

    /* How it works */
    .how-section { max-width: 920px; margin: 1.5rem auto; position: relative; z-index: 15; min-height: 10vh; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .how-title { text-align: center; color: #0a2540; font-size: 1.5rem; font-weight: 800; margin-bottom: 1rem; text-shadow: 0 2px 8px rgba(255,255,255,0.5); background: rgba(255,255,245,0.4); backdrop-filter: blur(20px); padding: 0.6rem 1.8rem; border-radius: 60px; display: inline-block; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    .how-steps { display: flex; justify-content: center; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
    .how-step { background: rgba(255,255,255,0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 1rem 1.4rem; text-align: center; width: 208px; box-shadow: 0 12px 28px rgba(0,0,0,0.15); }
    .how-step-icon { font-size: 1.8rem; }
    .how-step-title { font-weight: 700; color: #0a2540; margin-top: 0.3rem; font-size: 1.2rem; }
    .how-step-desc { font-size: 0.9rem; color: #475569; margin-top: 0.2rem; line-height: 1.3; }
    .how-arrow { font-size: 1.6rem; color: #ff5e00; font-weight: 800; }

    /* Feature cards */
    .features-title { text-align: center; color: #0a2540; font-size: 1.5rem; font-weight: 800; margin: 1.8rem auto 1rem; text-shadow: 0 2px 8px rgba(255,255,255,0.5); position: relative; z-index: 15; }
    .feature-card { background: rgba(255,255,255,0.92); backdrop-filter: blur(12px); border-radius: 20px; padding: 1.3rem 1.2rem; text-align: center; box-shadow: 0 15px 30px rgba(0,0,0,0.18); border: 1px solid rgba(255,255,255,0.6); height: 100%; transition: transform 0.2s ease, box-shadow 0.2s ease; }
    .feature-card:hover { transform: translateY(-6px); box-shadow: 0 20px 38px rgba(0,0,0,0.25); }
    .feature-icon { font-size: 2rem; }
    .feature-title { font-weight: 700; color: #0a2540; font-size: 1.05rem; margin-top: 0.4rem; }
    .feature-desc { font-size: 0.82rem; color: #475569; margin-top: 0.4rem; line-height: 1.4; min-height: 55px; }
    div[data-testid="column"] .stButton > button { font-size: 0.95rem !important; font-weight: 600 !important; padding: 0.5rem 1.2rem !important; border-radius: 30px !important; box-shadow: 0 8px 18px rgba(255,94,0,0.3) !important; width: 100%; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Push Streamlit's main block up so it overlays the city bg */
    .block-container {
        position: relative;
        z-index: 15;
        padding-top: 0 !important;
        margin-top: 0 !important;
        background: transparent !important;
    }

    /* Hide default Streamlit scrollable wrapper padding */
    section[data-testid="stAppViewContainer"] {
        background: transparent !important;
        overflow: hidden !important;
    }

    section[data-testid="stMain"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    /* Fix the content area to viewport, allow scroll for added sections */
    .main .block-container {
        max-width: 100% !important;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# City background — NO HTML comments inside this block
st.markdown(
    '<div class="city-bg">'
        '<div class="sky"></div>'
        '<div class="sun"></div>'
        '<div class="cloud" style="width:220px;height:70px;top:180px;left:-300px;opacity:0.7;animation:floatCloud 60s linear infinite;"></div>'
        '<div class="building" style="left:5%;width:90px;height:240px;"></div>'
        '<div class="building" style="left:15%;width:120px;height:320px;"></div>'
        '<div class="building" style="left:27%;width:80px;height:180px;"></div>'
        '<div class="building" style="left:37%;width:140px;height:360px;"></div>'
        '<div class="building" style="left:50%;width:110px;height:260px;"></div>'
        '<div class="building" style="left:62%;width:160px;height:390px;"></div>'
        '<div class="building" style="left:77%;width:90px;height:210px;"></div>'
        '<div class="building" style="left:88%;width:130px;height:330px;"></div>'
        '<div class="road"><div class="road-lane"></div></div>'
        '<div class="cars-layer">'
            '<div class="vehicle car1" style="bottom:20px;"></div>'
            '<div class="vehicle bus" style="bottom:60px;"></div>'
            '<div class="vehicle truck" style="bottom:100px;"></div>'
        '</div>'
    '</div>',
    unsafe_allow_html=True
)

import base64
import os

logo_path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f'<div class="hero">'
        f'<div style="display:flex; align-items:center; justify-content:center; gap:1rem;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="width:70px; height:70px; border-radius:50%; box-shadow:0 8px 24px rgba(0,0,0,0.25);">'
            f'<h1 style="margin:0;">AutoMR</h1>'
        f'</div>'
        '<div class="hero-subtitle">Generalized Metamorphic Testing Framework for<br>Regression-Based Autonomous Driving Models</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="overview">'
        '<h3 style="margin-top:0;color:#f97316;font-size:1.8rem;font-weight:700;">Project Overview</h3>'
        '<p style="font-size:1.05rem;line-height:1.6;color:#1f2a3f;">AutoMR is a comprehensive framework that tests the <strong>robustness and reliability</strong> of autonomous driving models using <strong>Metamorphic Testing</strong>.</p>'
        '<p style="font-size:1.05rem;line-height:1.6;color:#1f2a3f;margin-top:0.8rem;">It applies realistic transformations like lighting, weather, noise, and fog to driving scenes and checks whether the model maintains consistent behavior according to defined metamorphic relations.</p>'
        '<p style="font-size:1.05rem;line-height:1.6;color:#1f2a3f;margin-top:0.8rem;">The framework helps discover hidden failures that are often missed by traditional testing methods.</p>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="stats-strip">'
        f'<div class="stat-pill"><div class="stat-num">{model_count}</div><div class="stat-label">MODELS SUPPORTED</div></div>'
        f'<div class="stat-pill"><div class="stat-num">{transform_count}</div><div class="stat-label">TRANSFORMATIONS</div></div>'
        f'<div class="stat-pill"><div class="stat-num">{mr_count}</div><div class="stat-label">METAMORPHIC RELATIONS</div></div>'
        f'<div class="stat-pill"><div class="stat-num">{mr_category_count}</div><div class="stat-label">MR CATEGORIES</div></div>'
    f'</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="how-section">'
        '<div class="how-title">How It Works</div>'
        '<div class="how-steps">'
            '<div class="how-step"><div class="how-step-icon">📤</div><div class="how-step-title">1. Pick a Model</div><div class="how-step-desc">Choose a built-in model or upload your own.</div></div>'
            '<div class="how-arrow">→</div>'
            '<div class="how-step"><div class="how-step-icon">🌧️</div><div class="how-step-title">2. Apply Transform</div><div class="how-step-desc">Lighting, weather, noise, fog &amp; more.</div></div>'
            '<div class="how-arrow">→</div>'
            '<div class="how-step"><div class="how-step-icon">🔍</div><div class="how-step-title">3. Check Relations</div><div class="how-step-desc">Verify metamorphic relations hold.</div></div>'
            '<div class="how-arrow">→</div>'
            '<div class="how-step"><div class="how-step-icon">✅</div><div class="how-step-title">4. Get Score</div><div class="how-step-desc">Consistency &amp; robustness results.</div></div>'
        '</div>'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Start Testing Now", use_container_width=True, key="start_testing"):
        st.switch_page("pages/App.py")

st.markdown("---")
st.markdown(
    "<footer>AutoMR - Metamorphic Autodrive Testing Framework | Next-Gen Validation Suite</footer>",
    unsafe_allow_html=True
)