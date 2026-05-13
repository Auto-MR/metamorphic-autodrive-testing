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
    .hero { text-align: center; padding: 5rem 1rem 1.5rem 1rem; position: relative; z-index: 15; }
    .hero h1 { font-size: 4.8rem; font-weight: 800; background: linear-gradient(135deg, #0f172a, #1e293b); -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: -0.02em; animation: fadeSlideUp 0.8s ease; }
    @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }
    .hero-subtitle { font-size: 1.3rem; font-weight: 500; color: #0a2540; max-width: 800px; margin: 1rem auto; background: rgba(255,255,245,0.4); backdrop-filter: blur(12px); padding: 0.6rem 1.8rem; border-radius: 60px; display: inline-block; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .overview { background: rgba(255,255,255,0.92); backdrop-filter: blur(14px); padding: 2rem 2.5rem; border-radius: 24px; margin: 1.5rem auto; max-width: 880px; box-shadow: 0 25px 45px -12px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,200,0.5); position: relative; z-index: 15; transition: transform 0.25s ease; }
    .overview:hover { transform: scale(1.01); }
    .stButton > button { font-size: 1.6rem; font-weight: 700; padding: 1.2rem 3.5rem; border-radius: 60px; background: linear-gradient(105deg, #ff8c42, #ff5e00); color: white; box-shadow: 0 18px 35px rgba(255,94,0,0.35); transition: all 0.25s ease; border: none; }
    .stButton > button:hover { transform: scale(1.06); box-shadow: 0 22px 40px rgba(255,94,0,0.55); background: linear-gradient(105deg, #ff9a55, #ff6a20); }
    footer { text-align: center; padding: 2rem; color: #1f2a3e; background: rgba(0,0,0,0.08); backdrop-filter: blur(5px); margin-top: 2rem; position: relative; z-index: 15; font-weight: 500; }
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
        overflow: hidden !important;
    }

    /* Fix the content area to viewport, no scroll */
    .main .block-container {
        max-width: 100% !important;
        height: 100vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
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

st.markdown(
    '<div class="hero">'
        '<h1>AutoMR</h1>'
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

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Start Testing Now", use_container_width=True, key="start_testing"):
        st.switch_page("pages/App.py")

st.markdown("---")
st.markdown(
    "<footer>AutoMR — Metamorphic Autodrive Testing Framework | Next-Gen Validation Suite</footer>",
    unsafe_allow_html=True
)