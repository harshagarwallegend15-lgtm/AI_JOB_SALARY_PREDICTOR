import streamlit as st
import requests
from pathlib import Path
import pandas as pd
import streamlit.components.v1 as components

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SalaryIQ — AI Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://13.203.77.16:8000"

# Catchy AI/ML visual. Unsplash image URL referenced by an
# external project as a machine-learning visualization.
HERO_IMAGE = "https://images.unsplash.com/photo-1535378917042-10a22c95931a?auto=format&fit=crop&w=1800&q=85"

# Optional local training-data locations. The app does not fail
# if the CSV is not present.
DATA_CANDIDATES = [
    Path("ai_ds_job_salaries_2026.csv"),
    Path("source/ai_ds_job_salaries_2026.csv"),
    Path("data/ai_ds_job_salaries_2026.csv"),
]


# ============================================================
# ADVANCED CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 0%, rgba(61,116,255,.12), transparent 28%),
        radial-gradient(circle at 0% 35%, rgba(130,70,255,.10), transparent 25%),
        #070b14;
    color: #f6f8fc;
}

[data-testid="stHeader"] {
    background: rgba(7,11,20,.85);
}

.main .block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background: #0a0f1c;
    border-right: 1px solid rgba(255,255,255,.07);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

h1, h2, h3 {
    letter-spacing: -.03em;
}

.small-muted {
    color: #71809a;
    font-size: .88rem;
}

.section-heading {
    font-size: 1.55rem;
    font-weight: 800;
    margin: 1.8rem 0 .25rem;
}

.section-sub {
    color: #71809a;
    margin-bottom: 1.15rem;
}

div[data-testid="stForm"] {
    background: rgba(13,19,33,.82);
    border: 1px solid rgba(139,161,198,.12);
    border-radius: 24px;
    padding: 1.35rem 1.4rem 1.1rem;
    box-shadow: 0 20px 60px rgba(0,0,0,.18);
}

div[data-testid="stFormSubmitButton"] button {
    height: 3.2rem;
    border-radius: 13px;
    font-weight: 800;
    border: 1px solid rgba(110,150,255,.35);
    background: linear-gradient(135deg,#3b6df6,#6947e9);
    color: white;
    box-shadow: 0 12px 30px rgba(65,90,220,.22);
}

div[data-testid="stFormSubmitButton"] button:hover {
    border-color: rgba(160,180,255,.7);
    box-shadow: 0 16px 35px rgba(65,90,220,.32);
}

div[data-testid="stMetric"] {
    background: rgba(15,23,39,.92);
    border: 1px solid rgba(139,161,198,.12);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    min-height: 115px;
}

div[data-testid="stMetricLabel"] {
    color: #71809a;
}

div[data-testid="stMetricValue"] {
    color: #f7f9fd;
    font-weight: 800;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

div[data-testid="stImage"] img {
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.08);
}

hr {
    border-color: rgba(255,255,255,.07);
}

div[data-testid="stExpander"] {
    background: rgba(13,19,33,.7);
    border: 1px solid rgba(139,161,198,.1);
    border-radius: 16px;
}

.footer {
    text-align: center;
    color: #4e5d76;
    font-size: .78rem;
    padding-top: 2.8rem;
}

/* ============================================================
   PREMIUM VISUAL LAYER
   ============================================================ */

.glass-strip {
    display:flex;
    gap:12px;
    flex-wrap:wrap;
    margin:10px 0 4px;
}

div[data-testid="stImage"] img {
    transition: transform .25s ease, filter .25s ease;
}

div[data-testid="stImage"] img:hover {
    transform: scale(1.01);
    filter: saturate(1.08);
}

div[data-testid="stMetric"] {
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(110,150,255,.28);
    box-shadow: 0 15px 35px rgba(0,0,0,.20);
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div {
    border-radius: 12px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px;
}

@keyframes floatOrb {
    0%,100% { transform: translateY(0px); }
    50% { transform: translateY(-7px); }
}

@keyframes pulseGlow {
    0%,100% { opacity:.55; }
    50% { opacity:1; }
}

.premium-chip {
    padding:7px 11px;
    border-radius:10px;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(255,255,255,.07);
    color:#a9b8d0;
    font-size:11px;
    font-weight:700;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def render_component(html: str, height: int = 260):
    """Render isolated HTML so Markdown can never turn it into a code block."""
    components.html(html, height=height, scrolling=False)


def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def load_salary_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "salary_usd" in df.columns:
                    return df
            except Exception:
                pass
    return None


def predict_for_profile(payload):
    return requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=15,
    )


# ============================================================
# SIDEBAR — ADVANCED COMMAND CENTER
# ============================================================

with st.sidebar:

    # Brand
    components.html(
        """
        <div style="
            font-family:Inter,Arial,sans-serif;
            padding:8px 4px 20px 4px;
        ">
            <div style="
                display:flex;
                align-items:center;
                gap:12px;
                padding:14px;
                border:1px solid rgba(120,160,255,.16);
                border-radius:18px;
                background:
                    radial-gradient(circle at 85% 10%,
                    rgba(70,120,255,.20), transparent 45%),
                    linear-gradient(145deg,#111b30,#0b1220);
                box-shadow:0 14px 35px rgba(0,0,0,.25);
            ">
                <div style="
                    width:44px;
                    height:44px;
                    border-radius:14px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:23px;
                    background:rgba(85,125,255,.14);
                    border:1px solid rgba(120,160,255,.25);
                ">✦</div>

                <div>
                    <div style="
                        color:#fff;
                        font-size:18px;
                        font-weight:800;
                        letter-spacing:-.4px;
                    ">SalaryIQ</div>

                    <div style="
                        color:#7183a4;
                        font-size:10px;
                        font-weight:700;
                        letter-spacing:1.3px;
                        margin-top:3px;
                    ">AI MARKET INTELLIGENCE</div>
                </div>
            </div>
        </div>
        """,
        height=110,
        scrolling=False
    )

    # Live API status
    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
        api_online = health_response.status_code == 200
    except requests.exceptions.RequestException:
        api_online = False

    status_text = "CONNECTED" if api_online else "OFFLINE"
    status_color = "#58e6a8" if api_online else "#ff6b6b"
    status_bg = "rgba(55,220,155,.09)" if api_online else "rgba(255,90,90,.09)"

    components.html(
        f"""
        <div style="
            font-family:Inter,Arial,sans-serif;
            margin:4px 0 18px 0;
        ">
            <div style="
                color:#667795;
                font-size:10px;
                font-weight:800;
                letter-spacing:1.5px;
                margin:0 0 9px 3px;
            ">SYSTEM MONITOR</div>

            <div style="
                display:flex;
                align-items:center;
                justify-content:space-between;
                padding:12px 13px;
                border-radius:14px;
                background:{status_bg};
                border:1px solid {status_color}33;
            ">
                <div style="
                    display:flex;
                    align-items:center;
                    gap:9px;
                ">
                    <span style="
                        width:8px;
                        height:8px;
                        border-radius:50%;
                        background:{status_color};
                        box-shadow:0 0 12px {status_color};
                        display:inline-block;
                    "></span>

                    <span style="
                        color:#d9e1ef;
                        font-size:12px;
                        font-weight:700;
                    ">FastAPI</span>
                </div>

                <span style="
                    color:{status_color};
                    font-size:10px;
                    font-weight:800;
                    letter-spacing:1px;
                ">{status_text}</span>
            </div>
        </div>
        """,
        height=82,
        scrolling=False
    )

    # Navigation-style section
    st.markdown("#### ◈  Dashboard")

    nav_items = [
        ("⌂", "Salary Predictor", "Core prediction engine"),
        ("◌", "Market Profile", "Role & company analysis"),
        ("◈", "Model Insights", "ML-driven estimation"),
    ]

    for icon, title, subtitle in nav_items:
        components.html(
            f"""
            <div style="
                font-family:Inter,Arial,sans-serif;
                margin:5px 0;
                padding:10px 11px;
                border-radius:13px;
                background:rgba(255,255,255,.025);
                border:1px solid rgba(255,255,255,.045);
            ">
                <div style="
                    display:flex;
                    align-items:center;
                    gap:10px;
                ">
                    <span style="
                        width:29px;
                        height:29px;
                        border-radius:9px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:rgba(93,130,255,.10);
                        color:#91adff;
                        font-size:14px;
                    ">{icon}</span>

                    <div>
                        <div style="
                            color:#dce4f2;
                            font-size:11px;
                            font-weight:700;
                        ">{title}</div>

                        <div style="
                            color:#61718d;
                            font-size:9px;
                            margin-top:2px;
                        ">{subtitle}</div>
                    </div>
                </div>
            </div>
            """,
            height=54,
            scrolling=False
        )

    st.markdown("---")

    # Model information
    components.html(
        """
        <div style="
            font-family:Inter,Arial,sans-serif;
            margin-bottom:13px;
        ">
            <div style="
                color:#667795;
                font-size:10px;
                font-weight:800;
                letter-spacing:1.5px;
                margin-bottom:10px;
            ">MODEL CORE</div>

            <div style="
                padding:15px;
                border-radius:16px;
                background:
                    linear-gradient(145deg,
                    rgba(26,37,61,.90),
                    rgba(12,19,33,.90));
                border:1px solid rgba(120,150,210,.12);
            ">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:12px;
                ">
                    <span style="
                        color:#f2f5fb;
                        font-size:13px;
                        font-weight:800;
                    ">Salary Regression</span>

                    <span style="
                        color:#91aaff;
                        font-size:9px;
                        font-weight:800;
                        padding:5px 7px;
                        border-radius:7px;
                        background:rgba(100,135,255,.10);
                    ">ML</span>
                </div>

                <div style="
                    height:5px;
                    border-radius:10px;
                    background:#202b40;
                    overflow:hidden;
                    margin-bottom:8px;
                ">
                    <div style="
                        width:88%;
                        height:100%;
                        border-radius:10px;
                        background:linear-gradient(
                            90deg,#5579ff,#8ea7ff);
                    "></div>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    color:#63738f;
                    font-size:9px;
                ">
                    <span>MODEL READY</span>
                    <span>ACTIVE</span>
                </div>
            </div>
        </div>
        """,
        height=130,
        scrolling=False
    )

    # Architecture mini-card
    components.html(
        """
        <div style="
            font-family:Inter,Arial,sans-serif;
            padding:13px 14px;
            border-radius:15px;
            background:rgba(255,255,255,.022);
            border:1px solid rgba(255,255,255,.055);
            margin-bottom:15px;
        ">
            <div style="
                color:#6c7b96;
                font-size:9px;
                font-weight:800;
                letter-spacing:1.2px;
                margin-bottom:9px;
            ">PIPELINE</div>

            <div style="
                display:flex;
                align-items:center;
                justify-content:space-between;
                color:#aebbd0;
                font-size:9px;
            ">
                <span>INPUT</span>
                <span style="color:#53637d;">→</span>
                <span>API</span>
                <span style="color:#53637d;">→</span>
                <span>MODEL</span>
                <span style="color:#53637d;">→</span>
                <span>OUTPUT</span>
            </div>
        </div>
        """,
        height=70,
        scrolling=False
    )

    # Footer
    components.html(
        """
        <div style="
            font-family:Inter,Arial,sans-serif;
            text-align:center;
            padding:15px 3px 4px;
            border-top:1px solid rgba(255,255,255,.05);
        ">
            <div style="
                color:#52627d;
                font-size:9px;
                line-height:1.6;
            ">
                AI Salary Intelligence<br>
                FastAPI • ML • Streamlit
            </div>

            <div style="
                color:#3f4d65;
                font-size:8px;
                margin-top:7px;
            ">v1.0 • Production Dashboard</div>
        </div>
        """,
        height=78,
        scrolling=False
    )


# ============================================================
# PREMIUM HERO
# ============================================================

AI_NETWORK_IMAGE = (
    "https://images.unsplash.com/"
    "photo-1782330301019-5fc4ab9710a7"
    "?auto=format&fit=crop&w=1100&q=85"
)

CHIP_IMAGE = (
    "https://images.unsplash.com/"
    "photo-1781313201657-612a75a521fa"
    "?auto=format&fit=crop&w=900&q=85"
)

hero_left, hero_right = st.columns([1.08, .92], gap="large")

with hero_left:
    render_component(
        """
        <!doctype html>
        <html>
        <body style="margin:0;background:transparent;font-family:Inter,Arial,sans-serif;">
        <div style="
            position:relative;
            min-height:390px;
            overflow:hidden;
            padding:40px;
            box-sizing:border-box;
            border:1px solid rgba(130,160,220,.17);
            border-radius:30px;
            background:
              radial-gradient(circle at 90% 0%,rgba(73,120,255,.24),transparent 31%),
              radial-gradient(circle at 0% 100%,rgba(116,66,255,.13),transparent 35%),
              linear-gradient(145deg,#111b30,#090f1c);
            box-shadow:0 28px 80px rgba(0,0,0,.34);
            color:white;
        ">

          <div style="
              position:absolute;
              width:170px;height:170px;
              right:-45px;bottom:-55px;
              border-radius:50%;
              background:rgba(65,120,255,.14);
              filter:blur(2px);
              animation:floatOrb 5s ease-in-out infinite;
          "></div>

          <div style="
              position:absolute;
              width:90px;height:90px;
              right:110px;top:-35px;
              border-radius:50%;
              background:rgba(26,211,238,.10);
              animation:pulseGlow 3s ease-in-out infinite;
          "></div>

          <div style="
              display:inline-flex;
              align-items:center;
              gap:8px;
              padding:8px 13px;
              border-radius:999px;
              background:rgba(82,126,255,.11);
              border:1px solid rgba(100,145,255,.26);
              color:#a6beff;
              font-size:10px;
              font-weight:800;
              letter-spacing:1.5px;
          ">
            <span style="color:#68e6ff;">●</span>
            AI / ML • SALARY INTELLIGENCE
          </div>

          <div style="
              font-size:55px;
              line-height:1.01;
              font-weight:850;
              letter-spacing:-3px;
              margin-top:26px;
              position:relative;
          ">
            Know your<br>
            <span style="
                background:linear-gradient(90deg,#91b1ff,#b28cff,#62e6ff);
                -webkit-background-clip:text;
                background-clip:text;
                color:transparent;
            ">market value.</span>
          </div>

          <p style="
              max-width:650px;
              color:#8d9db8;
              font-size:15px;
              line-height:1.72;
              margin-top:22px;
              position:relative;
          ">
            Turn a professional profile into a data-driven salary estimate
            using role, experience, education, company profile, location
            and industry.
          </p>

          <div style="
              display:flex;
              gap:8px;
              flex-wrap:wrap;
              margin-top:22px;
              position:relative;
          ">
            <span style="padding:7px 10px;border-radius:9px;background:#121e34;border:1px solid rgba(255,255,255,.05);color:#aebed8;font-size:10px;">FASTAPI</span>
            <span style="padding:7px 10px;border-radius:9px;background:#121e34;border:1px solid rgba(255,255,255,.05);color:#aebed8;font-size:10px;">PYDANTIC</span>
            <span style="padding:7px 10px;border-radius:9px;background:#121e34;border:1px solid rgba(255,255,255,.05);color:#aebed8;font-size:10px;">REGRESSION ML</span>
            <span style="padding:7px 10px;border-radius:9px;background:#121e34;border:1px solid rgba(255,255,255,.05);color:#aebed8;font-size:10px;">STREAMLIT</span>
          </div>
        </div>
        </body>
        </html>
        """,
        height=435,
    )

with hero_right:

    render_component(
        f"""
        <!doctype html>
        <html>
        <body style="margin:0;background:transparent;font-family:Inter,Arial,sans-serif;">
        <div style="
            height:435px;
            position:relative;
            overflow:hidden;
            border-radius:30px;
            border:1px solid rgba(130,160,220,.16);
            background:#0a1020;
            box-shadow:0 28px 80px rgba(0,0,0,.32);
        ">

          <img src="{AI_NETWORK_IMAGE}" style="
              width:100%;
              height:100%;
              object-fit:cover;
              opacity:.72;
              display:block;
          ">

          <div style="
              position:absolute;
              inset:0;
              background:
                linear-gradient(180deg,rgba(5,10,22,.05),rgba(5,10,22,.78)),
                linear-gradient(90deg,rgba(5,10,22,.35),transparent);
          "></div>

          <div style="
              position:absolute;
              top:20px;
              left:20px;
              right:20px;
              display:flex;
              justify-content:space-between;
              align-items:center;
          ">
            <span style="
                padding:7px 10px;
                border-radius:9px;
                background:rgba(7,13,26,.65);
                backdrop-filter:blur(10px);
                color:#c4d3ee;
                font-size:9px;
                font-weight:800;
                letter-spacing:1.2px;
                border:1px solid rgba(255,255,255,.10);
            ">MODEL VISUALIZATION</span>

            <span style="
                width:10px;height:10px;border-radius:50%;
                background:#54e6ad;
                box-shadow:0 0 15px #54e6ad;
            "></span>
          </div>

          <div style="
              position:absolute;
              left:20px;
              right:20px;
              bottom:20px;
              padding:17px;
              border-radius:17px;
              background:rgba(7,12,24,.72);
              backdrop-filter:blur(14px);
              border:1px solid rgba(255,255,255,.10);
          ">
            <div style="
                color:#7889a8;
                font-size:9px;
                font-weight:800;
                letter-spacing:1.3px;
            ">PREDICTION ENGINE</div>

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:end;
                margin-top:7px;
            ">
              <div style="
                  color:white;
                  font-size:21px;
                  font-weight:800;
              ">47 model features</div>

              <div style="
                  color:#75dfff;
                  font-size:10px;
                  font-weight:800;
              ">LIVE PIPELINE</div>
            </div>
          </div>

        </div>
        </body>
        </html>
        """,
        height=435,
    )

# Visual technology strip
render_component(
    f"""
    <!doctype html>
    <html>
    <body style="margin:0;background:transparent;font-family:Inter,Arial,sans-serif;">
      <div style="
          display:grid;
          grid-template-columns:1.5fr 1fr 1fr;
          gap:12px;
          margin-top:12px;
      ">
        <div style="
            height:78px;
            overflow:hidden;
            position:relative;
            border-radius:17px;
            border:1px solid rgba(130,160,220,.10);
            background:#0d1525;
        ">
          <img src="{CHIP_IMAGE}" style="
              width:100%;height:100%;object-fit:cover;opacity:.42;
          ">
          <div style="
              position:absolute;inset:0;
              background:linear-gradient(90deg,#0b1220,transparent);
          "></div>
          <div style="
              position:absolute;left:14px;top:19px;
              color:#d9e4f8;font-size:11px;font-weight:800;
          ">INTELLIGENCE LAYER</div>
          <div style="
              position:absolute;left:14px;top:40px;
              color:#647590;font-size:9px;
          ">Data → Features → Model → Salary</div>
        </div>

        <div style="
            height:78px;padding:15px;box-sizing:border-box;
            border-radius:17px;border:1px solid rgba(130,160,220,.10);
            background:linear-gradient(145deg,#111b30,#0c1322);
        ">
          <div style="color:#637491;font-size:9px;font-weight:800;letter-spacing:1px;">INFERENCE</div>
          <div style="color:#fff;font-size:18px;font-weight:800;margin-top:7px;">Real-time</div>
          <div style="color:#62728e;font-size:9px;margin-top:2px;">FastAPI prediction</div>
        </div>

        <div style="
            height:78px;padding:15px;box-sizing:border-box;
            border-radius:17px;border:1px solid rgba(130,160,220,.10);
            background:linear-gradient(145deg,#111b30,#0c1322);
        ">
          <div style="color:#637491;font-size:9px;font-weight:800;letter-spacing:1px;">OUTPUT</div>
          <div style="color:#fff;font-size:18px;font-weight:800;margin-top:7px;">USD / year</div>
          <div style="color:#62728e;font-size:9px;margin-top:2px;">Market-value estimate</div>
        </div>
      </div>
    </body>
    </html>
    """,
    height=95,
)

# ============================================================
# VISUAL WORKFLOW
# ============================================================

st.markdown(
    '<div class="section-heading">⚡ From profile to prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-sub">A production-style inference pipeline built around your trained model.</div>',
    unsafe_allow_html=True
)

w1, w2, w3, w4 = st.columns(4, gap="medium")

workflow = [
    ("01", "👤", "Build profile", "Role, experience, education and company context."),
    ("02", "🛡️", "Validate", "Pydantic keeps the prediction request structured."),
    ("03", "🧠", "Transform", "Inputs become the exact model-ready feature vector."),
    ("04", "✦", "Predict", "The regression model returns your estimated salary."),
]

for col, (num, icon, title, desc) in zip([w1, w2, w3, w4], workflow):
    with col:
        render_component(
            f"""
            <!doctype html>
            <html>
            <body style="margin:0;background:transparent;font-family:Inter,Arial,sans-serif;">
            <div style="
                height:225px;
                box-sizing:border-box;
                padding:23px;
                position:relative;
                overflow:hidden;
                border-radius:22px;
                border:1px solid rgba(130,160,220,.13);
                background:
                    radial-gradient(circle at 100% 0%,rgba(78,120,255,.12),transparent 35%),
                    linear-gradient(145deg,#101827,#0b1220);
                color:white;
            ">
              <div style="
                  position:absolute;
                  right:-25px;top:-25px;
                  width:90px;height:90px;border-radius:50%;
                  border:1px solid rgba(100,140,255,.10);
              "></div>

              <div style="
                  width:42px;height:42px;border-radius:13px;
                  display:flex;align-items:center;justify-content:center;
                  background:rgba(90,125,255,.10);
                  border:1px solid rgba(100,140,255,.15);
                  font-size:20px;
              ">{icon}</div>

              <div style="
                  color:#61728e;font-size:9px;font-weight:800;
                  letter-spacing:1.5px;margin-top:16px;
              ">STEP {num}</div>

              <div style="
                  font-size:17px;font-weight:800;margin-top:6px;
              ">{title}</div>

              <div style="
                  font-size:11px;line-height:1.6;color:#74849e;
                  margin-top:8px;max-width:230px;
              ">{desc}</div>

              <div style="
                  position:absolute;left:23px;right:23px;bottom:18px;
                  height:3px;border-radius:9px;
                  background:linear-gradient(90deg,#496ff5,transparent);
                  opacity:.7;
              "></div>
            </div>
            </body>
            </html>
            """,
            height=240,
        )


# ============================================================
# INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-heading">🎯 Build your professional profile</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-sub">Enter your profile once. The backend handles validation, transformation and inference.</div>',
    unsafe_allow_html=True
)

# ============================================================
# INPUT FORM
# ============================================================

st.markdown('<div class="section-heading">🎯 Build your professional profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">All fields are sent directly to the FastAPI prediction endpoint.</div>',
    unsafe_allow_html=True,
)

with st.form("salary_prediction_form"):
    r1 = st.columns(3, gap="medium")

    with r1[0]:
        job_title = st.selectbox(
            "Job Title",
            [
                "AI Engineer",
                "Analytics Engineer",
                "Business Intelligence Analyst",
                "Computer Vision Engineer",
                "Data Analyst",
                "Data Engineer",
                "Data Science Manager",
                "Data Scientist",
                "LLM Engineer",
                "MLOps Engineer",
                "Machine Learning Engineer",
                "Research Scientist",
            ],
        )

    with r1[1]:
        experience_level = st.selectbox(
            "Experience Level",
            ["Entry", "Mid", "Senior", "Lead", "Executive"],
        )

    with r1[2]:
        years_experience = st.number_input(
            "Years of Experience",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=.5,
        )

    r2 = st.columns(3, gap="medium")

    with r2[0]:
        employment_type = st.selectbox(
            "Employment Type",
            ["Full-time", "Contract", "Freelance", "Part-time"],
        )

    with r2[1]:
        company_location = st.selectbox(
            "Company Location",
            ["US","IN","GB","CA","DE","FR","AU","SG","NL","PL","ES","BR"],
        )

    with r2[2]:
        employee_residence = st.selectbox(
            "Employee Residence",
            ["IN"],
        )

    r3 = st.columns(3, gap="medium")

    with r3[0]:
        education_level = st.selectbox(
            "Education Level",
            ["Bootcamp","Self-taught","Bachelors","Masters","PhD"],
        )

    with r3[1]:
        company_size = st.selectbox(
            "Company Size",
            ["S","M","L"],
            format_func=lambda x: {"S":"Small","M":"Medium","L":"Large"}[x],
        )

    with r3[2]:
        industry = st.selectbox(
            "Industry",
            [
                "Consulting","Education","Energy","Finance","Government",
                "Healthcare","Manufacturing","Media","Retail","Technology",
            ],
        )

    st.write("")
    submitted = st.form_submit_button("🚀  Predict my salary", use_container_width=True)


# ============================================================
# PREDICTION
# ============================================================

if submitted:
    payload = {
        "job_title": job_title,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_location": company_location,
        "industry": industry,
        "employee_residence": employee_residence,
        "education_level": education_level,
        "company_size": company_size,
        "years_experience": years_experience,
    }

    with st.spinner("Running model inference..."):
        try:
            response = predict_for_profile(payload)

            if response.status_code == 200:
                result = response.json()
                salary = float(result["predicted_salary_usd"])

                st.markdown("---")

                # Main KPI
                st.markdown('<div class="section-heading">💰 Your market-value estimate</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-sub">Output returned by the trained regression model.</div>',
                    unsafe_allow_html=True,
                )

                k1, k2, k3, k4 = st.columns(4, gap="medium")

                with k1:
                    st.metric("Estimated salary", f"${salary:,.0f}")

                with k2:
                    st.metric("Experience", f"{years_experience:g} yrs")

                with k3:
                    st.metric("Level", experience_level)

                with k4:
                    st.metric("Location", company_location)

                # Salary visual gauge — not a confidence interval.
                gauge_pct = min(max(salary / 250000 * 100, 4), 100)

                render_component(
                    f"""
                    <!doctype html>
                    <html>
                    <body style="margin:0;background:transparent;font-family:Inter,Arial,sans-serif;">
                    <div style="
                        margin-top:18px;
                        padding:23px;
                        border:1px solid rgba(130,160,220,.13);
                        border-radius:20px;
                        background:#0d1524;
                    ">
                      <div style="display:flex;justify-content:space-between;color:#8090aa;font-size:12px;font-weight:700;">
                        <span>MODEL OUTPUT VISUALIZATION</span>
                        <span>${salary:,.0f}</span>
                      </div>
                      <div style="height:14px;background:#182338;border-radius:99px;margin-top:14px;overflow:hidden;">
                        <div style="
                            width:{gauge_pct:.1f}%;
                            height:100%;
                            border-radius:99px;
                            background:linear-gradient(90deg,#416cf5,#8b5cf6,#22d3ee);
                        "></div>
                      </div>
                      <div style="display:flex;justify-content:space-between;color:#566680;font-size:10px;margin-top:8px;">
                        <span>$0</span><span>$125K</span><span>$250K+</span>
                      </div>
                    </div>
                    </body>
                    </html>
                    """,
                    height=125,
                )

                # Profile
                st.markdown('<div class="section-heading">📊 Prediction profile</div>', unsafe_allow_html=True)

                p1, p2, p3, p4 = st.columns(4, gap="medium")
                with p1:
                    st.metric("Job title", job_title)
                with p2:
                    st.metric("Education", education_level)
                with p3:
                    st.metric("Company", {"S":"Small","M":"Medium","L":"Large"}[company_size])
                with p4:
                    st.metric("Industry", industry)

                p5, p6, p7, p8 = st.columns(4, gap="medium")
                with p5:
                    st.metric("Employment", employment_type)
                with p6:
                    st.metric("Company location", company_location)
                with p7:
                    st.metric("Residence", employee_residence)
                with p8:
                    st.metric("Experience level", experience_level)

                # ====================================================
                # OPTIONAL DATA-DRIVEN CHARTS
                # ====================================================

                render_component(
                    f"""
                    <!doctype html>
                    <html>
                    <body style="margin:0;background:transparent;font-family:Inter,Arial,sans-serif;">
                    <div style="
                        padding:20px 22px;
                        border-radius:20px;
                        border:1px solid rgba(110,150,255,.13);
                        background:
                            radial-gradient(circle at 100% 0%,rgba(80,120,255,.14),transparent 32%),
                            linear-gradient(145deg,#101a2e,#0b1220);
                    ">
                      <div style="
                          display:flex;
                          justify-content:space-between;
                          align-items:center;
                          gap:15px;
                      ">
                        <div>
                          <div style="
                              color:#647593;font-size:9px;font-weight:800;
                              letter-spacing:1.5px;
                          ">MARKET INTELLIGENCE</div>
                          <div style="
                              color:#f4f7fc;font-size:18px;font-weight:800;
                              margin-top:5px;
                          ">See how salary signals move</div>
                        </div>

                        <div style="
                            padding:8px 10px;border-radius:9px;
                            background:rgba(81,117,255,.10);
                            border:1px solid rgba(100,140,255,.14);
                            color:#91aaff;font-size:9px;font-weight:800;
                        ">DATA-DRIVEN</div>
                      </div>
                    </div>
                    </body>
                    </html>
                    """,
                    height=105,
                )


                df = load_salary_data()

                if df is not None:
                    st.markdown(
                        '<div class="section-heading">📈 Salary intelligence</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="section-sub">Charts below use the local salary dataset when it is available.</div>',
                        unsafe_allow_html=True,
                    )

                    if "experience_level" in df.columns:
                        chart_df = (
                            df.groupby("experience_level")["salary_usd"]
                            .median()
                            .reindex(["Entry","Mid","Senior","Lead","Executive"])
                            .dropna()
                        )

                        if not chart_df.empty:
                            st.markdown("**Median salary by experience level**")
                            st.bar_chart(chart_df, height=300)

                    if "job_title" in df.columns:
                        top_jobs = (
                            df.groupby("job_title")["salary_usd"]
                            .median()
                            .sort_values(ascending=False)
                            .head(8)
                            .sort_values()
                        )

                        if not top_jobs.empty:
                            st.markdown("**Top job-title salary medians**")
                            st.bar_chart(top_jobs, height=320)

                    if "years_experience" in df.columns:
                        trend = (
                            df.assign(
                                experience_bucket=(df["years_experience"] // 2) * 2
                            )
                            .groupby("experience_bucket")["salary_usd"]
                            .median()
                            .sort_index()
                        )

                        if len(trend) >= 3:
                            st.markdown("**Salary trend across experience**")
                            st.line_chart(trend, height=300)

                else:
                    st.info(
                        "Optional dataset charts are ready. "
                        "Place your salary CSV at "
                        "`ai_ds_job_salaries_2026.csv` or `source/ai_ds_job_salaries_2026.csv` "
                        "to activate them."
                    )

                st.success("Prediction generated successfully.")

            elif response.status_code == 422:
                st.error("FastAPI rejected the request.")
                try:
                    st.json(response.json())
                except Exception:
                    st.code(response.text)

            else:
                st.error(f"Prediction failed with HTTP {response.status_code}.")
                try:
                    st.json(response.json())
                except Exception:
                    st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI is not reachable.")
            st.code("uvicorn app.api_design:app --reload")

        except requests.exceptions.Timeout:
            st.error("❌ FastAPI request timed out.")

        except Exception as exc:
            st.error("❌ Unexpected error.")
            st.exception(exc)


# ============================================================
# PREMIUM FOOTER
# ============================================================

render_component(
    """
    <!doctype html>
    <html>
    <body style="margin:0;background:transparent;font-family:Inter,Arial,sans-serif;">
      <div style="
          margin-top:30px;
          padding:25px;
          text-align:center;
          border-radius:22px;
          border:1px solid rgba(120,150,200,.08);
          background:linear-gradient(145deg,#0d1525,#09101d);
      ">
        <div style="
            color:#c4d0e4;font-size:12px;font-weight:800;
            letter-spacing:.5px;
        ">SalaryIQ</div>

        <div style="
            color:#53637d;font-size:9px;margin-top:7px;
        ">AI Salary Intelligence • Streamlit • FastAPI • Machine Learning</div>

        <div style="
            color:#35445d;font-size:8px;margin-top:9px;
        ">Built as an end-to-end ML deployment project</div>
      </div>
    </body>
    </html>
    """,
    height=125,
)
