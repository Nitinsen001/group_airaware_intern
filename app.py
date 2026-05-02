import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirAware · AQI Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MASTER CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background: #ffffff !important;
  color: #0f172a !important;
}

/* ── Hide chrome but KEEP sidebar expand button ── */
footer { display: none !important; }
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenuButton"],
[data-testid="stLogoSpacer"] { display: none !important; }

/* Keep the header strip but make it transparent */
header[data-testid="stHeader"] {
  background: transparent !important;
  height: 2.5rem !important;
}

/* ── Expand sidebar button (shows when sidebar collapsed) ── */
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: auto !important;
  height: auto !important;
  min-width: 36px !important;
  min-height: 36px !important;
  background: #0f172a !important;
  border: 1.5px solid #334155 !important;
  border-radius: 8px !important;
  color: #f1f5f9 !important;
  margin: 4px !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] button svg {
  fill: #f1f5f9 !important;
  stroke: #f1f5f9 !important;
  color: #f1f5f9 !important;
  width: 18px !important;
  height: 18px !important;
}


/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0f172a !important;
  border-right: 2px solid #1e293b !important;
  min-width: 260px !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

/* ── Sidebar collapse/expand toggle button ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
button[kind="header"] {
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  border-radius: 8px !important;
  color: #f1f5f9 !important;
}
[data-testid="stSidebarCollapseButton"] svg,
button[kind="header"] svg {
  fill: #f1f5f9 !important;
  stroke: #f1f5f9 !important;
}
/* The expand chevron that shows on the left edge when sidebar is collapsed */
[data-testid="collapsedControl"] {
  background: #0f172a !important;
  border: 1px solid #334155 !important;
  border-radius: 0 8px 8px 0 !important;
  color: #f1f5f9 !important;
}
[data-testid="collapsedControl"] svg {
  fill: #f1f5f9 !important;
}


/* ── Selectbox (sidebar dark) ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
  color: #f1f5f9 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
  color: #f1f5f9 !important;
  background: transparent !important;
}
[data-baseweb="popover"] { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px !important; }
[data-baseweb="menu"] { background: #1e293b !important; }
[data-baseweb="menu"] li { color: #f1f5f9 !important; }
[data-baseweb="menu"] li:hover { background: #334155 !important; }

/* ── Run button ── */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.8rem 1.5rem !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 4px 20px rgba(56,189,248,0.4) !important;
  transition: all 0.3s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 8px 28px rgba(56,189,248,0.55) !important;
}
.stButton > button p { color: #ffffff !important; font-weight: 700 !important; }

/* ── Main block padding ── */
.block-container { padding: 0 2rem 2rem !important; max-width: 1400px !important; }

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-10px); }
}
@keyframes pulse-ring {
  0%   { transform: scale(0.8); opacity: 0.6; }
  100% { transform: scale(1.6); opacity: 0; }
}
@keyframes shimmer {
  0%   { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}
@keyframes slideInLeft {
  from { opacity: 0; transform: translateX(-30px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes countUp { from { opacity: 0; } to { opacity: 1; } }

.animate-up  { animation: fadeUp 0.6s ease forwards; }
.animate-in  { animation: fadeIn 0.5s ease forwards; }
.float-anim  { animation: float 4s ease-in-out infinite; }

/* ── Glass card ── */
.glass {
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(226,232,240,0.8);
  border-radius: 20px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Staggered children ── */
.stagger > *:nth-child(1) { animation-delay: 0.05s; }
.stagger > *:nth-child(2) { animation-delay: 0.15s; }
.stagger > *:nth-child(3) { animation-delay: 0.25s; }
.stagger > *:nth-child(4) { animation-delay: 0.35s; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── AQI CONFIG (exact from model.py + UI display fields) ─────────────────────
# AQI Categories — exact from model.py
AQI_CATEGORIES = {
    'Good':                {'range': (0, 50),   'color': '#2ca02c', 'advice': 'Air quality is satisfactory. Enjoy outdoor activities!',
                           'bg': '#f0fdf4', 'border': '#86efac', 'icon': '😊', 'label': 'Good',        'hex3d': '0x2ca02c'},
    'Satisfactory':        {'range': (51, 100),  'color': '#90ee90', 'advice': 'Air quality is acceptable. Sensitive groups should limit outdoor exposure.',
                           'bg': '#dcfce7', 'border': '#6ee7b7', 'icon': '🙂', 'label': 'Satisfactory', 'hex3d': '0x90ee90'},
    'Moderately Polluted': {'range': (101, 200), 'color': '#ffd700', 'advice': 'Members of sensitive groups should reduce outdoor activities. Wear N95 masks if going out.',
                           'bg': '#fffbeb', 'border': '#fcd34d', 'icon': '😐', 'label': 'Moderate',     'hex3d': '0xffd700'},
    'Poor':                {'range': (201, 300), 'color': '#ff8c00', 'advice': 'Avoid outdoor activities. Wear N95/N99 masks. Use air purifiers indoors.',
                           'bg': '#fff7ed', 'border': '#fdba74', 'icon': '😷', 'label': 'Poor',         'hex3d': '0xff8c00'},
    'Very Poor':           {'range': (301, 400), 'color': '#ff4500', 'advice': 'Exposure may cause respiratory illness. Stay indoors. Use air purifiers. Consult doctor.',
                           'bg': '#fef2f2', 'border': '#fca5a5', 'icon': '🤢', 'label': 'Very Poor',    'hex3d': '0xff4500'},
    'Severe':              {'range': (401, 500), 'color': '#8b0000', 'advice': 'Serious health effects. Remain indoors. Wear N95/N99 masks. Minimize outdoor exposure.',
                           'bg': '#fee2e2', 'border': '#f87171', 'icon': '☠️', 'label': 'Severe',       'hex3d': '0x8b0000'},
    'Hazardous':           {'range': (501, 600), 'color': '#111111', 'advice': 'EMERGENCY! Avoid all outdoor activities. Wear protective masks. Stay in air-purified rooms.',
                           'bg': '#fce7e7', 'border': '#f87171', 'icon': '⛔', 'label': 'Hazardous',    'hex3d': '0x111111'},
}

def get_aqi_category(aqi_value):
    """Returns category and details for given AQI value — exact from model.py"""
    aqi_value = round(aqi_value)  # round to fix decimal gap between ranges
    for category, details in AQI_CATEGORIES.items():
        if details['range'][0] <= aqi_value <= details['range'][1]:
            return category, details
    return 'Hazardous', AQI_CATEGORIES['Hazardous']

# ── DATA & MODEL (exact lines from model.py) ───────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv('city_day.csv')

@st.cache_data(show_spinner=False)
def run_forecast(city_name, _df):
    # Filter and prepare data — exact from model.py
    city_data = _df[_df['City'] == city_name].copy()
    city_data['Date'] = pd.to_datetime(city_data['Date'])
    city_data = city_data.sort_values('Date').reset_index(drop=True)

    # Clean data — exact from model.py
    city_data['AQI'] = city_data['AQI'].fillna(method='ffill').fillna(method='bfill')
    city_data['AQI'] = city_data['AQI'].interpolate(method='linear')
    city_data = city_data.dropna(subset=['AQI']).reset_index(drop=True)

    # Remove outliers beacuse it makes the model skewed — exact from model.py
    Q1 = city_data['AQI'].quantile(0.25)
    Q3 = city_data['AQI'].quantile(0.75)
    IQR = Q3 - Q1
    city_data['AQI'] = city_data['AQI'].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

    # Prepare for Prophet — exact from model.py
    prophet_data = city_data[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})

    # Train Prophet model — exact from model.py
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_data)

    # Calculate RMSE — exact from model.py
    predictions = model.predict(prophet_data[['ds']])
    rmse = np.sqrt(mean_squared_error(prophet_data['y'].values, predictions['yhat'].values))

    # Make 10-day forecast — exact from model.py
    future_dates = model.make_future_dataframe(periods=10)
    forecast = model.predict(future_dates)
    forecast_10days = forecast[forecast['ds'] > prophet_data['ds'].max()][['ds', 'yhat']].reset_index(drop=True)

    # Clip values — exact from model.py
    min_aqi = prophet_data['y'].min()
    max_aqi = prophet_data['y'].max()
    forecast_10days['yhat'] = forecast_10days['yhat'].clip(lower=min_aqi * 0.5, upper=max_aqi * 1.2)

    # UI-only: add confidence bands and historical fit for charts
    hist90 = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90)
    conf = forecast[forecast['ds'] > prophet_data['ds'].max()][['ds', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)
    forecast_10days['yhat_lower'] = conf['yhat_lower'].values
    forecast_10days['yhat_upper'] = conf['yhat_upper'].values

    return prophet_data, hist90, forecast_10days, rmse

# ── 3D ORB COMPONENT ───────────────────────────────────────────────────────────
def three_js_orb(aqi_value: float, color_hex: str, label: str, height: int = 320):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background: transparent; overflow: hidden; }}
      canvas {{ display:block; }}
      #info {{
        position:absolute; top:50%; left:50%;
        transform: translate(-50%, -50%);
        text-align:center; pointer-events:none;
        font-family:'Inter',sans-serif;
      }}
      #aqi-num {{
        font-size: 3.2rem; font-weight: 900;
        color: {color_hex.replace('0x','#')};
        line-height: 1;
        text-shadow: 0 2px 20px {color_hex.replace('0x','#')}88;
        animation: pop 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards;
      }}
      #aqi-lbl {{
        font-size: 0.9rem; font-weight:700;
        color: #475569; margin-top:6px; letter-spacing:0.5px;
        text-transform: uppercase;
      }}
      @keyframes pop {{
        from {{ transform: scale(0.4); opacity:0; }}
        to   {{ transform: scale(1);   opacity:1; }}
      }}
    </style>
    </head>
    <body>
    <canvas id="c"></canvas>
    <div id="info">
      <div id="aqi-num">{aqi_value:.0f}</div>
      <div id="aqi-lbl">{label}</div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
      const W = window.innerWidth, H = window.innerHeight;
      const renderer = new THREE.WebGLRenderer({{ canvas: document.getElementById('c'), alpha: true, antialias: true }});
      renderer.setSize(W, H);
      renderer.setPixelRatio(window.devicePixelRatio);

      const scene  = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 100);
      camera.position.set(0, 0, 4.5);

      // Lights
      const ambient = new THREE.AmbientLight(0xffffff, 0.4);
      scene.add(ambient);
      const point1 = new THREE.PointLight({color_hex}, 2.5, 20);
      point1.position.set(3, 3, 3);
      scene.add(point1);
      const point2 = new THREE.PointLight(0xffffff, 1.2, 20);
      point2.position.set(-3, -2, 2);
      scene.add(point2);

      // Main orb   
      const geo = new THREE.SphereGeometry(1.4, 64, 64);
      const mat = new THREE.MeshPhongMaterial({{
        color: {color_hex},
        shininess: 120,
        specular: new THREE.Color(0xffffff),
        transparent: true,
        opacity: 0.88,
      }});
      const sphere = new THREE.Mesh(geo, mat);
      scene.add(sphere);

      // Outer glow ring
      const ringGeo = new THREE.TorusGeometry(1.75, 0.04, 16, 120);
      const ringMat = new THREE.MeshBasicMaterial({{ color: {color_hex}, transparent: true, opacity: 0.35 }});
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 4;
      scene.add(ring);

      // Second ring
      const ring2Geo = new THREE.TorusGeometry(2.1, 0.02, 16, 120);
      const ring2Mat = new THREE.MeshBasicMaterial({{ color: {color_hex}, transparent: true, opacity: 0.18 }});
      const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
      ring2.rotation.x = -Math.PI / 6;
      ring2.rotation.y = Math.PI / 4;
      scene.add(ring2);

      // Particles
      const partGeo = new THREE.BufferGeometry();
      const count = 280;
      const pos = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {{
        const theta = Math.random() * Math.PI * 2;
        const phi   = Math.acos(2 * Math.random() - 1);
        const r     = 2.2 + Math.random() * 1.4;
        pos[i*3]   = r * Math.sin(phi) * Math.cos(theta);
        pos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
        pos[i*3+2] = r * Math.cos(phi);
      }}
      partGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const partMat = new THREE.PointsMaterial({{ color: {color_hex}, size: 0.045, transparent: true, opacity: 0.55 }});
      const particles = new THREE.Points(partGeo, partMat);
      scene.add(particles);

      let t = 0;
      function animate() {{
        requestAnimationFrame(animate);
        t += 0.012;
        sphere.rotation.y  = t * 0.35;
        sphere.rotation.x  = Math.sin(t * 0.2) * 0.12;
        ring.rotation.z    = t * 0.18;
        ring2.rotation.y   = t * 0.22;
        particles.rotation.y = t * 0.08;
        particles.rotation.x = t * 0.04;
        point1.position.x = Math.sin(t * 0.5) * 4;
        point1.position.z = Math.cos(t * 0.5) * 4;
        renderer.render(scene, camera);
      }}
      animate();
    </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=False)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 4px 24px; text-align:center;">
      <div style="font-size:2.8rem; line-height:1; animation: float 3s ease-in-out infinite;">🌿</div>
      <div style="font-size:1.35rem; font-weight:900; color:#f1f5f9; margin-top:10px; letter-spacing:-0.5px;">AirAware</div>
      <div style="font-size:0.7rem; color:#64748b; margin-top:3px; letter-spacing:1.5px; text-transform:uppercase; font-weight:600;">AQI Intelligence</div>
    </div>
    <hr style="border:none; border-top:1px solid #1e293b; margin-bottom:20px;">
    <style>
      @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
    </style>
    """, unsafe_allow_html=True)

    df_raw = load_data()
    cities = sorted(df_raw['City'].unique())

    st.markdown('<p style="font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;">📍 City</p>', unsafe_allow_html=True)
    selected_city = st.selectbox("City", cities, label_visibility="collapsed",
                                 index=cities.index("Delhi") if "Delhi" in cities else 0)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    run_btn = st.button("🔍  Forecast AQI", use_container_width=True)

    st.markdown("""
    <hr style="border:none; border-top:1px solid #1e293b; margin:22px 0 16px;">
    <p style="font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:14px;">AQI Scale</p>
    """, unsafe_allow_html=True)

    for cat, d in AQI_CATEGORIES.items():
        lo, hi = d['range']
        hi_s = f"{hi}" if hi < 9000 else "600+"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:9px;margin-bottom:7px;padding:5px 8px;border-radius:8px;background:rgba(255,255,255,0.04);">
          <div style="width:9px;height:9px;border-radius:50%;background:{d['color']};flex-shrink:0;box-shadow:0 0 6px {d['color']}99;"></div>
          <span style="font-size:0.79rem;font-weight:600;color:#cbd5e1;flex:1;">{d['label']}</span>
          <span style="font-size:0.7rem;color:#475569;">{lo}–{hi_s}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:none; border-top:1px solid #1e293b; margin:16px 0 12px;">
    <div style="text-align:center;font-size:0.7rem;color:#475569;line-height:1.8;">
      <strong style="color:#38bdf8;">Facebook Prophet</strong> · AI Forecast<br>
      India City Day AQI Dataset
    </div>
    """, unsafe_allow_html=True)

# ── HERO BANNER ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  position:relative; overflow:hidden;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0c1445 100%);
  border-radius: 20px; padding: 28px 36px; margin-bottom: 16px;
  animation: fadeUp 0.7s ease forwards;
">
  <!-- bubble decorations -->
  <div style="position:absolute;top:-80px;right:-80px;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(56,189,248,0.15),transparent 70%);"></div>
  <div style="position:absolute;bottom:-60px;left:30%;width:240px;height:240px;border-radius:50%;background:radial-gradient(circle,rgba(129,140,248,0.12),transparent 70%);"></div>
  <div style="position:absolute;top:20px;left:48%;width:160px;height:160px;border-radius:50%;background:radial-gradient(circle,rgba(34,197,94,0.08),transparent 70%);"></div>

  <div style="position:relative;z-index:1;display:flex;align-items:center;gap:28px;flex-wrap:wrap;">
    <div style="flex:1;min-width:280px;">
      <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.25);border-radius:30px;padding:5px 14px;margin-bottom:16px;">
        <div style="width:7px;height:7px;border-radius:50%;background:#38bdf8;animation:pulse 2s infinite;"></div>
        <span style="font-size:0.72rem;color:#7dd3fc;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;">Live AQI Intelligence</span>
      </div>
      <h1 style="font-size:2.8rem;font-weight:900;color:#ffffff;letter-spacing:-1.5px;line-height:1.1;margin:0 0 12px;">
        Air<span style="color:#38bdf8;">Aware</span>
      </h1>
      <p style="font-size:1rem;color:#94a3b8;font-weight:400;line-height:1.7;max-width:480px;margin:0 0 24px;">
        AI-powered 10-day Air Quality forecasting using Facebook Prophet.<br>
        Select a city, run the model, and get instant health insights.
      </p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:10px 16px;">
          <div style="font-size:1.4rem;font-weight:900;color:#38bdf8;">60+</div>
          <div style="font-size:0.72rem;color:#64748b;font-weight:500;margin-top:1px;">Indian Cities</div>
        </div>
        <div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:10px 16px;">
          <div style="font-size:1.4rem;font-weight:900;color:#818cf8;">10</div>
          <div style="font-size:0.72rem;color:#64748b;font-weight:500;margin-top:1px;">Day Forecast</div>
        </div>
        <div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:10px 16px;">
          <div style="font-size:1.4rem;font-weight:900;color:#4ade80;">AI</div>
          <div style="font-size:0.72rem;color:#64748b;font-weight:500;margin-top:1px;">Prophet Model</div>
        </div>
      </div>
    </div>
    <!-- floating orb preview in hero -->
    <div style="flex-shrink:0;text-align:center;animation:float 4s ease-in-out infinite;">
      <div style="width:130px;height:130px;border-radius:50%;
        background: radial-gradient(circle at 35% 35%, #38bdf8, #0284c7 60%, #0c4a6e);
        box-shadow: 0 0 50px rgba(56,189,248,0.5), 0 0 100px rgba(56,189,248,0.2), inset 0 0 30px rgba(255,255,255,0.15);
        margin: 0 auto 10px;">
      </div>
      <div style="font-size:0.75rem;color:#475569;font-weight:600;letter-spacing:0.5px;">AQI ORBS</div>
    </div>
  </div>
</div>
<style>
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }
  @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
  @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
</style>
""", unsafe_allow_html=True)

# ── LANDING PAGE ───────────────────────────────────────────────────────────────
if not run_btn and "last_city" not in st.session_state:
    c1, c2, c3 = st.columns(3)
    features = [
        ("📊","#eff6ff","#1d4ed8","#bfdbfe","#dbeafe",
         "Prophet AI Engine",
         "Trained on years of city-level daily AQI records with yearly & weekly seasonality."),
        ("🩺","#f0fdf4","#15803d","#86efac","#dcfce7",
         "Health Advisories",
         "Real health guidance per AQI category — from outdoor safety to emergency protocols."),
        ("🏙️","#faf5ff","#7c3aed","#c4b5fd","#ede9fe",
         "60+ Indian Cities",
         "Historical AQI data for India's major cities with outlier-filtered time series."),
    ]
    for col, (icon, bg, color, border, light, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div style="
              background:{bg}; border:1.5px solid {border};
              border-radius:20px; padding:28px 22px;
              animation: fadeUp 0.6s ease forwards;
              transition: transform 0.2s, box-shadow 0.2s;
            " onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 40px rgba(0,0,0,0.1)';"
               onmouseout="this.style.transform='';this.style.boxShadow='';">
              <div style="
                width:54px; height:54px; border-radius:16px;
                background: linear-gradient(135deg, {color}, {color}cc);
                display:flex; align-items:center; justify-content:center;
                font-size:1.6rem; margin-bottom:16px;
                box-shadow: 0 6px 20px {color}44;
              ">{icon}</div>
              <div style="font-size:1rem;font-weight:800;color:#0f172a;margin-bottom:8px;">{title}</div>
              <div style="font-size:0.83rem;color:#64748b;line-height:1.65;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
      margin-top:24px;
      background: linear-gradient(135deg, #f8fafc, #f1f5f9);
      border: 2px dashed #cbd5e1; border-radius:20px;
      padding:48px; text-align:center;
      animation: fadeUp 0.8s 0.2s ease both;
    ">
      <div style="font-size:3rem;margin-bottom:16px;animation:float 3s ease-in-out infinite;">👈</div>
      <div style="font-size:1.25rem;font-weight:800;color:#0f172a;margin-bottom:10px;">
        Choose a City · Click Forecast
      </div>
      <div style="font-size:0.9rem;color:#64748b;line-height:1.8;max-width:400px;margin:0 auto;">
        Select any city from the dark sidebar on the left.<br>
        Hit <strong style="color:#0284c7;">🔍 Forecast AQI</strong> and watch the AI work.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Resolve city ──────────────────────────────────────────────────────────
    if run_btn:
        st.session_state["last_city"] = selected_city
    city = st.session_state.get("last_city", selected_city)

    with st.spinner(f"⚡ Training Prophet model for **{city}**…"):
        prophet_data, hist90, forecast_10days, rmse = run_forecast(city, df_raw)

    avg_aqi = forecast_10days['yhat'].mean()
    avg_cat, avg_d = get_aqi_category(avg_aqi)
    hist_avg = prophet_data['y'].mean()
    delta = avg_aqi - hist_avg

    # ── City header ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;animation:fadeUp 0.5s ease both;">
      <div style="
        width:50px;height:50px;border-radius:14px;
        background:linear-gradient(135deg,#0284c7,#7c3aed);
        display:flex;align-items:center;justify-content:center;
        font-size:1.3rem; box-shadow:0 6px 18px rgba(2,132,199,0.35);
      ">📍</div>
      <div>
        <div style="font-size:1.65rem;font-weight:900;color:#0f172a;letter-spacing:-0.5px;line-height:1.1;">{city}</div>
        <div style="font-size:0.78rem;color:#94a3b8;font-weight:500;margin-top:2px;">10-Day AQI Forecast  ·  Facebook Prophet AI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3D Orb + KPI row ──────────────────────────────────────────────────────
    col3d, col_kpis = st.columns([1, 2], gap="large")

    with col3d:
        st.markdown(f"""
        <div class="glass" style="padding:20px;text-align:center;animation:fadeUp 0.5s 0.1s ease both;">
          <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;margin-bottom:6px;">Live AQI Orb</div>
          <div style="font-size:0.85rem;font-weight:600;color:{avg_d['color']};margin-bottom:8px;">{city} · 10-Day Avg</div>
        </div>
        """, unsafe_allow_html=True)
        three_js_orb(avg_aqi, avg_d['hex3d'], avg_d['label'], height=300)
        st.markdown(f"""
        <div style="text-align:center;margin-top:-8px;animation:fadeIn 1s 0.5s ease both;">
          <div style="font-size:0.78rem;color:#94a3b8;font-weight:500;">{avg_d['icon']} {avg_d['advice'][:60]}…</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpis:
        # Animated KPI cards (2 cards only)
        kpi_data = [
            ("📅", "#eff6ff", "#1d4ed8", "#bfdbfe", "Forecast Days", "10", ""),
            ("📈", avg_d['bg'], avg_d['color'], avg_d['border'], "Avg AQI (10 Days)", f"{avg_aqi:.1f}", avg_cat),
        ]
        cols = st.columns(2)
        for col, (icon, bg, color, border, label, val, sub) in zip(cols, kpi_data):
            sub_html = f'<div style="font-size:0.75rem;font-weight:600;color:{"#16a34a" if "▼" in sub else "#dc2626" if "▲" in sub else "#94a3b8"};margin-top:3px;">{sub}</div>' if sub else ""
            with col:
                st.markdown(f"""
                <div style="
                  background:{bg}; border:1.5px solid {border};
                  border-radius:16px; padding:20px;
                  animation: fadeUp 0.5s ease both;
                  transition: transform 0.2s, box-shadow 0.2s;
                  height:100%;
                ">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                    <span style="font-size:0.68rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;line-height:1.3;">{label}</span>
                    <span style="font-size:1.2rem;">{icon}</span>
                  </div>
                  <div style="font-size:2.1rem;font-weight:900;color:{color};line-height:1;letter-spacing:-1px;">{val}</div>
                  {sub_html}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Health Advisory Banner ─────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
      position:relative; overflow:hidden;
      background: linear-gradient(135deg, {avg_d['color']}18, {avg_d['color']}08);
      border: 2px solid {avg_d['border']};
      border-radius:14px; padding:16px 20px; margin-bottom:14px;
      animation: fadeUp 0.6s 0.2s ease both;
    ">
      <div style="position:absolute;right:-30px;top:-30px;width:160px;height:160px;border-radius:50%;background:radial-gradient(circle,{avg_d['color']}22,transparent 70%);"></div>
      <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;position:relative;z-index:1;">
        <div style="width:64px;height:64px;border-radius:18px;background:{avg_d['color']};
          display:flex;align-items:center;justify-content:center;font-size:2rem;
          box-shadow:0 8px 24px {avg_d['color']}55;flex-shrink:0;">{avg_d['icon']}</div>
        <div style="flex:1;min-width:200px;">
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:#94a3b8;margin-bottom:4px;">Health Advisory · Overall Outlook</div>
          <div style="font-size:1.4rem;font-weight:900;color:{avg_d['color']};letter-spacing:-0.3px;margin-bottom:5px;">{avg_cat}</div>
          <div style="font-size:0.88rem;color:#475569;line-height:1.6;">{avg_d['advice']}</div>
        </div>
        <div style="background:white;border-radius:16px;padding:18px 26px;border:2px solid {avg_d['border']};text-align:center;box-shadow:0 4px 16px {avg_d['color']}22;flex-shrink:0;">
          <div style="font-size:3.2rem;font-weight:900;color:{avg_d['color']};line-height:1;letter-spacing:-2px;">{avg_aqi:.0f}</div>
          <div style="font-size:0.68rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">AQI Index</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 10-Day Forecast Chart ─────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:16px 18px 4px;margin-bottom:12px;animation:fadeUp 0.6s 0.3s ease both;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
      <div style="font-size:0.92rem;font-weight:800;color:#0f172a;margin-bottom:2px;">📈 10-Day AQI Forecast</div>
      <div style="font-size:0.74rem;color:#94a3b8;margin-bottom:10px;">Prophet model predictions with 95% confidence interval</div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_10days['ds'], forecast_10days['ds'][::-1]]),
        y=pd.concat([forecast_10days['yhat_upper'].clip(upper=600), forecast_10days['yhat_lower'].clip(lower=0)[::-1]]),
        fill='toself', fillcolor=f'rgba({int(avg_d["color"][1:3],16)},{int(avg_d["color"][3:5],16)},{int(avg_d["color"][5:7],16)},0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence Band', hoverinfo='skip',
    ))
    dot_colors = [get_aqi_category(v)[1]['color'] for v in forecast_10days['yhat']]
    fig.add_trace(go.Scatter(
        x=forecast_10days['ds'], y=forecast_10days['yhat'],
        mode='lines+markers+text',
        line=dict(color='#0284c7', width=3.5),
        marker=dict(size=13, color=dot_colors, line=dict(color='white', width=2.5)),
        text=[f"<b>{v:.0f}</b>" for v in forecast_10days['yhat']],
        textposition='top center',
        textfont=dict(size=12, color='#0f172a', family='Inter'),
        name='AQI Forecast',
        hovertemplate="<b>%{x|%a, %b %d}</b><br>AQI: <b>%{y:.1f}</b><extra></extra>",
    ))
    for val, col, lbl in [(50,'#16a34a','Good'),(100,'#15803d','Satisfactory'),
                          (200,'#b45309','Moderate'),(300,'#c2410c','Poor'),(400,'#dc2626','Very Poor')]:
        fig.add_hline(y=val, line_dash="dot", line_color=col, opacity=0.28,
                      annotation_text=lbl, annotation_position="right",
                      annotation_font_size=10, annotation_font_color=col)
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=370,
        margin=dict(l=10, r=120, t=24, b=10),
        font=dict(family='Inter', size=12, color='#334155'),
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', gridwidth=1,
                   tickformat='%b %d', showline=False, zeroline=False,
                   tickfont=dict(size=11, color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', gridwidth=1, range=[0, 540],
                   showline=False, zeroline=False, tickfont=dict(size=11, color='#64748b'),
                   title=dict(text='AQI', font=dict(size=11, color='#94a3b8'))),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                    font=dict(size=11, color='#334155'), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Historical + Table ────────────────────────────────────────────────────
    ch, ct = st.columns([3, 2], gap="large")

    with ch:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px 4px;
          box-shadow:0 2px 8px rgba(0,0,0,0.04);animation:fadeUp 0.6s 0.4s ease both;">
          <div style="font-size:0.92rem;font-weight:800;color:#0f172a;margin-bottom:2px;">📉 Historical Trend · Last 90 Days</div>
          <div style="font-size:0.74rem;color:#94a3b8;margin-bottom:10px;">Actual AQI vs Prophet model fit</div>
        """, unsafe_allow_html=True)
        tail = prophet_data.tail(90)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=tail['ds'], y=tail['y'], mode='lines', name='Actual',
            line=dict(color='#cbd5e1', width=2), fill='tozeroy',
            fillcolor='rgba(2,132,199,0.04)',
            hovertemplate="<b>%{x|%b %d}</b><br>AQI: %{y:.1f}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=hist90['ds'], y=hist90['yhat'], mode='lines', name='Model Fit',
            line=dict(color='#0284c7', width=2.5, dash='dot'),
            hovertemplate="<b>%{x|%b %d}</b><br>Fit: %{y:.1f}<extra></extra>",
        ))
        fig2.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', height=265,
            margin=dict(l=10, r=10, t=6, b=10),
            font=dict(family='Inter', size=11, color='#334155'),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickformat='%b %d',
                       tickfont=dict(size=10, color='#64748b'), showline=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                       tickfont=dict(size=10, color='#64748b'), showline=False, zeroline=False,
                       title=dict(text='AQI', font=dict(size=10, color='#94a3b8'))),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                        font=dict(size=10, color='#334155'), bgcolor='rgba(0,0,0,0)'),
            hovermode='x unified',
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with ct:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px 10px;
          box-shadow:0 2px 8px rgba(0,0,0,0.04);animation:fadeUp 0.6s 0.5s ease both;">
          <div style="font-size:0.92rem;font-weight:800;color:#0f172a;margin-bottom:2px;">📋 Day-by-Day</div>
          <div style="font-size:0.74rem;color:#94a3b8;margin-bottom:10px;">Next 10 days forecast detail</div>
        """, unsafe_allow_html=True)
        rows = ""
        for i, row in forecast_10days.iterrows():
            v = row['yhat']
            cat, d = get_aqi_category(v)
            rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9;transition:background 0.15s;"
                onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background=''">
              <td style="padding:9px 10px;font-size:0.82rem;font-weight:700;color:#0f172a;">
                Day {i+1}
                <span style="display:block;font-size:0.68rem;font-weight:500;color:#94a3b8;">{row['ds'].strftime('%b %d')}</span>
              </td>
              <td style="padding:9px 10px;font-size:1.05rem;font-weight:900;color:{d['color']};letter-spacing:-0.5px;">{v:.0f}</td>
              <td style="padding:9px 10px;">
                <span style="background:{d['bg']};color:{d['color']};border:1px solid {d['border']};
                  border-radius:20px;padding:3px 9px;font-size:0.7rem;font-weight:700;white-space:nowrap;">
                  {d['icon']} {d['label']}
                </span>
              </td>
            </tr>"""
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;">
          <thead><tr style="background:#f8fafc;">
            <th style="padding:8px 10px;text-align:left;font-size:0.67rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;">Date</th>
            <th style="padding:8px 10px;text-align:left;font-size:0.67rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;">AQI</th>
            <th style="padding:8px 10px;text-align:left;font-size:0.67rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;">Status</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Distribution chart ────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px 4px;
      box-shadow:0 2px 8px rgba(0,0,0,0.04);animation:fadeUp 0.6s 0.6s ease both;">
      <div style="font-size:0.92rem;font-weight:800;color:#0f172a;margin-bottom:2px;">📊 Historical AQI Distribution</div>
      <div style="font-size:0.74rem;color:#94a3b8;margin-bottom:10px;">Days per AQI category · entire historical dataset</div>
    """, unsafe_allow_html=True)
    bars = {}
    for _, row in prophet_data.iterrows():
        cat, _ = get_aqi_category(row['y'])
        bars[cat] = bars.get(cat, 0) + 1
    x_v = [c for c in AQI_CATEGORIES if c in bars]
    y_v = [bars[c] for c in x_v]
    c_v = [AQI_CATEGORIES[c]['color'] for c in x_v]
    fig3 = go.Figure(go.Bar(
        x=x_v, y=y_v, marker_color=c_v, marker_line_color='white', marker_line_width=2,
        text=y_v, textposition='outside',
        textfont=dict(size=11, color='#334155', family='Inter'),
        hovertemplate="<b>%{x}</b> — %{y} days<extra></extra>",
    ))
    fig3.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=250,
        margin=dict(l=10, r=10, t=8, b=10),
        font=dict(family='Inter', size=12, color='#334155'),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#334155'), showline=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickfont=dict(size=10, color='#64748b'),
                   showline=False, zeroline=False,
                   title=dict(text='Days', font=dict(size=10, color='#94a3b8'))),
        bargap=0.3,
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:32px;text-align:center;padding-bottom:16px;animation:fadeIn 1s ease both;">
      <span style="font-size:0.75rem;color:#cbd5e1;">
        AirAware &nbsp;·&nbsp;
        Built with <strong style="color:#0284c7;">Streamlit</strong> &amp;
        <strong style="color:#818cf8;">Facebook Prophet</strong> &amp;
        <strong style="color:#4ade80;">Three.js</strong>
        &nbsp;·&nbsp; India City Day AQI Dataset
      </span>
    </div>
    """, unsafe_allow_html=True)
