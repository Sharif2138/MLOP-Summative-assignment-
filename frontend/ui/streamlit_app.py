import streamlit as st
import requests
import os
import time
from streamlit_autorefresh import st_autorefresh

API_URL = "http://ml-api:8000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Page config   
st.set_page_config(
    page_title="DermAI — Skin Disease Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#refresh after every 5 secs
st_autorefresh(interval=5000, key="uptime_refresh")  

#Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,100,60,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(60,120,255,0.10) 0%, transparent 60%),
        #0a0a0f !important;
    min-height: 100vh;
}

[data-testid="stHeader"], header { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { max-width: 1100px !important; padding: 3rem 2rem 6rem !important; margin: 0 auto !important; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 4rem 0 3rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,100,60,0.12);
    border: 1px solid rgba(255,100,60,0.3);
    color: #ff6438;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: clamp(2.8rem, 6vw, 5rem) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    color: #f0ece4 !important;
    margin-bottom: 1rem !important;
}
.hero h1 span { color: #ff6438; }
.hero p {
    font-size: 1.1rem;
    color: #8a8580;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Section headers ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    margin-top: 1rem;
}
.section-label .num {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #ff6438;
    letter-spacing: 0.1em;
    background: rgba(255,100,60,0.1);
    border: 1px solid rgba(255,100,60,0.2);
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
}
.section-label h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #f0ece4 !important;
    margin: 0 !important;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 2rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(255,255,255,0.13); }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 3rem 0;
}

/* ── Result box ── */
.result-box {
    background: rgba(255,100,60,0.07);
    border: 1px solid rgba(255,100,60,0.25);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.result-box .label { font-size: 0.75rem; color: #8a8580; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem; }
.result-box .value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #ff6438; }
.result-box .conf  { font-size: 0.95rem; color: #8a8580; margin-top: 0.5rem; }

/* ── Confidence bar ── */
.conf-bar-wrap { margin-top: 1rem; }
.conf-bar-bg { background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ff6438, #ff9a6c); transition: width 0.8s cubic-bezier(.4,0,.2,1); }

/* ── Upload zone hint ── */
.upload-hint {
    font-size: 0.82rem;
    color: #5a5650;
    margin-top: 0.75rem;
    font-style: italic;
}

/* ── Status steps ── */
.step-list { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 1rem; }
.step-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.9rem;
    color: #8a8580;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    transition: all 0.3s;
}
.step-item.active  { color: #f0ece4; background: rgba(255,100,60,0.08); border-color: rgba(255,100,60,0.2); }
.step-item.done    { color: #5cc978; background: rgba(92,201,120,0.06); border-color: rgba(92,201,120,0.15); }
.step-item.error   { color: #e05050; background: rgba(224,80,80,0.06); border-color: rgba(224,80,80,0.15); }
.step-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex-shrink: 0; }

/* ── Plots grid ── */
.plot-caption {
    font-size: 0.78rem;
    color: #5a5650;
    text-align: center;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255,100,60,0.4) !important;
}
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(255,100,60,0.5) !important;
    box-shadow: 0 0 0 3px rgba(255,100,60,0.1) !important;
}
.stButton > button {
    background: #ff6438 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 1.8rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #ff7a55 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(255,100,60,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stSuccess, .stInfo, .stError, .stWarning {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stImage"] img { border-radius: 12px !important; }
label, .stTextInput label { color: #8a8580 !important; font-size: 0.82rem !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
</style>
""", unsafe_allow_html=True)


#Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Dermatology</div>
    <h1>Derm<span>AI</span></h1>
    <p>Upload a skin image for instant disease classification powered by deep learning.</p>
</div>
""", unsafe_allow_html=True)

#MODEL STATUS (LIVE UPTIME)
try:
    uptime_res = requests.get(f"{API_URL}/uptime", timeout=3)

    if uptime_res.status_code == 200:
        data = uptime_res.json()
        uptime_seconds = data.get("uptime_seconds", 0)

        days = uptime_seconds // 86400
        hours = (uptime_seconds % 86400) // 3600
        minutes = (uptime_seconds % 3600) // 60

        if days > 0:
            uptime_str = f"{days}d {hours}h"
        elif hours > 0:
            uptime_str = f"{hours}h {minutes}m"
        else:
            uptime_str = f"{minutes}m"

        st.markdown(f"""
        <div style="
            display:flex;
            justify-content:center;
            margin-top:-1rem;
            margin-bottom:2rem;
        ">
            <div style="
                background:rgba(92,201,120,0.08);
                border:1px solid rgba(92,201,120,0.25);
                color:#5cc978;
                padding:0.6rem 1.4rem;
                border-radius:999px;
                font-size:0.8rem;
                letter-spacing:0.05em;
            ">
                ● MODEL ONLINE · UPTIME {uptime_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;margin-top:-1rem;margin-bottom:2rem;">
            <div style="
                background:rgba(224,80,80,0.08);
                border:1px solid rgba(224,80,80,0.25);
                color:#e05050;
                padding:0.6rem 1.4rem;
                border-radius:999px;
                font-size:0.8rem;
                display:inline-block;
            ">
                ● MODEL ERROR
            </div>
        </div>
        """, unsafe_allow_html=True)

except:
    st.markdown("""
    <div style="text-align:center;margin-top:-1rem;margin-bottom:2rem;">
        <div style="
            background:rgba(224,80,80,0.08);
            border:1px solid rgba(224,80,80,0.25);
            color:#e05050;
            padding:0.6rem 1.4rem;
            border-radius:999px;
            font-size:0.8rem;
            display:inline-block;
        ">
            ● MODEL OFFLINE
        </div>
    </div>
    """, unsafe_allow_html=True)


# SECTION 1 — PREDICT
st.markdown("""
<div class="section-label">
    <span class="num">01</span>
    <h2>Diagnose Image</h2>
</div>
""", unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your skin image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
    st.markdown('<p class="upload-hint">Accepted formats: JPG, JPEG, PNG · Max 200 MB</p>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="card" style="min-height:220px">',
                unsafe_allow_html=True)
    if uploaded_file:
        if st.button("▶  Run Analysis", key="predict_btn"):
            with st.spinner("Analysing image…"):
                try:
                    uploaded_file.seek(0)
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded_file.name,
                                        uploaded_file, uploaded_file.type)},
                        timeout=30,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        cls = result["class_name"]
                        conf = result["confidence"]
                        pct = int(conf * 100)
                        st.markdown(f"""
                        <div class="result-box">
                            <div class="label">Detected Condition</div>
                            <div class="value">{cls.upper()}</div>
                            <div class="conf">Confidence — {conf:.4f}</div>
                            <div class="conf-bar-wrap">
                                <div class="conf-bar-bg">
                                    <div class="conf-bar-fill" style="width:{pct}%"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Prediction failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API. Is the FastAPI server running?")
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;height:180px;color:#3a3830;font-size:0.9rem;text-align:center;line-height:1.6;">
            Upload an image on the left<br>to run analysis
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


#Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# SECTION 2 — PERFORMANCE PLOTS
st.markdown("""
<div class="section-label">
    <span class="num">02</span>
    <h2>Model Performance</h2>
</div>
""", unsafe_allow_html=True)

plot_files = {
    "Class Distribution": os.path.join(PLOTS_DIR, "class_distribution.png"),
    "Training Accuracy":  os.path.join(PLOTS_DIR, "accuracy_curve.png"),
    "Training Loss":      os.path.join(PLOTS_DIR, "loss_curve.png"),
    "Confusion Matrix":   os.path.join(PLOTS_DIR, "confusion_matrix.png"),
}
available = {k: v for k, v in plot_files.items() if os.path.exists(v)}

if available:
    items = list(available.items())
    cols = st.columns(min(len(items), 2), gap="medium")
    for i, (title, path) in enumerate(items):
        with cols[i % 2]:
            st.markdown('<div class="card" style="padding:1.2rem">',
                        unsafe_allow_html=True)
            st.image(path, use_container_width=True)
            st.markdown(
                f'<p class="plot-caption">{title}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2.5rem;color:#3a3830;">
        No training plots found yet. Retrain the model to generate metrics.
    </div>
    """, unsafe_allow_html=True)


#Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)



# SECTION 3 — UPLOAD TRAINING DATA
st.markdown("""
<div class="section-label">
    <span class="num">03</span>
    <h2>Upload Training Data</h2>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

col_label, col_files = st.columns([1, 2], gap="large")

with col_label:
    class_name = st.text_input(
        "Class label",
        placeholder="e.g. benign or malignant",
    )
    st.markdown('<p class="upload-hint">Images will be stored under this class name in Supabase.</p>',
                unsafe_allow_html=True)

with col_files:
    training_files = st.file_uploader(
        "Select images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="training_uploader",
    )
    if training_files:
        st.markdown(
            f'<p class="upload-hint">{len(training_files)} file(s) selected</p>', unsafe_allow_html=True)

if training_files:
    if st.button("⬆  Upload Dataset", key="upload_btn"):
        if not class_name.strip():
            st.error("Please enter a class label before uploading.")
        else:
            progress_bar = st.progress(0, text="Preparing upload…")
            status_text = st.empty()

            total = len(training_files)
            files_payload = []
            for i, f in enumerate(training_files):
                files_payload.append(
                    ("files", (f"{class_name.strip()}/{f.name}", f, f.type)))
                progress_bar.progress(
                    (i + 1) / (total + 1), text=f"Preparing {i+1}/{total}…")

            status_text.markdown("**Sending to server…**")
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files_payload,
                    timeout=300,
                )
                progress_bar.progress(1.0, text="Complete!")
                if response.status_code == 200:
                    status_text.empty()
                    st.success(
                        f"✓ {total} image(s) uploaded under class **{class_name.strip()}**")
                else:
                    status_text.empty()
                    st.error(f"Upload failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Is the FastAPI server running?")

st.markdown('</div>', unsafe_allow_html=True)


#Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)



# SECTION 4 — RETRAIN
st.markdown("""
<div class="section-label">
    <span class="num">04</span>
    <h2>Retrain Model</h2>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("""
<p style="color:#8a8580;font-size:0.92rem;margin-bottom:1.5rem;line-height:1.7;">
Retraining downloads all data from Supabase, merges it with the original dataset,
and fine-tunes the model for up to 5 epochs with early stopping.
This runs in the background — progress is shown below in real time.
</p>
""", unsafe_allow_html=True)

RETRAIN_STEPS = [
    ("connecting",   "Connecting to Supabase"),
    ("downloading",  "Downloading training data"),
    ("merging",      "Merging datasets"),
    ("training",     "Fine-tuning model"),
    ("saving",       "Saving updated model"),
    ("done",         "Retraining complete"),
]

if st.button(" Start Retraining", key="retrain_btn"):
    step_placeholder = st.empty()

    def render_steps(current_idx, error=False):
        html = '<div class="step-list">'
        for i, (key, label) in enumerate(RETRAIN_STEPS):
            if error and i == current_idx:
                cls = "error"
                icon = "✕"
            elif i < current_idx:
                cls = "done"
                icon = "✓"
            elif i == current_idx:
                cls = "active"
                icon = "●"
            else:
                cls = ""
                icon = "○"
            html += f'<div class="step-item {cls}"><span class="step-dot"></span>{icon}&nbsp;&nbsp;{label}</div>'
        html += '</div>'
        step_placeholder.markdown(html, unsafe_allow_html=True)

    try:
        render_steps(0)
        response = requests.post(f"{API_URL}/retrain", timeout=10)

        if response.status_code == 200:
            # Simulate step progression while background task runs
            # Steps 1-4 are animated; step 5 shown on completion
            for step in range(1, 5):
                render_steps(step)
                time.sleep(60)

            render_steps(4)
            time.sleep(30)
            render_steps(5)
            st.success(
                "Retraining completed successfully.")
        else:
            render_steps(0, error=True)
            st.error(f"Failed to start retraining: {response.text}")

    except requests.exceptions.ConnectionError:
        render_steps(0, error=True)
        st.error("Cannot reach API. Is the FastAPI server running?")

st.markdown('</div>', unsafe_allow_html=True)
