import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Ahahajij182u2/local_doctor-360M"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Doctor",
    page_icon="🩺",
    layout="centered",
)

# ── Global CSS (full dark override) ───────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap" rel="stylesheet">

<style>
  /* DARK BASE — broad override for all Streamlit surfaces */
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"],
  .stApp,
  section.main,
  section.main > div,
  [data-testid="block-container"],
  .main .block-container {
    background-color: #0d1117 !important;
    color: #e6f4e6 !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  [data-testid="stSidebar"] { background-color: #0a0f0a !important; }

  /* Toolbar / decoration / status strip */
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] { background-color: #0d1117 !important; }

  /* Bottom bar (Streamlit >= 1.30) */
  [data-testid="stBottom"],
  [data-testid="stBottom"] > div,
  [data-testid="stBottomBlockContainer"] { background-color: #0d1117 !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 720px; }

  /* TYPOGRAPHY */
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.7rem;
    color: #e6f4e6;
    text-align: center;
    letter-spacing: -0.5px;
    margin: 0 0 0.2rem;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #6b8f6b;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 300;
  }

  /* LOGO */
  .logo-wrap { display: flex; justify-content: center; margin-bottom: 0.3rem; }

  /* DEVICE BADGE */
  .device-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #111a11;
    color: #69f0ae;
    border: 1px solid #1e3a1e;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 1.8rem;
  }
  .device-badge .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #69f0ae;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

  /* INPUT */
  /* Label */
  .stTextInput label,
  .stTextInput label p,
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label {
    color: #8aab8a !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  /* Input box — shallow selector catches all wrapper depths */
  .stTextInput input,
  [data-testid="stTextInput"] input,
  div[data-baseweb="input"] input {
    background: #111a11 !important;
    border: 1.5px solid #1e3a1e !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    color: #e6f4e6 !important;
    caret-color: #69f0ae !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .stTextInput input:focus,
  [data-testid="stTextInput"] input:focus,
  div[data-baseweb="input"] input:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 3px rgba(76,175,80,.18) !important;
    outline: none !important;
  }
  .stTextInput input::placeholder,
  [data-testid="stTextInput"] input::placeholder { color: #3a5a3a !important; }

  /* BaseWeb container background */
  div[data-baseweb="input"],
  div[data-baseweb="base-input"] {
    background: #111a11 !important;
    border-radius: 10px !important;
  }

  /* BUTTON — covers primary, secondary, tertiary variants */
  .stButton > button,
  [data-testid="baseButton-primary"],
  [data-testid="baseButton-secondary"],
  [data-testid="baseButton-tertiary"] {
    background: #1b5e20 !important;
    color: #e6f4e6 !important;
    border: 1px solid #2e7d32 !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: background 0.2s, transform 0.1s !important;
  }
  .stButton > button:hover,
  [data-testid="baseButton-primary"]:hover,
  [data-testid="baseButton-secondary"]:hover {
    background: #2e7d32 !important;
    transform: translateY(-1px);
    border-color: #43a047 !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* MODEL-LOADING OVERLAY */
  .loading-overlay {
    background: #111a11;
    border: 1.5px solid #1e3a1e;
    border-radius: 14px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
  }
  .loading-overlay .spinner {
    width: 46px; height: 46px;
    border: 3px solid #1a2e1a;
    border-top-color: #4caf50;
    border-radius: 50%;
    animation: spin 0.85s linear infinite;
    margin: 0 auto 1.1rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-overlay p  { color: #c8e6c9; font-size: 1rem; font-weight: 500; margin: 0 0 0.35rem; }
  .loading-overlay small { color: #3d5e3d; font-size: 0.8rem; }

  /* PROCESSING ANIMATION */
  .processing-label {
    font-size: 0.85rem;
    color: #69f0ae;
    font-weight: 500;
    text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: 0.02em;
  }
  .processing-bar {
    height: 3px;
    background: linear-gradient(90deg, #1b5e20, #69f0ae, #1b5e20);
    background-size: 200% 100%;
    animation: shimmer 1.3s ease infinite;
    border-radius: 99px;
    margin: 0 0 1.5rem;
  }
  @keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  /* RESPONSE CARD */
  .response-card {
    background: #0f1a0f;
    border: 1.5px solid #1e3a1e;
    border-left: 4px solid #4caf50;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    white-space: pre-wrap;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #c8e6c9;
  }

  /* WARNING BANNER */
  .warning-banner {
    background: #180a0a;
    border: 1.5px solid #4a1010;
    border-left: 4px solid #e53935;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-top: 1.25rem;
    color: #ef9a9a;
    font-size: 0.85rem;
    line-height: 1.65;
  }
  .warning-banner strong {
    display: block;
    margin-bottom: 0.2rem;
    font-size: 0.88rem;
    color: #f44336;
  }

  /* st.warning / st.alert — covers old and new Streamlit alert markup */
  [data-testid="stAlert"],
  [data-testid="stAlertContainer"],
  div[role="alert"] {
    background: #1a1500 !important;
    border-color: #4a3800 !important;
    color: #ffe082 !important;
    border-radius: 10px !important;
  }
  [data-testid="stAlert"] p,
  [data-testid="stAlertContainer"] p,
  div[role="alert"] p { color: #ffe082 !important; }

  /* DIVIDER */
  .soft-divider { border: none; border-top: 1px solid #182018; margin: 2rem 0 1.5rem; }

  /* SCROLLBAR */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #1e3a1e; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def device_label() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"⚡ GPU · {name}"
    return "💻 CPU"

def show_device_badge():
    label = device_label()
    st.markdown(
        f"""<div style="display:flex;justify-content:center">
              <span class="device-badge"><span class="dot"></span>{label}</span>
            </div>""",
        unsafe_allow_html=True,
    )


# ── Logo & header ──────────────────────────────────────────────────────────────

try:
    st.markdown('<div class="logo-wrap">', unsafe_allow_html=True)
    st.image("assets/logo.png", width=130)
    st.markdown('</div>', unsafe_allow_html=True)
except Exception:
    st.markdown(
        '<div style="text-align:center;font-size:3rem;margin-bottom:0.25rem">🩺</div>',
        unsafe_allow_html=True,
    )

st.markdown('<h1 class="hero-title">Local Doctor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">AI-powered symptom checker · runs entirely on your machine</p>',
    unsafe_allow_html=True,
)

show_device_badge()


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model

if "model_ready" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div class="loading-overlay">
      <div class="spinner"></div>
      <p>Initialising Local Doctor…</p>
      <small>Loading the 360 M-parameter model into memory. This takes a moment on first launch.</small>
    </div>
    """, unsafe_allow_html=True)

    tokenizer, model = load_model()
    st.session_state["model_ready"] = True
    loading_placeholder.empty()
else:
    tokenizer, model = load_model()


# ── Input form ─────────────────────────────────────────────────────────────────

st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

symptoms = st.text_input(
    "Describe your symptoms",
    placeholder="e.g. headache, fever, sore throat …",
    label_visibility="visible",
)

diagnose_clicked = st.button("🩺 Diagnose", use_container_width=True)


# ── Inference ──────────────────────────────────────────────────────────────────

if diagnose_clicked and symptoms.strip():
    status_label = st.empty()
    status_bar   = st.empty()

    status_label.markdown('<p class="processing-label">Analysing symptoms…</p>', unsafe_allow_html=True)
    status_bar.markdown('<div class="processing-bar"></div>', unsafe_allow_html=True)

    prompt = f"""<|im_start|>system
You are a medical assistant AI.

Rules:
- Predict the most likely disease from the symptoms.
- Give 3 to 5 basic precautions or first steps.
- Use only this format:

POSSIBLE DISEASE: ...

POSSIBLE PRECAUTIONS: ...

- If symptoms suggest an emergency, advise immediate medical help.
- Do not give unusual or unsafe advice.
<|im_end|>
<|im_start|>user
{symptoms}
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
        )

    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    response = decoded.split("<|im_end|>")[0].strip()

    # ── Swap processing indicator → done badge ─────────────────────────────
    status_bar.empty()
    status_label.markdown(
        '<p class="processing-label" style="color:#4caf50;">✔ Analysis complete</p>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="response-card">{response}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-banner">
      <strong>⚠ Disclaimer</strong>
      This response is generated by a very small AI model and may not be accurate.<br>
      It is intended for experimental purposes only.
    </div>
    """, unsafe_allow_html=True)

elif diagnose_clicked and not symptoms.strip():
    st.warning("Please enter at least one symptom before diagnosing.")
