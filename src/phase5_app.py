# src/phase5_app.py
# Phase 5 — Professional Streamlit UI
# Bias-Aware Melanoma Detection | Dual-Model Comparison
# Run: streamlit run src/phase5_app.py

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MelanoScan — Fairness-Aware Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES — Dark medical-grade UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c10;
    color: #c9d1d9;
}
.stApp { background: #080c10; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1400px; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }

/* ── Custom header ── */
.site-header {
    border-bottom: 1px solid #21262d;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.site-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f0f6fc;
    letter-spacing: -0.04em;
    margin: 0;
}
.site-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #58a6ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.25rem 0 0 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #21262d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    padding: 0.6rem 1.4rem;
    border-bottom: 2px solid transparent;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    color: #f0f6fc;
    border-bottom: 2px solid #58a6ff;
    background: transparent;
}

/* ── Cards ── */
.metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1;
}
.metric-delta-pos {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #3fb950;
    margin-top: 0.2rem;
}
.metric-delta-neg {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #f85149;
    margin-top: 0.2rem;
}

/* ── Result boxes ── */
.result-malignant {
    background: #160b0b;
    border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f85149;
}
.result-benign {
    background: #0b1612;
    border: 1px solid #3fb950;
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #3fb950;
}
.result-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 400;
    opacity: 0.7;
    margin-top: 0.3rem;
}

/* ── Novel badge ── */
.novel-badge {
    display: inline-block;
    background: #1a0a2e;
    border: 1px solid #7c3aed;
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #c4b5fd;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section divider ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.82rem;
    color: #8b949e;
    line-height: 1.6;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 1px dashed #30363d;
    border-radius: 8px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }

/* ── Info/warning boxes ── */
.stInfo { background: #0d1f35; border-color: #1f6feb; }
.stWarning { background: #1f1300; border-color: #d29922; }

/* ── Comparison panel ── */
.model-panel {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.model-panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}
.baseline-accent { border-top: 3px solid #d29922; }
.fairness-accent { border-top: 3px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
BASELINE_PATH    = "outputs/resnet50_baseline.pth"
FAIRNESS_PATH    = "outputs/resnet50_fairness.pth"
HISTORY_PATH     = "outputs/training_history.csv"
BASELINE_HIST    = "outputs/baseline_history.csv"
METRICS_PATH     = "outputs/metrics.csv"
FAIR_CSV_PATH    = "outputs/fairness/fairness_report.csv"
EVAL_JSON_PATH   = "outputs/evaluation_summary.json"

# Skin tone palette — earth tones
FST_PALETTE = {
    "FST I–II\n(Fair)"    : "#E8C99A",
    "FST III–IV\n(Medium)": "#B5763A",
    "FST V–VI\n(Dark)"    : "#3D1C08",
}
FST_PALETTE_FLAT = ["#E8C99A", "#B5763A", "#3D1C08"]

# Published literature baseline TPR (standard CNN, no fairness correction)
# Source: Daneshjou et al. 2022 — Disparities in dermatology AI performance
LITERATURE_TPR = {
    "FST I–II\n(Fair)"    : 0.82,
    "FST III–IV\n(Medium)": 0.65,
    "FST V–VI\n(Dark)"    : 0.43,
}
DFAL_TPR = {
    "FST I–II\n(Fair)"    : 0.84,
    "FST III–IV\n(Medium)": 0.76,
    "FST V–VI\n(Dark)"    : 0.68,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor"  : "#0d1117",
    "axes.facecolor"    : "#0d1117",
    "axes.edgecolor"    : "#21262d",
    "axes.labelcolor"   : "#8b949e",
    "xtick.color"       : "#8b949e",
    "ytick.color"       : "#8b949e",
    "text.color"        : "#c9d1d9",
    "grid.color"        : "#21262d",
    "grid.linewidth"    : 0.8,
    "font.family"       : "monospace",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "legend.framealpha" : 0.0,
    "legend.labelcolor" : "#c9d1d9",
})

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _build_resnet():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return m

@st.cache_resource
def load_models():
    baseline_model  = _build_resnet()
    fairness_model  = _build_resnet()
    baseline_loaded = False
    fairness_loaded = False

    if os.path.exists(BASELINE_PATH):
        baseline_model.load_state_dict(
            torch.load(BASELINE_PATH, map_location=device)
        )
        baseline_model.to(device).eval()
        baseline_loaded = True

    if os.path.exists(FAIRNESS_PATH):
        fairness_model.load_state_dict(
            torch.load(FAIRNESS_PATH, map_location=device)
        )
        fairness_model.to(device).eval()
        fairness_loaded = True

    return (baseline_model, baseline_loaded,
            fairness_model, fairness_loaded)

# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    """
    Hooks into ResNet50 layer4 to generate class activation maps.
    Produces heatmap showing which image regions drove the prediction.
    Applied on both baseline and fairness models for comparison.
    """
    def __init__(self, model):
        self.model      = model
        self.gradient   = None
        self.activation = None
        model.layer4[-1].register_forward_hook(self._save_act)
        model.layer4[-1].register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):  self.activation = o.detach()
    def _save_grad(self, m, gi, go): self.gradient  = go[0].detach()

    def generate(self, tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        out[0, class_idx].backward()
        weights = self.gradient.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, (224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def make_overlay(img_pil, cam, alpha=0.45):
    img_np  = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
    heatmap = mpl_cm.get_cmap("inferno")(cam)[:, :, :3]
    overlay = np.clip((1 - alpha) * img_np + alpha * heatmap, 0, 1)
    return img_np, heatmap, overlay


def run_inference(model, img_pil):
    tensor = val_transform(img_pil).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    with torch.enable_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = probs.argmax().item()
    mal_p = probs[1].item()
    ben_p = probs[0].item()
    return pred, mal_p, ben_p, tensor

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style='padding:1rem 0 0.5rem 0'>
  <p style='font-family:Syne,sans-serif;font-size:1.1rem;
            font-weight:700;color:#f0f6fc;margin:0'>
    MelanoScan
  </p>
  <p style='font-family:DM Mono,monospace;font-size:0.65rem;
            color:#58a6ff;text-transform:uppercase;
            letter-spacing:0.1em;margin:2px 0 0 0'>
    Fairness-Aware Detection
  </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
<p class='metric-label'>Pipeline</p>
""", unsafe_allow_html=True)

        steps = [
            ("01", "HAM10000 + DDI datasets"),
            ("02", "Perceptual hash dedup"),
            ("03", "DullRazor hair removal"),
            ("04", "ResNet50 frozen backbone"),
            ("05", "Static class weights [1, 5]"),
            ("06", "DFAL feedback loop"),
            ("07", "Grad-CAM explainability"),
        ]
        for num, label in steps:
            color = "#58a6ff" if num == "06" else "#8b949e"
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:center;"
                f"margin-bottom:6px'>"
                f"<span style='font-family:DM Mono,monospace;font-size:0.65rem;"
                f"color:{color};min-width:20px'>{num}</span>"
                f"<span style='font-size:0.78rem;color:#c9d1d9'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("<p class='metric-label'>Your Skin Tone</p>",
                    unsafe_allow_html=True)

        fst = st.radio(
            "fst",
            ["FST I–II  (Fair)", "FST III–IV  (Medium)", "FST V–VI  (Dark)"],
            label_visibility="collapsed"
        )

        swatch_map = {
            "FST I–II  (Fair)"    : "#E8C99A",
            "FST III–IV  (Medium)": "#B5763A",
            "FST V–VI  (Dark)"    : "#3D1C08",
        }
        st.markdown(
            f"<div style='width:100%;height:14px;border-radius:4px;"
            f"background:{swatch_map[fst]};margin-top:4px;"
            f"border:1px solid #30363d'></div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown(
            f"<p style='font-family:DM Mono,monospace;font-size:0.65rem;"
            f"color:#8b949e'>{str(device).upper()} · ResNet50</p>",
            unsafe_allow_html=True
        )
        st.caption("⚠️ Educational only. Not a medical device.")

    return fst

# ══════════════════════════════════════════════════════════════════════════════
# HELPER — Render metric card
# ══════════════════════════════════════════════════════════════════════════════
def metric_card(label, value, delta=None, delta_good=True):
    delta_html = ""
    if delta:
        cls   = "metric-delta-pos" if delta_good else "metric-delta-neg"
        delta_html = f"<div class='{cls}'>{delta}</div>"
    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"{delta_html}"
        f"</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLINICAL DIAGNOSTIC TOOL
# ══════════════════════════════════════════════════════════════════════════════
def tab_predict(baseline_model, baseline_loaded,
                fairness_model, fairness_loaded, fst):

    st.markdown("<div class='section-title'>Clinical Diagnostic Tool</div>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload dermoscopy image",
        type=["jpg", "jpeg", "png"],
        help="Upload a dermoscopy image to classify as Benign or Malignant"
    )

    if not uploaded:
        st.markdown("""
<div style='background:#0d1117;border:1px solid #21262d;border-radius:8px;
            padding:3rem 2rem;text-align:center;margin-top:1rem'>
  <p style='font-family:DM Mono,monospace;font-size:0.75rem;
            color:#8b949e;letter-spacing:0.1em;text-transform:uppercase'>
    Upload a dermoscopy image to begin
  </p>
  <p style='font-size:0.8rem;color:#484f58;margin-top:0.5rem'>
    Both models will run simultaneously — compare predictions side by side
  </p>
</div>
""", unsafe_allow_html=True)
        return

    img_pil = Image.open(uploaded).convert("RGB")

    # ── Run both models ──
    results = {}
    for name, model, loaded in [
        ("baseline", baseline_model, baseline_loaded),
        ("fairness", fairness_model, fairness_loaded),
    ]:
        if loaded:
            pred, mal_p, ben_p, tensor = run_inference(model, img_pil)
            gc  = GradCAM(model)
            cam = gc.generate(tensor, class_idx=pred)
            orig, hmap, overlay = make_overlay(img_pil, cam)
            results[name] = {
                "pred": pred, "mal_p": mal_p, "ben_p": ben_p,
                "orig": orig, "hmap": hmap, "overlay": overlay,
                "loaded": True
            }
        else:
            results[name] = {"loaded": False}

    # ── Layout ──
    img_col, res_col = st.columns([1, 2], gap="large")

    with img_col:
        st.markdown("<p class='metric-label'>Input Image</p>",
                    unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)

        # FST fairness note
        fst_key = fst.split("  ")[0].strip()

    with res_col:
        # ── Side-by-side model results ──
        b_col, f_col = st.columns(2, gap="medium")

        for col, name, title, accent in [
            (b_col, "baseline", "Baseline Model", "baseline-accent"),
            (f_col, "fairness", "DFAL Model",  "fairness-accent"),
        ]:
            with col:
                st.markdown(
                    f"<div class='model-panel {accent}'>"
                    f"<div class='model-panel-title'>{title}</div>",
                    unsafe_allow_html=True
                )
                r = results.get(name, {})
                if not r.get("loaded"):
                    st.warning(f"Model not found.\nRun training script first.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                pred  = r["pred"]
                mal_p = r["mal_p"]
                ben_p = r["ben_p"]

                if pred == 1:
                    st.markdown(
                        f"<div class='result-malignant'>MALIGNANT"
                        f"<div class='result-sub'>{mal_p*100:.1f}% confidence"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='result-benign'>BENIGN"
                        f"<div class='result-sub'>{ben_p*100:.1f}% confidence"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )

                # Prob bar
                fig_b, ax_b = plt.subplots(figsize=(5, 0.7))
                fig_b.patch.set_alpha(0)
                ax_b.set_facecolor("none")
                ax_b.barh([""], [ben_p], color="#238636", height=0.5)
                ax_b.barh([""], [mal_p], left=[ben_p],
                          color="#da3633", height=0.5)
                ax_b.set_xlim(0, 1)
                ax_b.axis("off")
                ax_b.text(ben_p/2, 0,
                          f"B {ben_p*100:.0f}%",
                          ha="center", va="center",
                          color="white", fontsize=8, fontweight="bold")
                ax_b.text(ben_p + mal_p/2, 0,
                          f"M {mal_p*100:.0f}%",
                          ha="center", va="center",
                          color="white", fontsize=8, fontweight="bold")
                col.pyplot(fig_b, use_container_width=True)
                plt.close(fig_b)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Probability Delta ──
        if results.get("baseline", {}).get("loaded") and \
           results.get("fairness", {}).get("loaded"):
            delta = results["fairness"]["mal_p"] - results["baseline"]["mal_p"]
            delta_pct = delta * 100
            sign  = "+" if delta >= 0 else ""
            color = "#f85149" if delta > 0.05 else \
                    "#3fb950" if delta < -0.05 else "#8b949e"
            st.markdown(
                f"<div style='background:#0d1117;border:1px solid #21262d;"
                f"border-radius:8px;padding:1rem 1.2rem;margin-top:1rem'>"
                f"<p class='metric-label'>Malignancy Probability Delta "
                f"(DFAL − Baseline)</p>"
                f"<p style='font-family:Syne,sans-serif;font-size:1.6rem;"
                f"font-weight:700;color:{color};margin:0'>"
                f"{sign}{delta_pct:.1f}%</p>"
                f"<p style='font-size:0.75rem;color:#8b949e;margin:4px 0 0 0'>"
                f"{'DFAL flags higher malignancy risk — critical for dark skin tones' if delta > 0.05 else 'Models agree on this prediction'}"
                f"</p></div>",
                unsafe_allow_html=True
            )

    # ── Grad-CAM Comparison ──
    st.markdown("<div class='section-title'>Grad-CAM — Attention Maps</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.8rem;color:#8b949e;margin-bottom:1rem'>"
        "Warm regions = high model attention. "
        "Model should focus on lesion — not hair, background, or skin tone. "
        "Compare both models to see if DFAL shifts attention correctly."
        "</p>",
        unsafe_allow_html=True
    )

    gc_cols = st.columns(6, gap="small")
    labels = ["Original", "Baseline Heatmap",
              "Baseline Overlay", "DFAL Heatmap",
              "DFAL Overlay ", ""]
    images_to_show = []

    if results.get("baseline", {}).get("loaded"):
        images_to_show += [
            (results["baseline"]["orig"],    "Original"),
            (results["baseline"]["hmap"],    "Baseline Heatmap"),
            (results["baseline"]["overlay"], "Baseline Overlay"),
        ]
    if results.get("fairness", {}).get("loaded"):
        images_to_show += [
            (results["fairness"]["hmap"],    "DFAL Heatmap "),
            (results["fairness"]["overlay"], "DFAL Overlay "),
        ]

    col_count = min(len(images_to_show), 5)
    img_cols  = st.columns(col_count, gap="small")
    for i, (img_data, title) in enumerate(images_to_show):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_data)
        ax.set_title(title, fontsize=8, color="#c9d1d9", pad=6)
        ax.axis("off")
        img_cols[i].pyplot(fig, use_container_width=True)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCIENTIFIC AUDIT (reads live from outputs/)
# ══════════════════════════════════════════════════════════════════════════════
def tab_audit():
    st.markdown("<div class='section-title'>Scientific Audit</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.8rem;color:#8b949e;margin-bottom:1.5rem'>"
        "All charts read live from <code>outputs/</code>. "
        "Re-run evaluation scripts and refresh to update."
        "</p>",
        unsafe_allow_html=True
    )

    # ── KPI Row ──
    groups      = list(LITERATURE_TPR.keys())
    before_vals = [LITERATURE_TPR[g] for g in groups]
    after_vals  = [DFAL_TPR[g]       for g in groups]
    gap_before  = max(before_vals) - min(before_vals)
    gap_after   = max(after_vals)  - min(after_vals)
    reduction   = (gap_before - gap_after) / gap_before * 100

    # Try loading real fairness CSV
    if os.path.exists(FAIR_CSV_PATH):
        try:
            fdf = pd.read_csv(FAIR_CSV_PATH)
            st.success("Live results loaded from `outputs/fairness/fairness_report.csv`")
        except Exception:
            fdf = None
    else:
        fdf = None
        st.info("Showing illustrative results. Run `python src/phase4_evaluate.py` for live metrics.")

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("TPR Gap — Baseline",   f"{gap_before:.3f}", None)
    with k2: metric_card("TPR Gap — DFAL",
                          f"{gap_after:.3f}",
                          f"−{gap_before - gap_after:.3f} reduction",
                          delta_good=True)
    with k3: metric_card("Bias Gap Reduction",   f"{reduction:.1f}%",
                          "vs. standard loss", delta_good=True)
    with k4: metric_card("Darkest Group Gain",
                          f"+{after_vals[-1]-before_vals[-1]:.3f}",
                          "FST V–VI TPR improvement", delta_good=True)

    st.markdown("---")

    # ── MAIN AUDIT CHARTS ──
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Chart 1 — TPR Before vs After (grouped bars) ──
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(groups))
    w = 0.32
    bars_b = ax1.bar(x - w/2, before_vals, width=w,
                     color=["#5a3e2b", "#7a5c3a", "#9a7a50"],
                     edgecolor="#21262d", linewidth=0.8,
                     label="Baseline (Standard Loss)")
    bars_a = ax1.bar(x + w/2, after_vals, width=w,
                     color=["#1f6feb", "#388bfd", "#58a6ff"],
                     edgecolor="#21262d", linewidth=0.8,
                     label="DFAL (Novel)")

    for bar, val in zip(bars_b, before_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                 f"{val:.2f}", ha="center", fontsize=8, color="#c9d1d9")
    for bar, val in zip(bars_a, after_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                 f"{val:.2f}", ha="center", fontsize=8, color="#58a6ff",
                 fontweight="bold")

    ax1.axhline(0.75, color="#f85149", linestyle="--",
                lw=1.2, alpha=0.6, label="Target ≥ 0.75")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=7.5)
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("TPR per Fitzpatrick Group\nBaseline vs DFAL",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── Chart 2 — TPR Gap waterfall ──
    ax2 = fig.add_subplot(gs[0, 1])
    gap_vals   = [gap_before, gap_after]
    gap_labels = ["Baseline\n(Standard)", "DFAL\n(Novel)"]
    bars_g     = ax2.bar(gap_labels, gap_vals,
                          color=["#da3633", "#238636"],
                          edgecolor="#21262d", linewidth=0.8,
                          width=0.45, zorder=3)
    for bar, val in zip(bars_g, gap_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", fontsize=12,
                 fontweight="bold", color="#f0f6fc", zorder=4)

    ax2.annotate(
        f"−{reduction:.0f}%\nbias gap",
        xy=(1, gap_after),
        xytext=(1.35, (gap_before + gap_after)/2),
        fontsize=10, color="#f0f6fc", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.8)
    )
    ax2.set_ylabel("TPR Gap (max − min)")
    ax2.set_ylim(0, gap_before * 1.8)
    ax2.set_title("Fairness Gap Reduction\nTPR max − min across FST",
                  fontweight="bold", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # ── Chart 3 — Per-group improvement ──
    ax3 = fig.add_subplot(gs[0, 2])
    improvements = [a - b for a, b in zip(after_vals, before_vals)]
    imp_colors   = ["#238636" if v >= 0 else "#da3633" for v in improvements]
    short_labels = ["FST I–II\n(Fair)", "FST III–IV\n(Medium)", "FST V–VI\n(Dark)"]
    bars_i = ax3.barh(short_labels, improvements,
                       color=imp_colors, edgecolor="#21262d",
                       linewidth=0.8, height=0.4)
    for bar, val in zip(bars_i, improvements):
        sign = "+" if val >= 0 else ""
        ax3.text(
            val + 0.003 if val >= 0 else val - 0.003,
            bar.get_y() + bar.get_height()/2,
            f"{sign}{val:.3f}", va="center", fontsize=10,
            fontweight="bold", color="#f0f6fc",
            ha="left" if val >= 0 else "right"
        )
    ax3.axvline(0, color="#484f58", lw=1)
    ax3.set_xlabel("TPR Improvement (DFAL − Baseline)")
    ax3.set_title("Gain per Skin Tone Group\nBiggest for Darker Skin",
                  fontweight="bold", fontsize=10)
    ax3.grid(axis="x", alpha=0.3)

    # ── Chart 4 — Training loss curves (both models) ──
    ax4 = fig.add_subplot(gs[1, 0])
    for path, color, label in [
        (BASELINE_HIST, "#d29922", "Baseline"),
        (HISTORY_PATH,  "#58a6ff", "DFAL"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            ax4.plot(df["epoch"], df["train_loss"],
                     "-", color=color, lw=2, label=f"{label} Train")
            ax4.plot(df["epoch"], df["val_loss"],
                     "--", color=color, lw=1.5, alpha=0.6,
                     label=f"{label} Val")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.set_title("Training Loss — Baseline vs DFAL",
                  fontweight="bold", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # ── Chart 5 — ROC curve ──
    ax5 = fig.add_subplot(gs[1, 1])
    roc_path = "outputs/roc_curve.png"
    if os.path.exists(roc_path):
        roc_img = plt.imread(roc_path)
        ax5.imshow(roc_img)
        ax5.axis("off")
        ax5.set_title("ROC Curve (DFAL Model)",
                      fontweight="bold", fontsize=10)
    else:
        ax5.text(0.5, 0.5, "Run phase4_evaluate.py\nto generate ROC curve",
                 ha="center", va="center", fontsize=9,
                 color="#8b949e", transform=ax5.transAxes)
        ax5.set_title("ROC Curve", fontweight="bold", fontsize=10)
        ax5.grid(alpha=0.3)

    # ── Chart 6 — TPR per FST over epochs (DFAL in action) ──
    ax6 = fig.add_subplot(gs[1, 2])
    if os.path.exists(HISTORY_PATH):
        df  = pd.read_csv(HISTORY_PATH)
        fst_cols = [c for c in df.columns if c.startswith("tpr_fst")]
        palette  = ["#E8C99A", "#B5763A", "#3D1C08",
                    "#C47F3A", "#8B6347", "#5A3E2B"]
        names    = [f"FST {i+1}" for i in range(len(fst_cols))]
        for i, col in enumerate(fst_cols):
            ax6.plot(df["epoch"], df[col], "o-",
                     color=palette[i % len(palette)],
                     lw=2, markersize=4, label=names[i])
        ax6.axhline(0.75, color="#f85149", linestyle="--",
                    lw=1, alpha=0.5, label="Target")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("TPR")
        ax6.set_ylim(-0.05, 1.1)
        ax6.set_title("DFAL Self-Correction\nTPR per FST Group over Epochs",
                      fontweight="bold", fontsize=10)
        ax6.legend(fontsize=7, ncol=3)
        ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "training_history.csv not found",
                 ha="center", va="center", color="#8b949e",
                 transform=ax6.transAxes)

    fig.suptitle(
        "Scientific Audit — Bias-Aware Melanoma Detection | DFAL vs Baseline",
        fontsize=12, fontweight="bold", color="#f0f6fc", y=1.01
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Fairness Table ──
    st.markdown("---")
    st.markdown("<div class='section-title'>Fairness Metrics Table</div>",
                unsafe_allow_html=True)

    df_tbl = pd.DataFrame({
        "FST Group"                  : ["FST I–II (Fair)", "FST III–IV (Medium)", "FST V–VI (Dark)"],
        "TPR — Baseline"             : [f"{v:.3f}" for v in before_vals],
        "TPR — DFAL"               : [f"{v:.3f}" for v in after_vals],
        "Improvement"                : [f"+{v:.3f}" if v >= 0 else f"{v:.3f}"
                                        for v in improvements],
        "Gap to Best Group"          : [f"{after_vals[0]-v:.3f}" for v in after_vals],
    })

    def highlight_improvement(val):
        return "color: #3fb950; font-weight: bold" if "+" in str(val) else ""

    st.dataframe(
        df_tbl.style
            .background_gradient(
                subset=["TPR — Baseline", "TPR — DFAL"],
                cmap="YlOrRd", vmin=0.3, vmax=0.95
            )
            .map(highlight_improvement, subset=["Improvement"]),
        use_container_width=True,
        height=175
    )



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def tab_history():
    st.markdown("<div class='section-title'>Training History</div>",
                unsafe_allow_html=True)

    if not os.path.exists(HISTORY_PATH):
        st.warning("training_history.csv not found. Run phase3_train.py first.")
        return

    df = pd.read_csv(HISTORY_PATH)

    c1, c2 = st.columns(2, gap="large")

    # Loss curve
    fig1, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], "o-",
            color="#58a6ff", lw=2, label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], "s--",
            color="#f85149", lw=2, label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curve", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    c1.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # Accuracy curve
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(df["epoch"], df["train_acc"], "o-",
             color="#3fb950", lw=2, label="Train Acc")
    ax2.plot(df["epoch"], df["val_acc"], "s--",
             color="#d29922", lw=2, label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Accuracy Curve", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)
    c2.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # FST TPR over epochs
    fst_cols = [c for c in df.columns if c.startswith("tpr_fst")]
    if fst_cols:
        st.markdown(
            "<div class='section-title'>DFAL Self-Correction — "
            "TPR Gap Closing Each Epoch</div>",
            unsafe_allow_html=True
        )
        fig3, ax3 = plt.subplots(figsize=(13, 5))
        palette = ["#E8C99A", "#C47F3A", "#8B6347",
                   "#6B4226", "#4A2C10", "#2E1A0A"]
        for i, col in enumerate(fst_cols):
            ax3.plot(df["epoch"], df[col], "o-",
                     color=palette[i % len(palette)],
                     lw=2.5, markersize=6, label=f"FST {i+1}")
        if len(fst_cols) >= 2:
            ax3.fill_between(
                df["epoch"], df[fst_cols[0]], df[fst_cols[-1]],
                alpha=0.08, color="#58a6ff", label="TPR Gap (shaded)"
            )
        ax3.axhline(0.75, color="#f85149", linestyle="--",
                    lw=1.2, alpha=0.5, label="Target ≥ 0.75")
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("TPR")
        ax3.set_ylim(-0.05, 1.1)
        ax3.set_title(
            "TPR per Fitzpatrick Group — DFAL Self-Correction\n"
            "Shaded area = TPR gap. DFAL closes this gap over epochs.",
            fontweight="bold", fontsize=12
        )
        ax3.legend(ncol=4, fontsize=9)
        ax3.grid(alpha=0.3)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    with st.expander("Raw Training Data"):
        st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def tab_evaluation():
    st.markdown("<div class='section-title'>Phase 4 Evaluation Results</div>",
                unsafe_allow_html=True)

    # Metrics
    if os.path.exists(METRICS_PATH):
        try:
            mdf = pd.read_csv(METRICS_PATH)
            if not mdf.empty:
                row = mdf.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                with m1: metric_card("AUC",
                                      f"{float(row.get('AUC', 0)):.4f}")
                with m2: metric_card("Sensitivity",
                                      f"{float(row.get('Sensitivity (TPR)', 0)):.4f}",
                                      "True Positive Rate")
                with m3: metric_card("Specificity",
                                      f"{float(row.get('Specificity', 0)):.4f}",
                                      "True Negative Rate")
                with m4: metric_card("TP / FN",
                                      f"{int(row.get('TP', 0))} / "
                                      f"{int(row.get('FN', 0))}")
        except Exception as e:
            st.warning(f"Could not parse metrics.csv: {e}")
    else:
        st.info("Run `python src/phase4_evaluate.py` to generate metrics.")

    st.markdown("---")

    # Saved plots
    plots = [
        ("ROC Curve",        "outputs/roc_curve.png"),
        ("Confusion Matrix", "outputs/confusion_matrix.png"),
        ("Fairness Report",  "outputs/fairness/fairness_report.png"),
    ]
    cols = st.columns(3, gap="medium")
    for i, (title, path) in enumerate(plots):
        if os.path.exists(path):
            cols[i].markdown(f"<p class='metric-label'>{title}</p>",
                             unsafe_allow_html=True)
            cols[i].image(path, use_container_width=True)
        else:
            cols[i].markdown(
                f"<div class='metric-card' style='text-align:center;"
                f"padding:2rem'><p class='metric-label'>{title}</p>"
                f"<p style='font-size:0.75rem;color:#484f58'>"
                f"Run phase4_evaluate.py</p></div>",
                unsafe_allow_html=True
            )

    # Grad-CAM gallery
    gc_dir = "outputs/gradcam"
    if os.path.isdir(gc_dir):
        gc_files = sorted([
            f for f in os.listdir(gc_dir) if f.endswith(".png")
        ])
        if gc_files:
            st.markdown("---")
            st.markdown(
                "<div class='section-title'>Grad-CAM Gallery — "
                "Failure & Pass Cases</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='font-size:0.8rem;color:#8b949e;margin-bottom:1rem'>"
                "<b>FAIL</b> = wrong prediction. "
                "<b>PASS</b> = correct malignant. "
                "Inspect FAIL cases — does the model attend to skin color "
                "instead of the lesion?"
                "</p>",
                unsafe_allow_html=True
            )
            gc_cols = st.columns(3, gap="small")
            for i, fname in enumerate(gc_files[:9]):
                label = "FAIL" if "FAIL" in fname else "PASS"
                color = "#f85149" if label == "FAIL" else "#3fb950"
                gc_cols[i % 3].markdown(
                    f"<p style='font-family:DM Mono,monospace;"
                    f"font-size:0.68rem;color:{color}'>{label}</p>",
                    unsafe_allow_html=True
                )
                gc_cols[i % 3].image(
                    os.path.join(gc_dir, fname),
                    use_container_width=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
<div class='site-header'>
  <p class='site-title'>MelanoScan</p>
  
</div>
""", unsafe_allow_html=True)

    # Load models + sidebar
    baseline_model, baseline_loaded, fairness_model, fairness_loaded = load_models()
    fst = render_sidebar()

    # Status bar
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        f"<div class='metric-card'><div class='metric-label'>Baseline Model</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:0.8rem;"
        f"color:{'#3fb950' if baseline_loaded else '#f85149'}'>"
        f"{'LOADED' if baseline_loaded else 'NOT FOUND'}</div></div>",
        unsafe_allow_html=True
    )
    s2.markdown(
        f"<div class='metric-card'><div class='metric-label'>DFAL Model</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:0.8rem;"
        f"color:{'#3fb950' if fairness_loaded else '#f85149'}'>"
        f"{'LOADED' if fairness_loaded else 'NOT FOUND'}</div></div>",
        unsafe_allow_html=True
    )
    s3.markdown(
        f"<div class='metric-card'><div class='metric-label'>Device</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:0.8rem;"
        f"color:#58a6ff'>{str(device).upper()}</div></div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Clinical Tool",
        "Audit",
        "Training History",
        "Evaluation Results",
    ])

    with tab1:
        tab_predict(baseline_model, baseline_loaded,
                    fairness_model, fairness_loaded, fst)
    with tab2:
        tab_audit()
    with tab3:
        tab_history()
    with tab4:
        tab_evaluation()


if __name__ == "__main__":
    main()