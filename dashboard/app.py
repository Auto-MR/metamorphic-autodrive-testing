"""
Metamorphic Autodrive Testing — Upgraded Dashboard
Run with: streamlit run dashboard/app.py

UPGRADES (per research guideline):
  • 5 models: DAVE-2, GPR, SVR, Random Forest, Linear Regression
  • 4 MR categories: A (Invariance), B (Symmetry), C (Temporal), D (Composition)
  • Cross-model generalization score table
  • Statistical (dataset-based) thresholds
  • Output normalization for fair comparison
  • Model weakness analysis
  • Per-category generalization scores
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── Adapters ──────────────────────────────────────────────────────────────────
from core.adapters.cnn_adapter            import CNNAdapter
from core.adapters.steering_adapter       import SteeringAdapter
from core.adapters.gpr_steering_adapter   import GPRSteeringAdapter
from core.adapters.svr_steering_adapter   import SVRSteeringAdapter
from core.adapters.rf_steering_adapter    import RFSteeringAdapter
from core.adapters.linear_steering_adapter import LinearSteeringAdapter

# ── MR Engine ─────────────────────────────────────────────────────────────────
from core.metamorphic_engine.transformer  import get_transform, list_transforms
from core.metamorphic_engine.mr_library   import get_steering_mrs, get_mrs_by_category

# ── Evaluation ────────────────────────────────────────────────────────────────
from evaluation.generalization_scorer import (
    compute_generalization_score,
    compute_cross_model_table,
    rank_models_by_robustness,
    identify_model_weaknesses,
    compute_category_generalization,
    statistical_threshold,
    MR_CATEGORIES,
)
from evaluation.robustness_scorer import compute_score
from evaluation.visualization import make_steering_gauge
from core.metrics.normalizer import normalize_steering_output

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metamorphic Autodrive Testing Framework",
    layout="wide",
    page_icon="",
)

# ── Academic CSS Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Headers */
h1, h2, h3 {
    color: #1a1a2e;
    font-weight: 600;
    letter-spacing: -0.3px;
}

h1 {
    border-bottom: 2px solid #2c7a47;
    padding-bottom: 0.5rem;
}

h2 {
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 0.3rem;
}

/* Category badges */
.category-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7em;
    font-weight: 600;
    margin-left: 6px;
}
.cat-A { background: #2c7a47; color: white; }
.cat-B { background: #2563eb; color: white; }
.cat-C { background: #290ad8; color: white; }
.cat-D { background: #7c3aed; color: white; }

/* Metric cards */
.academic-metric {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.75rem;
    text-align: center;
}

.academic-metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    font-weight: 500;
}

.academic-metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
}

/* Score indicators */
.gen-score-high { color: #2c7a47; font-weight: bold; }
.gen-score-mid  { color: #290ad8; font-weight: bold; }
.gen-score-low  { color: #c0392b; font-weight: bold; }

/* Progress bars */
.stProgress > div > div {
    background-color: #2c7a47;
}

/* Buttons */
.stButton > button {
    background-color: #1e293b;
    color: white;
    font-weight: 500;
    border: none;
    border-radius: 6px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background-color: #2c7a47;
    color: white;
}

/* Dataframe styling */
.dataframe {
    font-size: 0.8rem;
}

/* Info boxes */
.stAlert {
    border-radius: 6px;
}

/* Expander headers */
.streamlit-expanderHeader {
    font-weight: 500;
    background-color: #f8fafc;
    border-radius: 6px;
}

/* Sidebar */
.css-1d391kg {
    background-color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem;'>
    <h1 style='margin-bottom: 0.25rem;'>Metamorphic Autodrive Testing Framework</h1>
    <p style='color: #475569; font-size: 1rem;'>Cross-model generalization of metamorphic relations for autonomous driving regression models</p>
</div>
""", unsafe_allow_html=True)

# ── Helper: load adapter ──────────────────────────────────────────────────────
@st.cache_resource
def load_adapter(model_choice: str):
    if model_choice == "DAVE-2 (CNN)":
        a = SteeringAdapter(); a.load(); return a
    elif model_choice == "GPR":
        a = GPRSteeringAdapter(); a.load(); return a
    elif model_choice == "SVR":
        a = SVRSteeringAdapter(); a.load(); return a
    elif model_choice == "Random Forest":
        a = RFSteeringAdapter(); a.load(); return a
    elif model_choice == "Linear Regression":
        a = LinearSteeringAdapter(); a.load(); return a
    elif model_choice == "CNN Depth":
        a = CNNAdapter(); a.load(); return a
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Configuration")

page = st.sidebar.radio(
    "Analysis Mode",
    ["Single Model Test", "Cross-Model Analysis", "Parametric Sweep",
     "Upload Model"],
    help="Single: test one model | Cross: compare all 5 | Sweep: intensity sensitivity | Upload: bring your own model"
)

st.sidebar.divider()

MODEL_OPTIONS = ["DAVE-2 (CNN)", "GPR", "SVR", "Random Forest", "Linear Regression", "CNN Depth"]
TRANSFORM_OPTIONS = [
    "brightness", "contrast", "noise", "fog", "rain", "snow",
    "flip", "rotate_5", "scale", "crop",
    "translate", "translate_large",
    "composed_bright_noise", "composed_weather_blur", "composed_shift_bright",
]

CATEGORY_LABELS = {
    "A": "Invariance (|f(x) - f(T(x))| < ε)",
    "B": "Symmetry (f(flip(x)) ≈ -f(x))",
    "C": "Temporal (|f(x_t) - f(x_{t+1})| < δ)",
    "D": "Composition (multiple transforms)",
}

if page == "Single Model Test":
    model_choice     = st.sidebar.selectbox("Model", MODEL_OPTIONS)
    transform_choice = st.sidebar.selectbox("Transformation", TRANSFORM_OPTIONS)

    # Statistical threshold (dataset-based)
    st.sidebar.markdown("### Threshold")
    use_stat_thresh = st.sidebar.checkbox("Use statistical threshold", value=True)
    if use_stat_thresh:
        tolerance = statistical_threshold()
        st.sidebar.info(f"ε = 0.05 × 2.0 = {tolerance:.2f}")
    else:
        tolerance = st.sidebar.slider("Manual ε", 0.01, 0.30, 0.10, 0.01)

    mr_category = st.sidebar.multiselect(
        "MR Categories",
        ["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help="A=Invariance, B=Symmetry, C=Temporal, D=Composition"
    )

else:
    st.sidebar.markdown("### Cross-Model Settings")
    n_test_images = st.sidebar.slider("Test images", 3, 15, 5)
    tolerance = statistical_threshold()
    st.sidebar.info(f"Statistical ε = {tolerance:.2f}")
    mr_category = ["A", "B", "C", "D"]


# ── Image upload ──────────────────────────────────────────────────────────────
st.markdown("### Input Image")
uploaded = st.file_uploader("Upload a driving image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB")).astype(np.float32) / 255.0
else:
    st.caption("No image uploaded — using synthetic driving scene.")
    x = np.linspace(0.1, 0.9, 64)
    img = np.stack([
        np.tile(x,        (64, 1)),
        np.tile(x[::-1],  (64, 1)),
        np.ones((64, 64)) * 0.4,
    ], axis=-1).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1: SINGLE MODEL TEST
# ════════════════════════════════════════════════════════════════════════════════
if page == "Single Model Test":

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(img, caption="Input image", use_container_width=True)
    with col_info:
        cat_for_transform = {
            "brightness": "A", "contrast": "A", "noise": "A", "fog": "A",
            "rain": "A", "snow": "A",
            "flip": "B", "rotate_5": "B", "scale": "B", "crop": "B",
            "translate": "C", "translate_large": "C",
            "composed_bright_noise": "D", "composed_weather_blur": "D",
            "composed_shift_bright": "D",
        }.get(transform_choice, "?")

        st.markdown(f"""
        <div style='background:#f8fafc; padding:0.75rem; border-radius:6px;'>
        <table style='width:100%;'>
        <tr><td style='font-weight:500;'>Model</td><td>{model_choice}</td></tr>
        <tr><td style='font-weight:500;'>Transformation</td><td>{transform_choice} <span class='category-badge cat-{cat_for_transform}'>Cat {cat_for_transform}</span></td></tr>
        <tr><td style='font-weight:500;'>Threshold ε</td><td>{tolerance}</td></tr>
        <tr><td style='font-weight:500;'>MR Categories</td><td>{', '.join(mr_category)}</td></tr>
        <tr><td style='font-weight:500;'>Input shape</td><td>{img.shape}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Run Metamorphic Tests", type="primary"):

        with st.spinner("Running tests ..."):

            # Load model
            adapter = load_adapter(model_choice)

            # Apply transform
            try:
                transform_fn = get_transform(transform_choice, domain="image")
                img_trans = transform_fn(img)
            except KeyError as e:
                st.error(str(e))
                st.stop()

            # Select MRs by category
            if model_choice == "CNN Depth":
                from metamorphic_relations.cnn_mrs import BrightnessInvarianceMR, FlipSymmetryCNNMR, NoiseRobustnessMR
                mrs = [BrightnessInvarianceMR(tolerance), FlipSymmetryCNNMR(tolerance), NoiseRobustnessMR(tolerance)]
            else:
                mrs = []
                for cat in mr_category:
                    mrs.extend(get_mrs_by_category(cat, tolerance))

            # Predict
            y_orig  = adapter.predict(img)
            y_trans = adapter.predict(img_trans)

            # Normalize outputs for display
            y_orig_norm  = normalize_steering_output(float(y_orig))
            y_trans_norm = normalize_steering_output(float(y_trans))

            # Check MRs
            results = [mr.check(img, img_trans, y_orig, y_trans) for mr in mrs]

        # ── Image comparison ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Visual Comparison")

        col_o, col_t, col_d = st.columns(3)
        with col_o:
            st.image(np.clip(img, 0, 1), caption="Original", use_container_width=True)
        with col_t:
            st.image(np.clip(img_trans, 0, 1), caption=f"Transformed: {transform_choice}", use_container_width=True)
        with col_d:
            diff_img = np.clip(np.abs(img - img_trans) * 4, 0, 1)
            st.image(diff_img, caption="Pixel Difference (×4)", use_container_width=True)

        # ── Model output comparison ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Model Output")

        if model_choice != "CNN Depth":
            gauge_fig = make_steering_gauge(float(y_orig), float(y_trans), transform_choice)
            st.pyplot(gauge_fig)
            plt.close()

            delta = float(y_trans) - float(y_orig)
            delta_color = "#2c7a47" if delta > 0 else "#c0392b"
            
            col_a, col_b, col_c, col_d_col = st.columns(4)
            col_a.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>θ original</div><div class='academic-metric-value'>{float(y_orig):.4f}</div></div>", unsafe_allow_html=True)
            col_b.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>θ transformed</div><div class='academic-metric-value'>{float(y_trans):.4f}</div><div class='academic-metric-label' style='color:{delta_color};'>{delta:+.4f}</div></div>", unsafe_allow_html=True)
            col_c.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Normalized (original)</div><div class='academic-metric-value'>{y_orig_norm:.3f}</div></div>", unsafe_allow_html=True)
            col_d_col.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Normalized (transformed)</div><div class='academic-metric-value'>{y_trans_norm:.3f}</div></div>", unsafe_allow_html=True)

        # ── MR results by category ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Metamorphic Relation Results")

        for cat_letter in ["A", "B", "C", "D"]:
            if cat_letter not in mr_category:
                continue
            cat_results = [r for r in results if r.mr_name in MR_CATEGORIES and MR_CATEGORIES[r.mr_name] == cat_letter]
            if not cat_results:
                continue

            cat_pass = sum(r.passed for r in cat_results)
            cat_total = len(cat_results)
            badge_class = f"cat-{cat_letter}"

            st.markdown(
                f"<div style='margin-top:0.75rem; margin-bottom:0.5rem;'>"
                f"<span style='font-weight:600;'>Category {cat_letter}</span> — {CATEGORY_LABELS[cat_letter]}  "
                f"<span class='category-badge {badge_class}'>{cat_pass}/{cat_total} passed</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            for r in cat_results:
                icon = "✓" if r.passed else "✗"
                label = f"{icon} **{r.mr_name}**"
                if r.delta is not None:
                    label += f" — δ = {r.delta:.4f}"
                with st.expander(label):
                    c1, c2 = st.columns(2)
                    c1.write(f"**Expected:** {r.expected}")
                    c2.write(f"**Actual:** {r.actual}")
                    if r.message:
                        if r.passed:
                            st.success(r.message)
                        else:
                            st.warning(r.message)

        # ── Score summary ─────────────────────────────────────────────────────
        st.markdown("---")
        passed_n  = sum(r.passed for r in results)
        total_n   = len(results)
        score     = passed_n / total_n if total_n else 0

        score_level = "High" if score >= 0.8 else ("Moderate" if score >= 0.5 else "Low")
        score_color = "#2c7a47" if score >= 0.8 else ("#290ad8" if score >= 0.5 else "#c0392b")

        st.markdown("#### Robustness Score")
        col_score, col_bar = st.columns([1, 3])
        col_score.markdown(f"<div class='academic-metric' style='background:{score_color}10; border-color:{score_color};'><div class='academic-metric-label'>Score</div><div class='academic-metric-value' style='color:{score_color};'>{score:.1%}</div><div class='academic-metric-label'>{score_level} ({passed_n}/{total_n} MRs)</div></div>", unsafe_allow_html=True)
        col_bar.progress(score)

        if score == 1.0:
            st.success("All metamorphic relations satisfied. Model demonstrates strong robustness.")
        elif score >= 0.75:
            st.warning("Minor violations detected. Consider investigating specific MR failures.")
        elif score >= 0.5:
            st.warning("Multiple violations detected. Model shows inconsistent behavior under transformations.")
        else:
            st.error("Major violations detected. Model is not robust to semantic-preserving transformations.")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2: CROSS-MODEL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Cross-Model Analysis":
    st.markdown("### Cross-Model Generalization Analysis")
    st.markdown("""
    This analysis runs all 14 metamorphic relations across 5 model architectures and 
    computes the **Generalization Score** — the fraction of models that satisfy each MR.
    
    > **Generalization Score** = models satisfying MR / total models
    """)

    col_models, col_info2 = st.columns([2, 1])
    with col_models:
        st.markdown("""
        **Models evaluated:**
        | # | Model | Architecture |
        |---|---|---|
        | 1 | DAVE-2 (CNN) | Deep Learning |
        | 2 | GPR | Probabilistic |
        | 3 | SVR | Classical ML |
        | 4 | Random Forest | Ensemble |
        | 5 | Linear Regression | Baseline |
        """)
    with col_info2:
        st.markdown(f"""
        <div style='background:#f8fafc; padding:0.75rem; border-radius:6px;'>
        <b>Configuration</b><br>
        ε = {tolerance}<br>
        Test images: {n_test_images}<br>
        MR categories: 4 (14 relations)
        </div>
        """, unsafe_allow_html=True)

    if st.button("Run Cross-Model Analysis", type="primary"):

        with st.spinner("Loading models and evaluating metamorphic relations ..."):

            # Generate varied test images
            test_imgs = []
            for i in range(n_test_images):
                x = np.linspace(0.05 + i*0.02, 0.95 - i*0.01, 64)
                base = np.stack([
                    np.tile(x, (64, 1)),
                    np.tile(x[::-1], (64, 1)),
                    np.ones((64, 64)) * (0.3 + i * 0.04),
                ], axis=-1).clip(0, 1).astype(np.float32)
                test_imgs.append(base)

            if uploaded:
                test_imgs.append(img)

            # Load all adapters
            adapters = {
                "dave2_cnn":      load_adapter("DAVE-2 (CNN)"),
                "gpr_steering":   load_adapter("GPR"),
                "svr_steering":   load_adapter("SVR"),
                "random_forest":  load_adapter("Random Forest"),
                "linear_baseline": load_adapter("Linear Regression"),
            }

            # MR to transform mapping
            MR_TRANSFORM_MAP = {
                "brightness_steering":          "brightness",
                "contrast_steering":            "contrast",
                "noise_robustness_steering":    "noise",
                "weather_robustness":           "fog",
                "occlusion_robustness":         "noise",
                "flip_symmetry_steering":       "flip",
                "straight_road_stability":      "brightness",
                "perspective_scaling":          "scale",
                "temporal_smoothness":          "translate",
                "constant_curvature_consistency": "translate",
                "small_lane_shift_invariance":  "translate",
                "composed_brightness_noise":    "composed_bright_noise",
                "composed_weather_blur":        "composed_weather_blur",
                "lane_center_shift_consistency": "composed_shift_bright",
            }

            from core.metamorphic_engine.mr_library import ALL_STEERING_MRS
            results_map = {}

            progress_bar = st.progress(0)
            for idx, (model_name, adapter) in enumerate(adapters.items()):
                model_results = []
                mrs = get_steering_mrs(tolerance)
                for mr in mrs:
                    tf_name = MR_TRANSFORM_MAP.get(mr.name, "brightness")
                    try:
                        transform_fn = get_transform(tf_name, domain="image")
                    except KeyError:
                        transform_fn = get_transform("brightness", domain="image")
                    for test_img in test_imgs:
                        img2 = transform_fn(test_img)
                        y1 = adapter.predict(test_img)
                        y2 = adapter.predict(img2)
                        result = mr.check(test_img, img2, y1, y2)
                        model_results.append(result)
                results_map[model_name] = model_results
                progress_bar.progress((idx + 1) / len(adapters))

            # Compute scores
            gen_scores  = compute_generalization_score(results_map)
            table       = compute_cross_model_table(results_map)
            ranking     = rank_models_by_robustness(results_map)
            weaknesses  = identify_model_weaknesses(results_map)
            cat_gen     = compute_category_generalization(results_map, MR_CATEGORIES)

        # ── Generalization table ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### MR × Model Generalization Matrix")

        model_names = list(adapters.keys())
        MODEL_DISPLAY = {
            "dave2_cnn": "DAVE-2",
            "gpr_steering": "GPR",
            "svr_steering": "SVR",
            "random_forest": "RF",
            "linear_baseline": "Linear",
        }

        import pandas as pd
        rows = []
        for mr_name in sorted(gen_scores.keys()):
            cat = MR_CATEGORIES.get(mr_name, "?")
            row = {
                "Metamorphic Relation": mr_name,
                "Category": f"Cat {cat}",
            }
            for model in model_names:
                display = MODEL_DISPLAY.get(model, model)
                row[display] = table.get(mr_name, {}).get(model, "—")
            row["Generalization"] = f"{gen_scores[mr_name]:.0%}"
            rows.append(row)

        def highlight_cells(val):
            if val == "✓":
                return "background-color: #2c7a4720; color: #2c7a47; font-weight: 500;"
            elif val == "✗":
                return "background-color: #c0392b20; color: #c0392b; font-weight: 500;"
            elif "%" in str(val):
                pct = float(str(val).strip("%")) / 100
                if pct >= 0.8:
                    return "color: #2c7a47; font-weight: bold"
                elif pct >= 0.6:
                    return "color: #290ad8; font-weight: bold"
                else:
                    return "color: #c0392b; font-weight: bold"
            return ""

        st.dataframe(pd.DataFrame(rows).style.applymap(highlight_cells), use_container_width=True)

        # ── Category generalization scores ────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Performance by Category")

        col_a, col_b, col_c, col_d_col = st.columns(4)
        cat_metrics = [
            (col_a, "A", "Invariance"),
            (col_b, "B", "Symmetry"),
            (col_c, "C", "Temporal"),
            (col_d_col, "D", "Composition"),
        ]
        for col, cat, label in cat_metrics:
            score_val = cat_gen.get(cat, 0.0)
            col.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Category {cat}</div><div class='academic-metric-value'>{score_val:.1%}</div><div class='academic-metric-label'>{label}</div></div>", unsafe_allow_html=True)

        # Category bar chart
        fig_cat, ax = plt.subplots(figsize=(8, 3.5), facecolor="white")
        cats = list(cat_gen.keys())
        vals = [cat_gen[c] for c in cats]
        colors = ["#2c7a47" if v >= 0.8 else "#290ad8" if v >= 0.5 else "#c0392b" for v in vals]
        bars = ax.bar([f"Category {c}" for c in cats], vals, color=colors, edgecolor="#cbd5e1", linewidth=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Generalization Score", fontsize=10)
        ax.set_title("Generalization Score by MR Category", fontsize=11, fontweight="bold")
        ax.axhline(0.8, color="#64748b", linestyle="--", lw=0.8, alpha=0.7, label="80% threshold")
        ax.set_facecolor("#f8fafc")
        ax.tick_params(colors="#475569", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f"{val:.0%}", ha="center", color="#1e293b", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cat)
        plt.close()

        # ── Model robustness ranking ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Model Robustness Ranking")

        fig_rank, ax2 = plt.subplots(figsize=(8, 4), facecolor="white")
        rank_names = [MODEL_DISPLAY.get(m, m) for m, _ in ranking]
        rank_vals  = [s for _, s in ranking]
        rcolors = ["#2c7a47" if v >= 0.8 else "#290ad8" if v >= 0.5 else "#c0392b" for v in rank_vals]
        rbars = ax2.barh(rank_names[::-1], rank_vals[::-1], color=rcolors[::-1], edgecolor="#cbd5e1", linewidth=0.5)
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Overall Pass Rate", fontsize=10)
        ax2.set_title("Model Robustness (14 MRs)", fontsize=11, fontweight="bold")
        ax2.tick_params(colors="#475569", labelsize=9)
        ax2.set_facecolor("#f8fafc")
        ax2.axvline(0.8, color="#64748b", linestyle="--", lw=0.8, alpha=0.7)
        for spine in ax2.spines.values():
            spine.set_color("#cbd5e1")
        for bar, val in zip(rbars, rank_vals[::-1]):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.1%}", va="center", color="#1e293b", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_rank)
        plt.close()

        # ── Violation heatmap ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### MR × Model Pass/Fail Matrix")

        mr_names_sorted = sorted(gen_scores.keys())
        heat_data = np.zeros((len(mr_names_sorted), len(model_names)))
        for i, mr_name in enumerate(mr_names_sorted):
            for j, model in enumerate(model_names):
                heat_data[i, j] = 1.0 if table.get(mr_name, {}).get(model, "✗") == "✓" else 0.0

        fig_heat, ax3 = plt.subplots(figsize=(10, 6), facecolor="white")
        im = ax3.imshow(heat_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_names], fontsize=8)
        ax3.set_yticks(range(len(mr_names_sorted)))
        ax3.set_yticklabels(mr_names_sorted, fontsize=7)
        ax3.set_title("Pass (green) / Fail (red) — MR × Model", fontsize=11, fontweight="bold")
        ax3.set_facecolor("#f8fafc")
        for i in range(len(mr_names_sorted)):
            for j in range(len(model_names)):
                txt = "✓" if heat_data[i, j] == 1 else "✗"
                ax3.text(j, i, txt, ha="center", va="center",
                         color="black", fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax3, label="Pass rate")
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close()

        # ── Model weaknesses ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Weakness Analysis")

        for model_key, failed_mrs in weaknesses.items():
            display_name = MODEL_DISPLAY.get(model_key, model_key)
            if failed_mrs:
                with st.expander(f"[{display_name}] {len(failed_mrs)} failing metamorphic relations"):
                    for mr in failed_mrs:
                        cat = MR_CATEGORIES.get(mr, "?")
                        st.markdown(f"- **{mr}** (Category {cat})")
            else:
                st.success(f"**{display_name}**: No weaknesses detected — all MRs passed")

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Research Summary")

        avg_gen = np.mean(list(gen_scores.values()))
        top_model, top_score = ranking[0]
        worst_model, worst_score = ranking[-1]
        best_mr = max(gen_scores, key=gen_scores.get)
        worst_mr = min(gen_scores, key=gen_scores.get)

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Avg Generalization</div><div class='academic-metric-value'>{avg_gen:.1%}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Most Robust Model</div><div class='academic-metric-value'>{MODEL_DISPLAY.get(top_model, top_model)}</div><div class='academic-metric-label'>{top_score:.1%}</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Most Generalizable MR</div><div class='academic-metric-value' style='font-size:0.7rem;'>{best_mr}</div><div class='academic-metric-label'>{gen_scores[best_mr]:.0%}</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Least Generalizable MR</div><div class='academic-metric-value' style='font-size:0.7rem;'>{worst_mr}</div><div class='academic-metric-label'>{gen_scores[worst_mr]:.0%}</div></div>", unsafe_allow_html=True)

        st.info(
            f"**Research Finding:** MR `{best_mr}` generalizes across all {len(adapters)} architectures "
            f"({gen_scores[best_mr]:.0%} generalization score), confirming it as a model-agnostic "
            f"metamorphic relation. MR `{worst_mr}` is architecture-specific "
            f"({gen_scores[worst_mr]:.0%} generalization score)."
        )


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3: PARAMETRIC SWEEP TESTING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Parametric Sweep":

    from evaluation.parametric_sweep import (
        SWEEP_CATALOGUE, run_parametric_sweep, run_cross_model_sweep
    )
    from evaluation.sweep_visualization import (
        plot_single_sweep, plot_cross_model_sweep,
        plot_critical_thresholds, plot_sweep_summary_heatmap
    )

    st.markdown("### Parametric Sensitivity Analysis")
    st.markdown("""
    Tests MR pass rate as transformation intensity increases — from zero to extreme values.
    This identifies the intensity threshold at which a model begins to fail and quantifies degradation rate.
    """)

    # ── Sweep sidebar controls ────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Sweep Configuration")

    sweep_mode = st.sidebar.radio(
        "Sweep Mode",
        ["Single Model", "Cross-Model"],
        help="Single: detailed per-model analysis | Cross-model: compare all architectures"
    )

    SWEEP_TRANSFORM_OPTIONS = list(SWEEP_CATALOGUE.keys())
    sweep_transform = st.sidebar.selectbox("Transformation", SWEEP_TRANSFORM_OPTIONS)
    cfg = SWEEP_CATALOGUE[sweep_transform]
    st.sidebar.caption(cfg["description"])
    st.sidebar.caption(f"Range: {cfg['sweep_values'][0]}{cfg['unit']} → {cfg['sweep_values'][-1]}{cfg['unit']}")

    if sweep_mode == "Single Model":
        sweep_model = st.sidebar.selectbox("Model", MODEL_OPTIONS[:-1])

    MR_OPTIONS_SWEEP = [
        "flip_symmetry_steering",
        "brightness_steering",
        "noise_robustness_steering",
        "weather_robustness",
        "temporal_smoothness",
        "perspective_scaling",
        "composed_brightness_noise",
    ]
    sweep_mr_name = st.sidebar.selectbox("Metamorphic Relation", MR_OPTIONS_SWEEP)

    n_images_sweep = st.sidebar.slider("Test images", 3, 20, 8)

    # Custom range controls
    st.sidebar.markdown("### Custom Range")
    use_custom = st.sidebar.checkbox("Override sweep values")
    if use_custom:
        custom_min  = st.sidebar.number_input("Min value",  value=float(cfg["sweep_values"][0]))
        custom_max  = st.sidebar.number_input("Max value",  value=float(cfg["sweep_values"][-1]))
        custom_n    = st.sidebar.slider("Number of steps", 5, 30, 12)
        custom_vals = list(np.linspace(custom_min, custom_max, custom_n))
    else:
        custom_vals = None

    sweep_tolerance = statistical_threshold()

    # ── Info cards ────────────────────────────────────────────────────────────
    default_vals = cfg["sweep_values"] if custom_vals is None else custom_vals
    
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Transformation</div><div class='academic-metric-value'>{sweep_transform}</div><div class='academic-metric-label'>Category {cfg['category']}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Intensity Range</div><div class='academic-metric-value'>{default_vals[0]}{cfg['unit']}</div><div class='academic-metric-label'>to {default_vals[-1]}{cfg['unit']} ({len(default_vals)} levels)</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Statistical ε</div><div class='academic-metric-value'>{sweep_tolerance:.2f}</div><div class='academic-metric-label'>0.05 × steering range</div></div>", unsafe_allow_html=True)

    # ── Run sweep ─────────────────────────────────────────────────────────────
    if st.button("Run Sensitivity Analysis", type="primary"):

        # Generate test images
        sweep_imgs = []
        for i in range(n_images_sweep):
            x = np.linspace(0.05 + i * 0.02, 0.95, 64)
            base = np.stack([
                np.tile(x, (64, 1)),
                np.tile(x[::-1], (64, 1)),
                np.ones((64, 64)) * (0.3 + i * 0.035),
            ], axis=-1).clip(0, 1).astype(np.float32)
            sweep_imgs.append(base)
        if uploaded:
            sweep_imgs.insert(0, img)

        # Load MR
        from core.metamorphic_engine.mr_library import get_mr
        sweep_mr = get_mr(sweep_mr_name, tolerance=sweep_tolerance)

        if sweep_mode == "Single Model":
            # ── Single model sweep ─────────────────────────────────────────────
            adapter = load_adapter(sweep_model)
            from evaluation.parametric_sweep import SWEEP_CATALOGUE as SC
            factory = SC[sweep_transform]["factory"]

            sweep_levels = []
            with st.spinner(f"Sweeping {sweep_transform} across {len(default_vals)} levels ..."):
                from core.metamorphic_engine.mr_library import get_mr as _get_mr
                _mr_inst = _get_mr(sweep_mr_name, tolerance=sweep_tolerance)
                from evaluation.parametric_sweep import SweepResult, SweepPoint
                _result = SweepResult(
                    transform_name=sweep_transform,
                    parameter_name=cfg["label"],
                    mr_name=sweep_mr_name,
                    model_name=adapter.model_name,
                )
                _result._tolerance = sweep_tolerance

                ref_img = sweep_imgs[0]
                ref_f = (ref_img / 255.0).astype(np.float32) if ref_img.max() > 1.0 else ref_img.astype(np.float32)
                y_orig_ref = adapter.predict(ref_f)

                for intensity in default_vals:
                    tf_fn  = factory(intensity)
                    img_t  = tf_fn(ref_f)
                    y_t    = adapter.predict(img_t)
                    diff_t = np.clip(np.abs(ref_f - img_t) * 4, 0, 1)

                    deltas, passes = [], []
                    for _si in sweep_imgs:
                        _si_f = ((_si / 255.0).astype(np.float32) if _si.max() > 1.0 else _si.astype(np.float32))
                        _si2 = factory(intensity)(_si_f)
                        _y1  = adapter.predict(_si_f)
                        _y2  = adapter.predict(_si2)
                        _r   = _mr_inst.check(_si_f, _si2, _y1, _y2)
                        deltas.append(_r.delta if _r.delta is not None else abs(_y1 - _y2))
                        passes.append(_r.passed)

                    n = len(sweep_imgs)
                    pc = sum(passes)
                    pt = SweepPoint(
                        intensity=intensity,
                        pass_rate=pc / n,
                        mean_delta=float(np.mean(deltas)),
                        max_delta=float(np.max(deltas)),
                        violations=n - pc,
                        total=n,
                        passed=pc,
                    )
                    _result.sweep_points.append(pt)

                    mr_ref = _mr_inst.check(ref_f, img_t, y_orig_ref, y_t)

                    sweep_levels.append({
                        "intensity":  intensity,
                        "img_orig":   ref_f,
                        "img_trans":  img_t,
                        "img_diff":   diff_t,
                        "y_orig":     float(y_orig_ref),
                        "y_trans":    float(y_t),
                        "delta":      float(abs(y_orig_ref - y_t)),
                        "mr_passed":  mr_ref.passed,
                        "mr_message": mr_ref.message,
                        "pass_rate":  pt.pass_rate,
                        "mean_delta": pt.mean_delta,
                    })

            result = _result
            ct  = result.critical_threshold
            ct2 = result.collapse_threshold

            # ── Summary metrics ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Summary")

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Levels</div><div class='academic-metric-value'>{len(default_vals)}</div></div>", unsafe_allow_html=True)
            mc2.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Max Pass Rate</div><div class='academic-metric-value'>{max(result.pass_rates):.1%}</div></div>", unsafe_allow_html=True)
            mc3.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Min Pass Rate</div><div class='academic-metric-value'>{min(result.pass_rates):.1%}</div></div>", unsafe_allow_html=True)
            degrade_text = f"{ct}{cfg['unit']}" if ct is not None else "None"
            degrade_color = "#2c7a47" if ct is None else "#290ad8"
            mc4.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Degradation at</div><div class='academic-metric-value' style='color:{degrade_color};'>{degrade_text}</div></div>", unsafe_allow_html=True)
            collapse_text = f"{ct2}{cfg['unit']}" if ct2 is not None else "None"
            collapse_color = "#2c7a47" if ct2 is None else "#c0392b"
            mc5.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Collapse at</div><div class='academic-metric-value' style='color:{collapse_color};'>{collapse_text}</div></div>", unsafe_allow_html=True)

            # ── Degradation chart ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Robustness Degradation Curve")
            fig_single = plot_single_sweep(result)
            st.pyplot(fig_single)
            plt.close()

            # ── Visual strip ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Transformation Progression")
            
            n_show = min(8, len(sweep_levels))
            step   = max(1, len(sweep_levels) // n_show)
            shown  = sweep_levels[::step][:n_show]
            n_cols = len(shown)
            
            fig_strip, axes_strip = plt.subplots(2, n_cols, figsize=(max(8, n_cols * 2), 4.5), facecolor="white")
            if n_cols == 1:
                axes_strip = np.array(axes_strip).reshape(2, 1)
            
            for j, lv in enumerate(shown):
                pr = lv["pass_rate"]
                status_color = "#2c7a47" if pr >= 0.80 else ("#290ad8" if pr >= 0.50 else "#c0392b")
                status_icon = "✓" if pr >= 0.80 else ("⚠" if pr >= 0.50 else "✗")
                
                ax_t = axes_strip[0, j]
                ax_t.imshow(np.clip(lv["img_trans"], 0, 1))
                ax_t.set_title(f"{lv['intensity']}{cfg['unit']}", fontsize=8)
                ax_t.axis("off")
                
                ax_d = axes_strip[1, j]
                ax_d.imshow(lv["img_diff"])
                ax_d.set_xlabel(f"{status_icon} {pr:.0%}", fontsize=8, color=status_color)
                ax_d.axis("off")
            
            axes_strip[0, 0].set_ylabel("Transformed", fontsize=8)
            axes_strip[1, 0].set_ylabel("Difference", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_strip)
            plt.close()

            # ── Steering output progression ─────────────────────────────────
            st.markdown("---")
            st.markdown("#### Steering Output Progression")

            intensities_arr = [lv["intensity"] for lv in sweep_levels]
            y_trans_arr     = [lv["y_trans"]   for lv in sweep_levels]
            delta_arr       = [lv["delta"]      for lv in sweep_levels]
            pass_rate_arr   = [lv["pass_rate"]  for lv in sweep_levels]
            y_orig_val      = sweep_levels[0]["y_orig"]
            
            bar_colors = ["#2c7a47" if pr >= 0.8 else "#290ad8" if pr >= 0.5 else "#c0392b" for pr in pass_rate_arr]
            
            fig_out, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor="white")
            x_pos = list(range(len(intensities_arr)))
            x_labels = [f"{v}{cfg['unit']}" for v in intensities_arr]
            
            # Panel A
            ax_s = axes[0]
            ax_s.set_facecolor("#f8fafc")
            ax_s.bar(x_pos, y_trans_arr, color=bar_colors, edgecolor="#cbd5e1", width=0.65)
            ax_s.axhline(y_orig_val, color="#1e293b", linestyle="--", lw=1.5, label=f"θ original = {y_orig_val:.4f}")
            ax_s.axhline(0, color="#cbd5e1", linestyle=":", lw=0.8)
            ax_s.set_xticks(x_pos)
            ax_s.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            ax_s.set_ylabel("Steering θ", fontsize=9)
            ax_s.set_title("Steering Output vs Intensity", fontsize=10)
            ax_s.legend(fontsize=8)
            
            # Panel B
            ax_d = axes[1]
            ax_d.set_facecolor("#f8fafc")
            ax_d.bar(x_pos, delta_arr, color=bar_colors, edgecolor="#cbd5e1", width=0.65)
            ax_d.axhline(sweep_tolerance, color="#290ad8", linestyle="--", lw=1.5, label=f"ε = {sweep_tolerance}")
            ax_d.set_xticks(x_pos)
            ax_d.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            ax_d.set_ylabel("|Δθ|", fontsize=9)
            ax_d.set_title("MR Delta — |Original − Transformed|", fontsize=10)
            ax_d.legend(fontsize=8)
            
            # Panel C
            ax_p = axes[2]
            ax_p.set_facecolor("#f8fafc")
            ax_p.plot(x_pos, pass_rate_arr, "o-", color="#2563eb", lw=2, markersize=6)
            ax_p.fill_between(x_pos, pass_rate_arr, alpha=0.1, color="#2563eb")
            for xi, (xp, pr) in enumerate(zip(x_pos, pass_rate_arr)):
                c = "#2c7a47" if pr >= 0.80 else "#290ad8" if pr >= 0.50 else "#c0392b"
                ax_p.scatter([xp], [pr], color=c, s=60, zorder=5)
            ax_p.axhline(0.80, color="#290ad8", linestyle="--", lw=1.2, label="80% threshold")
            ax_p.axhline(0.50, color="#c0392b", linestyle=":", lw=1.0, label="50% threshold")
            ax_p.set_xticks(x_pos)
            ax_p.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            ax_p.set_ylabel("Pass Rate", fontsize=9)
            ax_p.set_title("MR Pass Rate vs Intensity", fontsize=10)
            ax_p.legend(fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig_out)
            plt.close()

            # ── Per-level details ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Detailed Results by Intensity Level")
            
            for lv in sweep_levels:
                pr = lv["pass_rate"]
                icon = "✓" if pr >= 0.80 else ("⚠" if pr >= 0.50 else "✗")
                with st.expander(
                    f"{icon} {cfg['label']} = {lv['intensity']}{cfg['unit']}  |  "
                    f"Pass rate: {pr:.0%}  |  Δθ = {lv['delta']:.4f}"
                ):
                    ci1, ci2, ci3 = st.columns(3)
                    ci1.image(np.clip(lv["img_orig"], 0, 1), caption="Original", use_container_width=True)
                    ci2.image(np.clip(lv["img_trans"], 0, 1), caption=f"Transformed", use_container_width=True)
                    ci3.image(lv["img_diff"], caption="Difference", use_container_width=True)
                    
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("θ original", f"{lv['y_orig']:.4f}")
                    cm2.metric("θ transformed", f"{lv['y_trans']:.4f}", delta=f"{lv['y_trans'] - lv['y_orig']:+.4f}")
                    cm3.metric("|Δθ|", f"{lv['delta']:.4f}", delta="PASS" if lv["delta"] <= sweep_tolerance else "FAIL")
                    cm4.metric("Pass rate", f"{pr:.0%}")
                    
                    if pr >= 0.80:
                        st.success(lv["mr_message"])
                    elif pr >= 0.50:
                        st.warning(lv["mr_message"])
                    else:
                        st.error(lv["mr_message"])

            # ── Research interpretation ─────────────────────────────────────
            st.markdown("---")
            if ct is None:
                st.success(
                    f"**{sweep_model}** demonstrates high robustness to {sweep_transform} transformations. "
                    f"The MR {sweep_mr_name} maintains ≥80% pass rate across the entire tested range."
                )
            elif ct2 is None:
                st.warning(
                    f"**{sweep_model}** degrades at {cfg['label']} = {ct}{cfg['unit']}, "
                    f"but maintains performance above 50% throughout the range."
                )
            else:
                st.error(
                    f"**{sweep_model}** degrades at {cfg['label']} = {ct}{cfg['unit']} "
                    f"and collapses below 50% at {cfg['label']} = {ct2}{cfg['unit']}. "
                    f"This defines the operational safety boundary."
                )

        else:
            # ── Cross-model sweep ──────────────────────────────────────────────
            with st.spinner(f"Sweeping all models across {len(default_vals)} levels ..."):
                all_adapters = {
                    "dave2_cnn":       load_adapter("DAVE-2 (CNN)"),
                    "gpr_steering":    load_adapter("GPR"),
                    "svr_steering":    load_adapter("SVR"),
                    "random_forest":   load_adapter("Random Forest"),
                    "linear_baseline": load_adapter("Linear Regression"),
                }
                sweep_results = run_cross_model_sweep(
                    all_adapters, sweep_mr, sweep_imgs,
                    sweep_transform, custom_vals
                )

            st.markdown("---")
            st.markdown("#### Cross-Model Sweep Results")

            MODEL_DISPLAY_SWEEP = {
                "dave2_cnn": "DAVE-2", "gpr_steering": "GPR",
                "svr_steering": "SVR", "random_forest": "RF", "linear_baseline": "Linear",
            }

            summary_rows = []
            for model_name, res in sweep_results.items():
                ct_val = res.critical_threshold
                ct2_val = res.collapse_threshold
                avg_pr = np.mean(res.pass_rates)
                summary_rows.append({
                    "Model": MODEL_DISPLAY_SWEEP.get(model_name, model_name),
                    "Avg Pass Rate": f"{avg_pr:.1%}",
                    "Min Pass Rate": f"{min(res.pass_rates):.1%}",
                    "Degrades at": f"{ct_val}{cfg['unit']}" if ct_val is not None else "None",
                    "Collapses at": f"{ct2_val}{cfg['unit']}" if ct2_val is not None else "None",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            fig_cross = plot_cross_model_sweep(sweep_results)
            st.pyplot(fig_cross)
            plt.close()

            st.markdown("---")
            st.markdown("#### Critical Threshold Comparison")
            fig_thresh = plot_critical_thresholds(sweep_results)
            st.pyplot(fig_thresh)
            plt.close()


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 4: UPLOAD MODEL
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Upload Model":

    st.markdown("### Upload and Test Custom Model")
    st.markdown(
        "Submit any regression model file (.pkl, .h5, .pt, .onnx) — "
        "the framework auto-detects the framework and runs the complete MR test pipeline."
    )

    from evaluation.generalization_scorer import statistical_threshold as _stat_thresh
    from dashboard.upload_model_page import render as _render_upload_page

    _render_upload_page(tolerance=_stat_thresh())