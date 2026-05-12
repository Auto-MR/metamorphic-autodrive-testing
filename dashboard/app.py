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
    page_icon="🚗",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.category-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75em;
    font-weight: bold;
    margin-left: 6px;
}
.cat-A { background: #1a472a; color: #90EE90; }
.cat-B { background: #1a2a47; color: #87CEEB; }
.cat-C { background: #47381a; color: #FFD700; }
.cat-D { background: #3a1a3a; color: #DDA0DD; }
.gen-score-high { color: #2ecc71; font-weight: bold; }
.gen-score-mid  { color: #f39c12; font-weight: bold; }
.gen-score-low  { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🚗 Metamorphic Autodrive Testing Framework")
st.caption("Cross-model generalization of metamorphic relations for autonomous driving regression models")

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
st.sidebar.header("⚙️ Configuration")

page = st.sidebar.radio(
    "Mode",
    ["Single Model Test", "Cross-Model Analysis", "Parametric Sweep",
     "🆕 Upload & Test Your Model"],
    help="Upload & Test: bring your own .pkl/.h5/.pt/.onnx model and run the full MR pipeline on it."
)

st.sidebar.divider()

MODEL_OPTIONS = ["DAVE-2 (CNN)", "GPR", "SVR", "Random Forest", "Linear Regression", "CNN Depth"]
TRANSFORM_OPTIONS = [
    "— Category A: Invariance —",
    "brightness", "contrast", "noise", "fog", "rain", "snow",
    "— Category B: Symmetry —",
    "flip", "rotate_5", "scale", "crop",
    "— Category C: Temporal —",
    "translate", "translate_large",
    "— Category D: Composition —",
    "composed_bright_noise", "composed_weather_blur", "composed_shift_bright",
]
VALID_TRANSFORMS = [t for t in TRANSFORM_OPTIONS if not t.startswith("—")]

CATEGORY_LABELS = {
    "A": "Invariance (|f(x) - f(T(x))| < ε)",
    "B": "Symmetry (f(flip(x)) ≈ -f(x))",
    "C": "Temporal (|f(x_t) - f(x_{t+1})| < δ)",
    "D": "Composition (multiple transforms)",
}

if page == "Single Model Test":
    model_choice     = st.sidebar.selectbox("Model", MODEL_OPTIONS)
    transform_choice = st.sidebar.selectbox(
        "Transform", TRANSFORM_OPTIONS,
        format_func=lambda x: x if not x.startswith("—") else x,
    )
    if transform_choice.startswith("—"):
        transform_choice = "brightness"

    # Statistical threshold (dataset-based)
    st.sidebar.subheader("Threshold (ε)")
    use_stat_thresh = st.sidebar.checkbox("Use statistical threshold (ε = 0.05 × range)", value=True)
    if use_stat_thresh:
        tolerance = statistical_threshold()
        st.sidebar.info(f"ε = 0.05 × 2.0 = **{tolerance:.2f}**")
    else:
        tolerance = st.sidebar.slider("Manual ε", 0.01, 0.30, 0.10, 0.01)

    mr_category = st.sidebar.multiselect(
        "MR Categories to test",
        ["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help="A=Invariance, B=Symmetry, C=Temporal, D=Composition"
    )

else:
    # Cross-model mode
    st.sidebar.subheader("Cross-Model Analysis")
    n_test_images = st.sidebar.slider("Test images (synthetic)", 3, 15, 5)
    tolerance = statistical_threshold()
    st.sidebar.info(f"Statistical ε = **{tolerance:.2f}** (0.05 × steering range)")
    mr_category = ["A", "B", "C", "D"]


# ── Image upload ──────────────────────────────────────────────────────────────
st.subheader("📸 Input Image")
uploaded = st.file_uploader("Upload a driving image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB")).astype(np.float32) / 255.0
else:
    st.info("No image uploaded — using synthetic driving scene for demo.")
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
        | Property | Value |
        |---|---|
        | Shape | `{img.shape}` |
        | Model | `{model_choice}` |
        | Transform | `{transform_choice}` (Category {cat_for_transform}) |
        | Threshold ε | `{tolerance}` |
        | MR Categories | `{', '.join(mr_category)}` |
        """)

    if st.button("▶️ Run Metamorphic Tests", type="primary"):

        with st.spinner("Running tests..."):

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
        st.subheader("🖼️ Visual Comparison")

        col_o, col_t, col_d = st.columns(3)
        with col_o:
            st.image(np.clip(img, 0, 1), caption="Original", use_container_width=True)
        with col_t:
            st.image(np.clip(img_trans, 0, 1), caption=f"Transformed ({transform_choice})", use_container_width=True)
        with col_d:
            diff_img = np.clip(np.abs(img - img_trans) * 4, 0, 1)
            st.image(diff_img, caption="Pixel Diff (×4 amplified)", use_container_width=True)

        # ── Model output comparison ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Model Output Comparison")

        if model_choice != "CNN Depth":
            gauge_fig = make_steering_gauge(float(y_orig), float(y_trans), transform_choice)
            st.pyplot(gauge_fig)
            plt.close()

            col_a, col_b, col_c, col_d_col = st.columns(4)
            col_a.metric("Steering (original)",    f"{float(y_orig):.4f}")
            col_b.metric("Steering (transformed)", f"{float(y_trans):.4f}",
                         delta=f"{float(y_trans) - float(y_orig):.4f}")
            col_c.metric("Normalized y (original)", f"{y_orig_norm:.3f}",
                         help="(y - (-1)) / (1 - (-1))")
            col_d_col.metric("Normalized y (transformed)", f"{y_trans_norm:.3f}",
                              help="Normalized to [0,1] for cross-model comparison")

        # ── MR results by category ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Metamorphic Relation Results")

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
                f"**Category {cat_letter}** — {CATEGORY_LABELS[cat_letter]}  "
                f"<span class='category-badge {badge_class}'>{cat_pass}/{cat_total} passed</span>",
                unsafe_allow_html=True,
            )

            for r in cat_results:
                icon = "✅" if r.passed else "❌"
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

        st.subheader("🏆 Robustness Score")
        col_score, col_bar = st.columns([1, 3])
        col_score.metric(
            "Robustness Score", f"{score:.1%}",
            f"{passed_n}/{total_n} MRs passed"
        )
        col_bar.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)
        col_bar.progress(score)

        if score == 1.0:
            st.success("✅ All MRs satisfied — model is robust to this transform.")
        elif score >= 0.75:
            st.warning("⚠️ Minor violations — model shows some sensitivity.")
        elif score >= 0.5:
            st.warning("⚠️ Multiple violations — partial robustness.")
        else:
            st.error("❌ Major violations — model is not robust to this transform.")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2: CROSS-MODEL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Cross-Model Analysis":
    st.subheader("🔁 Cross-Model Generalization Analysis")
    st.markdown("""
    This runs all **14 MRs** across **5 model architectures** simultaneously and 
    computes the **Generalization Score** — the fraction of models that satisfy each MR.

    > **Generalization Score** = models satisfying MR / total models
    """)

    col_models, col_info2 = st.columns([2, 1])
    with col_models:
        st.markdown("""
        **Models compared:**
        | # | Model | Type |
        |---|---|---|
        | 1 | DAVE-2 (CNN) | Deep Learning |
        | 2 | GPR | Probabilistic |
        | 3 | SVR | Classical ML |
        | 4 | Random Forest | Ensemble |
        | 5 | Linear Regression | Baseline |
        """)
    with col_info2:
        st.markdown(f"""
        **Settings:**
        - ε = `{tolerance}` (statistical)
        - Test images: `{n_test_images}`
        - Total MRs: `14` across 4 categories
        """)

    if st.button("▶️ Run Cross-Model Analysis", type="primary"):

        with st.spinner("Loading all 5 models and running 14 MRs..."):

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

            # Also include uploaded image if available
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

            # Collect results per model
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
        st.subheader("📋 Cross-Model Generalization Table")
        st.markdown("*Each row = one MR. Each column = one model. Score = fraction of models passing.*")

        model_names = list(adapters.keys())
        MODEL_DISPLAY = {
            "dave2_cnn": "DAVE-2",
            "gpr_steering": "GPR",
            "svr_steering": "SVR",
            "random_forest": "RF",
            "linear_baseline": "Linear",
        }

        # Build table data
        import pandas as pd
        rows = []
        for mr_name in sorted(gen_scores.keys()):
            cat = MR_CATEGORIES.get(mr_name, "?")
            row = {
                "MR Name": mr_name,
                "Category": f"Cat {cat}",
            }
            for model in model_names:
                display = MODEL_DISPLAY.get(model, model)
                row[display] = table.get(mr_name, {}).get(model, "—")
            row["Gen. Score"] = f"{gen_scores[mr_name]:.0%}"
            rows.append(row)

        df = pd.DataFrame(rows)

        def highlight_cells(val):
            if val == "✓":
                return "background-color: #1a3a1a; color: #90EE90"
            elif val == "✗":
                return "background-color: #3a1a1a; color: #FF9999"
            elif "%" in str(val):
                pct = float(str(val).strip("%")) / 100
                if pct >= 0.8:
                    return "color: #2ecc71; font-weight: bold"
                elif pct >= 0.6:
                    return "color: #f39c12; font-weight: bold"
                else:
                    return "color: #e74c3c; font-weight: bold"
            return ""

        st.dataframe(df.style.applymap(highlight_cells), use_container_width=True)

        # ── Category generalization scores ────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Generalization by MR Category")

        col_a, col_b, col_c, col_d_col = st.columns(4)
        cat_metrics = [
            (col_a, "A", "Invariance"),
            (col_b, "B", "Symmetry"),
            (col_c, "C", "Temporal"),
            (col_d_col, "D", "Composition"),
        ]
        for col, cat, label in cat_metrics:
            score_val = cat_gen.get(cat, 0.0)
            col.metric(
                f"Cat {cat}: {label}",
                f"{score_val:.1%}",
                help=CATEGORY_LABELS[cat]
            )

        # Category bar chart
        fig_cat, ax = plt.subplots(figsize=(8, 3), facecolor="#1a1a2e")
        cats = list(cat_gen.keys())
        vals = [cat_gen[c] for c in cats]
        colors = ["#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.5 else "#e74c3c" for v in vals]
        bars = ax.bar([f"Cat {c}\n{CATEGORY_LABELS[c][:12]}..." for c in cats], vals, color=colors, edgecolor="#444")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Generalization Score", color="white")
        ax.set_title("Generalization Score by MR Category", color="white", fontweight="bold")
        ax.axhline(0.8, color="grey", linestyle="--", lw=0.8, label="80% threshold")
        ax.tick_params(colors="white")
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.0%}", ha="center", color="white", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cat)
        plt.close()

        # ── Model robustness ranking ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏆 Model Robustness Ranking")

        fig_rank, ax2 = plt.subplots(figsize=(9, 4), facecolor="#1a1a2e")
        rank_names = [MODEL_DISPLAY.get(m, m) for m, _ in ranking]
        rank_vals  = [s for _, s in ranking]
        rcolors = ["#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.5 else "#e74c3c" for v in rank_vals]
        rbars = ax2.barh(rank_names[::-1], rank_vals[::-1], color=rcolors[::-1], edgecolor="#444")
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Overall Pass Rate", color="white")
        ax2.set_title("Model Robustness Ranking (all 14 MRs)", color="white", fontweight="bold")
        ax2.tick_params(colors="white")
        ax2.set_facecolor("#16213e")
        ax2.axvline(0.8, color="grey", linestyle="--", lw=0.8)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")
        for bar, val in zip(rbars, rank_vals[::-1]):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.1%}", va="center", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_rank)
        plt.close()

        # ── Generalization score heatmap ──────────────────────────────────────
        st.markdown("---")
        st.subheader("🔥 MR × Model Violation Heatmap")

        mr_names_sorted = sorted(gen_scores.keys())
        heat_data = np.zeros((len(mr_names_sorted), len(model_names)))
        for i, mr_name in enumerate(mr_names_sorted):
            for j, model in enumerate(model_names):
                heat_data[i, j] = 1.0 if table.get(mr_name, {}).get(model, "✗") == "✓" else 0.0

        fig_heat, ax3 = plt.subplots(figsize=(10, 7), facecolor="#1a1a2e")
        im = ax3.imshow(heat_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_names], color="white")
        ax3.set_yticks(range(len(mr_names_sorted)))
        ax3.set_yticklabels(mr_names_sorted, color="white", fontsize=8)
        ax3.set_title("Pass (green) / Fail (red) — MR × Model", color="white", fontweight="bold")
        ax3.set_facecolor("#16213e")
        for i in range(len(mr_names_sorted)):
            for j in range(len(model_names)):
                txt = "✓" if heat_data[i, j] == 1 else "✗"
                ax3.text(j, i, txt, ha="center", va="center",
                         color="white", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax3).ax.yaxis.set_tick_params(color="white")
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close()

        # ── Model weaknesses ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚠️ Model Weakness Analysis")
        st.markdown("*Identifies which MRs each model fails — enables targeted robustness improvements.*")

        for model_key, failed_mrs in weaknesses.items():
            display_name = MODEL_DISPLAY.get(model_key, model_key)
            if failed_mrs:
                with st.expander(f"❌ **{display_name}** — {len(failed_mrs)} weakness(es)"):
                    for mr in failed_mrs:
                        cat = MR_CATEGORIES.get(mr, "?")
                        st.markdown(f"- `{mr}` (Category {cat})")
            else:
                st.success(f"✅ **{display_name}** — No weaknesses detected (all MRs passed)")

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 Research Summary")

        avg_gen = np.mean(list(gen_scores.values()))
        top_model, top_score = ranking[0]
        worst_model, worst_score = ranking[-1]
        best_mr = max(gen_scores, key=gen_scores.get)
        worst_mr = min(gen_scores, key=gen_scores.get)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg. Generalization Score", f"{avg_gen:.1%}")
        col2.metric("Most Robust Model", MODEL_DISPLAY.get(top_model, top_model), f"{top_score:.1%}")
        col3.metric("Most Generalizable MR", best_mr.replace("_", " "), f"{gen_scores[best_mr]:.0%}")
        col4.metric("Least Generalizable MR", worst_mr.replace("_", " "), f"{gen_scores[worst_mr]:.0%}")

        st.info(
            f"**Research Finding:** MR `{best_mr}` generalizes across **all {len(adapters)} architectures** "
            f"({gen_scores[best_mr]:.0%} generalization score), confirming it as a model-agnostic "
            f"metamorphic relation. MR `{worst_mr}` is architecture-specific "
            f"({gen_scores[worst_mr]:.0%} generalization score)."
        )


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3: PARAMETRIC SWEEP TESTING  (supervisor requirement)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Parametric Sweep":

    from evaluation.parametric_sweep import (
        SWEEP_CATALOGUE, run_parametric_sweep, run_cross_model_sweep
    )
    from evaluation.sweep_visualization import (
        plot_single_sweep, plot_cross_model_sweep,
        plot_critical_thresholds, plot_sweep_summary_heatmap
    )

    st.subheader("📈 Parametric Sweep Testing")
    st.markdown("""
    Tests MR pass rate as transformation intensity increases — from zero to extreme values.
    This answers: *"At what intensity does a model start failing, and how quickly does it degrade?"*
    """)

    # ── Sweep sidebar controls ────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Sweep Settings")

    sweep_mode = st.sidebar.radio(
        "Sweep Mode",
        ["Single Model", "Cross-Model (all 5)"],
        help="Single: one model, detailed view. Cross-model: compare all architectures."
    )

    SWEEP_TRANSFORM_OPTIONS = list(SWEEP_CATALOGUE.keys())
    sweep_transform = st.sidebar.selectbox(
        "Transform to sweep",
        SWEEP_TRANSFORM_OPTIONS,
        help="Each transform is swept across a range of intensity values"
    )
    cfg = SWEEP_CATALOGUE[sweep_transform]
    st.sidebar.markdown(f"*{cfg['description']}*")
    st.sidebar.markdown(f"**Range:** {cfg['sweep_values'][0]}{cfg['unit']} → {cfg['sweep_values'][-1]}{cfg['unit']}")

    if sweep_mode == "Single Model":
        sweep_model = st.sidebar.selectbox("Model", MODEL_OPTIONS[:-1])   # exclude CNN Depth

    MR_OPTIONS_SWEEP = [
        "flip_symmetry_steering",
        "brightness_steering",
        "noise_robustness_steering",
        "weather_robustness",
        "temporal_smoothness",
        "perspective_scaling",
        "composed_brightness_noise",
    ]
    sweep_mr_name = st.sidebar.selectbox("MR to test", MR_OPTIONS_SWEEP)

    n_images_sweep = st.sidebar.slider("Test images", 3, 20, 8)

    # Custom range controls
    st.sidebar.subheader("Custom Range (optional)")
    use_custom = st.sidebar.checkbox("Override default sweep values")
    if use_custom:
        custom_min  = st.sidebar.number_input("Min value",  value=float(cfg["sweep_values"][0]))
        custom_max  = st.sidebar.number_input("Max value",  value=float(cfg["sweep_values"][-1]))
        custom_n    = st.sidebar.slider("Number of steps", 5, 30, 12)
        custom_vals = list(np.linspace(custom_min, custom_max, custom_n))
    else:
        custom_vals = None

    sweep_tolerance = statistical_threshold()

    # ── Info cards ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    default_vals = cfg["sweep_values"] if custom_vals is None else custom_vals
    col1.metric("Transform", sweep_transform)
    col1.caption(f"Category {cfg['category']} MR")
    col2.metric("Intensity Range",
                f"{default_vals[0]}{cfg['unit']} → {default_vals[-1]}{cfg['unit']}")
    col2.caption(f"{len(default_vals)} test levels")
    col3.metric("Statistical ε", f"{sweep_tolerance:.2f}")
    col3.caption("= 0.05 × steering range")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    if st.button("▶️ Run Parametric Sweep", type="primary"):

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
            sweep_imgs.insert(0, img)  # Put uploaded image FIRST so it becomes ref_img

        # Load MR
        from core.metamorphic_engine.mr_library import get_mr
        sweep_mr = get_mr(sweep_mr_name, tolerance=sweep_tolerance)

        if sweep_mode == "Single Model":
            # ── Single model sweep ─────────────────────────────────────────────
            # Run the sweep AND collect per-level visuals + steering outputs
            adapter = load_adapter(sweep_model)
            from evaluation.parametric_sweep import SWEEP_CATALOGUE as SC
            factory = SC[sweep_transform]["factory"]

            sweep_levels = []   # list of dicts, one per intensity level
            with st.spinner(f"Sweeping {sweep_transform} across {len(default_vals)} levels..."):
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

                # Use first test image as the representative visual
                ref_img = sweep_imgs[0]
                # Normalise once — all parametric factories expect float32 [0,1]
                ref_f = (ref_img / 255.0).astype(np.float32) if ref_img.max() > 1.0 else ref_img.astype(np.float32)
                y_orig_ref = adapter.predict(ref_f)

                for intensity in default_vals:
                    tf_fn  = factory(intensity)
                    img_t  = tf_fn(ref_f)
                    y_t    = adapter.predict(img_t)
                    diff_t = np.clip(np.abs(ref_f - img_t) * 4, 0, 1)

                    # Run MR across all test images for stats
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

                    # MR result on the reference image for per-level detail
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

            # ── 1. SUMMARY METRICS ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Sweep Overview")

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Levels tested",   len(default_vals))
            mc2.metric("Max Pass Rate",   f"{max(result.pass_rates):.1%}")
            mc3.metric("Min Pass Rate",   f"{min(result.pass_rates):.1%}")
            mc3.caption("at max intensity")
            mc4.metric(
                "⚠️ Degrades at",
                f"{ct}{cfg['unit']}" if ct is not None else "Never",
                help="Pass rate first drops below 80%"
            )
            mc5.metric(
                "💥 Collapses at",
                f"{ct2}{cfg['unit']}" if ct2 is not None else "Never",
                help="Pass rate first drops below 50%"
            )

            # ── 2. DEGRADATION CHART ───────────────────────────────────────────
            st.markdown("---")
            st.subheader("📈 Robustness Degradation Curve")
            fig_single = plot_single_sweep(result)
            st.pyplot(fig_single)
            plt.close()

            # ── 3. VISUAL IMAGE STRIP — matplotlib figure, proper size ─────────
            st.markdown("---")
            st.subheader("🖼️ Visual Transform Progression")
            st.caption(
                "Row 1 = transformed image at each intensity. "
                "Row 2 = pixel diff vs original (×4 amplified). "
                "Badge = MR pass rate."
            )

            # Pick a representative subset (max 8 levels)
            n_show = min(8, len(sweep_levels))
            step   = max(1, len(sweep_levels) // n_show)
            shown  = sweep_levels[::step][:n_show]

            n_cols = len(shown)
            fig_strip, axes_strip = plt.subplots(
                2, n_cols,
                figsize=(max(10, n_cols * 2.2), 5.5),
                facecolor="#1a1a2e",
            )
            fig_strip.suptitle(
                f"Transform Progression — {sweep_transform}  |  Model: {sweep_model}",
                color="white", fontsize=11, fontweight="bold", y=1.01,
            )
            # Ensure axes_strip is always 2D
            if n_cols == 1:
                axes_strip = np.array(axes_strip).reshape(2, 1)

            for j, lv in enumerate(shown):
                pr = lv["pass_rate"]
                status_color = (
                    "#2ecc71" if pr >= 0.80 else
                    "#f39c12" if pr >= 0.50 else
                    "#e74c3c"
                )
                status_icon = "✅" if pr >= 0.80 else ("⚠️" if pr >= 0.50 else "❌")

                # Row 1 — transformed image
                ax_t = axes_strip[0, j]
                ax_t.imshow(np.clip(lv["img_trans"], 0, 1))
                ax_t.set_title(
                    f"{lv['intensity']}{cfg['unit']}",
                    color="white", fontsize=9, pad=3,
                )
                ax_t.axis("off")

                # Row 2 — pixel diff ×4
                ax_d = axes_strip[1, j]
                ax_d.imshow(lv["img_diff"])
                ax_d.set_xlabel(
                    f"{status_icon} {pr:.0%}",
                    color=status_color, fontsize=9, labelpad=4,
                )
                ax_d.axis("off")

                # Coloured border to indicate pass/fail
                for spine in ax_t.spines.values():
                    spine.set_edgecolor(status_color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)

            # Row labels on the left
            axes_strip[0, 0].set_ylabel("Transformed", color="white", fontsize=9)
            axes_strip[1, 0].set_ylabel("Diff ×4", color="white", fontsize=9)

            plt.tight_layout(pad=0.5)
            st.pyplot(fig_strip)
            plt.close()

            # ── 4. STEERING OUTPUT PROGRESSION ────────────────────────────────
            st.markdown("---")
            st.subheader("🎯 Steering Output at Each Intensity Level")
            st.caption(
                "How the model's steering prediction changes as transform intensity increases. "
                "The original prediction is the dashed baseline."
            )

            # Steering angle bar chart across levels
            intensities_arr = [lv["intensity"] for lv in sweep_levels]
            y_trans_arr     = [lv["y_trans"]   for lv in sweep_levels]
            delta_arr       = [lv["delta"]      for lv in sweep_levels]
            pass_rate_arr   = [lv["pass_rate"]  for lv in sweep_levels]
            y_orig_val      = sweep_levels[0]["y_orig"]

            bar_colors = [
                "#2ecc71" if pr >= 0.80 else
                "#f39c12" if pr >= 0.50 else
                "#e74c3c"
                for pr in pass_rate_arr
            ]

            fig_out, axes = plt.subplots(3, 1, figsize=(12, 9), facecolor="#1a1a2e")
            fig_out.suptitle(
                f"Steering Output Progression — {sweep_model}  |  Transform: {sweep_transform}  |  MR: {sweep_mr_name}",
                color="white", fontsize=11, fontweight="bold", y=0.99,
            )

            x_pos = list(range(len(intensities_arr)))
            x_labels = [f"{v}{cfg['unit']}" for v in intensities_arr]

            # Panel A — Steering value at each level
            ax_s = axes[0]
            ax_s.set_facecolor("#16213e")
            ax_s.bar(x_pos, y_trans_arr, color=bar_colors, edgecolor="#333", width=0.65, label="θ (transformed)")
            ax_s.axhline(y_orig_val, color="white", linestyle="--", lw=1.5, label=f"θ original = {y_orig_val:.4f}")
            ax_s.axhline(0, color="#555", linestyle=":", lw=0.8)
            ax_s.set_xticks(x_pos)
            ax_s.set_xticklabels(x_labels, color="white", fontsize=8, rotation=30, ha="right")
            ax_s.set_ylabel("Steering θ", color="white")
            ax_s.set_title("Steering Output vs Intensity", color="white", fontsize=9)
            ax_s.tick_params(colors="white")
            ax_s.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
            for sp in ax_s.spines.values():
                sp.set_edgecolor("#444")

            # Panel B — Delta |θ_orig - θ_trans|
            ax_d = axes[1]
            ax_d.set_facecolor("#16213e")
            ax_d.bar(x_pos, delta_arr, color=bar_colors, edgecolor="#333", width=0.65, label="|Δθ| per level")
            ax_d.axhline(sweep_tolerance, color="#f39c12", linestyle="--", lw=1.5,
                         label=f"Tolerance ε = {sweep_tolerance}")
            ax_d.set_xticks(x_pos)
            ax_d.set_xticklabels(x_labels, color="white", fontsize=8, rotation=30, ha="right")
            ax_d.set_ylabel("|Δθ|", color="white")
            ax_d.set_title("MR Delta — |Original − Transformed Steering|", color="white", fontsize=9)
            ax_d.tick_params(colors="white")
            ax_d.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
            for sp in ax_d.spines.values():
                sp.set_edgecolor("#444")

            # Panel C — Pass rate line
            ax_p = axes[2]
            ax_p.set_facecolor("#16213e")
            ax_p.plot(x_pos, pass_rate_arr, "o-", color="#3498db", lw=2.5, markersize=8)
            ax_p.fill_between(x_pos, pass_rate_arr, alpha=0.15, color="#3498db")
            for xi, (xp, pr) in enumerate(zip(x_pos, pass_rate_arr)):
                c = "#2ecc71" if pr >= 0.80 else "#f39c12" if pr >= 0.50 else "#e74c3c"
                ax_p.scatter([xp], [pr], color=c, s=80, zorder=5)
            ax_p.axhline(0.80, color="#f39c12", linestyle="--", lw=1.2, label="80% threshold")
            ax_p.axhline(0.50, color="#e74c3c", linestyle=":",  lw=1.0, label="50% threshold")
            ax_p.set_xticks(x_pos)
            ax_p.set_xticklabels(x_labels, color="white", fontsize=8, rotation=30, ha="right")
            ax_p.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_p.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color="white")
            ax_p.set_ylabel("Pass Rate", color="white")
            ax_p.set_title("MR Pass Rate vs Intensity", color="white", fontsize=9)
            ax_p.tick_params(colors="white")
            ax_p.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
            for sp in ax_p.spines.values():
                sp.set_edgecolor("#444")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            st.pyplot(fig_out)
            plt.close()

            # ── 5. PER-LEVEL DETAIL (expander per intensity) ──────────────────
            st.markdown("---")
            st.subheader("🔬 Per-Level Detailed Results")
            st.caption("Expand any intensity level to see the image comparison, steering output, and MR verdict.")

            import pandas as pd
            for lv in sweep_levels:
                pr    = lv["pass_rate"]
                icon  = "✅" if pr >= 0.80 else ("⚠️" if pr >= 0.50 else "❌")
                label_exp = (
                    f"{icon} **{cfg['label']} = {lv['intensity']}{cfg['unit']}**"
                    f"  —  pass rate {pr:.0%}"
                    f"  |  θ_orig = {lv['y_orig']:.4f}"
                    f"  |  θ_trans = {lv['y_trans']:.4f}"
                    f"  |  |Δθ| = {lv['delta']:.4f}"
                )
                with st.expander(label_exp):
                    # Image row
                    ci1, ci2, ci3 = st.columns(3)
                    with ci1:
                        st.image(np.clip(lv["img_orig"],  0, 1), caption="Original", use_container_width=True)
                    with ci2:
                        st.image(np.clip(lv["img_trans"], 0, 1),
                                 caption=f"Transformed ({sweep_transform} = {lv['intensity']}{cfg['unit']})",
                                 use_container_width=True)
                    with ci3:
                        st.image(lv["img_diff"], caption="Pixel Diff (×4 amplified)", use_container_width=True)

                    # Steering gauge
                    gauge = make_steering_gauge(lv["y_orig"], lv["y_trans"],
                                               f"{sweep_transform}={lv['intensity']}{cfg['unit']}")
                    st.pyplot(gauge)
                    plt.close()

                    # Metrics row
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("θ original",    f"{lv['y_orig']:.4f}")
                    cm2.metric("θ transformed", f"{lv['y_trans']:.4f}",
                               delta=f"{lv['y_trans'] - lv['y_orig']:+.4f}")
                    cm3.metric("|Δθ|",          f"{lv['delta']:.4f}",
                               delta="PASS ✅" if lv["delta"] <= sweep_tolerance else "FAIL ❌",
                               delta_color="normal" if lv["delta"] <= sweep_tolerance else "inverse")
                    cm4.metric("MR Pass Rate",  f"{pr:.0%}",
                               delta=f"ε = {sweep_tolerance}")

                    # MR verdict
                    if pr >= 0.80:
                        st.success(lv["mr_message"] or f"MR satisfied at {cfg['label']} = {lv['intensity']}{cfg['unit']}")
                    elif pr >= 0.50:
                        st.warning(lv["mr_message"] or f"Partial violation at {cfg['label']} = {lv['intensity']}{cfg['unit']}")
                    else:
                        st.error(lv["mr_message"] or f"MR violated at {cfg['label']} = {lv['intensity']}{cfg['unit']}")

            # ── 6. RAW DATA TABLE ──────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🗂️ Raw Sweep Data Table")
            rows_data = []
            for pt, lv in zip(result.sweep_points, sweep_levels):
                rows_data.append({
                    f"{cfg['label']} ({cfg['unit']})": pt.intensity,
                    "θ original":   f"{lv['y_orig']:.4f}",
                    "θ transformed": f"{lv['y_trans']:.4f}",
                    "|Δθ| (ref img)": f"{lv['delta']:.4f}",
                    "Mean |Δθ| (all imgs)": f"{pt.mean_delta:.4f}",
                    "Max |Δθ|":     f"{pt.max_delta:.4f}",
                    "Pass Rate":    f"{pt.pass_rate:.1%}",
                    "Violations":   f"{pt.violations}/{pt.total}",
                    "Status":       "✅ Robust"   if pt.pass_rate >= 0.80
                                    else ("⚠️ Degraded" if pt.pass_rate >= 0.50
                                    else "❌ Collapsed"),
                })
            st.dataframe(pd.DataFrame(rows_data), use_container_width=True)

            # ── 7. RESEARCH INTERPRETATION ─────────────────────────────────────
            st.markdown("---")
            st.subheader("🔬 Research Interpretation")

            if ct is None:
                st.success(
                    f"✅ **{sweep_model}** is highly robust to `{sweep_transform}` — "
                    f"MR `{sweep_mr_name}` never drops below 80% even at maximum intensity "
                    f"({default_vals[-1]}{cfg['unit']})."
                )
            elif ct2 is None:
                st.warning(
                    f"⚠️ **{sweep_model}** starts degrading at `{sweep_transform}` = **{ct}{cfg['unit']}**, "
                    f"but never completely collapses (pass rate stays above 50%). "
                    f"Acceptable degradation — model retains partial robustness."
                )
            else:
                st.error(
                    f"❌ **{sweep_model}** degrades at `{sweep_transform}` = **{ct}{cfg['unit']}** "
                    f"and completely collapses at **{ct2}{cfg['unit']}**. "
                    f"This is a safety-critical failure boundary."
                )

        else:
            # ── Cross-model sweep ──────────────────────────────────────────────
            with st.spinner(f"Sweeping all 5 models across {len(default_vals)} intensity levels..."):
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

            # ── Summary table ──────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Cross-Model Sweep Results")

            import pandas as pd
            MODEL_DISPLAY_SWEEP = {
                "dave2_cnn":       "DAVE-2",
                "gpr_steering":    "GPR",
                "svr_steering":    "SVR",
                "random_forest":   "RF",
                "linear_baseline": "Linear",
            }

            summary_rows = []
            for model_name, res in sweep_results.items():
                ct  = res.critical_threshold
                ct2 = res.collapse_threshold
                avg_pr = np.mean(res.pass_rates)
                summary_rows.append({
                    "Model":              MODEL_DISPLAY_SWEEP.get(model_name, model_name),
                    "Avg Pass Rate":      f"{avg_pr:.1%}",
                    "Min Pass Rate":      f"{min(res.pass_rates):.1%}",
                    "Degrades at":        f"{ct}{cfg['unit']}"  if ct  is not None else "Never",
                    "Collapses at":       f"{ct2}{cfg['unit']}" if ct2 is not None else "Never",
                    "Robustness":         "✅ High" if avg_pr >= 0.8
                                          else ("⚠️ Medium" if avg_pr >= 0.5
                                          else "❌ Low"),
                })
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True)

            # ── Cross-model degradation chart ──────────────────────────────────
            fig_cross = plot_cross_model_sweep(sweep_results)
            st.pyplot(fig_cross)
            plt.close()

            # ── Critical threshold comparison ──────────────────────────────────
            st.markdown("---")
            st.subheader("🏆 Critical Threshold Comparison")
            st.markdown("*Which model tolerates the highest intensity before failing?*")

            fig_thresh = plot_critical_thresholds(sweep_results)
            st.pyplot(fig_thresh)
            plt.close()

            # ── Per-intensity generalization ───────────────────────────────────
            st.markdown("---")
            st.subheader("📉 Generalization Score vs Intensity")
            st.markdown(
                "*At each intensity level: what fraction of models still pass the MR?*  \n"
                "This shows whether an MR remains generalizable as conditions worsen."
            )

            intensities = list(sweep_results.values())[0].intensities
            gen_at_intensity = []
            for i, intensity in enumerate(intensities):
                passes = sum(
                    1 for res in sweep_results.values()
                    if res.sweep_points[i].pass_rate >= 0.5
                )
                gen_at_intensity.append(passes / len(sweep_results))

            fig_gen, ax_gen = plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
            ax_gen.set_facecolor("#16213e")
            ax_gen.plot(intensities, gen_at_intensity, "D-", color="#9b59b6", lw=2.5, markersize=7)
            ax_gen.fill_between(intensities, gen_at_intensity, alpha=0.2, color="#9b59b6")
            ax_gen.axhline(0.8, color="#f39c12", linestyle="--", lw=1.2, label="80% generalization")
            ax_gen.set_xlabel(f"{cfg['label']}{' (' + cfg['unit'] + ')' if cfg['unit'] else ''}", color="white")
            ax_gen.set_ylabel("Generalization Score", color="white")
            ax_gen.set_ylim(-0.05, 1.10)
            ax_gen.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_gen.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color="white")
            ax_gen.tick_params(colors="white")
            ax_gen.set_title(
                f"MR Generalization Score vs {cfg['label']} — MR: {sweep_mr_name}",
                color="white", fontweight="bold"
            )
            ax_gen.legend(facecolor="#16213e", labelcolor="white")
            for sp in ax_gen.spines.values():
                sp.set_edgecolor("#444")
            plt.tight_layout()
            st.pyplot(fig_gen)
            plt.close()

            # ── Raw data per model ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🗂️ Detailed Sweep Data per Model")
            tabs = st.tabs([MODEL_DISPLAY_SWEEP.get(m, m) for m in sweep_results])
            for tab, (model_name, res) in zip(tabs, sweep_results.items()):
                with tab:
                    tab_rows = []
                    for pt in res.sweep_points:
                        tab_rows.append({
                            f"{cfg['label']} ({cfg['unit']})": pt.intensity,
                            "Pass Rate":   f"{pt.pass_rate:.1%}",
                            "Mean |Δθ|":  f"{pt.mean_delta:.4f}",
                            "Max |Δθ|":   f"{pt.max_delta:.4f}",
                            "Violations":  f"{pt.violations}/{pt.total}",
                        })
                    st.dataframe(pd.DataFrame(tab_rows), use_container_width=True)

            # ── Research interpretation ────────────────────────────────────────
            st.markdown("---")
            st.subheader("🔬 Research Findings")

            best_model   = max(sweep_results, key=lambda m: np.mean(sweep_results[m].pass_rates))
            worst_model  = min(sweep_results, key=lambda m: np.mean(sweep_results[m].pass_rates))
            final_gen    = gen_at_intensity[-1] if gen_at_intensity else 0.0
            initial_gen  = gen_at_intensity[0]  if gen_at_intensity else 0.0

            st.info(
                f"**MR `{sweep_mr_name}` under `{sweep_transform}` sweep:**  \n"
                f"- Most robust model: **{MODEL_DISPLAY_SWEEP.get(best_model, best_model)}** "
                f"(avg pass rate: {np.mean(sweep_results[best_model].pass_rates):.1%})  \n"
                f"- Most fragile model: **{MODEL_DISPLAY_SWEEP.get(worst_model, worst_model)}** "
                f"(avg pass rate: {np.mean(sweep_results[worst_model].pass_rates):.1%})  \n"
                f"- Generalization drops from **{initial_gen:.0%}** (no transform) to "
                f"**{final_gen:.0%}** (max intensity {default_vals[-1]}{cfg['unit']})  \n"
                f"- This MR {'remains generalizable' if final_gen >= 0.6 else 'becomes model-specific'} "
                f"at extreme intensities."
            )


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 4: UPLOAD & TEST YOUR MODEL
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🆕 Upload & Test Your Model":

    st.subheader("🆕 Upload & Test Your Own Model")
    st.caption(
        "Bring any regression model (.pkl, .h5, .pt, .onnx) — "
        "the framework auto-detects the framework and runs the full 14-MR test pipeline."
    )

    from evaluation.generalization_scorer import statistical_threshold as _stat_thresh
    from dashboard.upload_model_page import render as _render_upload_page

    _render_upload_page(tolerance=_stat_thresh())