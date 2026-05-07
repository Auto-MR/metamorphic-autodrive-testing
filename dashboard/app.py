"""
Dashboard metamorphic testing.
Run with: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from PIL import Image

from core.adapters.cnn_adapter          import CNNAdapter
from core.adapters.steering_adapter     import SteeringAdapter
from core.metamorphic_engine.transformer import get_transform
from metamorphic_relations.steering_mrs import FlipSymmetrySteeringMR, TemporalSmoothnessMR, BrightnessSteeringMR
#from metamorphic_relations.steering_mrs  import FlipSymmetrySteeringMR, SmoothnessSteeringMR, BrightnessSteeringMR
from metamorphic_relations.cnn_mrs       import BrightnessInvarianceMR, FlipSymmetryCNNMR, NoiseRobustnessMR
from core.adapters.gpr_steering_adapter import GPRSteeringAdapter
from core.pipeline.run_single_test       import run_test
from evaluation.visualization import make_comparison_figure, make_steering_gauge
from metamorphic_relations.steering_mrs import (
    FlipSymmetrySteeringMR,
    SmallLaneShiftInvarianceMR,
    BrightnessSteeringMR,
    WeatherRobustnessMR,
    TemporalSmoothnessMR,
    ConstantCurvatureConsistencyMR,
    OcclusionRobustnessMR,
    LaneCenterShiftConsistencyMR,
    PerspectiveScalingMR,
    StraightRoadStabilityMR,
) 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Metamorphic Autodrive Tester", layout="wide")

st.title("Metamorphic Autodrive Testing Framework")
st.caption("Validate autonomous driving models using metamorphic relations.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
model_choice     = st.sidebar.selectbox("Model", ["steering_regression", "cnn_depth", "gaussian_process_regression"])
transform_choice = st.sidebar.selectbox("Transform", ["flip", "brightness", "noise", "rotate_5", "translate","rain", "snow", "fog"])
tolerance        = st.sidebar.slider("MR Tolerance", 0.01, 0.30, 0.10, 0.01)

# ── Image upload ──────────────────────────────────────────────────────────────
st.subheader("Input Image")
uploaded = st.file_uploader("Upload a driving image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB")).astype(np.float32) / 255.0
else:
    st.info("No image uploaded. using a synthetic gradient image for demo.")
    # Synthetic image with left-right gradient (more interesting than random noise)
    x = np.linspace(0.1, 0.9, 64)
    img = np.stack([
        np.tile(x,        (64, 1)),
        np.tile(x[::-1],  (64, 1)),
        np.ones((64, 64)) * 0.4,
    ], axis=-1).astype(np.float32)

# ── Show input image ──────────────────────────────────────────────────────────
col_img, col_info = st.columns([1, 2])
with col_img:
    st.image(img, caption="Input image", use_container_width=True)
with col_info:
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Shape | `{img.shape}` |
    | dtype | `{img.dtype}` |
    | min / max | `{img.min():.3f}` / `{img.max():.3f}` |
    | Model | `{model_choice}` |
    | Transform | `{transform_choice}` |
    | Tolerance | `{tolerance}` |
    """)

# ── Run button ────────────────────────────────────────────────────────────────
if st.button(" Run Metamorphic Tests", type="primary"):

    with st.spinner("Running tests..."):

        # ── Apply transform ───────────────────────────────────────────────────
        transform   = get_transform(transform_choice, domain="image")
        img_trans   = transform(img)

        # ── Load model & MRs ──────────────────────────────────────────────────
        if model_choice == "steering_regression":
            adapter = SteeringAdapter(); adapter.load()
            mrs = [FlipSymmetrySteeringMR(tolerance),
                SmallLaneShiftInvarianceMR(tolerance),
                BrightnessSteeringMR(tolerance),
                WeatherRobustnessMR(tolerance),
                TemporalSmoothnessMR(tolerance),
                ConstantCurvatureConsistencyMR(tolerance),
                OcclusionRobustnessMR(tolerance),
                LaneCenterShiftConsistencyMR(tolerance),
                PerspectiveScalingMR(tolerance),
                StraightRoadStabilityMR(tolerance),]
            
        elif model_choice == "gaussian_process_regression":
            adapter = GPRSteeringAdapter()
            adapter.load(r"C:\Users\user\Downloads\Fyp_metamopic\gpr_steering_model.pkl")

            mrs = [
                FlipSymmetrySteeringMR(tolerance),
                BrightnessSteeringMR(tolerance),
                WeatherRobustnessMR(tolerance),
                TemporalSmoothnessMR(tolerance),
                OcclusionRobustnessMR(tolerance),
                PerspectiveScalingMR(tolerance),
                StraightRoadStabilityMR(tolerance),
            ]
        else:
            adapter = CNNAdapter()
            adapter.load()
            mrs = [
                BrightnessInvarianceMR(tolerance),
                FlipSymmetryCNNMR(tolerance),
                NoiseRobustnessMR(tolerance),
            ]
        

        # ── Run predictions ───────────────────────────────────────────────────
        y_orig  = adapter.predict(img)
        y_trans = adapter.predict(img_trans)

        # ── Check MRs ─────────────────────────────────────────────────────────
        results = [mr.check(img, img_trans, y_orig, y_trans) for mr in mrs]

    # ── SECTION 1: Image comparison ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Visual Comparison")

    col_o, col_t, col_d = st.columns(3)

    with col_o:
        st.image(np.clip(img, 0, 1), caption="Original", use_container_width=True)

    with col_t:
        st.image(np.clip(img_trans, 0, 1), caption=f"Transformed ({transform_choice})", use_container_width=True)

    with col_d:
        # Pixel-level diff (amplified ×4 so subtle changes are visible)
        diff_img = np.clip(np.abs(img - img_trans) * 4, 0, 1)
        st.image(diff_img, caption="Pixel Diff (×4 amplified)", use_container_width=True)

    # ── SECTION 2: Model output comparison ────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Output Comparison")

    if model_choice == "cnn_depth":
        # y_orig and y_trans are 2D depth maps
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#1a1a2e")
        fig.suptitle("Depth Map Comparison", color="white", fontsize=13, fontweight="bold")

        titles  = ["Depth (original)", f"Depth ({transform_choice})", "|Diff| Heatmap"]
        maps    = [y_orig, y_trans, np.abs(y_orig - y_trans)]
        cmaps   = ["plasma", "plasma", "hot"]
        vmaxes  = [1.0, 1.0, 0.30]

        for ax, data, title, cmap, vmax in zip(axes, maps, titles, cmaps, vmaxes):
            ax.set_facecolor("#16213e")
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        mean_diff = float(np.abs(y_orig - y_trans).mean())
        fig.text(0.5, 0.01,
                 f"Mean absolute depth difference: {mean_diff:.4f}",
                 ha="center", color="#f39c12" if mean_diff > 0.05 else "#2ecc71",
                 fontsize=11, fontweight="bold")

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        st.pyplot(fig)
        plt.close()

    else:
        # Steering: show gauges
        gauge_fig = make_steering_gauge(float(y_orig), float(y_trans), transform_choice)
        st.pyplot(gauge_fig)
        import matplotlib.pyplot as plt
        plt.close()

        # Also numeric comparison
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Steering (original)",    f"{float(y_orig):.4f}")
        col_b.metric("Steering (transformed)", f"{float(y_trans):.4f}",
                     delta=f"{float(y_trans) - float(y_orig):.4f}")
        col_c.metric("|y + y_flip| (antisymmetry)",
                     f"{abs(float(y_orig) + float(y_trans)):.4f}",
                     delta="✅ OK" if abs(float(y_orig) + float(y_trans)) < tolerance else "❌ Violated")

    # ── SECTION 3: Full comparison panel (saved figure) ───────────────────────
    st.markdown("---")
    st.subheader("Full Metamorphic Panel")

    panel_fig = make_comparison_figure(
        original        = img,
        transformed     = img_trans,
        transform_name  = transform_choice,
        output_orig     = y_orig  if model_choice == "cnn_depth" else None,
        output_trans    = y_trans if model_choice == "cnn_depth" else None,
        model_name      = model_choice,
        mr_results      = results,
    )
    st.pyplot(panel_fig)
    import matplotlib.pyplot as plt
    plt.close()

    # ── SECTION 4: MR results table ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Metamorphic Relation Results")

    for r in results:
        icon = "✅" if r.passed else "❌"
        with st.expander(
            f"{icon}  **{r.mr_name}**  —  delta = {r.delta:.4f}" if r.delta is not None
            else f"{icon}  **{r.mr_name}**"
        ):
            c1, c2 = st.columns(2)
            c1.write(f"**Expected:** {r.expected}")
            c2.write(f"**Actual:**   {r.actual}")
            if r.message:
                st.info(r.message)

    # ── SECTION 5: Score summary ──────────────────────────────────────────────
    st.markdown("---")
    passed = sum(r.passed for r in results)
    score  = passed / len(results)

    col_score, col_bar = st.columns([1, 2])
    col_score.metric("Robustness Score", f"{score:.1%}",
                     f"{passed}/{len(results)} MRs passed")

    # Progress bar colour-coded
    col_bar.markdown(
        f"<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True
    )
    col_bar.progress(score)

    if score == 1.0:
        st.success("All metamorphic relations satisfied. Model appears robust to this transform.")
    elif score >= 0.6:
        st.warning("Some MRs violated. Model shows partial sensitivity to this transform.")
    else:
        st.error("Multiple MR violations Model is not robust to this transform.")
