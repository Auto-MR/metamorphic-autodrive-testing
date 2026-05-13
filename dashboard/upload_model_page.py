"""
Upload & Test Your Model — Streamlit page module.

Import and call  render()  from app.py.

Workflow:
  1. Upload model file  (.pkl / .h5 / .pt / .onnx / .joblib / .keras)
  2. Configure display name, input shape, output scale
  3. Load & validate — instant smoke-test prediction displayed
  4. Choose MR categories and transforms to test
  5. Run full MR battery — same pipeline as built-in models
  6. (Optional) Upload a second model and compare side-by-side
  7. (Optional) Run parametric sweep on the uploaded model
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

from core.adapters.user_model_adapter import UserModelAdapter, SUPPORTED_EXTENSIONS
from core.adapters.user_model_registry import get_registry
from core.metamorphic_engine.transformer import get_transform
from core.metamorphic_engine.mr_library import get_steering_mrs, get_mrs_by_category, get_mr
from evaluation.generalization_scorer import (
    compute_generalization_score, compute_cross_model_table,
    rank_models_by_robustness, identify_model_weaknesses,
    compute_category_generalization, statistical_threshold, MR_CATEGORIES,
)
from evaluation.visualization import make_steering_gauge
from evaluation.parametric_sweep import SWEEP_CATALOGUE, SweepResult, SweepPoint
from evaluation.sweep_visualization import plot_single_sweep
from core.metrics.normalizer import normalize_steering_output


# ── constants ─────────────────────────────────────────────────────────────────

CATEGORY_LABELS = {
    "A": "Invariance  |f(x) − f(T(x))| < ε",
    "B": "Symmetry    f(flip(x)) ≈ −f(x)",
    "C": "Temporal    |f(x_t) − f(x_{t+1})| < δ",
    "D": "Composition  multiple transforms",
}

MR_TRANSFORM_MAP = {
    "brightness_steering":            "brightness",
    "contrast_steering":              "contrast",
    "noise_robustness_steering":      "noise",
    "weather_robustness":             "fog",
    "occlusion_robustness":           "noise",
    "flip_symmetry_steering":         "flip",
    "straight_road_stability":        "brightness",
    "perspective_scaling":            "scale",
    "temporal_smoothness":            "translate",
    "constant_curvature_consistency": "translate",
    "small_lane_shift_invariance":    "translate",
    "composed_brightness_noise":      "composed_bright_noise",
    "composed_weather_blur":          "composed_weather_blur",
    "lane_center_shift_consistency":  "composed_shift_bright",
}

FW_ICONS = {
    "sklearn/pickle": "sklearn",
    "sklearn/joblib": "sklearn",
    "keras":          "keras",
    "pytorch":        "pytorch",
    "onnx":           "onnx",
    "unknown":        "?",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _synthetic_img() -> np.ndarray:
    x = np.linspace(0.1, 0.9, 64)
    return np.stack([
        np.tile(x, (64, 1)),
        np.tile(x[::-1], (64, 1)),
        np.ones((64, 64)) * 0.4,
    ], axis=-1).astype(np.float32)


def _normalise(img: np.ndarray) -> np.ndarray:
    return (img / 255.0).astype(np.float32) if img.max() > 1.0 else img.astype(np.float32)


def _status_badge(passed: bool) -> str:
    return "✓" if passed else "✗"


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#479baa"
    if score >= 0.5:
        return "#290ad8"
    return "#c0392b"


def _run_mrs(adapter, img, mr_categories, tolerance) -> list:
    mrs = []
    for cat in mr_categories:
        mrs.extend(get_mrs_by_category(cat, tolerance))
    results = []
    for mr in mrs:
        tf_name = MR_TRANSFORM_MAP.get(mr.name, "brightness")
        try:
            tf_fn = get_transform(tf_name, domain="image")
        except KeyError:
            tf_fn = get_transform("brightness", domain="image")
        img_f = _normalise(img)
        img2  = tf_fn(img_f)
        y1    = adapter.predict(img_f)
        y2    = adapter.predict(img2)
        results.append(mr.check(img_f, img2, y1, y2))
    return results


# ── academic styling ─────────────────────────────────────────────────────────

def _apply_academic_styling():
    """Apply clean academic styling to the page."""
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
        border-bottom: 2px solid #479baa;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 0.3rem;
    }
    
    /* Section containers */
    .academic-section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Metric cards */
    .academic-metric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .academic-metric-label {
        font-size: 0.75rem;
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
    
    /* Status indicators */
    .status-pass {
        color: #479baa;
        font-weight: 600;
    }
    
    .status-fail {
        color: #c0392b;
        font-weight: 600;
    }
    
    .status-warning {
        color: #290ad8;
        font-weight: 600;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        font-weight: 500;
        background-color: #f8fafc;
        border-radius: 6px;
    }
    
    /* Dataframes */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #479baa;
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
        background-color: #479baa;
        color: white;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 6px;
    }
    
    /* Metrics grid */
    div[data-testid="column"] {
        background: transparent;
    }
    </style>
    """, unsafe_allow_html=True)


def _section_header(title: str, description: str = None):
    """Render a section header with academic styling."""
    st.markdown(f"### {title}")
    if description:
        st.markdown(f"<p style='color:#475569; font-size:0.9rem; margin-bottom:1rem;'>{description}</p>", 
                   unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Upload & Register
# ═══════════════════════════════════════════════════════════════════════════════

def _section_upload() -> None:
    _section_header(
        "Model Upload",
        "Upload a trained regression model for evaluation against metamorphic relations."
    )
    
    st.markdown("""
    <div style='background:#f8fafc; padding:1rem; border-radius:8px; margin-bottom:1rem; font-size:0.85rem;'>
    <b>Supported formats:</b> pickle (.pkl, .joblib) | Keras/TensorFlow (.h5, .keras) | 
    PyTorch (.pt, .pth) | ONNX (.onnx) | TF Checkpoint bundle (.zip)<br>
    <span style='color:#475569;'>
    ⓘ For TensorFlow <code>.ckpt</code> checkpoints, zip all companion files together:<br>
    &nbsp;&nbsp;&nbsp;<code>zip model_ckpt.zip model.ckpt.meta model.ckpt.index model.ckpt.data-00000-of-00001</code><br>
    &nbsp;&nbsp;&nbsp;Or zip a SavedModel directory: <code>zip -r saved_model.zip saved_model/</code>
    </span>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_cfg = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Model file",
            type=["pkl", "joblib", "h5", "keras", "pt", "pth", "onnx", "zip"],
            help="Maximum file size: 500 MB. For PyTorch, use torch.save(model, path) not state_dict. For TF checkpoints, zip the .meta/.index/.data-* files together.",
        )

    with col_cfg:
        display_name = st.text_input(
            "Display name",
            value=uploaded_file.name.rsplit(".", 1)[0] if uploaded_file else "uploaded_model",
            help="Identifier used throughout the dashboard.",
        )
        output_scale = st.selectbox(
            "Output normalization",
            ["auto", "normalized (−1 to 1)", "degrees (±25°)", "radians"],
            help="'auto' attempts automatic detection based on model output range.",
        )
        col_h, col_w = st.columns(2)
        input_h = col_h.number_input("Input height (px)", value=66, min_value=32, max_value=512)
        input_w = col_w.number_input("Input width (px)", value=200, min_value=32, max_value=1024)

    if uploaded_file is None:
        st.info("Upload a model file to begin evaluation.")
        return

    registry = get_registry()

    if st.button("Load and Register Model", type="primary"):
        with st.spinner(f"Loading {uploaded_file.name} ..."):
            scale_map = {
                "auto": "auto",
                "normalized (−1 to 1)": "normalized",
                "degrees (±25°)": "degrees",
                "radians": "radians",
            }
            try:
                adapter = UserModelAdapter.from_file_bytes(
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.read(),
                    display_name=display_name,
                    input_h=int(input_h),
                    input_w=int(input_w),
                    output_scale=scale_map[output_scale],
                )
                adapter.load()

                if not adapter.is_loaded:
                    st.error(f"Failed to load model: {adapter.load_error}")
                    return

                # Smoke-test with synthetic image
                smoke_img = _synthetic_img()
                smoke_pred = adapter.predict(smoke_img)

                registry.add(adapter)
                
                st.success(
                    f"**{display_name}** registered successfully. "
                    f"Framework: {adapter.framework}. "
                    f"Smoke-test output: {smoke_pred:.4f}"
                )

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Registered Models Panel
# ═══════════════════════════════════════════════════════════════════════════════

def _section_registry() -> None:
    registry = get_registry()
    if len(registry) == 0:
        return

    st.markdown("---")
    _section_header(f"Registered Models ({len(registry)})", 
                   "Models currently loaded in the evaluation environment.")

    for info in registry.list_models():
        framework_label = info["framework"]
        status = "Ready" if info["loaded"] else f"Error: {info['error']}"
        status_color = "#1c9ba0" if info["loaded"] else "#c0392b"
        
        with st.expander(f"[{framework_label}] {info['name']} — {status}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Framework</div><div class='academic-metric-value'>{info['framework']}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Input</div><div class='academic-metric-value'>{info['input_h']}×{info['input_w']}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Status</div><div class='academic-metric-value' style='color:{status_color};'>{'✓ Loaded' if info['loaded'] else '✗ Failed'}</div></div>", unsafe_allow_html=True)
            
            if not info["loaded"]:
                c4.error(info["error"])
            
            if st.button(f"Remove {info['name']}", key=f"rm_{info['name']}"):
                registry.remove(info["name"])
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Single Model MR Test
# ═══════════════════════════════════════════════════════════════════════════════

def _section_single_test(tolerance: float) -> None:
    registry = get_registry()
    if len(registry) == 0:
        st.info("Register at least one model to run metamorphic tests.")
        return

    st.markdown("---")
    _section_header(
        "Metamorphic Testing",
        "Evaluate model robustness by checking invariance under semantic-preserving transformations."
    )

    col_sel, col_img = st.columns([1, 1])

    with col_sel:
        model_name = st.selectbox("Target model", registry.names(), key="single_sel")
        mr_cats    = st.multiselect(
            "MR categories",
            ["A", "B", "C", "D"],
            default=["A", "B", "C", "D"],
            key="single_cats",
            help="A: Invariance | B: Symmetry | C: Temporal | D: Composition"
        )
        transform_choice = st.selectbox(
            "Preview transformation",
            ["brightness", "contrast", "noise", "fog", "rain", "snow",
             "flip", "rotate_5", "scale", "crop", "translate",
             "composed_bright_noise", "composed_weather_blur", "composed_shift_bright"],
            key="single_tf",
            help="Apply transformation to test image for visual verification."
        )

    with col_img:
        up_img = st.file_uploader(
            "Test image (optional)",
            type=["jpg", "jpeg", "png"],
            key="single_img",
            help="Upload custom test image. If not provided, synthetic test image will be used."
        )
        if up_img:
            img = np.array(Image.open(up_img).convert("RGB")).astype(np.float32) / 255.0
            st.session_state['uploaded_test_img'] = img
            st.caption("Custom image loaded and stored for subsequent tests.")
        else:
            img = _synthetic_img()
            st.caption("Using synthetic test pattern. Upload custom image for domain-specific evaluation.")
        st.image(np.clip(img, 0, 1), caption="Test input", use_container_width=True)

    if st.button("Run Metamorphic Tests", type="primary", key="run_single"):
        adapter = registry.get(model_name)

        with st.spinner("Evaluating metamorphic relations ..."):
            img_f = _normalise(img)
            try:
                tf_fn   = get_transform(transform_choice, domain="image")
                img_t   = tf_fn(img_f)
            except KeyError:
                img_t   = img_f
            y_orig  = adapter.predict(img_f)
            y_trans = adapter.predict(img_t)

            results = _run_mrs(adapter, img_f, mr_cats, tolerance)

        # ── Visual comparison ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Visual Transformation")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<p style='text-align:center; font-size:0.8rem; color:#475569;'>Original</p>", unsafe_allow_html=True)
            st.image(np.clip(img_f, 0, 1), use_container_width=True)
        with c2:
            st.markdown(f"<p style='text-align:center; font-size:0.8rem; color:#475569;'>Transformed: {transform_choice}</p>", unsafe_allow_html=True)
            st.image(np.clip(img_t, 0, 1), use_container_width=True)
        with c3:
            st.markdown("<p style='text-align:center; font-size:0.8rem; color:#475569;'>Difference (enhanced)</p>", unsafe_allow_html=True)
            diff = np.clip(np.abs(img_f - img_t) * 4, 0, 1)
            st.image(diff, use_container_width=True)

        # ── Steering gauge ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Model Output")
        gauge_fig = make_steering_gauge(float(y_orig), float(y_trans), transform_choice)
        st.pyplot(gauge_fig)
        plt.close()

        # Calculate delta and determine color
        delta = y_trans - y_orig
        delta_color = "#479baa" if delta > 0 else "#c0392b"
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>θ original</div><div class='academic-metric-value'>{y_orig:.4f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>θ transformed</div><div class='academic-metric-value'>{y_trans:.4f}</div><div class='academic-metric-label' style='color:{delta_color};'>{delta:+.4f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Normalized original</div><div class='academic-metric-value'>{normalize_steering_output(float(y_orig)):.3f}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Normalized transformed</div><div class='academic-metric-value'>{normalize_steering_output(float(y_trans)):.3f}</div></div>", unsafe_allow_html=True)

        # ── MR results ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Metamorphic Relation Results")

        for cat in ["A", "B", "C", "D"]:
            if cat not in mr_cats:
                continue
            cat_res = [r for r in results if MR_CATEGORIES.get(r.mr_name) == cat]
            if not cat_res:
                continue
            n_pass = sum(r.passed for r in cat_res)
            pass_color = "#479baa" if n_pass == len(cat_res) else ("#290ad8" if n_pass > 0 else "#c0392b")
            
            st.markdown(
                f"<div style='margin-top:0.75rem; margin-bottom:0.5rem;'>"
                f"<span style='font-weight:600;'>Category {cat}</span> — {CATEGORY_LABELS[cat]} "
                f"<span style='color:{pass_color};'>({n_pass}/{len(cat_res)} passed)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            for r in cat_res:
                icon = _status_badge(r.passed)
                lbl  = f"{icon} {r.mr_name}"
                if r.delta is not None:
                    lbl += f"  —  δ = {r.delta:.4f}"
                with st.expander(lbl):
                    ca, cb = st.columns(2)
                    ca.markdown(f"**Expected:** {r.expected}")
                    cb.markdown(f"**Actual:** {r.actual}")
                    if r.passed:
                        st.success(r.message)
                    else:
                        st.error(r.message)

        # ── Score ─────────────────────────────────────────────────────────────
        st.markdown("---")
        n_pass = sum(r.passed for r in results)
        score  = n_pass / len(results) if results else 0
        
        score_color = _score_color(score)
        score_level = "High" if score >= 0.8 else ("Moderate" if score >= 0.5 else "Low")
        
        st.markdown("#### Robustness Score")
        col_a, col_b = st.columns([1, 3])
        col_a.markdown(f"<div class='academic-metric' style='background: {score_color}10; border-color: {score_color};'><div class='academic-metric-label'>Score</div><div class='academic-metric-value' style='color:{score_color};'>{score:.1%}</div><div class='academic-metric-label'>{score_level} ({n_pass}/{len(results)} MRs)</div></div>", unsafe_allow_html=True)
        col_b.progress(score)

        if score == 1.0:
            st.success("All metamorphic relations satisfied. Model demonstrates strong robustness.")
        elif score >= 0.75:
            st.warning("Minor violations detected. Consider investigating specific MR failures.")
        elif score >= 0.5:
            st.warning("Multiple violations detected. Model shows inconsistent behavior under transformations.")
        else:
            st.error("Major violations detected. Model is not robust to semantic-preserving transformations.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Cross-Model Comparison (user models vs built-in)
# ═══════════════════════════════════════════════════════════════════════════════

def _section_cross_compare(tolerance: float) -> None:
    registry = get_registry()
    if len(registry) < 2:
        st.info("Register at least 2 models to enable cross-model comparison.")
        return

    st.markdown("---")
    _section_header(
        "Cross-Model Comparison",
        "Compare all registered models across the complete metamorphic test suite."
    )

    n_imgs = st.slider("Number of test images", 3, 15, 5, key="cc_imgs", 
                       help="Multiple test images improve statistical reliability.")

    if st.button("Run Comparison", type="primary", key="run_cross"):
        adapters = registry.all_adapters()

        # Generate test images
        test_imgs = []
        
        if 'uploaded_test_img' in st.session_state:
            test_imgs.append(st.session_state['uploaded_test_img'])
        
        for i in range(n_imgs):
            x = np.linspace(0.05 + i * 0.02, 0.95, 64)
            base = np.stack([
                np.tile(x, (64, 1)),
                np.tile(x[::-1], (64, 1)),
                np.ones((64, 64)) * (0.3 + i * 0.04),
            ], axis=-1).clip(0, 1).astype(np.float32)
            test_imgs.append(base)

        results_map: dict = {}
        pb = st.progress(0)

        with st.spinner("Evaluating all models across all metamorphic relations ..."):
            all_mrs = []
            for cat in "ABCD":
                all_mrs.extend(get_mrs_by_category(cat, tolerance))

            for idx, (name, adapter) in enumerate(adapters.items()):
                model_results = []
                for mr in all_mrs:
                    tf_name = MR_TRANSFORM_MAP.get(mr.name, "brightness")
                    try:
                        tf_fn = get_transform(tf_name, domain="image")
                    except KeyError:
                        tf_fn = get_transform("brightness", domain="image")
                    for si in test_imgs:
                        si_f  = _normalise(si)
                        si2   = tf_fn(si_f)
                        y1    = adapter.predict(si_f)
                        y2    = adapter.predict(si2)
                        model_results.append(mr.check(si_f, si2, y1, y2))
                results_map[name] = model_results
                pb.progress((idx + 1) / len(adapters))
                pb.caption(f"Evaluated: {name}")

        gen_scores = compute_generalization_score(results_map)
        table      = compute_cross_model_table(results_map)
        ranking    = rank_models_by_robustness(results_map)
        weaknesses = identify_model_weaknesses(results_map)
        cat_gen    = compute_category_generalization(results_map, MR_CATEGORIES)

        # ── Generalization table ──────────────────────────────────────────────
        import pandas as pd
        st.markdown("---")
        st.markdown("#### MR × Model Generalization Matrix")

        model_names = list(adapters.keys())
        rows = []
        for mr_name in sorted(gen_scores):
            cat = MR_CATEGORIES.get(mr_name, "?")
            row = {"Metamorphic Relation": mr_name, "Category": f"Cat {cat}"}
            for m in model_names:
                row[m] = table.get(mr_name, {}).get(m, "—")
            row["Generalization Score"] = f"{gen_scores[mr_name]:.0%}"
            rows.append(row)

        def _style_dataframe(val):
            if val == "✓":
                return "background-color: #479baa20; color: #479baa; font-weight: 500;"
            if val == "✗":
                return "background-color: #c0392b20; color: #c0392b; font-weight: 500;"
            if "%" in str(val):
                p = float(str(val).strip("%")) / 100
                if p >= 0.8:
                    return "color: #479baa; font-weight: 600;"
                elif p >= 0.6:
                    return "color: #290ad8; font-weight: 600;"
                else:
                    return "color: #c0392b; font-weight: 600;"
            return ""

        st.dataframe(
            pd.DataFrame(rows).style.applymap(_style_dataframe),
            use_container_width=True,
        )

        # ── Category bar chart ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Performance by Category")

        c1, c2, c3, c4 = st.columns(4)
        cat_labels = ["Invariance", "Symmetry", "Temporal", "Composition"]
        for col, cat, label in zip([c1, c2, c3, c4], "ABCD", cat_labels):
            col.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Category {cat}</div><div class='academic-metric-value'>{cat_gen.get(cat, 0):.1%}</div><div class='academic-metric-label'>{label}</div></div>", unsafe_allow_html=True)

        fig_cat, ax = plt.subplots(figsize=(8, 3.5), facecolor="white")
        cats  = list(cat_gen)
        vals  = [cat_gen[c] for c in cats]
        cols  = [_score_color(v) for v in vals]
        bars  = ax.bar([f"Category {c}" for c in cats], vals, color=cols, edgecolor="#cbd5e1", linewidth=0.5)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color="#64748b", linestyle="--", lw=0.8, alpha=0.7, label="High robustness threshold")
        ax.set_ylabel("Generalization Score", fontsize=10)
        ax.set_facecolor("#f8fafc")
        ax.tick_params(colors="#475569", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{v:.0%}", ha="center", color="#1e293b", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cat)
        plt.close()

        # ── Ranking ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Model Robustness Ranking")
        fig_rank, ax2 = plt.subplots(figsize=(8, max(3, len(ranking) * 0.6)), facecolor="white")
        rnames = [m for m, _ in ranking][::-1]
        rvals  = [s for _, s in ranking][::-1]
        rcols  = [_score_color(v) for v in rvals]
        rbars  = ax2.barh(rnames, rvals, color=rcols, edgecolor="#cbd5e1", linewidth=0.5)
        ax2.set_xlim(0, 1.05)
        ax2.axvline(0.8, color="#64748b", linestyle="--", lw=0.8, alpha=0.7, label="High robustness")
        ax2.axvline(0.5, color="#290ad8", linestyle="--", lw=0.8, alpha=0.7, label="Moderate threshold")
        ax2.set_xlabel("Aggregate Robustness Score", fontsize=10)
        ax2.set_facecolor("#f8fafc")
        ax2.tick_params(colors="#475569", labelsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color("#cbd5e1")
        ax2.spines["bottom"].set_color("#cbd5e1")
        for b, v in zip(rbars, rvals):
            ax2.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                     f"{v:.1%}", va="center", color="#1e293b", fontsize=9)
        ax2.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_rank)
        plt.close()

        # ── Weaknesses ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Weakness Analysis")
        for mname, failed in weaknesses.items():
            if failed:
                with st.expander(f"[{mname}] {len(failed)} failing metamorphic relations"):
                    for mr in failed:
                        cat = MR_CATEGORIES.get(mr, "?")
                        st.markdown(f"- **{mr}** (Category {cat})")
            else:
                st.success(f"**{mname}**: No weaknesses detected — all metamorphic relations satisfied.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Parametric Sweep on User Model
# ═══════════════════════════════════════════════════════════════════════════════

def _section_sweep(tolerance: float) -> None:
    registry = get_registry()
    if len(registry) == 0:
        return

    st.markdown("---")
    _section_header(
        "Parametric Sensitivity Analysis",
        "Identify critical thresholds where model performance degrades under increasing transformation intensity."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sweep_model = col_a.selectbox("Target model", registry.names(), key="sw_model")
    with col_b:
        sweep_tf    = col_b.selectbox("Transformation", list(SWEEP_CATALOGUE.keys()), key="sw_tf")
    with col_c:
        MR_OPTIONS = [
            "brightness_steering", "flip_symmetry_steering",
            "noise_robustness_steering", "weather_robustness",
            "temporal_smoothness", "perspective_scaling",
        ]
        sweep_mr_name = col_c.selectbox("Metamorphic relation", MR_OPTIONS, key="sw_mr")

    cfg         = SWEEP_CATALOGUE[sweep_tf]
    n_sw_imgs   = st.slider("Number of test images", 3, 20, 6, key="sw_imgs",
                            help="Multiple images improve statistical reliability of pass rate estimation.")
    st.caption(f"Sweep range: {cfg['sweep_values'][0]}{cfg['unit']} → {cfg['sweep_values'][-1]}{cfg['unit']}")

    if st.button("Run Sensitivity Analysis", type="primary", key="run_sweep"):
        adapter  = registry.get(sweep_model)
        factory  = cfg["factory"]
        values   = cfg["sweep_values"]
        sweep_mr = get_mr(sweep_mr_name, tolerance=tolerance)

        # Build test images
        sweep_imgs = []
        
        if 'uploaded_test_img' in st.session_state:
            sweep_imgs.append(st.session_state['uploaded_test_img'])
        
        for i in range(n_sw_imgs):
            x = np.linspace(0.05 + i * 0.02, 0.95, 64)
            base = np.stack([
                np.tile(x, (64, 1)),
                np.tile(x[::-1], (64, 1)),
                np.ones((64, 64)) * (0.3 + i * 0.035),
            ], axis=-1).clip(0, 1).astype(np.float32)
            sweep_imgs.append(base)

        ref_f      = _normalise(sweep_imgs[0])
        y_orig_ref = adapter.predict(ref_f)

        result = SweepResult(
            transform_name=sweep_tf,
            parameter_name=cfg["label"],
            mr_name=sweep_mr_name,
            model_name=sweep_model,
        )
        sweep_levels = []

        pb2 = st.progress(0)
        for idx, intensity in enumerate(values):
            tf_fn  = factory(intensity)
            img_t  = tf_fn(ref_f)
            y_t    = adapter.predict(img_t)
            diff_t = np.clip(np.abs(ref_f - img_t) * 4, 0, 1)

            deltas, passes = [], []
            for si in sweep_imgs:
                si_f  = _normalise(si)
                si2   = factory(intensity)(si_f)
                y1_   = adapter.predict(si_f)
                y2_   = adapter.predict(si2)
                r_    = sweep_mr.check(si_f, si2, y1_, y2_)
                deltas.append(r_.delta if r_.delta is not None else abs(y1_ - y2_))
                passes.append(r_.passed)

            n, pc = len(sweep_imgs), sum(passes)
            pt = SweepPoint(
                intensity=intensity, pass_rate=pc / n,
                mean_delta=float(np.mean(deltas)), max_delta=float(np.max(deltas)),
                violations=n - pc, total=n, passed=pc,
            )
            result.sweep_points.append(pt)

            mr_ref = sweep_mr.check(ref_f, img_t, y_orig_ref, y_t)
            sweep_levels.append({
                "intensity": intensity, "img_orig": ref_f,
                "img_trans": img_t, "img_diff": diff_t,
                "y_orig": float(y_orig_ref), "y_trans": float(y_t),
                "delta": float(abs(y_orig_ref - y_t)),
                "pass_rate": pt.pass_rate, "mr_message": mr_ref.message,
            })
            pb2.progress((idx + 1) / len(values))
            pb2.caption(f"Evaluating: {intensity}{cfg['unit']}")

        ct  = result.critical_threshold
        ct2 = result.collapse_threshold

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Summary")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Sweep levels</div><div class='academic-metric-value'>{len(values)}</div></div>", unsafe_allow_html=True)
        mc2.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Max pass rate</div><div class='academic-metric-value'>{max(result.pass_rates):.1%}</div></div>", unsafe_allow_html=True)
        
        degrade_text = f"{ct}{cfg['unit']}" if ct is not None else "None"
        degrade_color = "#479baa" if ct is None else "#290ad8"
        mc3.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Degradation at</div><div class='academic-metric-value' style='color:{degrade_color};'>{degrade_text}</div></div>", unsafe_allow_html=True)
        
        collapse_text = f"{ct2}{cfg['unit']}" if ct2 is not None else "None"
        collapse_color = "#479baa" if ct2 is None else "#c0392b"
        mc4.markdown(f"<div class='academic-metric'><div class='academic-metric-label'>Collapse at</div><div class='academic-metric-value' style='color:{collapse_color};'>{collapse_text}</div></div>", unsafe_allow_html=True)

        # ── Degradation curve ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Degradation Curve")
        fig_sw = plot_single_sweep(result)
        st.pyplot(fig_sw)
        plt.close()

        # ── Visual strip ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Transformation Progression")
        n_show  = min(8, len(sweep_levels))
        step    = max(1, len(sweep_levels) // n_show)
        shown   = sweep_levels[::step][:n_show]

        n_cols = len(shown)
        fig_strip, axes_strip = plt.subplots(
            2, n_cols, figsize=(max(8, n_cols * 2), 5), facecolor="white"
        )
        if n_cols == 1:
            axes_strip = np.array(axes_strip).reshape(2, 1)
        fig_strip.suptitle(
            f"Visual progression of {sweep_tf} transformation — {sweep_model}",
            fontsize=10, fontweight="normal", color="#1e293b"
        )
        for j, lv in enumerate(shown):
            pr = lv["pass_rate"]
            sc = _score_color(pr)
            si = "✓" if pr >= 0.80 else ("⚠" if pr >= 0.50 else "✗")
            ax_t = axes_strip[0, j]
            ax_t.imshow(np.clip(lv["img_trans"], 0, 1))
            ax_t.set_title(f"{lv['intensity']}{cfg['unit']}", fontsize=8, color="#475569")
            ax_t.axis("off")
            ax_d = axes_strip[1, j]
            ax_d.imshow(lv["img_diff"])
            ax_d.set_xlabel(f"{si} {pr:.0%}", fontsize=8, color=sc)
            ax_d.axis("off")
            ax_d.set_xticks([])
            ax_d.set_yticks([])
        axes_strip[0, 0].set_ylabel("Transformed", fontsize=8, color="#475569")
        axes_strip[1, 0].set_ylabel("Difference", fontsize=8, color="#475569")
        plt.tight_layout()
        st.pyplot(fig_strip)
        plt.close()

        # ── Per-level expanders ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Detailed Results by Intensity Level")
        for lv in sweep_levels:
            pr   = lv["pass_rate"]
            icon = "✓" if pr >= 0.80 else ("⚠" if pr >= 0.50 else "✗")
            with st.expander(
                f"{icon} {cfg['label']} = {lv['intensity']}{cfg['unit']}  |  "
                f"Pass rate: {pr:.0%}  |  "
                f"Δθ = {lv['delta']:.4f}"
            ):
                ca, cb, cc = st.columns(3)
                ca.image(np.clip(lv["img_orig"],  0, 1), caption="Original input", use_container_width=True)
                cb.image(np.clip(lv["img_trans"], 0, 1), caption=f"Transformed ({sweep_tf} = {lv['intensity']}{cfg['unit']})", use_container_width=True)
                cc.image(lv["img_diff"], caption="Difference (enhanced)", use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("θ (original)", f"{lv['y_orig']:.4f}")
                m2.metric("θ (transformed)", f"{lv['y_trans']:.4f}",
                          delta=f"{lv['y_trans'] - lv['y_orig']:+.4f}")
                m3.metric("|Δθ|", f"{lv['delta']:.4f}",
                          delta="PASS" if lv["delta"] <= tolerance else "FAIL",
                          delta_color="normal" if lv["delta"] <= tolerance else "inverse")
                m4.metric("Pass rate", f"{pr:.0%}")

                if pr >= 0.80:
                    st.success(lv["mr_message"])
                elif pr >= 0.50:
                    st.warning(lv["mr_message"])
                else:
                    st.error(lv["mr_message"])

        # ── Research interpretation ───────────────────────────────────────────
        st.markdown("---")
        if ct is None:
            st.success(
                f"**{sweep_model}** demonstrates high robustness to {sweep_tf} transformations. "
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER  (called from app.py)
# ═══════════════════════════════════════════════════════════════════════════════

def render(tolerance: float | None = None) -> None:
    """Entry point — call this from the main app.py."""
    if tolerance is None:
        from evaluation.generalization_scorer import statistical_threshold
        tolerance = statistical_threshold()

    # Apply academic styling
    _apply_academic_styling()
    
    # Page header
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.25rem;'>Model Evaluation Framework</h1>
        <p style='color: #475569; font-size: 1rem;'>Metamorphic testing for regression models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Workflow guide
    with st.expander("Workflow Overview", expanded=False):
        st.markdown("""
        **1. Model Upload** — Upload a trained regression model (.pkl, .h5, .pt, .onnx)
        
        **2. Configuration** — Set display name, input dimensions, and output scaling
        
        **3. Metamorphic Testing** — Evaluate against 14 metamorphic relations across 4 categories
        
        **4. Cross-Model Comparison** — Compare multiple models head-to-head (if ≥2 registered)
        
        **5. Sensitivity Analysis** — Identify critical thresholds via parametric sweep
        """)
    
    st.markdown("---")
    
    _section_upload()
    _section_registry()
    _section_single_test(tolerance)
    _section_cross_compare(tolerance)
    _section_sweep(tolerance)