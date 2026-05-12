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
    "sklearn/pickle": "🧮",
    "sklearn/joblib": "🧮",
    "keras":          "🧠",
    "pytorch":        "🔥",
    "onnx":           "⚙️",
    "unknown":        "❓",
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
    return "✅" if passed else "❌"


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#2ecc71"
    if score >= 0.5:
        return "#f39c12"
    return "#e74c3c"


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


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Upload & Register
# ═══════════════════════════════════════════════════════════════════════════════

def _section_upload() -> None:
    st.subheader("📤 Step 1 — Upload Your Model")
    st.markdown("""
    Upload any regression model file.  The adapter auto-detects the framework
    from the file extension and routes to the correct loader.

    | Extension | Framework |
    |---|---|
    | `.pkl` `.joblib` | scikit-learn / any pickle-able model |
    | `.h5` `.keras` | TensorFlow / Keras |
    | `.pt` `.pth` | PyTorch (full model, **not** state_dict) |
    | `.onnx` | ONNX Runtime |
    """)

    col_up, col_cfg = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Choose model file",
            type=["pkl", "joblib", "h5", "keras", "pt", "pth", "onnx"],
            help="Max ~500 MB.  For PyTorch, save with torch.save(model, path) not state_dict.",
        )

    with col_cfg:
        display_name = st.text_input(
            "Model display name",
            value=uploaded_file.name.rsplit(".", 1)[0] if uploaded_file else "my_model",
            help="Used as label throughout the dashboard.",
        )
        output_scale = st.selectbox(
            "Output scale",
            ["auto", "normalized (−1 to 1)", "degrees (±25°)", "radians"],
            help="'auto' works for most models.",
        )
        col_h, col_w = st.columns(2)
        input_h = col_h.number_input("Input height (px)", value=66,  min_value=32, max_value=512)
        input_w = col_w.number_input("Input width  (px)", value=200, min_value=32, max_value=1024)

    if uploaded_file is None:
        st.info("↑ Upload a model file to get started.")
        return

    registry = get_registry()

    if st.button("⚡ Load & Register Model", type="primary"):
        with st.spinner(f"Loading {uploaded_file.name} …"):
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
                    st.error(f"❌ Failed to load model: {adapter.load_error}")
                    return

                # Smoke-test with synthetic image
                smoke_img = _synthetic_img()
                smoke_pred = adapter.predict(smoke_img)

                registry.add(adapter)
                st.success(
                    f"✅ **{display_name}** registered!  "
                    f"Framework: **{adapter.framework}**  |  "
                    f"Smoke-test prediction: **{smoke_pred:.4f}**"
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
    st.subheader(f"🗂️ Registered Models  ({len(registry)})")

    for info in registry.list_models():
        icon = FW_ICONS.get(info["framework"], "❓")
        status = "✅ Ready" if info["loaded"] else f"❌ {info['error']}"
        with st.expander(f"{icon} **{info['name']}**  —  {info['framework']}  |  {status}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Framework", info["framework"])
            c2.metric("Input shape", f"{info['input_h']} × {info['input_w']}")
            c3.metric("Status", "Loaded" if info["loaded"] else "Error")
            if not info["loaded"]:
                c4.error(info["error"])
            if st.button(f"🗑️ Remove  {info['name']}", key=f"rm_{info['name']}"):
                registry.remove(info["name"])
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Single Model MR Test
# ═══════════════════════════════════════════════════════════════════════════════

def _section_single_test(tolerance: float) -> None:
    registry = get_registry()
    if len(registry) == 0:
        st.info("Register at least one model above to run MR tests.")
        return

    st.markdown("---")
    st.subheader("🔬 Step 2 — Run MR Tests on Your Model")

    col_sel, col_img = st.columns([1, 1])

    with col_sel:
        model_name = st.selectbox("Select model", registry.names(), key="single_sel")
        mr_cats    = st.multiselect(
            "MR Categories",
            ["A", "B", "C", "D"],
            default=["A", "B", "C", "D"],
            key="single_cats",
        )
        transform_choice = st.selectbox(
            "Preview transform",
            ["brightness", "contrast", "noise", "fog", "rain", "snow",
             "flip", "rotate_5", "scale", "crop", "translate",
             "composed_bright_noise", "composed_weather_blur", "composed_shift_bright"],
            key="single_tf",
        )

    with col_img:
        up_img = st.file_uploader(
            "Upload test image (optional)",
            type=["jpg", "jpeg", "png"],
            key="single_img",
        )
        if up_img:
            img = np.array(Image.open(up_img).convert("RGB")).astype(np.float32) / 255.0
            # Store in session_state so sweep and cross-compare can use it
            st.session_state['uploaded_test_img'] = img
            st.info("✅ Image stored — will be used in Sweep & Cross-Compare.")
        else:
            st.info("Using synthetic image.")
            img = _synthetic_img()
        st.image(np.clip(img, 0, 1), caption="Test image", use_container_width=True)

    if st.button("▶️ Run MR Tests", type="primary", key="run_single"):
        adapter = registry.get(model_name)

        with st.spinner("Running …"):
            img_f = _normalise(img)
            # Apply selected preview transform
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
        st.subheader("🖼️ Visual Comparison")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(np.clip(img_f, 0, 1), caption="Original", use_container_width=True)
        with c2:
            st.image(np.clip(img_t, 0, 1), caption=f"Transformed ({transform_choice})", use_container_width=True)
        with c3:
            diff = np.clip(np.abs(img_f - img_t) * 4, 0, 1)
            st.image(diff, caption="Pixel Diff ×4", use_container_width=True)

        # ── Steering gauge ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Steering Output")
        gauge_fig = make_steering_gauge(float(y_orig), float(y_trans), transform_choice)
        st.pyplot(gauge_fig)
        plt.close()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("θ original",    f"{y_orig:.4f}")
        c2.metric("θ transformed", f"{y_trans:.4f}",
                  delta=f"{y_trans - y_orig:+.4f}")
        c3.metric("Normalised original",    f"{normalize_steering_output(float(y_orig)):.3f}")
        c4.metric("Normalised transformed", f"{normalize_steering_output(float(y_trans)):.3f}")

        # ── MR results ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧪 Metamorphic Relation Results")

        for cat in ["A", "B", "C", "D"]:
            if cat not in mr_cats:
                continue
            cat_res = [r for r in results if MR_CATEGORIES.get(r.mr_name) == cat]
            if not cat_res:
                continue
            n_pass = sum(r.passed for r in cat_res)
            st.markdown(
                f"**Category {cat}** — {CATEGORY_LABELS[cat]}  "
                f"&nbsp;`{n_pass}/{len(cat_res)} passed`",
                unsafe_allow_html=True,
            )
            for r in cat_res:
                icon = _status_badge(r.passed)
                lbl  = f"{icon} **{r.mr_name}**"
                if r.delta is not None:
                    lbl += f"  —  δ = {r.delta:.4f}"
                with st.expander(lbl):
                    ca, cb = st.columns(2)
                    ca.write(f"**Expected:** {r.expected}")
                    cb.write(f"**Actual:** {r.actual}")
                    if r.passed:
                        st.success(r.message)
                    else:
                        st.warning(r.message)

        # ── Score ─────────────────────────────────────────────────────────────
        st.markdown("---")
        n_pass = sum(r.passed for r in results)
        score  = n_pass / len(results) if results else 0
        st.subheader("🏆 Robustness Score")
        ca, cb = st.columns([1, 3])
        ca.metric("Score", f"{score:.1%}", f"{n_pass}/{len(results)} MRs passed")
        cb.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)
        cb.progress(score)

        if score == 1.0:
            st.success("✅ All MRs satisfied.")
        elif score >= 0.75:
            st.warning("⚠️ Minor violations.")
        elif score >= 0.5:
            st.warning("⚠️ Multiple violations.")
        else:
            st.error("❌ Major violations — model is not robust to these transforms.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Cross-Model Comparison (user models vs built-in)
# ═══════════════════════════════════════════════════════════════════════════════

def _section_cross_compare(tolerance: float) -> None:
    registry = get_registry()
    if len(registry) < 2:
        st.info("Register at least **2 models** to run cross-model comparison.")
        return

    st.markdown("---")
    st.subheader("🔁 Step 3 — Cross-Model Comparison")
    st.markdown(
        "Compare all registered user models head-to-head on all 14 MRs "
        "using the same Generalization Score as the built-in analysis."
    )

    n_imgs = st.slider("Synthetic test images", 3, 15, 5, key="cc_imgs")

    if st.button("▶️ Run Cross-Model Comparison", type="primary", key="run_cross"):
        adapters = registry.all_adapters()

        # Generate test images
        test_imgs = []
        
        # Check if a real image was uploaded and stored in session_state
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

        with st.spinner("Running all MRs across all models …"):
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

        gen_scores = compute_generalization_score(results_map)
        table      = compute_cross_model_table(results_map)
        ranking    = rank_models_by_robustness(results_map)
        weaknesses = identify_model_weaknesses(results_map)
        cat_gen    = compute_category_generalization(results_map, MR_CATEGORIES)

        # ── Generalization table ──────────────────────────────────────────────
        import pandas as pd
        st.markdown("---")
        st.subheader("📋 MR × Model Generalization Table")

        model_names = list(adapters.keys())
        rows = []
        for mr_name in sorted(gen_scores):
            cat = MR_CATEGORIES.get(mr_name, "?")
            row = {"MR Name": mr_name, "Category": f"Cat {cat}"}
            for m in model_names:
                row[m] = table.get(mr_name, {}).get(m, "—")
            row["Gen. Score"] = f"{gen_scores[mr_name]:.0%}"
            rows.append(row)

        def _highlight(val):
            if val == "✓":
                return "background-color:#1a3a1a;color:#90EE90"
            if val == "✗":
                return "background-color:#3a1a1a;color:#FF9999"
            if "%" in str(val):
                p = float(str(val).strip("%")) / 100
                c = "#2ecc71" if p >= 0.8 else ("#f39c12" if p >= 0.6 else "#e74c3c")
                return f"color:{c};font-weight:bold"
            return ""

        st.dataframe(
            pd.DataFrame(rows).style.applymap(_highlight),
            use_container_width=True,
        )

        # ── Category bar chart ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Generalization by Category")

        c1, c2, c3, c4 = st.columns(4)
        for col, cat, label in zip([c1, c2, c3, c4], "ABCD",
                                    ["Invariance", "Symmetry", "Temporal", "Composition"]):
            col.metric(f"Cat {cat}: {label}", f"{cat_gen.get(cat, 0):.1%}")

        fig_cat, ax = plt.subplots(figsize=(8, 3), facecolor="#1a1a2e")
        cats  = list(cat_gen)
        vals  = [cat_gen[c] for c in cats]
        cols  = [_score_color(v) for v in vals]
        bars  = ax.bar([f"Cat {c}" for c in cats], vals, color=cols, edgecolor="#444")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color="grey", linestyle="--", lw=0.8)
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{v:.0%}", ha="center", color="white", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cat)
        plt.close()

        # ── Ranking ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏆 Model Robustness Ranking")
        fig_rank, ax2 = plt.subplots(figsize=(9, max(3, len(ranking) * 0.8)),
                                      facecolor="#1a1a2e")
        rnames = [m for m, _ in ranking][::-1]
        rvals  = [s for _, s in ranking][::-1]
        rcols  = [_score_color(v) for v in rvals]
        rbars  = ax2.barh(rnames, rvals, color=rcols, edgecolor="#444")
        ax2.set_xlim(0, 1.05)
        ax2.axvline(0.8, color="grey", linestyle="--", lw=0.8)
        ax2.set_facecolor("#16213e")
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#444")
        for b, v in zip(rbars, rvals):
            ax2.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                     f"{v:.1%}", va="center", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_rank)
        plt.close()

        # ── Weaknesses ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚠️ Model Weakness Analysis")
        for mname, failed in weaknesses.items():
            if failed:
                with st.expander(f"❌ **{mname}** — {len(failed)} weakness(es)"):
                    for mr in failed:
                        cat = MR_CATEGORIES.get(mr, "?")
                        st.markdown(f"- `{mr}` (Category {cat})")
            else:
                st.success(f"✅ **{mname}** — No weaknesses detected")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Parametric Sweep on User Model
# ═══════════════════════════════════════════════════════════════════════════════

def _section_sweep(tolerance: float) -> None:
    registry = get_registry()
    if len(registry) == 0:
        return

    st.markdown("---")
    st.subheader("📈 Step 4 — Parametric Sweep on Your Model")
    st.markdown(
        "Sweep a transform intensity from low to high and watch the MR pass rate degrade. "
        "Identifies the exact intensity threshold where your model starts failing."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sweep_model = col_a.selectbox("Model", registry.names(), key="sw_model")
    with col_b:
        sweep_tf    = col_b.selectbox("Transform", list(SWEEP_CATALOGUE.keys()), key="sw_tf")
    with col_c:
        MR_OPTIONS = [
            "brightness_steering", "flip_symmetry_steering",
            "noise_robustness_steering", "weather_robustness",
            "temporal_smoothness", "perspective_scaling",
        ]
        sweep_mr_name = col_c.selectbox("MR to test", MR_OPTIONS, key="sw_mr")

    cfg         = SWEEP_CATALOGUE[sweep_tf]
    n_sw_imgs   = st.slider("Test images", 3, 20, 6, key="sw_imgs")
    st.caption(f"Range: {cfg['sweep_values'][0]}{cfg['unit']} → {cfg['sweep_values'][-1]}{cfg['unit']}")

    if st.button("▶️ Run Sweep", type="primary", key="run_sweep"):
        adapter  = registry.get(sweep_model)
        factory  = cfg["factory"]
        values   = cfg["sweep_values"]
        sweep_mr = get_mr(sweep_mr_name, tolerance=tolerance)

        # Build test images
        sweep_imgs = []
        
        # Check if a real image was uploaded and stored in session_state
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

        ct  = result.critical_threshold
        ct2 = result.collapse_threshold

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Levels",    len(values))
        mc2.metric("Max pass rate", f"{max(result.pass_rates):.1%}")
        mc3.metric("⚠️ Degrades at",
                   f"{ct}{cfg['unit']}"  if ct  is not None else "Never")
        mc4.metric("💥 Collapses at",
                   f"{ct2}{cfg['unit']}" if ct2 is not None else "Never")

        # ── Degradation curve ─────────────────────────────────────────────────
        st.markdown("---")
        fig_sw = plot_single_sweep(result)
        st.pyplot(fig_sw)
        plt.close()

        # ── Visual strip ──────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🖼️ Visual Transform Progression")
        n_show  = min(8, len(sweep_levels))
        step    = max(1, len(sweep_levels) // n_show)
        shown   = sweep_levels[::step][:n_show]

        n_cols = len(shown)
        fig_strip, axes_strip = plt.subplots(
            2, n_cols, figsize=(max(10, n_cols * 2.2), 5.5), facecolor="#1a1a2e"
        )
        if n_cols == 1:
            axes_strip = np.array(axes_strip).reshape(2, 1)
        fig_strip.suptitle(
            f"Transform Progression — {sweep_tf}  |  {sweep_model}",
            color="white", fontsize=11, fontweight="bold",
        )
        for j, lv in enumerate(shown):
            pr = lv["pass_rate"]
            sc = _score_color(pr)
            si = "✅" if pr >= 0.80 else ("⚠️" if pr >= 0.50 else "❌")
            ax_t = axes_strip[0, j]
            ax_t.imshow(np.clip(lv["img_trans"], 0, 1))
            ax_t.set_title(f"{lv['intensity']}{cfg['unit']}", color="white", fontsize=9)
            ax_t.axis("off")
            ax_d = axes_strip[1, j]
            ax_d.imshow(lv["img_diff"])
            ax_d.set_xlabel(f"{si} {pr:.0%}", color=sc, fontsize=9)
            ax_d.axis("off")
        axes_strip[0, 0].set_ylabel("Transformed", color="white", fontsize=8)
        axes_strip[1, 0].set_ylabel("Diff ×4",     color="white", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_strip)
        plt.close()

        # ── Per-level expanders ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Per-Level Detail")
        for lv in sweep_levels:
            pr   = lv["pass_rate"]
            icon = "✅" if pr >= 0.80 else ("⚠️" if pr >= 0.50 else "❌")
            with st.expander(
                f"{icon} **{cfg['label']} = {lv['intensity']}{cfg['unit']}**  "
                f"|  pass rate {pr:.0%}  "
                f"|  θ_orig={lv['y_orig']:.4f}  "
                f"|  θ_trans={lv['y_trans']:.4f}  "
                f"|  |Δθ|={lv['delta']:.4f}"
            ):
                ca, cb, cc = st.columns(3)
                ca.image(np.clip(lv["img_orig"],  0, 1), caption="Original",          use_container_width=True)
                cb.image(np.clip(lv["img_trans"], 0, 1), caption=f"{sweep_tf} = {lv['intensity']}{cfg['unit']}", use_container_width=True)
                cc.image(lv["img_diff"],                 caption="Pixel Diff ×4",     use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("θ original",    f"{lv['y_orig']:.4f}")
                m2.metric("θ transformed", f"{lv['y_trans']:.4f}",
                          delta=f"{lv['y_trans'] - lv['y_orig']:+.4f}")
                m3.metric("|Δθ|", f"{lv['delta']:.4f}",
                          delta="PASS ✅" if lv["delta"] <= tolerance else "FAIL ❌",
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
                f"✅ **{sweep_model}** is highly robust to `{sweep_tf}` — "
                f"MR `{sweep_mr_name}` never drops below 80% across the full range."
            )
        elif ct2 is None:
            st.warning(
                f"⚠️ **{sweep_model}** degrades at `{sweep_tf}` = **{ct}{cfg['unit']}**, "
                f"but never collapses below 50%."
            )
        else:
            st.error(
                f"❌ **{sweep_model}** degrades at **{ct}{cfg['unit']}** "
                f"and collapses at **{ct2}{cfg['unit']}**. Safety-critical failure boundary."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER  (called from app.py)
# ═══════════════════════════════════════════════════════════════════════════════

def render(tolerance: float | None = None) -> None:
    """Entry point — call this from the main app.py."""
    if tolerance is None:
        from evaluation.generalization_scorer import statistical_threshold
        tolerance = statistical_threshold()

    st.markdown("""
    <style>
    .upload-step {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2E75B6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### How it works
    1. **Upload** any `.pkl / .h5 / .pt / .onnx` regression model file
    2. **Load & validate** — instant smoke-test confirms the model works
    3. **Run MR tests** — all 14 metamorphic relations, same pipeline as built-in models
    4. **Cross-compare** — if you upload multiple models, compare them head-to-head
    5. **Parametric sweep** — find the exact intensity threshold where your model fails
    """)

    _section_upload()
    _section_registry()
    _section_single_test(tolerance)
    _section_cross_compare(tolerance)
    _section_sweep(tolerance)
