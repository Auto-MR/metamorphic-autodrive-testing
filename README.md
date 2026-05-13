# Metamorphic Autodrive Testing Framework

A research framework for evaluating the robustness of **regression-based autonomous driving models** using metamorphic testing. The framework tests whether models produce consistent, semantically correct predictions when their inputs are transformed in controlled ways — without needing ground-truth labels for every test case.

---

## Research Context

Traditional software testing requires known expected outputs. For machine learning models, this is often impossible — you cannot label every transformed driving scene. **Metamorphic testing** solves this by defining relationships between inputs and outputs that _must_ hold regardless of the specific input value.

For example: if you horizontally flip a road image, the steering angle must flip its sign. If you slightly brighten the image, the steering angle must not change significantly. These are **Metamorphic Relations (MRs)** — testable invariants that hold across all valid inputs.

The central research question this framework addresses is: **do these invariants hold universally across different ML architectures, or are they architecture-specific?** The answer is captured by the Generalization Score.

---

## Key Concepts

### Generalization Score

The primary research metric introduced by this project:

```
GeneralizationScore(MR) = models_satisfying_MR / total_models
```

A score of 1.0 means every model architecture satisfies the relation — it is a universal property of the task, not a quirk of one model. A low score reveals that the relation is architecture-dependent, which has safety implications.

### Statistical Threshold

Tolerances are not hardcoded per model. Instead, they are derived from the steering output range:

```
ε = fraction × steering_range = 0.05 × 2.0 = 0.10
```

This makes evaluation fair across architectures that may have different output magnitudes.

### Output Normalization

All model outputs are normalized to `[-1, 1]` before comparison, enabling fair cross-model evaluation regardless of whether a model outputs degrees, radians, or normalized angles.

---

## Models Under Test

Five built-in models are included, covering the major ML paradigms:

| #   | Model                       | Type          | Adapter                 |
| --- | --------------------------- | ------------- | ----------------------- |
| 1   | DAVE-2 (CNN)                | Deep Learning | `SteeringAdapter`       |
| 2   | Gaussian Process Regression | Probabilistic | `GPRSteeringAdapter`    |
| 3   | Support Vector Regression   | Classical ML  | `SVRSteeringAdapter`    |
| 4   | Random Forest               | Ensemble      | `RFSteeringAdapter`     |
| 5   | Linear Regression           | Baseline      | `LinearSteeringAdapter` |

Additional model stubs are included for CNN depth estimation (`CNNAdapter`) and LSTM trajectory prediction (`LSTMAdapter`).

You can also upload your own trained model through the dashboard — see the [Upload Your Own Model](#upload-your-own-model) section.

---

## Metamorphic Relations (14 total)

MRs are grouped into four categories based on the type of invariance they test.

### Category A — Invariance `|f(x) − f(T(x))| < ε`

Environmental changes that a safe model must be robust to:

| MR                          | Transform              | Tolerance | Real-world Scenario  |
| --------------------------- | ---------------------- | --------- | -------------------- |
| `brightness_steering`       | Brightness ×1.3        | 0.10      | Tunnels, time of day |
| `contrast_steering`         | Contrast ×1.5          | 0.10      | Overcast vs. sunny   |
| `noise_robustness_steering` | Gaussian noise σ=0.025 | 0.12      | Sensor noise         |
| `weather_robustness`        | Fog intensity=0.45     | 0.20      | Fog, haze            |
| `occlusion_robustness`      | Noise patch            | 0.18      | Partial obstruction  |

### Category B — Symmetry `f(flip(x)) ≈ −f(x)`

Structural properties of the driving geometry:

| MR                                    | Transform              | Tolerance | Real-world Scenario    |
| ------------------------------------- | ---------------------- | --------- | ---------------------- |
| `flip_symmetry_steering` _(critical)_ | Horizontal flip        | 0.10      | Mirror-image road      |
| `straight_road_stability`             | Brightness on straight | 0.05      | Motorway cruise        |
| `perspective_scaling`                 | Scale ×1.1             | 0.12      | Camera zoom / distance |

### Category C — Temporal `|f(x_t) − f(x_{t+1})| < δ`

Consistency across sequential frames — models must not make sudden jumps:

| MR                               | Transform          | Tolerance | Real-world Scenario       |
| -------------------------------- | ------------------ | --------- | ------------------------- |
| `temporal_smoothness`            | Small translation  | 0.08      | Frame-to-frame continuity |
| `constant_curvature_consistency` | Translate on curve | 0.05      | Sustained cornering       |
| `small_lane_shift_invariance`    | Lateral shift      | 0.15      | Lane centering            |

### Category D — Composition (combined transforms)

Tests robustness under multiple simultaneous degradations:

| MR                              | Transforms         | Tolerance | Real-world Scenario |
| ------------------------------- | ------------------ | --------- | ------------------- |
| `composed_brightness_noise`     | Brightness + Noise | 0.15      | Night sensor noise  |
| `composed_weather_blur`         | Fog + Blur         | 0.22      | Rain on lens        |
| `lane_center_shift_consistency` | Shift + Brightness | 0.25      | Lane drift in glare |

---

## Project Structure

```
metamorphic-autodrive-testing/
│
├── configs/
│   ├── models_config.yaml       # Model paths, architectures, adapters
│   ├── mr_config.yaml           # MR definitions, categories, tolerances
│   └── thresholds.yaml          # Pass/fail thresholds, generalization boundaries
│
├── core/
│   ├── adapters/
│   │   ├── base_adapter.py              # Abstract base class for all adapters
│   │   ├── steering_adapter.py          # DAVE-2 CNN adapter
│   │   ├── gpr_steering_adapter.py      # Gaussian Process adapter
│   │   ├── svr_steering_adapter.py      # SVR adapter
│   │   ├── rf_steering_adapter.py       # Random Forest adapter
│   │   ├── linear_steering_adapter.py   # Linear Regression adapter
│   │   ├── cnn_adapter.py               # Generic CNN adapter
│   │   ├── lstm_adapter.py              # LSTM adapter
│   │   ├── user_model_adapter.py        # Dynamic adapter for user-uploaded models
│   │   └── user_model_registry.py       # In-session model registry
│   │
│   ├── metamorphic_engine/
│   │   ├── mr_base.py           # MetamorphicRelation and MRResult base classes
│   │   ├── mr_library.py        # Factory registry for all 14+ MRs
│   │   ├── transformer.py       # Image transformation functions
│   │   └── evaluator.py         # Runs MR checks and collects results
│   │
│   ├── metrics/
│   │   ├── normalizer.py        # Steering output normalization to [-1, 1]
│   │   └── steering_metrics.py  # MAE, delta, pass rate helpers
│   │
│   └── pipeline/
│       ├── run_single_test.py   # CLI: test one model against all MRs
│       ├── run_batch_tests.py   # CLI: batch test multiple models
│       └── cross_model_tests.py # CLI: full cross-model generalization study
│
├── metamorphic_relations/
│   ├── steering_mrs.py          # All 14 steering MRs (Cat A/B/C/D)
│   ├── cnn_mrs.py               # CNN-specific MRs
│   ├── lstm_mrs.py              # LSTM-specific MRs
│   └── cross_model_mrs.py       # Cross-modal MRs (depth + steering)
│
├── evaluation/
│   ├── generalization_scorer.py  # MAIN CONTRIBUTION — GeneralizationScore
│   ├── robustness_scorer.py      # Per-model robustness score
│   ├── parametric_sweep.py       # Sweep transform intensity vs. pass rate
│   ├── sweep_visualization.py    # Degradation curve plots
│   ├── visualization.py          # Steering gauges, comparison charts
│   ├── report_generator.py       # Export results to report format
│   └── results_logger.py         # CSV/JSON logging
│
├── dashboard/
│   ├── app.py                   # Streamlit app entry point
│   └── upload_model_page.py     # Upload & test your own model page
│
├── models/
│   ├── steering_regression/     # DAVE-2, GPR, SVR, RF, Linear weights
│   ├── cnn_depth/               # CNN depth estimation stub
│   └── lstm_trajectory/         # LSTM trajectory prediction stub
│
├── transformations/
│   └── noise_models.py          # Image noise and corruption transforms
│
├── tests/
│   ├── test_models.py           # Unit tests for all adapters
│   └── test_mr_engine.py        # Unit tests for MR engine
│
└── requirements.txt
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

TensorFlow is optional — required only if loading DAVE-2 or checkpoint-based models:

```bash
pip install tensorflow>=2.13
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

### CLI — single model test

```bash
python -m core.pipeline.run_single_test
```

### CLI — full cross-model generalization study

```bash
python -m core.pipeline.cross_model_tests
```

---

## Dashboard Walkthrough

The Streamlit dashboard has five sections, each building on the last.

**1. Model Upload** — Upload a trained model file. Configure its display name, input image dimensions, and output scale. The adapter auto-detects the framework from the file extension and runs a smoke test with a synthetic image to confirm the model loads correctly.

**2. Registered Models Panel** — Shows all models currently loaded in the session, their framework, input shape, and load status. Models can be removed individually.

**3. Metamorphic Testing** — Select a registered model, choose which MR categories to test (A/B/C/D), and optionally upload a real road image. The panel shows the original image, the transformed image, the difference map, a steering gauge comparing before/after predictions, and per-MR pass/fail results with delta values. A robustness score is computed at the end.

**4. Cross-Model Comparison** — Requires two or more registered models. Runs all MRs across all models simultaneously and produces: a MR × Model generalization matrix (✓/✗ per cell), per-category bar chart, model robustness ranking chart, and weakness analysis listing which MRs each model fails.

**5. Parametric Sensitivity Analysis** — Sweeps a single transformation parameter (e.g. brightness from 0.5× to 3.0×) across a range of intensities and plots how pass rate degrades. Identifies two critical thresholds: the degradation point (pass rate drops below 80%) and the collapse point (drops below 50%). This defines the operational safety boundary for each model.

---

## Upload Your Own Model

The framework accepts user-uploaded models and evaluates them against the same 14 MRs as the built-in models.

### Supported formats

| Format                | Extension       | Notes                                         |
| --------------------- | --------------- | --------------------------------------------- |
| Scikit-learn / pickle | `.pkl`          | Any pickle-serializable model                 |
| Joblib                | `.joblib`       | Preferred for scikit-learn                    |
| Keras / TensorFlow    | `.h5`, `.keras` | Keras `save()` format                         |
| PyTorch               | `.pt`, `.pth`   | Use `torch.save(model, path)`, not state_dict |
| ONNX                  | `.onnx`         | Framework-agnostic                            |
| TF Checkpoint bundle  | `.zip`          | See below                                     |

### TensorFlow checkpoint upload

TF checkpoints consist of multiple companion files. Bundle them into a single ZIP before uploading:

```bash
# Raw checkpoint (3 files):
zip model_ckpt.zip model.ckpt.meta model.ckpt.index model.ckpt.data-00000-of-00001

# SavedModel directory:
zip -r saved_model.zip saved_model/
```

Upload the `.zip` file through the dashboard. Both raw checkpoint and SavedModel layouts are detected automatically.

### Model requirements

The adapter expects the model to accept a single input: a float32 image of shape `(1, H, W, 3)` with pixel values in `[0, 1]`, and return a single float steering angle. If your model outputs degrees or unnormalized values, select the appropriate output normalization option in the upload panel.

---

## Evaluation Metrics

| Metric                | Formula                               | Meaning                           |
| --------------------- | ------------------------------------- | --------------------------------- |
| Generalization Score  | `models_passing_MR / total_models`    | How universally an MR holds       |
| Robustness Score      | `MRs_passed / total_MRs`              | How robust a single model is      |
| Statistical Threshold | `ε = 0.05 × steering_range`           | Tolerance derived from data range |
| Critical Threshold    | Sweep intensity where pass rate < 80% | Operational boundary              |
| Collapse Threshold    | Sweep intensity where pass rate < 50% | Safety limit                      |

### Generalization thresholds (from `thresholds.yaml`)

| Score  | Interpretation                                |
| ------ | --------------------------------------------- |
| ≥ 80%  | Generalizable — MR holds across architectures |
| 40–80% | Architecture-dependent — some models fail     |
| < 40%  | Model-specific — not a universal property     |

### Expected pass rates by category

| Category        | Expected | Reason                                        |
| --------------- | -------- | --------------------------------------------- |
| A — Invariance  | 85%      | Environmental robustness is usually learnable |
| B — Symmetry    | 90%      | Geometric properties should be universal      |
| C — Temporal    | 75%      | Architecture-dependent on sequence modelling  |
| D — Composition | 65%      | Most challenging — combined degradations      |

---

## Adding a New Model

1. Create an adapter in `core/adapters/` inheriting from `BaseAdapter`.
2. Implement `load(weights_path)` and `predict(image) -> float`.
3. Register it in `configs/models_config.yaml`.
4. Import it in `core/adapters/__init__.py`.

The adapter will then be automatically picked up by the cross-model pipeline.

## Adding a New Metamorphic Relation

1. Add a new class in `metamorphic_relations/steering_mrs.py` (or a new file for a different domain), inheriting from `MetamorphicRelation`.
2. Implement `check(x, x2, y, y2) -> MRResult`.
3. Register it in `core/metamorphic_engine/mr_library.py` under `ALL_MRS` and `STEERING_MRS_BY_CATEGORY`.
4. Add its category mapping to `evaluation/generalization_scorer.py` under `MR_CATEGORIES`.
5. Add its config entry to `configs/mr_config.yaml`.

---

## Dependencies

| Package                   | Purpose                              |
| ------------------------- | ------------------------------------ |
| `numpy`, `scipy`          | Numerical computation                |
| `scikit-learn`            | SVR, RF, GPR, Linear models          |
| `opencv-python-headless`  | Image resizing and transformation    |
| `Pillow`                  | Image loading                        |
| `streamlit`               | Interactive dashboard                |
| `pandas`                  | Cross-model comparison tables        |
| `matplotlib`              | Charts, steering gauges, sweep plots |
| `pyyaml`                  | Config loading                       |
| `tensorflow` _(optional)_ | DAVE-2 CNN and checkpoint loading    |
| `pytest`                  | Unit testing                         |

---

## License

See `LICENSE.txt` for terms.
