# Metamorphic Autodrive Testing Framework

A generalized metamorphic testing framework for **regression-based autonomous driving models**, 
supporting cross-model comparison across 5 ML architectures.

## Research Contribution

> *"Generalized Metamorphic Relations for Regression-Based Autonomous Driving Models"*

This framework introduces:
- **Model-agnostic MRs** — same relation works across all architectures
- **Cross-model generalization score** — `models_satisfying_MR / total_models`
- **4 MR categories** — Invariance (A), Symmetry (B), Temporal (C), Composition (D)
- **Statistical thresholds** — `ε = 0.05 × steering_range` (no per-model hardcoding)
- **Output normalization** — enables fair comparison across architectures

## Models Under Test

| # | Model | Type | Adapter |
|---|---|---|---|
| 1 | DAVE-2 (CNN) | Deep Learning | `SteeringAdapter` |
| 2 | GPR | Probabilistic | `GPRSteeringAdapter` |
| 3 | SVR | Classical ML | `SVRSteeringAdapter` |
| 4 | Random Forest | Ensemble | `RFSteeringAdapter` |
| 5 | Linear Regression | Baseline | `LinearSteeringAdapter` |

## Metamorphic Relations (14 total, 4 categories)

### Category A — Invariance `|f(x) - f(T(x))| < ε`
| MR | Transform | Tolerance |
|---|---|---|
| brightness_steering | Brightness ×1.3 | 0.10 |
| contrast_steering | Contrast ×1.5 | 0.10 |
| noise_robustness_steering | Gaussian noise σ=0.025 | 0.12 |
| weather_robustness | Fog (intensity=0.45) | 0.20 |
| occlusion_robustness | Noise patch | 0.18 |

### Category B — Symmetry `|f(flip(x)) + f(x)| < ε`
| MR | Transform | Tolerance |
|---|---|---|
| flip_symmetry_steering *(CRITICAL)* | Horizontal flip | 0.10 |
| straight_road_stability | Straight road → θ≈0 | 0.05 |
| perspective_scaling | Scale ×1.1 | 0.12 |

### Category C — Temporal `|f(x_t) - f(x_{t+1})| < δ`
| MR | Transform | Tolerance |
|---|---|---|
| temporal_smoothness | Small translation | 0.08 |
| constant_curvature_consistency | Translate on curve | 0.05 |
| small_lane_shift_invariance | Lateral shift | 0.15 |

### Category D — Composition (combined transforms)
| MR | Transforms | Tolerance |
|---|---|---|
| composed_brightness_noise | Brightness + Noise | 0.15 |
| composed_weather_blur | Fog + Blur | 0.22 |
| lane_center_shift_consistency | Shift + Brightness | 0.25 |

## Quick Start

```bash
pip install -r requirements.txt

# Interactive dashboard
streamlit run dashboard/app.py

# Cross-model evaluation (CLI)
python -m core.pipeline.cross_model_tests
```

## Project Structure

```
core/
  adapters/           → DAVE-2, GPR, SVR, RandomForest, Linear, CNN, LSTM
  metamorphic_engine/ → MR base, library (14 MRs), evaluator, transformer
  metrics/            → normalizer, consistency, steering metrics
  pipeline/           → single test, batch tests, cross-model tests
metamorphic_relations/ → steering_mrs (4 categories), cnn_mrs, lstm_mrs, cross_model_mrs
transformations/      → image, geometric, noise, sequence transforms
evaluation/
  generalization_scorer.py  ← MAIN CONTRIBUTION
  robustness_scorer.py
  visualization.py
  report_generator.py
  results_logger.py
configs/              → mr_config.yaml, models_config.yaml, thresholds.yaml
dashboard/            → Streamlit UI with cross-model analysis page
experiments/          → Jupyter notebooks for thesis experiments
```

## Generalization Score

The key research metric:

```
GeneralizationScore(MR) = models_satisfying_MR / total_models
```

Example:
| MR | DAVE-2 | GPR | SVR | RF | Linear | Score |
|---|---|---|---|---|---|---|
| brightness_steering | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| flip_symmetry | ✓ | ✓ | ✓ | ✗ | ✓ | 80% |
| weather_robustness | ✓ | ✗ | ✓ | ✗ | ✓ | 60% |
