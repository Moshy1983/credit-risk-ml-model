# System Architecture

## Overview

The Credit Risk Default Prediction system is a **full-stack ML application** that combines a Python-based machine learning pipeline with an interactive React frontend. This document describes the architectural decisions, data flow, and component interactions.

## Design Principles

1. **Reproducibility**: Every step of the ML pipeline is scripted and deterministic
2. **Explainability**: SHAP analysis is integrated at both training and inference time
3. **Separation of Concerns**: Model training, evaluation, and deployment are decoupled
4. **Portfolio-Ready**: Code is production-quality with proper typing, documentation, and structure

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User / Portfolio Visitor                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    React Interactive Explorer (Frontend)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Overview │ │ Simulator│ │ SHAP Viz │ │  Tuning  │ │ Metrics  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                              │                                      │
│                    ┌─────────┴─────────┐                          │
│                    │   Express API      │                          │
│                    │  /api/analyze-risk │                          │
│                    └─────────┬─────────┘                          │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Python ML Pipeline (Backend)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Data Gen     │  │ Model Train  │  │ Evaluation & SHAP        │ │
│  │ (Synthetic)  │  │ (XGBoost)    │  │ (Plots & Metrics)        │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│         │                │                        │                 │
│         └────────────────┼────────────────────────┘                 │
│                          │                                          │
│               ┌──────────┴──────────┐                               │
│               │  Model Artifacts    │                               │
│               │  (artifacts/*.pkl)  │                               │
│               └─────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Training Pipeline

```
Raw Synthetic Data → Preprocessing → Feature Engineering → 
  → Optuna Tuning → XGBoost Training → Model Artifacts
                                       → Evaluation Plots
                                       → SHAP Visualizations
```

**Key Files:**
- `model/generate_synthetic_data.py` — Creates realistic credit application data
- `model/train.py` — Full training pipeline with Optuna optimization
- `model/evaluate.py` — Generates ROC, PR, confusion matrix, threshold plots
- `model/shap_analysis.py` — Produces SHAP summary, bar, waterfall plots

### 2. Inference Flow (Simulator)

The frontend simulator implements a **mathematical approximation** of the trained XGBoost model using a log-odds linear combination. This design choice was made to:
- Keep the frontend lightweight (no Python backend required for inference)
- Demonstrate deep understanding of how XGBoost predictions work
- Provide instant, interactive feedback without API latency

```
User Adjusts Sliders → Frontend Calculates Log-Odds → 
  Sigmoid → Risk Probability → SHAP Waterfall → SVG Render
```

**Log-Odds Model (Frontend):**
```
logOdds = -2.4                           # baseline
        - 2.5*(EXT1 - 0.5)               # bureau score 1
        - 3.2*(EXT2 - 0.5)               # bureau score 2
        - 3.8*(EXT3 - 0.5)               # bureau score 3
        + 3.5*(ANNUITY/INCOME - 0.15)    # debt burden
        - 0.05*(AGE - 45)                # age protective
        - 0.12*(EMPLOYMENT - 6)          # job stability
        + education_effect               # education level
        + gender_effect                  # statistical gender factor

risk = 1 / (1 + exp(-logOdds))
```

### 3. AI Analysis Flow

```
User Clicks "Analyze" → POST /api/analyze-risk → 
  Gemini API Prompt Engineering → Credit Committee Brief → JSON Response
```

## Component Architecture

### Frontend (React)

| Component | Responsibility | Key Tech |
|-----------|---------------|----------|
| `App.tsx` | Tab routing, layout shell, AnimatePresence | React, Framer Motion |
| `LandingTab.tsx` | Hero, feature cards, FAQ | React |
| `SimulatorTab.tsx` | Form controls, risk calculation, waterfall SVG, heatmap | React, SVG |
| `ExplainabilityTab.tsx` | SHAP bar chart, beeswarm scatter | React, SVG, Framer Motion |
| `TuningTab.tsx` | Trial sparkline, parameter inspector | React, SVG |
| `EvaluationTab.tsx` | ROC/PR curves, threshold slider, confusion matrix | React, SVG, Framer Motion |
| `DocumentationTab.tsx` | Model specs, preprocessing, compliance | React |

**State Management:** All state is local (useState). No external state library needed due to the focused, tabbed nature of the application.

**Styling Strategy:** Tailwind CSS with a custom theme defined in `index.css`:
- Primary: Indigo (`#4f46e5`)
- Success: Emerald (`#10b981`)
- Warning: Amber (`#f59e0b`)
- Danger: Rose (`#f43f5e`)
- Typography: Inter (sans), JetBrains Mono (monospace)

### Backend (Express + Gemini)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analyze-risk` | POST | AI-powered credit analysis using Gemini 3.5 Flash |
| `/*` | GET | SPA fallback (production) |

**Gemini Prompt Engineering:**
The prompt is carefully structured to elicit expert-level credit analysis:
- Client profile details formatted as structured data
- Model risk probability and SHAP key factors
- Three required sections: Risk Assessment, Decision Strategy, Risk Mitigation
- Constraints: 120–150 words, professional tone, no markdown headers

### Python ML Pipeline

| Module | Purpose | Key Libraries |
|--------|---------|--------------|
| `config.py` | Centralized configuration | pathlib |
| `generate_synthetic_data.py` | Realistic dataset generation | numpy, pandas |
| `train.py` | Training pipeline | xgboost, optuna, sklearn |
| `evaluate.py` | Evaluation & plotting | matplotlib, seaborn, sklearn |
| `shap_analysis.py` | SHAP explainability | shap, matplotlib |

## Key Architectural Decisions

### 1. Why XGBoost over Deep Learning?

Tabular financial data lacks spatial or temporal structure. GBDT algorithms consistently outperform neural networks on tabular datasets due to:
- Sharp decision boundaries on discontinuous, heavy-tailed numeric values
- Native handling of missing values
- Built-in feature importance via gain/split metrics
- Faster training and inference

### 2. Why Synthetic Data?

The original Home Credit dataset is a Kaggle competition dataset with usage restrictions. We generate synthetic data that:
- Preserves the same statistical structure (8% default rate, feature correlations)
- Is fully redistributable and reproducible
- Matches the documented feature distributions in the model card

### 3. Why Frontend Approximation?

The frontend uses a mathematical approximation rather than calling the actual model. This enables:
- Instant feedback without network latency
- Interactive sliders that update in real time
- Demonstration of understanding XGBoost's log-odds mechanics
- A lightweight deployment that doesn't require a Python runtime

The approximation is calibrated to match the trained model's behavior across the feature space.

### 4. Why SHAP for Explainability?

SHAP (Shapley Additive exPlanations) was chosen because:
- **Mathematically sound**: Based on coalitional game theory
- **Local consistency**: Every prediction has an exact additive explanation
- **Regulatory compliance**: Provides defensible reasoning for credit decisions (FCRA, GDPR)
- **Global + local**: Supports both summary plots and individual waterfalls

## Security & Compliance Considerations

- **No real PII**: All data is synthetic
- **Protected attribute monitoring**: Gender and age correlations are documented for disparate impact analysis
- **Probability calibration**: Ensures estimated probabilities map to empirical default rates
- **Decision transparency**: Every prediction includes SHAP-based reason codes
- **API key management**: Gemini API key is stored in environment variables, never committed

## Scalability Notes

This is a **portfolio/demo architecture**. For production scale:
- Deploy the Python model via FastAPI with ONNX runtime for sub-10ms inference
- Use Redis for caching frequent predictions
- Store SHAP values in a feature store for real-time lookup
- Add model monitoring with Evidently or WhyLabs for drift detection
- Implement A/B testing for threshold decisions

## File Organization Rationale

```
model/          → ML pipeline (self-contained, can run independently)
src/            → React frontend (Vite project, standard structure)
notebooks/      → Jupyter notebooks for interactive exploration
docs/           → Documentation (architecture, model card)
server.ts       → Express server (minimal, focused)
```

This separation allows the ML pipeline to be reused independently, while the frontend focuses on visualization and interaction.
