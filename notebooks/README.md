# Credit Risk Default Prediction — Jupyter Notebook

## Interactive ML Notebook

This notebook demonstrates the complete credit risk modeling pipeline interactively:

1. **Data Exploration** — Understand the Home Credit synthetic dataset
2. **Feature Engineering** — Build ratio-based and interaction features
3. **Model Training** — XGBoost with Optuna hyperparameter tuning
4. **Evaluation** — ROC/PR curves, confusion matrices, threshold analysis
5. **Explainability** — SHAP values for global and local model explanations

## Run Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook notebooks/credit_risk_model.ipynb
```

## Key Sections

### Data Overview
Explore the distribution of applicant demographics, credit amounts, and default rates.

### Feature Engineering
- `ANNUITY_TO_INCOME`: Debt burden ratio
- `EMPLOYMENT_RATE`: Job stability indicator
- `EXT_SOURCE_*`: External bureau score ensemble

### Model Architecture
```python
from xgboost import XGBClassifier
from optuna import create_study

# TPE Bayesian optimization for hyperparameters
study = create_study(direction="maximize")
study.optimize(objective, n_trials=15)
```

### Evaluation Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | >0.75 | Discrimination ability across thresholds |
| AUC-PR | >0.50 | Precision-recall on imbalanced data |
| LogLoss | <0.30 | Probability calibration quality |

### SHAP Explainability
- **Global**: Mean absolute SHAP values rank feature importance
- **Local**: Waterfall plots explain individual predictions
- **Dependence**: Partial dependence plots show feature interactions

## Output Artifacts

All generated plots and model files are saved to:
- `model/results/` — Evaluation plots and SHAP visualizations
- `model/artifacts/` — Serialized model and metadata
- `data/` — Processed datasets

## Next Steps

- Deploy the interactive explorer at `npm run dev`
- Explore the React frontend in `src/`
- Review the model card in `docs/MODEL_CARD.md`
