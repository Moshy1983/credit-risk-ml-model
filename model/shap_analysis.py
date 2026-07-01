#!/usr/bin/env python3
"""
SHAP Explainability Analysis

Generates SHAP-based model explanations:
- Global feature importance (mean absolute SHAP values)
- SHAP beeswarm summary plot
- Individual SHAP waterfall plots for sample predictions
- SHAP dependence plots for key features

Usage:
    python model/shap_analysis.py
"""

import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, RANDOM_SEED

warnings.filterwarnings("ignore")


def load_model_and_data():
    """Load trained model and test data."""
    from train import CreditRiskModel
    
    model_path = MODEL_DIR / "xgboost_credit_risk.pkl"
    data_path = DATA_DIR / "credit_risk_synthetic.csv"
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    pipeline = CreditRiskModel()
    X, y = pipeline.preprocess(df)
    
    # Sample 1000 for SHAP analysis (speed)
    X_sample = X.sample(n=min(1000, len(X)), random_state=RANDOM_SEED)
    
    return model, X, X_sample


def plot_shap_summary(model, X_sample, save_path):
    """Generate SHAP summary (beeswarm) plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="dot",
        show=False,
        color_bar_label="Feature Value"
    )
    plt.title("SHAP Summary Plot — Feature Impact on Default Risk", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary: {save_path}")
    
    return explainer, shap_values


def plot_shap_bar(model, X_sample, save_path):
    """Generate SHAP bar plot (mean absolute importance)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Mean Absolute SHAP Values — Global Feature Importance", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP bar plot: {save_path}")


def plot_shap_waterfall(explainer, X_sample, idx, save_path):
    """Generate SHAP waterfall plot for a single prediction."""
    shap_values = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_sample.iloc[idx],
            feature_names=X_sample.columns
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall — Individual Prediction #{idx}", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP waterfall: {save_path}")


def save_shap_importance_csv(explainer, X_sample, save_path):
    """Save SHAP importance values as CSV."""
    shap_values = explainer.shap_values(X_sample)
    
    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
        "shap_importance_pct": mean_abs_shap / mean_abs_shap.sum() * 100
    }).sort_values("mean_abs_shap", ascending=False)
    
    importance_df.to_csv(save_path, index=False)
    print(f"Saved SHAP importance CSV: {save_path}")
    return importance_df


if __name__ == "__main__":
    print("=" * 60)
    print("  SHAP Explainability Analysis")
    print("=" * 60)
    
    model, X_full, X_sample = load_model_and_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    explainer, shap_values = plot_shap_summary(model, X_sample, RESULTS_DIR / "shap_summary.png")
    plot_shap_bar(model, X_sample, RESULTS_DIR / "shap_bar.png")
    
    # Waterfall for first 3 samples
    for i in range(3):
        plot_shap_waterfall(explainer, X_sample, i, RESULTS_DIR / f"shap_waterfall_{i}.png")
    
    # Save importance CSV
    save_shap_importance_csv(explainer, X_sample, RESULTS_DIR / "shap_importance.csv")
    
    print("\n" + "=" * 60)
    print("  SHAP Analysis Complete")
    print("=" * 60)
