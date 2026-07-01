#!/usr/bin/env python3
"""
Model Evaluation Script

Comprehensive evaluation of the trained credit risk model including:
- ROC-AUC and PR-AUC curves
- Confusion matrix at multiple thresholds
- Classification report
- Calibration analysis
- Threshold trade-off analysis

Usage:
    python model/evaluate.py
"""

import json
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss, brier_score_loss
)
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, RANDOM_SEED, TARGET

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_model_and_data():
    """Load trained model and test data."""
    model_path = MODEL_DIR / "xgboost_credit_risk.pkl"
    data_path = DATA_DIR / "credit_risk_synthetic.csv"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # Preprocess (simple version matching train.py logic)
    from train import CreditRiskModel
    pipeline = CreditRiskModel()
    X, y = pipeline.preprocess(df)
    
    # Split same way as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    return model, X_test, y_test


def plot_roc_curve(y_true, y_proba, save_path):
    """Plot ROC curve and save."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#4f46e5", linewidth=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#cbd5e1", linestyle="--", linewidth=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#4f46e5")
    
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve — Credit Risk Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve: {save_path}")


def plot_precision_recall(y_true, y_proba, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#e11d48", linewidth=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.axhline(baseline, color="#cbd5e1", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline:.1%})")
    ax.fill_between(recall, precision, alpha=0.1, color="#e11d48")
    
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.set_title("Precision-Recall Curve — Credit Risk Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curve: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Non-Default (0)", "Default (1)"],
                yticklabels=["Non-Default (0)", "Default (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Threshold = 0.50)", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def plot_threshold_tradeoff(y_true, y_proba, save_path):
    """Plot threshold vs. precision/recall trade-off."""
    thresholds = np.arange(0.05, 0.95, 0.05)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, "o-", color="#e11d48", linewidth=2, label="Precision", markersize=4)
    ax.plot(thresholds, recalls, "s-", color="#4f46e5", linewidth=2, label="Recall", markersize=4)
    ax.plot(thresholds, f1s, "^-", color="#059669", linewidth=2, label="F1-Score", markersize=4)
    ax.axvline(0.50, color="#94a3b8", linestyle="--", linewidth=1, label="Default Threshold (0.50)")
    
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Trade-off Analysis", fontsize=14, fontweight="bold")
    ax.legend(loc="center right", fontsize=11)
    ax.set_xlim([0.05, 0.90])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold trade-off: {save_path}")


def generate_report(y_true, y_proba, y_pred):
    """Generate comprehensive evaluation report."""
    report = {
        "dataset_size": len(y_true),
        "default_rate": float(y_true.mean()),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    report_path = RESULTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report: {report_path}")
    
    return report


if __name__ == "__main__":
    print("=" * 60)
    print("  Credit Risk Model — Evaluation")
    print("=" * 60)
    
    model, X_test, y_test = load_model_and_data()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_roc_curve(y_test, y_proba, RESULTS_DIR / "roc_curve.png")
    plot_precision_recall(y_test, y_proba, RESULTS_DIR / "pr_curve.png")
    plot_confusion_matrix(y_test, y_pred, RESULTS_DIR / "confusion_matrix.png")
    plot_threshold_tradeoff(y_test, y_proba, RESULTS_DIR / "threshold_tradeoff.png")
    
    # Generate report
    report = generate_report(y_test, y_proba, y_pred)
    
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"AUC-ROC:  {report['auc_roc']:.4f}")
    print(f"AUC-PR:   {report['auc_pr']:.f}")
    print(f"LogLoss:  {report['log_loss']:.4f}")
    print(f"Brier:    {report['brier_score']:.4f}")
    print("\nClassification Report (threshold=0.50):")
    print(classification_report(y_test, y_pred, target_names=["Non-Default", "Default"]))
    print("=" * 60)
