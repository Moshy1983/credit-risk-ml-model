#!/usr/bin/env python3
"""
Configuration for the Credit Risk Model training pipeline.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model" / "artifacts"
RESULTS_DIR = ROOT_DIR / "model" / "results"

# Ensure directories exist
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset configuration
RANDOM_SEED = 42
N_SAMPLES = 50_000
DEFAULT_RATE = 0.08  # ~8% default rate as per Home Credit dataset

# Feature configuration
FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "ANNUITY_TO_INCOME",
    "AGE_YEARS",
    "EMPLOYMENT_RATE",
    "AMT_CREDIT",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
    "AMT_GOODS_PRICE",
]

CATEGORICAL_FEATURES = ["NAME_EDUCATION_TYPE"]
NUMERICAL_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# Target column
TARGET = "TARGET"

# Model configuration
MODEL_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "verbosity": 0,
    "random_state": RANDOM_SEED,
}

# Optuna tuning configuration
OPTUNA_CONFIG = {
    "n_trials": 15,
    "timeout": 3600,
    "direction": "maximize",
    "metric": "auc",
}

# Hyperparameter search space
HYPERPARAM_SPACE = {
    "learning_rate": (0.01, 0.2, "log"),
    "max_depth": (3, 10, "int"),
    "n_estimators": (100, 600, "int"),
    "subsample": (0.5, 1.0, "uniform"),
    "colsample_bytree": (0.5, 1.0, "uniform"),
    "min_child_weight": (1, 10, "int"),
    "gamma": (0.0, 0.5, "uniform"),
    "reg_alpha": (1e-8, 1.0, "log"),
    "reg_lambda": (1e-8, 1.0, "log"),
}

# Train/test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Threshold for classification
DEFAULT_THRESHOLD = 0.50
