#!/usr/bin/env python3
"""
XGBoost Credit Risk Model Training Pipeline with Optuna Hyperparameter Tuning

This script implements the full ML training pipeline:
  1. Load synthetic credit data
  2. Preprocess (imputation, encoding, feature engineering)
  3. Hyperparameter optimization with Optuna (TPE Bayesian search)
  4. Train final XGBoost model with best hyperparameters
  5. Save model artifacts and training metadata

Usage:
    python model/train.py
"""

import json
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, MODEL_DIR, RESULTS_DIR, RANDOM_SEED, FEATURES, TARGET,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, MODEL_CONFIG,
    OPTUNA_CONFIG, HYPERPARAM_SPACE, TEST_SIZE, VALIDATION_SIZE
)


class CreditRiskModel:
    """End-to-end credit risk model training pipeline."""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = None
        self.best_params = None
        self.cv_scores = []
        self.feature_importance = None
        
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load and validate dataset."""
        df = pd.read_csv(filepath)
        print(f"Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
        print(f"Default rate: {df[TARGET].mean():.2%}")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocess data: handle missing values, encode categoricals, engineer features."""
        df = df.copy()
        
        # Select features used by model
        use_cols = [c for c in FEATURES if c in df.columns] + [TARGET]
        df = df[use_cols].copy()
        
        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        # Impute numerical features (XGBoost handles missing natively, but we fill for consistency)
        for col in NUMERICAL_FEATURES:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Clip extreme values
        df["ANNUITY_TO_INCOME"] = df["ANNUITY_TO_INCOME"].clip(upper=2.0)
        df["EMPLOYMENT_RATE"] = df["EMPLOYMENT_RATE"].clip(upper=1.0)
        
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        
        print(f"Features after preprocessing: {list(X.columns)}")
        print(f"Missing values: {X.isnull().sum().sum()}")
        return X, y
    
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective function for hyperparameter optimization."""
        params = {
            "objective": MODEL_CONFIG["objective"],
            "eval_metric": MODEL_CONFIG["eval_metric"],
            "tree_method": "hist",
            "random_state": RANDOM_SEED,
            "verbosity": 0,
            # Hyperparameters to optimize
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            # Class imbalance handling
            "scale_pos_weight": len(y[y == 0]) / len(y[y == 1]),
        }
        
        # Cross-validation with early stopping
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            scores.append(auc)
        
        return np.mean(scores)
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run Optuna Bayesian optimization to find best hyperparameters."""
        print("\n" + "=" * 60)
        print("  Optuna Hyperparameter Optimization (TPE)")
        print("=" * 60)
        
        study = optuna.create_study(
            direction=OPTUNA_CONFIG["direction"],
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=OPTUNA_CONFIG["n_trials"],
            timeout=OPTUNA_CONFIG["timeout"],
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"\nBest AUC-ROC: {study.best_value:.4f}")
        print(f"Best Hyperparameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        
        # Save optimization history
        history = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": [
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in study.trials if t.value is not None
            ]
        }
        
        with open(RESULTS_DIR / "optuna_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        return self.best_params
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Train final model with best hyperparameters on full training set."""
        print("\n" + "=" * 60)
        print("  Training Final Model")
        print("=" * 60)
        
        # Add fixed params
        final_params = {
            **self.best_params,
            "objective": MODEL_CONFIG["objective"],
            "eval_metric": MODEL_CONFIG["eval_metric"],
            "tree_method": "hist",
            "random_state": RANDOM_SEED,
            "verbosity": 0,
            "scale_pos_weight": len(y[y == 0]) / len(y[y == 1]),
        }
        
        # Split into train/validation for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_SEED
        )
        
        self.model = xgb.XGBClassifier(**final_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_logloss = log_loss(y_val, y_val_proba)
        
        print(f"Validation AUC-ROC: {val_auc:.4f}")
        print(f"Validation LogLoss: {val_logloss:.4f}")
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        print("\nFeature Importance (Gain):")
        for _, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<20} {row['importance']:.4f}")
        
        return self.model
    
    def save_artifacts(self, feature_names: list[str]) -> None:
        """Save model and metadata."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = MODEL_DIR / "xgboost_credit_risk.json"
        self.model.save_model(str(model_path))
        print(f"\nModel saved to: {model_path}")
        
        # Save sklearn-compatible model
        joblib_path = MODEL_DIR / "xgboost_credit_risk.pkl"
        joblib.dump(self.model, joblib_path)
        print(f"Model (joblib) saved to: {joblib_path}")
        
        # Save metadata
        metadata = {
            "model_type": "xgboost.XGBClassifier",
            "objective": MODEL_CONFIG["objective"],
            "features": feature_names,
            "best_params": self.best_params,
            "feature_importance": self.feature_importance.to_dict(orient="records"),
        }
        
        metadata_path = MODEL_DIR / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    def run_full_pipeline(self, data_path: Path) -> None:
        """Execute the complete training pipeline."""
        print("=" * 60)
        print("  Credit Risk Model — Training Pipeline")
        print("=" * 60)
        
        # 1. Load data
        df = self.load_data(data_path)
        
        # 2. Preprocess
        X, y = self.preprocess(df)
        
        # 3. Optimize hyperparameters
        self.optimize_hyperparameters(X, y)
        
        # 4. Train final model
        self.train_final_model(X, y)
        
        # 5. Save artifacts
        self.save_artifacts(list(X.columns))
        
        print("\n" + "=" * 60)
        print("  Training Complete")
        print("=" * 60)


if __name__ == "__main__":
    # Generate data first if it doesn't exist
    data_path = DATA_DIR / "credit_risk_synthetic.csv"
    if not data_path.exists():
        print("Dataset not found. Generating synthetic data...")
        from generate_synthetic_data import generate_synthetic_credit_data, save_data
        df = generate_synthetic_credit_data()
        save_data(df, data_path)
    
    pipeline = CreditRiskModel()
    pipeline.run_full_pipeline(data_path)
