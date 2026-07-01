#!/usr/bin/env python3
"""
Synthetic Credit Risk Dataset Generator

Generates a synthetic dataset mimicking the Kaggle Home Credit Default Risk
competition structure. Features are engineered to match real-world credit bureau
patterns with realistic correlations between applicant demographics and default risk.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, RANDOM_SEED, N_SAMPLES, DEFAULT_RATE, FEATURES, TARGET
)

np.random.seed(RANDOM_SEED)


def generate_synthetic_credit_data(n_samples: int = N_SAMPLES, default_rate: float = DEFAULT_RATE) -> pd.DataFrame:
    """Generate synthetic credit application data with realistic correlations."""
    
    n_defaults = int(n_samples * default_rate)
    n_non_defaults = n_samples - n_defaults
    
    # Generate base population (non-defaulters) — more stable profiles
    base_income = np.random.lognormal(mean=11.2, sigma=0.6, size=n_non_defaults)
    base_age = np.random.normal(loc=42, scale=10, size=n_non_defaults)
    base_employment = np.random.gamma(shape=3, scale=3, size=n_non_defaults)
    
    # Generate defaulter population — riskier profiles
    def_income = np.random.lognormal(mean=10.5, sigma=0.7, size=n_defaults)
    def_age = np.random.normal(loc=32, scale=8, size=n_defaults)
    def_employment = np.random.gamma(shape=1.5, scale=2, size=n_defaults)
    
    # Combine
    income = np.concatenate([base_income, def_income])
    age = np.concatenate([base_age, def_age])
    employment_length = np.concatenate([base_employment, def_employment])
    
    # Clip to realistic ranges
    age = np.clip(age, 21, 68)
    employment_length = np.clip(employment_length, 0, 45)
    
    # Credit amount correlates with income but with noise
    credit_multiplier = np.random.normal(3.0, 1.5, size=n_samples)
    credit_multiplier = np.clip(credit_multiplier, 0.5, 8.0)
    credit_amount = income * credit_multiplier
    
    # Goods price is typically ~90% of credit amount
    goods_price = credit_amount * np.random.uniform(0.85, 0.95, size=n_samples)
    
    # Annuity is roughly 10-20% of credit amount (annualized loan payment)
    annuity_rate = np.random.uniform(0.06, 0.18, size=n_samples)
    annuity = credit_amount * annuity_rate
    
    # External bureau scores (0-1, higher = better credit)
    # Defaulters have lower scores
    ext_source_1 = np.concatenate([
        np.random.beta(5, 2, size=n_non_defaults),   # higher for non-defaulters
        np.random.beta(2, 4, size=n_defaults)         # lower for defaulters
    ])
    ext_source_2 = np.concatenate([
        np.random.beta(6, 2, size=n_non_defaults),
        np.random.beta(2, 5, size=n_defaults)
    ])
    ext_source_3 = np.concatenate([
        np.random.beta(7, 2, size=n_non_defaults),
        np.random.beta(2, 6, size=n_defaults)
    ])
    
    # Education type — defaulters skew toward lower education
    edu_weights_non_def = [0.05, 0.45, 0.25, 0.15, 0.10]  # Lower, Secondary, Higher, Academic, Incomplete
    edu_weights_def = [0.20, 0.40, 0.20, 0.10, 0.10]
    
    education_categories = ["Lower secondary", "Secondary / special education", "Higher education", "Academic degree", "Incomplete higher"]
    
    edu_non_def = np.random.choice(education_categories, size=n_non_defaults, p=edu_weights_non_def)
    edu_def = np.random.choice(education_categories, size=n_defaults, p=edu_weights_def)
    education = np.concatenate([edu_non_def, edu_def])
    
    # Shuffle to mix defaulters and non-defaulters
    indices = np.random.permutation(n_samples)
    
    # Calculate derived features
    annuity_to_income = annuity / income
    employment_rate = employment_length / age
    days_employed = -employment_length * 365
    days_birth = -age * 365
    
    # Build DataFrame
    df = pd.DataFrame({
        "SK_ID_CURR": range(100000, 100000 + n_samples),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_BIRTH": days_birth.astype(int),
        "DAYS_EMPLOYED": days_employed.astype(int),
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "NAME_EDUCATION_TYPE": education,
    })
    
    # Add derived features
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYMENT_RATE"] = -df["DAYS_EMPLOYED"] / df["AGE_YEARS"] / 365
    
    # Target
    df["TARGET"] = 0
    df.loc[indices[:n_defaults], "TARGET"] = 1
    df = df.iloc[indices].reset_index(drop=True)
    
    # Add some noise/missing values (realistic)
    mask = np.random.random(size=n_samples) < 0.02
    df.loc[mask, "EXT_SOURCE_1"] = np.nan
    mask = np.random.random(size=n_samples) < 0.015
    df.loc[mask, "EXT_SOURCE_2"] = np.nan
    mask = np.random.random(size=n_samples) < 0.01
    df.loc[mask, "EXT_SOURCE_3"] = np.nan
    
    return df


def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """Save dataset to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df):,} rows to {filepath}")
    print(f"  - Default rate: {df['TARGET'].mean():.1%}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Synthetic Credit Risk Dataset Generator")
    print("=" * 60)
    
    df = generate_synthetic_credit_data()
    
    output_path = DATA_DIR / "credit_risk_synthetic.csv"
    save_data(df, output_path)
    
    print("\nFeature Summary:")
    print(df[FEATURES + [TARGET]].describe())
    print(f"\nDataset saved to: {output_path}")
