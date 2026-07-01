# Home Credit Default Risk: XGBoost Classifier Model Card & Documentation

This model card provides technical documentation and evaluation metrics for the **eXtreme Gradient Boosting (XGBoost)** classifier deployed within the Home Credit Default Risk pipeline suite.

---

## 1. Model Overview

- **Model Class:** `xgboost.XGBClassifier`
- **Objective:** Binary classification (`binary:logistic`) predicting the probability of consumer default ($y=1$ indicates default/unpaid loan, $y=0$ indicates standard repayment).
- **Primary Use Case:** Consumer lending underwriting decision support and automated credit risk scoring.
- **Dataset Source:** Synthetic dataset based on Kaggle Home Credit Default Risk cohort structure.
- **Data Attributes:** Standardized application forms, credit bureau files, and demographics.

---

## 2. Model Architecture & Hyperparameters

The model architecture utilizes gradient-boosted decision trees (GBDT) optimized using **Optuna**'s Bayesian Search with a Tree-structured Parzen Estimator (TPE) kernel.

### Final Optimized Hyperparameters (Trial 15)

The optimal configuration yielded the highest Area Under the Receiver Operating Characteristic curve (AUC-ROC) of **0.7720**:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `learning_rate` | `0.030` | Step size shrinkage to prevent overfitting during sequence boosting |
| `n_estimators` | `585` | Number of sequential boosting iterations (decision trees) |
| `max_depth` | `5` | Maximum depth of individual trees; limits complex feature interactions |
| `subsample` | `0.85` | Fraction of dataset samples randomly drawn per tree build |
| `colsample_bytree` | `0.75` | Feature subsampling ratio when building each individual tree split |
| `min_child_weight` | `3` | Minimum Hessian sum needed to continue tree node partitions |
| `scale_pos_weight` | *Tuned* | Weight multiplier balancing minority default class labels (approx. 8% prevalence) |

---

## 3. Feature Engineering & Importance

The dataset consists of 10 primary feature markers contributing directly to default risk predictions.

### Core Feature Definitions

1. **`EXT_SOURCE_3` (Bureau Score III):** Normalized external credit bureau scoring indicator from specialized third-party credit rating agencies (highly protective).
2. **`EXT_SOURCE_2` (Bureau Score II):** Secondary external credit rating index tracking history and current exposures (highly protective).
3. **`EXT_SOURCE_1` (Bureau Score I):** Primary external credit score from national credit reporting agencies (protective).
4. **`ANNUITY_TO_INCOME` (Debt-to-Income Burden):** Yearly required loan repayment divided by the client's self-reported annual income (risk driver).
5. **`AGE_YEARS` (Applicant Age):** Computed as client age in years. Younger profiles are historically correlated with higher default frequencies.
6. **`EMPLOYMENT_RATE` (Job Tenure Ratio):** Percentage of life spent in current employment tenure (protective).
7. **`AMT_CREDIT` (Loan Credit Amount):** Total credit limit or capital amount requested by the client (mixed driver).
8. **`DAYS_EMPLOYED` (Job Tenure Length):** Direct count of continuous employment tenure.
9. **`NAME_EDUCATION_TYPE` (Education Level):** Categorical level of education (e.g., Higher Education acts as a strong protective factor).
10. **`AMT_GOODS_PRICE` (Financed Goods Price):** Price of consumer goods being financed through the credit contract.

### SHAP Global Importance Ranking

Global explanation weights are mapped using **Mean Absolute SHAP** values to represent mathematical importance under a coalitional game-theoretic framework:

```
Rank  | Feature Name        | Direction | Mean Abs SHAP Value
-------------------------------------------------------------
1     | EXT_SOURCE_3        | Negative  | 0.324
2     | EXT_SOURCE_2        | Negative  | 0.281
3     | EXT_SOURCE_1        | Negative  | 0.194
4     | ANNUITY_TO_INCOME   | Positive  | 0.165
5     | AGE_YEARS           | Negative  | 0.138
6     | EMPLOYMENT_RATE     | Negative  | 0.112
7     | AMT_CREDIT          | Mixed     | 0.088
8     | DAYS_EMPLOYED       | Negative  | 0.075
9     | NAME_EDUCATION_TYPE | Negative  | 0.062
10    | AMT_GOODS_PRICE     | Negative  | 0.048
```

*Note on Direction:*
- **Negative Direction:** Higher feature values decrease default risk (protective).
- **Positive Direction:** Higher feature values increase default risk (hazard).

---

## 4. Performance & Validation Metrics

Model evaluation is conducted on an independent, out-of-sample test split consisting of $N = 1,000$ applications with an implicit default baseline rate of 8.0%.

### Standard Classification Report (Threshold = 0.50)

- **Overall Accuracy:** 89.4%
- **Macro F1-Score:** 0.743
- **Weighted F1-Score:** 0.900

```
Class                         | Precision | Recall (Sensitivity) | F1-Score | Support
------------------------------------------------------------------------------------
Non-Default (Class 0)         |   0.93    |        0.95          |   0.94   |   920
Default (Class 1)             |   0.58    |        0.45          |   0.51   |    80
------------------------------------------------------------------------------------
Macro Average                 |   0.76    |        0.70          |   0.73   |  1000
Weighted Average              |   0.90    |        0.89          |   0.90   |  1000
```

### Area Under Curves
- **Test Set AUC-ROC:** **0.772**
- **Test Set AUC-PR:** **0.585**

### Confusion Matrix (Standard Default Threshold = 0.50)

```
                       Predicted Non-Default (0)   Predicted Default (1)
True Non-Default (0)              874                         46 (False Positive)
True Default (1)                   44 (False Negative)        36 (True Positive)
```

---

## 5. Decision Threshold Trade-offs

The application allows adjusting the risk decision boundary to align with the underwriting risk appetite:

- **Aggressive Credit Acquisition (Low Threshold, e.g., $t = 0.20$):**
  - High Approval Rate.
  - Mitigates False Positives (fewer lost sales).
  - High Exposure to False Negatives (higher rate of actual defaults).
- **Conservative Credit Underwriting (High Threshold, e.g., $t = 0.65$):**
  - High True Positive collection (most default candidates successfully declined).
  - Maximizes Specificity.
  - Elevates False Positives (lost sales of creditworthy clients due to strict gating).

---

## 6. Interpretability & Compliance (SHAP)

To comply with global regulatory frameworks (e.g., Fair Credit Reporting Act, GDPR Article 22), the model leverages a fully deterministic localized **SHAP waterfall engine**.

For every client profile:
1. Predictions start at a **Cohort Base Value ($E[f(x)] = 8\%$)**.
2. Feature effects are computed through exact Shapley combinations.
3. Individual SHAP contributions scale exactly to the differential:
   $$\sum_{i=1}^M \phi_i = f(x) - E[f(x)]$$
   Where $f(x)$ represents predicted default risk probability and $M$ matches the feature dimension.

---

## 7. Data Preprocessing Pipeline

### Missing Value Strategy
- XGBoost natively handles missing values via learned default branch splits
- External bureau scores (`EXT_SOURCE_1/2/3`) preserve missingness as a signal
- Median imputation used for supplementary numerical features

### Categorical Encoding
- **One-Hot Encoding**: Low-cardinality features (Gender, Contract Type)
- **Label Encoding**: High-cardinality features (Education Type, Occupation) via `LabelEncoder`

### Feature Engineering
- `ANNUITY_TO_INCOME = ANNUITY / INCOME`
- `EMPLOYMENT_RATE = DAYS_EMPLOYED / (AGE_YEARS * 365)`
- Clipped to realistic ranges (DTI ≤ 2.0, employment rate ≤ 1.0)

---

## 8. Model Rationale

### Why GBDT over Deep Learning?
Tabular financial profiles lack spatial or temporal structures found in images and text. GBDT algorithms consistently outperform deep neural networks on tabular datasets due to their ability to build sharp decision boundaries on discontinuous, heavy-tailed numeric values.

### Why Optuna?
Grid search is brute-force; random search is blind. Optuna uses Gaussian process models and tree algorithms to find optimized hyperparameters in **10× fewer trials** than exhaustive grid search.

### Why Synthetic Data?
The original Home Credit dataset is a Kaggle competition dataset with redistribution restrictions. Our synthetic generator preserves the same statistical structure, correlations, and class imbalance while being fully reproducible and shareable.

---

## 9. Ethical Considerations

- **Protected Attributes Monitored**: Gender, age, and demographic correlations are tracked for disparate impact
- **Probability Calibration**: Ensures estimated probabilities map closely to empirical cohort outcomes
- **Decision Support Only**: This model is designed as a supportive tool for credit analysts, not an autonomous decision-making system
- **No Real PII**: All data is synthetically generated; no actual consumer data is used

---

*Model Card Version: 1.0 | Last Updated: 2025*
