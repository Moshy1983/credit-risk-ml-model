/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ClientProfile {
  gender: "F" | "M";
  age: number; // in years (maps to DAYS_BIRTH = -age * 365)
  employmentLength: number; // in years (maps to DAYS_EMPLOYED = -employmentLength * 365)
  income: number; // AMT_INCOME_TOTAL
  creditAmount: number; // AMT_CREDIT
  annuity: number; // AMT_ANNUITY
  goodsPrice: number; // AMT_GOODS_PRICE
  contractType: "Cash loans" | "Revolving loans";
  occupationType: string;
  educationType: string;
  extSource1: number; // score between 0 and 1
  extSource2: number; // score between 0 and 1
  extSource3: number; // score between 0 and 1
}

export interface ShapValue {
  name: string;
  val: number; // SHAP contribution value (log-odds impact or margin impact)
  featureValue: string | number;
}

export interface TuningTrial {
  trial: number;
  value: number; // ROC-AUC
  params: {
    learning_rate: number;
    max_depth: number;
    n_estimators: number;
    subsample: number;
    colsample_bytree: number;
    min_child_weight: number;
  };
}
