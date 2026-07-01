import { BookOpen, Database, Cpu, ShieldAlert, GitBranch, ArrowRight, Table, Settings } from "lucide-react";

export default function DocumentationTab() {
  return (
    <div className="space-y-8" id="documentation-tab-root">
      
      {/* Overview Block */}
      <div className="bg-gradient-to-r from-indigo-500/10 to-blue-500/10 border border-slate-200 rounded-xl p-6 relative overflow-hidden">
        <div className="max-w-3xl space-y-2">
          <div className="text-indigo-700 text-xs font-mono font-bold uppercase tracking-wider">
            Model Card & Technical Report
          </div>
          <h3 className="text-xl font-bold text-slate-800">Home Credit Default Risk Modeling Rationale</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            This dossier outlines the technical specifications, architectural decisions, dataset engineering, and regulatory alignment principles supporting the XGBoost default prediction system. Built for credit risk committees and compliance audits.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Specs & Rationale */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Section 1: Technical Specifications */}
          <div className="bg-white border border-slate-200 shadow-sm rounded-xl p-6 space-y-4">
            <div className="border-b border-slate-100 pb-3">
              <h4 className="text-sm font-bold text-slate-800">1. Technical Specifications</h4>
            </div>

            <p className="text-xs text-slate-500 leading-relaxed">
              The credit evaluation pipeline is anchored on an extreme gradient boosting model optimized via Bayesian search. This configuration achieves superior tabular representation by successively fitting weak trees on residual errors.
            </p>

            <div className="overflow-x-auto border border-slate-150 rounded-lg">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-150 font-mono text-slate-500">
                    <th className="p-2.5 font-semibold">Specification</th>
                    <th className="p-2.5 font-semibold">Value / Target</th>
                    <th className="p-2.5 font-semibold">Methodology</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-600">
                  <tr>
                    <td className="p-2.5 font-semibold text-slate-800">Model Engine</td>
                    <td className="p-2.5 font-mono">XGBoost 2.0+</td>
                    <td className="p-2.5">Gradient Boosted Decision Trees (GBDT)</td>
                  </tr>
                  <tr>
                    <td className="p-2.5 font-semibold text-slate-800">Objective Function</td>
                    <td className="p-2.5 font-mono">binary:logistic</td>
                    <td className="p-2.5">Probability estimation of default event (y is 0 or 1)</td>
                  </tr>
                  <tr>
                    <td className="p-2.5 font-semibold text-slate-800">Hyperparameter Tuning</td>
                    <td className="p-2.5 font-mono">Optuna TPE</td>
                    <td className="p-2.5">Bayesian Optimization over 15 targeted trials</td>
                  </tr>
                  <tr>
                    <td className="p-2.5 font-semibold text-slate-800">Optimal AUC-ROC</td>
                    <td className="p-2.5 font-mono text-emerald-600 font-bold">0.7720</td>
                    <td className="p-2.5">Cross-validated out-of-sample performance</td>
                  </tr>
                  <tr>
                    <td className="p-2.5 font-semibold text-slate-800">Evaluation Metric</td>
                    <td className="p-2.5 font-mono">binary_logloss</td>
                    <td className="p-2.5">Strict probability alignment and calibration</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="bg-slate-50 border border-slate-150 rounded-lg p-3 space-y-2">
              <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wide">
                Selected Tree Hyperparameters
              </span>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="p-2 bg-white rounded border border-slate-100 text-center">
                  <div className="text-[10px] text-slate-400 font-mono">Learning Rate</div>
                  <div className="font-bold text-slate-800 font-mono">0.030</div>
                </div>
                <div className="p-2 bg-white rounded border border-slate-100 text-center">
                  <div className="text-[10px] text-slate-400 font-mono">Max Depth</div>
                  <div className="font-bold text-slate-800 font-mono">5</div>
                </div>
                <div className="p-2 bg-white rounded border border-slate-100 text-center">
                  <div className="text-[10px] text-slate-400 font-mono">N Estimators</div>
                  <div className="font-bold text-slate-800 font-mono">585</div>
                </div>
                <div className="p-2 bg-white rounded border border-slate-100 text-center">
                  <div className="text-[10px] text-slate-400 font-mono">Subsample</div>
                  <div className="font-bold text-slate-800 font-mono">0.85</div>
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Data Preprocessing Steps */}
          <div className="bg-white border border-slate-200 shadow-sm rounded-xl p-6 space-y-4">
            <div className="border-b border-slate-100 pb-3">
              <h4 className="text-sm font-bold text-slate-800">2. Data Preprocessing & Pipeline Pipeline</h4>
            </div>

            <p className="text-xs text-slate-500 leading-relaxed">
              Standard consumer finance datasets suffer from severe class imbalances (~8% default rate), missing values across third-party sources, and high-cardinality categorical attributes. Our structured preprocessing resolves these issues prior to model ingestion:
            </p>

            <div className="space-y-3.5">
              
              {/* Step 1 */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-6 h-6 rounded-full bg-indigo-50 border border-indigo-200 flex items-center justify-center text-xs font-mono font-bold text-indigo-700">1</div>
                  <div className="w-0.5 flex-1 bg-slate-150 my-1"></div>
                </div>
                <div className="space-y-1">
                  <h5 className="text-xs font-bold text-slate-800">Missing Value Strategy (Imputation)</h5>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Rather than traditional mean/median imputation which can destroy variance, XGBoost natively handles missing values by learning default directional tree branch splits. For supplementary external indexes (e.g., <code className="font-mono bg-slate-50 px-1 py-0.5 rounded text-indigo-700 text-[10px]">EXT_SOURCE_1/2/3</code>), missing identifiers are preserved and guided through learned paths that minimize loss during training.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-6 h-6 rounded-full bg-indigo-50 border border-indigo-200 flex items-center justify-center text-xs font-mono font-bold text-indigo-700">2</div>
                  <div className="w-0.5 flex-1 bg-slate-150 my-1"></div>
                </div>
                <div className="space-y-1">
                  <h5 className="text-xs font-bold text-slate-800">Categorical Attribute Encoding</h5>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Nominal and high-cardinality values such as Occupation and Education types are transformed into target-aligned encodings or binary maps:
                  </p>
                  <ul className="list-disc pl-4 text-xs text-slate-500 space-y-1">
                    <li><strong>One-Hot Encoding:</strong> Applied to low-cardinality values (Gender, Contract Type) to avoid imposing arbitrary ordinal rankings.</li>
                    <li><strong>Target/Frequency Mapping:</strong> Utilized for high-cardinality indicators like occupation categories to prevent dimensional bloat.</li>
                  </ul>
                </div>
              </div>

              {/* Step 3 */}
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-6 h-6 rounded-full bg-indigo-50 border border-indigo-200 flex items-center justify-center text-xs font-mono font-bold text-indigo-700">3</div>
                </div>
                <div className="space-y-1">
                  <h5 className="text-xs font-bold text-slate-800">Feature Engineering Ratios</h5>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Financial health cannot be evaluated by static assets alone. We synthesize relative indicators that capture the actual burden on client cash flow:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 pt-1.5">
                    <div className="p-2 border border-slate-100 rounded bg-slate-50 font-mono text-[11px] text-slate-600">
                      <strong>Annuity to Income (DTI):</strong> <br />
                      <code className="text-indigo-600 font-bold">ANNUITY_TO_INCOME = ANNUITY / INCOME</code>
                    </div>
                    <div className="p-2 border border-slate-100 rounded bg-slate-50 font-mono text-[11px] text-slate-600">
                      <strong>Employment Rate:</strong> <br />
                      <code className="text-indigo-600 font-bold">EMPLOYMENT_RATE = DAYS_EMPLOYED / (AGE_YEARS * 365)</code>
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </div>

        </div>

        {/* Right Column: Rationale, Audit Alignment */}
        <div className="space-y-6">
          
          {/* Section 3: Model Rationale */}
          <div className="bg-white border border-slate-200 shadow-sm rounded-xl p-6 space-y-4">
            <div className="border-b border-slate-100 pb-3">
              <h4 className="text-sm font-bold text-slate-800">3. Model Rationale</h4>
            </div>

            <div className="space-y-3.5 text-xs text-slate-500 leading-relaxed">
              <div>
                <h5 className="font-semibold text-slate-700 mb-1">Why GBDT over Deep Learning?</h5>
                <p>
                  Tabular financial profiles lack spatial or temporal structures found in images and text. GBDT algorithms consistently outperform deep neural networks on tabular datasets due to their ability to build sharp decision boundaries on discontinuous, heavy-tailed numeric values.
                </p>
              </div>

              <div>
                <h5 className="font-semibold text-slate-700 mb-1">Regulatory Explainability (TreeSHAP)</h5>
                <p>
                  Traditional "black-box" predictions are illegal under modern fair-lending frameworks (e.g., FCRA, GDPR). By implementing <strong>TreeSHAP</strong> (a localized coalitional game-theoretic framework), we compute exact Shapley values for each individual candidate. This provides mathematically sound, additive justifications for every score, enabling precise credit-denial reason codes.
                </p>
              </div>

              <div className="p-3 bg-indigo-50/50 border border-indigo-100 rounded-lg text-[11px] space-y-1">
                <span className="font-bold text-indigo-800 block">Shapley Additive Completeness</span>
                <p className="text-indigo-950">
                  The sum of individual feature SHAP contributions is mathematically guaranteed to equal the offset between the specific client prediction and the background cohort's average default risk probability.
                </p>
              </div>
            </div>
          </div>

          {/* Compliance & Risk Management Card */}
          <div className="bg-white border border-slate-200 shadow-sm rounded-xl p-6 space-y-4">
            <div className="border-b border-slate-100 pb-3">
              <h4 className="text-sm font-bold text-slate-800">Compliance & Guardrails</h4>
            </div>

            <div className="space-y-3.5 text-xs text-slate-500 leading-relaxed">
              <p>
                To maintain ethical underwriting and avoid systemic bias:
              </p>
              
              <div className="space-y-3">
                <div>
                  <strong className="text-slate-700 block">Protected Attributes Gated:</strong>
                  <p className="mt-0.5">Sensitive metrics like gender, age group correlations, and demographic indicators are monitored for disparate impact using demographic parity checks.</p>
                </div>

                <div>
                  <strong className="text-slate-700 block">Probability Calibration:</strong>
                  <p className="mt-0.5">Risk models are calibrated so that estimated probabilities map closely to empirical cohort outcomes, ensuring fair premium tiering.</p>
                </div>
              </div>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
