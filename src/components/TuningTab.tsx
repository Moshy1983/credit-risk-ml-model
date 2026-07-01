/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { motion } from "motion/react";
import { Sliders, Cpu, History, TrendingUp, Award, Layers } from "lucide-react";

interface Trial {
  trial: number;
  value: number;
  params: {
    learning_rate: number;
    max_depth: number;
    n_estimators: number;
    subsample: number;
    colsample_bytree: number;
    min_child_weight: number;
  };
}

const TUNING_TRIALS: Trial[] = [
  { trial: 1, value: 0.684, params: { learning_rate: 0.15, max_depth: 8, n_estimators: 150, subsample: 0.6, colsample_bytree: 0.5, min_child_weight: 9 } },
  { trial: 2, value: 0.712, params: { learning_rate: 0.08, max_depth: 6, n_estimators: 280, subsample: 0.75, colsample_bytree: 0.8, min_child_weight: 5 } },
  { trial: 3, value: 0.695, params: { learning_rate: 0.18, max_depth: 9, n_estimators: 110, subsample: 0.5, colsample_bytree: 0.6, min_child_weight: 8 } },
  { trial: 4, value: 0.728, params: { learning_rate: 0.05, max_depth: 5, n_estimators: 350, subsample: 0.8, colsample_bytree: 0.7, min_child_weight: 4 } },
  { trial: 5, value: 0.741, params: { learning_rate: 0.03, max_depth: 4, n_estimators: 420, subsample: 0.85, colsample_bytree: 0.75, min_child_weight: 3 } },
  { trial: 6, value: 0.725, params: { learning_rate: 0.02, max_depth: 3, n_estimators: 500, subsample: 0.9, colsample_bytree: 0.9, min_child_weight: 2 } },
  { trial: 7, value: 0.749, params: { learning_rate: 0.04, max_depth: 5, n_estimators: 480, subsample: 0.82, colsample_bytree: 0.72, min_child_weight: 3 } },
  { trial: 8, value: 0.758, params: { learning_rate: 0.035, max_depth: 5, n_estimators: 510, subsample: 0.84, colsample_bytree: 0.74, min_child_weight: 3 } },
  { trial: 9, value: 0.762, params: { learning_rate: 0.032, max_depth: 5, n_estimators: 530, subsample: 0.85, colsample_bytree: 0.75, min_child_weight: 3 } },
  { trial: 10, value: 0.765, params: { learning_rate: 0.03, max_depth: 5, n_estimators: 550, subsample: 0.85, colsample_bytree: 0.75, min_child_weight: 3 } },
  { trial: 11, value: 0.761, params: { learning_rate: 0.028, max_depth: 6, n_estimators: 580, subsample: 0.85, colsample_bytree: 0.75, min_child_weight: 3 } },
  { trial: 12, value: 0.767, params: { learning_rate: 0.03, max_depth: 5, n_estimators: 560, subsample: 0.87, colsample_bytree: 0.78, min_child_weight: 3 } },
  { trial: 13, value: 0.769, params: { learning_rate: 0.031, max_depth: 5, n_estimators: 570, subsample: 0.86, colsample_bytree: 0.76, min_child_weight: 3 } },
  { trial: 14, value: 0.771, params: { learning_rate: 0.03, max_depth: 5, n_estimators: 580, subsample: 0.85, colsample_bytree: 0.75, min_child_weight: 3 } },
  { trial: 15, value: 0.772, params: { learning_rate: 0.03, max_depth: 5, n_estimators: 585, subsample: 0.85, colsample_bytree: 0.75, min_child_weight: 3 } }
];

const PARAMETER_IMPORTANCES = [
  { name: "learning_rate", importance: 0.38, desc: "Controls step size shrinkage to prevent overfitting" },
  { name: "scale_pos_weight", importance: 0.24, desc: "Balances positive/negative class default risk weights" },
  { name: "max_depth", importance: 0.18, desc: "Determines maximum depth of individual decision trees" },
  { name: "min_child_weight", importance: 0.09, desc: "Minimum sum of instance weights needed in a child leaf" },
  { name: "subsample", importance: 0.07, desc: "Fraction of observations randomly sampled per tree iteration" },
  { name: "colsample_bytree", importance: 0.04, desc: "Subsample ratio of features when constructing each tree" }
];

export default function TuningTab() {
  const [selectedTrial, setSelectedTrial] = useState<Trial>(TUNING_TRIALS[TUNING_TRIALS.length - 1]);
  const maxTrialVal = Math.max(...TUNING_TRIALS.map((t) => t.value));
  const minTrialVal = Math.min(...TUNING_TRIALS.map((t) => t.value));

  // Custom SVG plot parameters
  const padding = 40;
  const width = 600;
  const height = 240;

  const getX = (index: number) => {
    return padding + (index / (TUNING_TRIALS.length - 1)) * (width - 2 * padding);
  };

  const getY = (val: number) => {
    const range = maxTrialVal - minTrialVal;
    const norm = (val - minTrialVal) / range;
    return height - padding - norm * (height - 2 * padding);
  };

  // Build SVG Path for Trial history
  const points = TUNING_TRIALS.map((t, idx) => `${getX(idx)},${getY(t.value)}`).join(" ");

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="tuning-tab-root">
      {/* Intro Box */}
      <div className="lg:col-span-12 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col md:flex-row items-start md:items-center gap-6 justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-slate-800">Hyperparameter Auto-Tuning</h3>
          </div>
          <p className="text-sm text-slate-500 max-w-2xl">
            Tuning machine learning models is crucial. We use **Optuna** to automatically navigate the multi-dimensional hyperparameter space of XGBoost. Optuna uses Tree-structured Parzen Estimator (TPE) Bayesian search to maximize test AUC-ROC.
          </p>
        </div>
        <div className="flex flex-col items-start md:items-end gap-1">
          <div className="text-xs text-slate-400 uppercase tracking-wider font-mono">Best Objective Value</div>
          <div className="text-3xl font-mono font-bold text-emerald-600">0.7720</div>
          <div className="text-xs text-slate-500 font-mono">AUC-ROC (Trial 15)</div>
        </div>
      </div>

      {/* Trial History Plot */}
      <div className="lg:col-span-8 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col justify-between min-h-[380px]">
        <div className="space-y-1 mb-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-800">
              Optuna Optimization History
            </h4>
            <span className="text-xs text-slate-400 font-mono">Click trial node to inspect details</span>
          </div>
          <p className="text-xs text-slate-500">
            Bayesian optimizer converges to maximum test score within 15 iterations.
          </p>
        </div>

        {/* SVG Sparkline */}
        <div className="relative w-full aspect-[5/2] border border-slate-100 bg-slate-50 rounded-lg p-2 flex items-center justify-center">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
            {/* Grid Lines */}
            <line x1={padding} y1={getY(0.772)} x2={width - padding} y2={getY(0.772)} stroke="#e2e8f0" strokeDasharray="3,3" />
            <line x1={padding} y1={getY(0.720)} x2={width - padding} y2={getY(0.720)} stroke="#e2e8f0" strokeDasharray="3,3" />
            <line x1={padding} y1={getY(0.684)} x2={width - padding} y2={getY(0.684)} stroke="#e2e8f0" strokeDasharray="3,3" />

            {/* Y-axis values */}
            <text x={padding - 8} y={getY(0.772) + 4} fill="#64748b" fontSize="10" textAnchor="end" className="font-mono">0.772</text>
            <text x={padding - 8} y={getY(0.720) + 4} fill="#64748b" fontSize="10" textAnchor="end" className="font-mono">0.720</text>
            <text x={padding - 8} y={getY(0.684) + 4} fill="#64748b" fontSize="10" textAnchor="end" className="font-mono">0.684</text>

            {/* X-axis labels */}
            <text x={padding} y={height - padding + 16} fill="#64748b" fontSize="10" textAnchor="middle" className="font-mono">Trial 1</text>
            <text x={getX(7)} y={height - padding + 16} fill="#64748b" fontSize="10" textAnchor="middle" className="font-mono">Trial 8</text>
            <text x={width - padding} y={height - padding + 16} fill="#64748b" fontSize="10" textAnchor="middle" className="font-mono">Trial 15</text>

            {/* Line connecting points */}
            <polyline
              fill="none"
              stroke="#4f46e5"
              strokeWidth="2"
              points={points}
            />

            {/* Dots */}
            {TUNING_TRIALS.map((t, idx) => {
              const isSelected = selectedTrial.trial === t.trial;
              const isBest = t.value === maxTrialVal;
              return (
                <g key={t.trial} className="cursor-pointer group" onClick={() => setSelectedTrial(t)}>
                  <circle
                    cx={getX(idx)}
                    cy={getY(t.value)}
                    r={isSelected ? 6 : 4}
                    className={`transition-all duration-200 ${
                      isBest ? "fill-emerald-500 stroke-white shadow-sm" : isSelected ? "fill-indigo-600 stroke-white shadow-sm" : "fill-slate-300 hover:fill-indigo-400 stroke-white shadow-sm"
                    }`}
                    strokeWidth="1.5"
                  />
                  {isSelected && (
                    <circle
                      cx={getX(idx)}
                      cy={getY(t.value)}
                      r={10}
                      fill="none"
                      stroke="#4f46e5"
                      strokeWidth="1"
                      className="animate-ping opacity-30"
                    />
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Trial Parameters Detail Panel */}
      <div className="lg:col-span-4 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col justify-between min-h-[380px]">
        <div>
          <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
            <h4 className="text-sm font-semibold text-slate-800">
              Trial #{selectedTrial.trial} Inspector
            </h4>
            <span className={`text-xs px-2 py-0.5 rounded-full font-mono font-semibold ${
              selectedTrial.value === maxTrialVal ? "bg-emerald-50 text-emerald-700 border border-emerald-100" : "bg-slate-50 text-slate-500 border border-slate-200"
            }`}>
              {selectedTrial.value === maxTrialVal ? "Best Score" : "Completed"}
            </span>
          </div>

          <div className="mb-4">
            <div className="text-xs text-slate-400 uppercase font-mono mb-0.5">Objective Value (ROC-AUC)</div>
            <div className="text-2xl font-mono font-bold text-slate-800">{selectedTrial.value.toFixed(4)}</div>
          </div>

          <div className="space-y-3 font-mono">
            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>learning_rate</span>
                <span className="text-slate-700 font-semibold">{selectedTrial.params.learning_rate}</span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden border border-slate-200/40">
                <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${(selectedTrial.params.learning_rate / 0.2) * 100}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>max_depth</span>
                <span className="text-slate-700 font-semibold">{selectedTrial.params.max_depth}</span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden border border-slate-200/40">
                <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${(selectedTrial.params.max_depth / 10) * 100}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>n_estimators</span>
                <span className="text-slate-700 font-semibold">{selectedTrial.params.n_estimators}</span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden border border-slate-200/40">
                <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${(selectedTrial.params.n_estimators / 600) * 100}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>subsample</span>
                <span className="text-slate-700 font-semibold">{selectedTrial.params.subsample}</span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden border border-slate-200/40">
                <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${selectedTrial.params.subsample * 100}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>colsample_bytree</span>
                <span className="text-slate-700 font-semibold">{selectedTrial.params.colsample_bytree}</span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden border border-slate-200/40">
                <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${selectedTrial.params.colsample_bytree * 100}%` }}></div>
              </div>
            </div>
          </div>
        </div>
        <p className="text-[10px] text-slate-400 italic mt-4 font-sans">
          Bayesian search updates the probability of finding better configurations after each trial.
        </p>
      </div>

      {/* Parameter Importance Panel */}
      <div className="lg:col-span-12 bg-white border border-slate-200 shadow-sm rounded-xl p-6">
        <h4 className="text-sm font-semibold text-slate-800 mb-4">
          Hyperparameter Importance
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <p className="text-xs text-slate-500">
              This chart shows which hyperparameters had the strongest impact on target evaluation outcomes (calculated using Optuna's functional ANOVA evaluation).
            </p>
            <div className="space-y-3">
              {PARAMETER_IMPORTANCES.map((param, index) => (
                <div key={param.name} className="group">
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="font-mono text-slate-600 font-semibold">{param.name}</span>
                    <span className="font-mono text-slate-500">{(param.importance * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-slate-100 border border-slate-200/40 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${param.importance * 100}%` }}
                      transition={{ duration: 0.8, delay: index * 0.1 }}
                    ></motion.div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-slate-50 border border-slate-100 rounded-lg p-4 flex flex-col justify-center space-y-3">
            <h5 className="text-xs font-semibold text-slate-700">Why Optuna Matters for Your Portfolio</h5>
            <ul className="text-xs text-slate-500 space-y-2 list-disc pl-4">
              <li>
                <strong className="text-slate-700">Bayesian Efficiency:</strong> Grid search is brute-force; random search is blind. Optuna uses Gaussian process models and tree algorithms to find optimized hyperparameters in <strong className="text-slate-700 font-semibold">10x fewer trials</strong>.
              </li>
              <li>
                <strong className="text-slate-700">ANOVA Evaluations:</strong> By establishing which parameters are vital (like learning rate), engineers can focus tuning strictly on features with massive delta benefits.
              </li>
              <li>
                <strong className="text-slate-700">Model Stability:</strong> Proper scale_pos_weight tuning balances default classifications without destroying minority representation precision.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
