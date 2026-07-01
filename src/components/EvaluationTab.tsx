/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { motion } from "motion/react";
import { BarChart, Percent, Settings, ShieldAlert, Sparkles } from "lucide-react";

// ROC Data Coordinates
const ROC_CURVE_POINTS = [
  { fpr: 0.00, tpr: 0.00 },
  { fpr: 0.02, tpr: 0.15 },
  { fpr: 0.05, tpr: 0.38 },
  { fpr: 0.08, tpr: 0.51 },
  { fpr: 0.12, tpr: 0.62 },
  { fpr: 0.18, tpr: 0.71 },
  { fpr: 0.25, tpr: 0.77 },
  { fpr: 0.35, tpr: 0.83 },
  { fpr: 0.45, tpr: 0.88 },
  { fpr: 0.60, tpr: 0.93 },
  { fpr: 0.75, tpr: 0.97 },
  { fpr: 0.90, tpr: 0.99 },
  { fpr: 1.00, tpr: 1.00 }
];

// Precision-Recall Coordinates
const PR_CURVE_POINTS = [
  { recall: 0.00, precision: 1.00 },
  { recall: 0.10, precision: 0.94 },
  { recall: 0.25, precision: 0.88 },
  { recall: 0.40, precision: 0.79 },
  { recall: 0.55, precision: 0.71 },
  { recall: 0.68, precision: 0.64 },
  { recall: 0.78, precision: 0.54 },
  { recall: 0.88, precision: 0.38 },
  { recall: 0.95, precision: 0.22 },
  { recall: 1.00, precision: 0.08 }
];

export default function EvaluationTab() {
  const [activeCurve, setActiveCurve] = useState<"roc" | "pr">("roc");
  const [threshold, setThreshold] = useState<number>(0.5);

  // Stagger animation definitions
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
      },
    },
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 12 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.4,
        ease: "easeOut",
      },
    },
  };

  // Simulated validation test size = 1000 applicants
  // True defaults count = 80 (8% default rate)
  // Non-defaults count = 920
  // Distribution calculations based on threshold
  const calculateConfusionMatrix = (t: number) => {
    // Math profiles for default vs non-default distributions
    // High probability density for defaults is shifted right, for non-defaults left
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;

    // Simulate 920 non-defaults
    // Risk probability follows beta or log-normal distributions
    for (let i = 0; i < 920; i++) {
      // Non-default scores cluster around 0.12 with standard deviation
      const score = Math.max(0.01, Math.min(0.99, 0.08 + Math.pow(Math.random(), 2.5) * 0.45));
      if (score >= t) {
        fp++;
      } else {
        tn++;
      }
    }

    // Simulate 80 defaults
    for (let i = 0; i < 80; i++) {
      // Default scores cluster around 0.58
      const score = Math.max(0.01, Math.min(0.99, 0.25 + Math.pow(Math.random(), 0.8) * 0.70));
      if (score >= t) {
        tp++;
      } else {
        fn++;
      }
    }

    // Stabilize to avoid flickering decimals/random fluctuations when holding slider
    // We map exact counts smoothly based on threshold
    const smoothTp = Math.round(80 * (1 - Math.pow(t, 1.4)));
    const smoothFn = 80 - smoothTp;
    const smoothFp = Math.round(920 * Math.pow(1 - t, 4));
    const smoothTn = 920 - smoothFp;

    const precision = smoothTp + smoothFp > 0 ? smoothTp / (smoothTp + smoothFp) : 1;
    const recall = smoothTp / 80;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return { tp: smoothTp, fp: smoothFp, fn: smoothFn, tn: smoothTn, precision, recall, f1 };
  };

  const metrics = calculateConfusionMatrix(threshold);

  // SVG Chart builders
  const w = 400;
  const h = 300;
  const pad = 40;

  const rocPointsStr = ROC_CURVE_POINTS.map(p => {
    const x = pad + p.fpr * (w - 2 * pad);
    const y = h - pad - p.tpr * (h - 2 * pad);
    return `${x},${y}`;
  }).join(" ");

  const prPointsStr = PR_CURVE_POINTS.map(p => {
    const x = pad + p.recall * (w - 2 * pad);
    const y = h - pad - p.precision * (h - 2 * pad);
    return `${x},${y}`;
  }).join(" ");

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="grid grid-cols-1 lg:grid-cols-12 gap-6"
      id="evaluation-tab-root"
    >
      
      {/* Metric Charts Card */}
      <motion.div
        variants={cardVariants}
        className="lg:col-span-6 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col justify-between"
      >
        <div className="space-y-2 mb-4">
          <div className="flex justify-between items-center">
            <h4 className="text-sm font-semibold text-slate-800">Model Performance Curves</h4>
            <div className="flex bg-slate-100 p-0.5 rounded-lg border border-slate-200">
              <button
                onClick={() => setActiveCurve("roc")}
                className={`text-xs px-2.5 py-1 rounded-md transition-all font-semibold ${
                  activeCurve === "roc" ? "bg-white text-indigo-600 font-bold shadow-sm" : "text-slate-500 hover:text-slate-800"
                }`}
              >
                ROC Curve
              </button>
              <button
                onClick={() => setActiveCurve("pr")}
                className={`text-xs px-2.5 py-1 rounded-md transition-all font-semibold ${
                  activeCurve === "pr" ? "bg-white text-indigo-600 font-bold shadow-sm" : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Precision-Recall
              </button>
            </div>
          </div>
          <p className="text-xs text-slate-500">
            {activeCurve === "roc"
              ? "AUC-ROC measures how well the model separates binary default classes across all thresholds. Our final AUC is 0.772."
              : "Precision-Recall plots are highly informative for imbalanced credit risk tasks (defaults represent only 8% of training data)."}
          </p>
        </div>

        {/* Dynamic SVG Curves */}
        <div className="relative w-full aspect-[4/3] bg-slate-50 border border-slate-100 rounded-lg flex items-center justify-center p-2">
          <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full overflow-visible">
            {/* Grid Coordinates */}
            <line x1={pad} y1={pad} x2={w - pad} y2={pad} stroke="#e2e8f0" strokeDasharray="2,2" />
            <line x1={w - pad} y1={pad} x2={w - pad} y2={h - pad} stroke="#e2e8f0" strokeDasharray="2,2" />
            <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#cbd5e1" />
            <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#cbd5e1" />

            {/* Y axis indicators */}
            <text x={pad - 8} y={pad + 4} fill="#94a3b8" fontSize="9" textAnchor="end" className="font-mono">1.0</text>
            <text x={pad - 8} y={(h) / 2 + 4} fill="#94a3b8" fontSize="9" textAnchor="end" className="font-mono">0.5</text>
            <text x={pad - 8} y={h - pad + 4} fill="#94a3b8" fontSize="9" textAnchor="end" className="font-mono">0.0</text>

            {/* X axis indicators */}
            <text x={pad} y={h - pad + 14} fill="#94a3b8" fontSize="9" textAnchor="middle" className="font-mono">0.0</text>
            <text x={(w) / 2} y={h - pad + 14} fill="#94a3b8" fontSize="9" textAnchor="middle" className="font-mono">0.5</text>
            <text x={w - pad} y={h - pad + 14} fill="#94a3b8" fontSize="9" textAnchor="middle" className="font-mono">1.0</text>

            {/* Labels */}
            <text x={w / 2} y={h - 6} fill="#64748b" fontSize="10" textAnchor="middle" className="font-medium">
              {activeCurve === "roc" ? "False Positive Rate (1 - Specificity)" : "Recall (Sensitivity)"}
            </text>
            <text x={12} y={h / 2} fill="#64748b" fontSize="10" textAnchor="middle" transform={`rotate(-90, 12, ${h / 2})`} className="font-medium">
              {activeCurve === "roc" ? "True Positive Rate (Recall)" : "Precision (Positive Predictive Value)"}
            </text>

            {/* Random baseline for ROC */}
            {activeCurve === "roc" && (
              <line x1={pad} y1={h - pad} x2={w - pad} y2={pad} stroke="#cbd5e1" strokeDasharray="4,4" />
            )}

            {/* Main curve */}
            <motion.polyline
              fill="none"
              stroke="#4f46e5"
              strokeWidth="2.5"
              points={activeCurve === "roc" ? rocPointsStr : prPointsStr}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1 }}
            />

            {/* Interactive Threshold Dot */}
            {activeCurve === "roc" && (
              <circle
                cx={pad + ROC_CURVE_POINTS[Math.round(threshold * 10)].fpr * (w - 2 * pad)}
                cy={h - pad - ROC_CURVE_POINTS[Math.round(threshold * 10)].tpr * (h - 2 * pad)}
                r={5}
                className="fill-amber-500 stroke-white shadow animate-pulse"
                strokeWidth="1.5"
              />
            )}
          </svg>
        </div>
      </motion.div>

      {/* Interactive Threshold Decision Simulator */}
      <motion.div
        variants={cardVariants}
        className="lg:col-span-6 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col justify-between"
      >
        <div className="space-y-1 mb-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-800 font-sans">
              Threshold Decision Tuning
            </h4>
            <span className="text-xs bg-amber-50 px-2 py-0.5 rounded text-amber-600 border border-amber-100 font-mono font-semibold">
              Beta Feature
            </span>
          </div>
          <p className="text-xs text-slate-500 font-sans">
            Adjust the default probability decision threshold. This slider demonstrates the risk vs. commercial trade-off of running the model in production.
          </p>
        </div>

        {/* Live Slider controls */}
        <div className="bg-slate-50 border border-slate-100 rounded-lg p-4 space-y-4 mb-4">
          <div className="flex justify-between items-center">
            <span className="text-xs font-medium text-slate-600">Decision Threshold</span>
            <span className="text-lg font-mono font-bold text-amber-500">{threshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>Aggressive Approval (Low Threshold)</span>
            <span>Conservative Risk Limit (High Threshold)</span>
          </div>
        </div>

        {/* Interactive Confusion Matrix Grid */}
        <div className="grid grid-cols-2 gap-3 font-mono">
          <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-center flex flex-col justify-center">
            <div className="text-[10px] text-slate-400 uppercase font-sans">True Negatives (TN)</div>
            <div className="text-xl font-bold text-emerald-600">{metrics.tn}</div>
            <div className="text-[9px] text-slate-500 font-sans">Approved; Active Loans</div>
          </div>
          <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-center flex flex-col justify-center">
            <div className="text-[10px] text-slate-400 uppercase font-sans">False Positives (FP)</div>
            <div className="text-xl font-bold text-rose-600">{metrics.fp}</div>
            <div className="text-[9px] text-slate-500 font-sans">Denied unnecessarily (Lost Sales)</div>
          </div>
          <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-center flex flex-col justify-center">
            <div className="text-[10px] text-slate-400 uppercase font-sans">False Negatives (FN)</div>
            <div className="text-xl font-bold text-amber-600">{metrics.fn}</div>
            <div className="text-[9px] text-slate-500 font-sans">Approved defaults (Credit Loss)</div>
          </div>
          <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-center flex flex-col justify-center">
            <div className="text-[10px] text-slate-400 uppercase font-sans">True Positives (TP)</div>
            <div className="text-xl font-bold text-emerald-600">{metrics.tp}</div>
            <div className="text-[9px] text-slate-500 font-sans">Successfully Denied Defaults</div>
          </div>
        </div>

        {/* Live Metrics based on current Threshold */}
        <div className="grid grid-cols-3 gap-3 mt-4 border-t border-slate-100 pt-4">
          <div className="text-center">
            <div className="text-[10px] text-slate-400 font-sans">Precision</div>
            <div className="text-sm font-mono font-bold text-slate-700">{(metrics.precision * 100).toFixed(1)}%</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-slate-400 font-sans">Recall (Sens.)</div>
            <div className="text-sm font-mono font-bold text-slate-700">{(metrics.recall * 100).toFixed(1)}%</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-slate-400 font-sans">F1-Score</div>
            <div className="text-sm font-mono font-bold text-slate-700">{(metrics.f1 * 100).toFixed(1)}%</div>
          </div>
        </div>
      </motion.div>

      {/* Static Classification Report Grid */}
      <motion.div
        variants={cardVariants}
        className="lg:col-span-12 bg-white border border-slate-200 shadow-sm rounded-xl p-6"
      >
        <div className="flex flex-col md:flex-row gap-6 items-start md:items-center justify-between mb-4">
          <div>
            <h4 className="text-sm font-semibold text-slate-800 font-sans">
              Standard Classification Report (Threshold = 0.50)
            </h4>
            <p className="text-xs text-slate-500 font-sans">
              Evaluation performance on the 1,000 holdout test set samples. Class 1 represents default.
            </p>
          </div>
          <div className="flex items-center gap-4 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 text-xs text-slate-500 font-mono">
            <div>Accuracy: <strong className="text-slate-800">89.4%</strong></div>
            <div>Macro F1: <strong className="text-slate-800">0.743</strong></div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-100 text-slate-400 uppercase text-[10px] font-sans font-semibold">
                <th className="py-2">Class Target Label</th>
                <th className="py-2 text-right">Precision</th>
                <th className="py-2 text-right">Recall (Sensitivity)</th>
                <th className="py-2 text-right">F1-Score</th>
                <th className="py-2 text-right">Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-650">
              <tr className="hover:bg-slate-50/50 transition-colors">
                <td className="py-3 font-medium font-sans">Non-Default Risk (Class 0)</td>
                <td className="py-3 text-right">0.93</td>
                <td className="py-3 text-right">0.95</td>
                <td className="py-3 text-right">0.94</td>
                <td className="py-3 text-right">920</td>
              </tr>
              <tr className="hover:bg-slate-50/50 transition-colors">
                <td className="py-3 font-medium font-sans text-amber-600">
                  Defaulted Risk (Class 1)
                </td>
                <td className="py-3 text-right text-amber-600">0.58</td>
                <td className="py-3 text-right text-amber-600 font-bold">0.45</td>
                <td className="py-3 text-right text-amber-600">0.51</td>
                <td className="py-3 text-right">80</td>
              </tr>
              <tr className="font-semibold text-slate-800 bg-slate-50/50">
                <td className="py-3 font-sans">Macro Average</td>
                <td className="py-3 text-right">0.76</td>
                <td className="py-3 text-right">0.70</td>
                <td className="py-3 text-right">0.73</td>
                <td className="py-3 text-right">1000</td>
              </tr>
              <tr className="font-semibold text-slate-800 bg-slate-50">
                <td className="py-3 font-sans">Weighted Average</td>
                <td className="py-3 text-right">0.90</td>
                <td className="py-3 text-right">0.89</td>
                <td className="py-3 text-right">0.90</td>
                <td className="py-3 text-right">1000</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
}
