/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { motion } from "motion/react";
import { Info, HelpCircle, GitPullRequest, Award, Zap } from "lucide-react";

interface FeatureImportance {
  name: string;
  importance: number; // absolute SHAP value
  importancePct: number;
  direction: "positive" | "negative" | "mixed";
  description: string;
}

const SHAP_IMPORTANCES: FeatureImportance[] = [
  { name: "EXT_SOURCE_3", importance: 0.324, importancePct: 100, direction: "negative", description: "External credit bureau scoring 3 (Higher score significantly decreases risk)" },
  { name: "EXT_SOURCE_2", importance: 0.281, importancePct: 86.7, direction: "negative", description: "External credit bureau scoring 2 (Higher score significantly decreases risk)" },
  { name: "EXT_SOURCE_1", importance: 0.194, importancePct: 59.8, direction: "negative", description: "External credit bureau scoring 1 (Higher score significantly decreases risk)" },
  { name: "ANNUITY_TO_INCOME", importance: 0.165, importancePct: 50.9, direction: "positive", description: "Yearly debt payment divided by total income (Higher ratio increases default risk)" },
  { name: "AGE_YEARS", importance: 0.138, importancePct: 42.5, direction: "negative", description: "Client's age (Younger clients generally demonstrate higher default risk)" },
  { name: "EMPLOYMENT_RATE", importance: 0.112, importancePct: 34.5, direction: "negative", description: "Ratio of years employed vs age (Higher relative job stability decreases risk)" },
  { name: "AMT_CREDIT", importance: 0.088, importancePct: 27.1, direction: "mixed", description: "Total loan credit amount (Higher credit causes higher payments but correlates with income)" },
  { name: "DAYS_EMPLOYED", importance: 0.075, importancePct: 23.1, direction: "negative", description: "Total continuous years of employment (Longer history decreases risk)" },
  { name: "NAME_EDUCATION_TYPE", importance: 0.062, importancePct: 19.1, direction: "negative", description: "Higher academic degree type (Academic degree significantly decreases default rates)" },
  { name: "AMT_GOODS_PRICE", importance: 0.048, importancePct: 14.8, direction: "negative", description: "Price of the consumer goods being financed" }
];

// Beeswarm points simulator
// Each feature has ~15 simulated dots with coordinates and color (feature values)
interface BeeswarmPoint {
  x: number; // SHAP value (-0.5 to +0.5)
  yOffset: number; // cluster spread
  isHighValue: boolean; // Red (true) or Blue (false)
}

const generateBeeswarmPoints = (direction: "positive" | "negative" | "mixed"): BeeswarmPoint[] => {
  const points: BeeswarmPoint[] = [];
  const count = 15;
  
  for (let i = 0; i < count; i++) {
    const featureValNorm = i / (count - 1); // 0 to 1
    const isHighValue = featureValNorm > 0.45;
    
    // Spread calculation
    let shapValue = 0;
    if (direction === "positive") {
      // High feature value (Red) -> positive SHAP (increases risk)
      shapValue = (featureValNorm - 0.5) * 0.8 + npRandom(-0.08, 0.08);
    } else if (direction === "negative") {
      // High feature value (Red) -> negative SHAP (decreases risk)
      shapValue = (0.5 - featureValNorm) * 0.8 + npRandom(-0.08, 0.08);
    } else {
      // Mixed
      shapValue = (featureValNorm - 0.5) * 0.3 + npRandom(-0.25, 0.25);
    }
    
    // Create density effect on yOffset (cluster points closer together if shapValue is near mean)
    const distFromCenter = Math.abs(shapValue);
    const yOffset = npRandom(-12, 12) * (1.2 - distFromCenter);
    
    points.push({ x: shapValue, yOffset, isHighValue });
  }
  
  return points.sort((a, b) => a.x - b.x);
};

const npRandom = (min: number, max: number) => {
  return Math.random() * (max - min) + min;
};

export default function ExplainabilityTab() {
  const [hoveredFeature, setHoveredFeature] = useState<FeatureImportance | null>(null);

  // Layout parameters for Beeswarm plot
  const w = 550;
  const h = 420;
  const paddingX = 140; // Space for feature labels
  const paddingY = 30;

  const getBeeswarmX = (shapVal: number) => {
    // Maps -0.6 to +0.6 SHAP to coordinate space
    const range = 1.2;
    const norm = (shapVal + 0.6) / range;
    return paddingX + norm * (w - paddingX - 40);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="explainability-tab-root">
      
      {/* Overview Block */}
      <div className="lg:col-span-12 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col md:flex-row items-start md:items-center gap-6 justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-slate-800">Model Explainability (SHAP Platform)</h3>
          </div>
          <p className="text-sm text-slate-500 max-w-2xl">
            In credit risk models, explanations are legal mandates, not just optional insights. We use **SHAP (SHapley Additive exPlanations)** to break down predictions into individual feature contributions. SHAP is mathematically grounded in game theory and guarantees local and global consistency.
          </p>
        </div>
      </div>

      {/* Global Feature Importance Bar Chart */}
      <div className="lg:col-span-6 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col justify-between">
        <div>
          <h4 className="text-sm font-semibold text-slate-800 mb-2">
            Global Feature Importance (Mean Absolute SHAP)
          </h4>
          <p className="text-xs text-slate-500 mb-4">
            Shows which customer attributes had the absolute highest average impact on loan risk predictions. Hover over any bar to inspect its definition.
          </p>
          
          <div className="space-y-4">
            {SHAP_IMPORTANCES.map((item) => (
              <div
                key={item.name}
                className="group cursor-pointer"
                onMouseEnter={() => setHoveredFeature(item)}
                onMouseLeave={() => setHoveredFeature(null)}
              >
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className={`font-mono transition-colors ${hoveredFeature?.name === item.name ? "text-indigo-600 font-bold" : "text-slate-600"}`}>{item.name}</span>
                  <span className="font-mono text-slate-400 text-[10px]">{item.importance.toFixed(3)}</span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden border border-slate-200/40 relative">
                  <motion.div
                    className={`h-full rounded-full ${
                      item.direction === "negative" 
                        ? "bg-gradient-to-r from-indigo-600 to-indigo-400" 
                        : item.direction === "positive" 
                        ? "bg-gradient-to-r from-rose-600 to-rose-400"
                        : "bg-gradient-to-r from-purple-600 to-purple-400"
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${item.importancePct}%` }}
                    transition={{ duration: 0.8 }}
                  ></motion.div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Feature Definition Box */}
        <div className="mt-6 p-3 bg-slate-50 border border-slate-100 rounded-lg min-h-[76px] flex items-start">
          <div className="text-xs">
            {hoveredFeature ? (
              <>
                <div className="font-semibold font-mono text-slate-700">{hoveredFeature.name}</div>
                <p className="text-slate-500 mt-0.5">{hoveredFeature.description}</p>
              </>
            ) : (
              <p className="text-slate-400 italic">Hover over any feature bar above to view detailed credit risk definitions and correlates.</p>
            )}
          </div>
        </div>
      </div>

      {/* SHAP Beeswarm Summary Plot */}
      <div className="lg:col-span-6 bg-white border border-slate-200 shadow-sm rounded-xl p-6 flex flex-col justify-between">
        <div>
          <h4 className="text-sm font-semibold text-slate-800 mb-1">
            SHAP Beeswarm Summary Plot
          </h4>
          <p className="text-xs text-slate-500 mb-4">
            Shows how feature values impact default risk. Every dot is a customer.
          </p>
        </div>

        {/* Beeswarm SVG Canvas */}
        <div className="relative w-full aspect-[11/9] border border-slate-100 bg-slate-50 rounded-lg p-2 overflow-visible">
          {/* Legend */}
          <div className="absolute top-2 right-4 flex items-center gap-3 text-[9px] font-mono">
            <div className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 bg-indigo-500 rounded-full"></span>
              <span className="text-slate-500">Low Feature Value</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 bg-rose-500 rounded-full"></span>
              <span className="text-slate-500">High Feature Value</span>
            </div>
          </div>

          <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full overflow-visible">
            {/* Central Vertical Axis Line (SHAP = 0) */}
            <line x1={getBeeswarmX(0)} y1={paddingY} x2={getBeeswarmX(0)} y2={h - paddingY} stroke="#cbd5e1" strokeDasharray="3,3" />
            <text x={getBeeswarmX(0)} y={h - 6} fill="#94a3b8" fontSize="8" textAnchor="middle" className="font-mono">SHAP = 0.0 (No Effect)</text>
            <text x={getBeeswarmX(-0.4)} y={h - 6} fill="#4f46e5" fontSize="8" textAnchor="middle" className="font-mono">← Decreases Risk</text>
            <text x={getBeeswarmX(0.4)} y={h - 6} fill="#e11d48" fontSize="8" textAnchor="middle" className="font-mono">Increases Risk →</text>

            {/* Render Features Rows */}
            {SHAP_IMPORTANCES.map((item, idx) => {
              const rowY = paddingY + idx * ((h - 2 * paddingY - 10) / (SHAP_IMPORTANCES.length - 1)) + 5;
              const points = generateBeeswarmPoints(item.direction);
              
              return (
                <g key={item.name}>
                  {/* Row Line */}
                  <line x1={paddingX} y1={rowY} x2={w - 20} y2={rowY} stroke="#f1f5f9" strokeWidth="0.5" />
                  
                  {/* Feature Label */}
                  <text
                    x={paddingX - 12}
                    y={rowY + 3}
                    fill="#64748b"
                    fontSize="9"
                    textAnchor="end"
                    className="font-mono hover:fill-indigo-600 cursor-pointer transition-all"
                    onMouseEnter={() => setHoveredFeature(item)}
                    onMouseLeave={() => setHoveredFeature(null)}
                  >
                    {item.name}
                  </text>

                  {/* Scatter Dots */}
                  {points.map((pt, pIdx) => {
                    const cx = getBeeswarmX(pt.x);
                    const cy = rowY + pt.yOffset;
                    return (
                      <circle
                        key={pIdx}
                        cx={cx}
                        cy={cy}
                        r={3}
                        className={`transition-all duration-300 ${
                          pt.isHighValue 
                            ? "fill-rose-500 hover:fill-rose-400" 
                            : "fill-indigo-500 hover:fill-indigo-400"
                        }`}
                        style={{ opacity: 0.75 }}
                      />
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </div>

        <p className="text-[10px] text-slate-400 italic mt-4 font-sans">
          Example: High scores on EXT_SOURCE_3 (Red dots) push values to the left (negative SHAP values), indicating a strong protective factor that decreases default probability.
        </p>
      </div>
    </div>
  );
}
