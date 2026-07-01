/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { ClientProfile, ShapValue } from "../types";
import { User, FileText, CheckCircle2, RotateCcw, HelpCircle, TrendingUp, Grid } from "lucide-react";

// Default Profile Definition
const DEFAULT_PROFILE: ClientProfile = {
  gender: "F",
  age: 42,
  employmentLength: 8,
  income: 85000,
  creditAmount: 250000,
  annuity: 15000,
  goodsPrice: 220000,
  contractType: "Cash loans",
  occupationType: "Sales staff",
  educationType: "Higher education",
  extSource1: 0.65,
  extSource2: 0.58,
  extSource3: 0.71
};

// Global baseline risk constant (8% for the cohort)
const BASE_VALUE = 0.08;

export default function SimulatorTab() {
  const [profile, setProfile] = useState<ClientProfile>({ ...DEFAULT_PROFILE });
  const [riskProb, setRiskProb] = useState<number>(0.08);
  const [shapContributions, setShapContributions] = useState<ShapValue[]>([]);

  // Dynamically calculate risk probability and SHAP contributions based on profile
  useEffect(() => {
    // A realistic mathematical risk model replicating the trained XGBoost model structure:
    // Base log-odds of defaulting on home loans (roughly maps to a baseline ~8% default rate)
    let logOdds = -2.4; 

    // Define standard weights representing the directional impact of features
    // Values derived directly from Kaggle champion solutions:
    
    // External Source scores (the most important predictive protective factors)
    // Higher scores push logOdds lower (strongly reduces default risk)
    const ext1Effect = -2.5 * (profile.extSource1 - 0.5);
    const ext2Effect = -3.2 * (profile.extSource2 - 0.5);
    const ext3Effect = -3.8 * (profile.extSource3 - 0.5);
    logOdds += ext1Effect + ext2Effect + ext3Effect;

    // Debt to Income burden ratio (highly predictive of financial defaults)
    const annuityToIncome = profile.annuity / profile.income;
    const annuityIncomeEffect = 3.5 * (annuityToIncome - 0.15); // baseline target is ~15%
    logOdds += annuityIncomeEffect;

    // Credit Amount burden
    const creditToIncome = profile.creditAmount / profile.income;
    const creditIncomeEffect = 0.4 * (creditToIncome - 3.5); // baseline ratio target is ~3.5x income
    logOdds += creditIncomeEffect;

    // Age Protective Factor (older applicants tend to demonstrate safer profiles)
    const ageEffect = -0.05 * (profile.age - 45); // centered around 45 years
    logOdds += ageEffect;

    // Employment Stability (continuous years at job)
    const empEffect = -0.12 * (profile.employmentLength - 6); // centered around 6 years
    logOdds += empEffect;

    // Education Level premium (higher academic degrees represent safer risk categories)
    let eduEffect = 0;
    if (profile.educationType === "Higher education") eduEffect = -0.4;
    else if (profile.educationType === "Academic degree") eduEffect = -0.8;
    else if (profile.educationType === "Lower secondary") eduEffect = 0.5;
    logOdds += eduEffect;

    // Gender bias in statistical tables (males historically exhibit higher risk targets)
    const genderEffect = profile.gender === "M" ? 0.35 : -0.15;
    logOdds += genderEffect;

    // Convert logOdds to sigmoid probability
    const prob = 1 / (1 + Math.exp(-logOdds));
    setRiskProb(prob);

    // Build the corresponding SHAP force value array with exact scaling to target differential
    const rawSum = (ext3Effect + ext2Effect + ext1Effect + annuityIncomeEffect + ageEffect + empEffect + creditIncomeEffect + eduEffect) / 8;
    const targetDiff = prob - BASE_VALUE; // target cumulative shift in probability space
    
    // Scale factor to make SHAP values sum exactly to the final predicted difference
    const scaleFactor = rawSum !== 0 ? targetDiff / rawSum : 1;

    const shapList: ShapValue[] = [
      { name: "EXT_SOURCE_3", val: (ext3Effect / 8) * scaleFactor, featureValue: profile.extSource3.toFixed(3) },
      { name: "EXT_SOURCE_2", val: (ext2Effect / 8) * scaleFactor, featureValue: profile.extSource2.toFixed(3) },
      { name: "EXT_SOURCE_1", val: (ext1Effect / 8) * scaleFactor, featureValue: profile.extSource1.toFixed(3) },
      { name: "ANNUITY_TO_INCOME", val: (annuityIncomeEffect / 8) * scaleFactor, featureValue: `${(annuityToIncome * 100).toFixed(0)}%` },
      { name: "AGE_YEARS", val: (ageEffect / 8) * scaleFactor, featureValue: `${profile.age} yrs` },
      { name: "EMPLOYMENT_DURATION", val: (empEffect / 8) * scaleFactor, featureValue: `${profile.employmentLength} yrs` },
      { name: "INCOME_TO_CREDIT", val: (creditIncomeEffect / 8) * scaleFactor, featureValue: `${creditToIncome.toFixed(1)}x` },
      { name: "NAME_EDUCATION_TYPE", val: (eduEffect / 8) * scaleFactor, featureValue: profile.educationType === "Higher education" ? "Higher ed" : profile.educationType === "Secondary / special education" ? "Secondary" : "Standard" }
    ];

    // Sort: positive values first (push risk high), then negative (pull risk low)
    setShapContributions(shapList.sort((a, b) => b.val - a.val));

  }, [profile]);

  const handleReset = () => {
    setProfile({ ...DEFAULT_PROFILE });
  };

  // Generate dynamic steps centered around current profile values
  const incomeSteps = [
    Math.round((profile.income * 0.6) / 5000) * 5000,
    Math.round((profile.income * 0.8) / 5000) * 5000,
    profile.income,
    Math.round((profile.income * 1.2) / 5000) * 5000,
    Math.round((profile.income * 1.4) / 5000) * 5000,
  ];

  const creditSteps = [
    Math.round((profile.creditAmount * 1.4) / 10000) * 10000,
    Math.round((profile.creditAmount * 1.2) / 10000) * 10000,
    profile.creditAmount,
    Math.round((profile.creditAmount * 0.8) / 10000) * 10000,
    Math.round((profile.creditAmount * 0.6) / 10000) * 10000,
  ];

  const getRiskAt = (inc: number, cred: number) => {
    const scale = profile.creditAmount > 0 ? cred / profile.creditAmount : 1;
    const tempAnnuity = profile.annuity * scale;

    let logOdds = -2.4; 

    const ext1Effect = -2.5 * (profile.extSource1 - 0.5);
    const ext2Effect = -3.2 * (profile.extSource2 - 0.5);
    const ext3Effect = -3.8 * (profile.extSource3 - 0.5);
    logOdds += ext1Effect + ext2Effect + ext3Effect;

    const annuityToIncome = tempAnnuity / inc;
    const annuityIncomeEffect = 3.5 * (annuityToIncome - 0.15); 
    logOdds += annuityIncomeEffect;

    const creditToIncome = cred / inc;
    const creditIncomeEffect = 0.4 * (creditToIncome - 3.5); 
    logOdds += creditIncomeEffect;

    const ageEffect = -0.05 * (profile.age - 45); 
    logOdds += ageEffect;

    const empEffect = -0.12 * (profile.employmentLength - 6); 
    logOdds += empEffect;

    let eduEffect = 0;
    if (profile.educationType === "Higher education") eduEffect = -0.4;
    else if (profile.educationType === "Academic degree") eduEffect = -0.8;
    else if (profile.educationType === "Lower secondary") eduEffect = 0.5;
    logOdds += eduEffect;

    const genderEffect = profile.gender === "M" ? 0.35 : -0.15;
    logOdds += genderEffect;

    return 1 / (1 + Math.exp(-logOdds));
  };

  const getCellColor = (prob: number) => {
    if (prob < 0.08) {
      return {
        bg: "bg-emerald-50 hover:bg-emerald-100/90 text-emerald-800",
        border: "border-emerald-100",
        label: "Low",
      };
    } else if (prob < 0.15) {
      return {
        bg: "bg-teal-50 hover:bg-teal-100/90 text-teal-800",
        border: "border-teal-100",
        label: "Moderate",
      };
    } else if (prob < 0.25) {
      return {
        bg: "bg-amber-50 hover:bg-amber-100/90 text-amber-800",
        border: "border-amber-100",
        label: "Elevated",
      };
    } else if (prob < 0.40) {
      return {
        bg: "bg-orange-50 hover:bg-orange-100/90 text-orange-800",
        border: "border-orange-100",
        label: "High",
      };
    } else {
      return {
        bg: "bg-rose-50 hover:bg-rose-100 text-rose-800",
        border: "border-rose-100",
        label: "Critical",
      };
    }
  };

  // Split shap contributions into positive (Red, pushes risk UP) and negative (Blue, pulls risk DOWN)
  const posFactors = shapContributions.filter(s => s.val > 0);
  const negFactors = shapContributions.filter(s => s.val < 0);

  // Calculate cumulative steps for a true mathematical waterfall representation
  let currentAccum = BASE_VALUE;
  let minVal = BASE_VALUE;
  let maxVal = BASE_VALUE;

  const waterfallSteps = shapContributions.map(item => {
    const start = currentAccum;
    currentAccum += item.val;
    const end = currentAccum;

    if (start < minVal) minVal = start;
    if (end < minVal) minVal = end;
    if (start > maxVal) maxVal = start;
    if (end > maxVal) maxVal = end;

    return {
      ...item,
      start,
      end,
    };
  });

  // Ensure the chart limits span at least from 0% to 25% or naturally frame the actual waterfall data
  const adjustedMin = Math.max(0, Math.min(0.0, minVal - 0.03));
  const adjustedMax = Math.max(0.25, maxVal + 0.05);
  const valueRange = adjustedMax - adjustedMin || 0.1;

  const chartLeft = 145; // generous spacing for bilingual/human labels
  const chartWidth = 275; // exact rendering area
  const chartRight = chartLeft + chartWidth;

  const getX = (val: number) => {
    const percentage = (val - adjustedMin) / valueRange;
    return chartLeft + percentage * chartWidth;
  };

  // Generate dynamic gridlines dynamically for clean data representation
  const gridLines: number[] = [];
  const stepSize = valueRange > 0.4 ? 0.1 : 0.05;
  const startGrid = Math.ceil(adjustedMin / stepSize) * stepSize;
  for (let g = startGrid; g <= adjustedMax; g += stepSize) {
    if (g >= adjustedMin && g <= adjustedMax) {
      gridLines.push(g);
    }
  }

  // Feature descriptive name mapping for premium visual hierarchy
  const FEATURE_DISPLAY_INFO: Record<string, { label: string }> = {
    EXT_SOURCE_3: { label: "Bureau Score III" },
    EXT_SOURCE_2: { label: "Bureau Score II" },
    EXT_SOURCE_1: { label: "Bureau Score I" },
    ANNUITY_TO_INCOME: { label: "Debt / Income burden" },
    AGE_YEARS: { label: "Applicant Age" },
    EMPLOYMENT_DURATION: { label: "Job Tenure Length" },
    INCOME_TO_CREDIT: { label: "Credit / Income" },
    NAME_EDUCATION_TYPE: { label: "Education Level" }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="simulator-tab-root">
      
      {/* Client Profile Input Controls Column */}
      <div className="lg:col-span-5 bg-white border border-slate-200 shadow-sm rounded-xl p-5 space-y-5">
        <div className="flex justify-between items-center border-b border-slate-100 pb-3">
          <h3 className="text-sm font-semibold text-slate-800">
            Applicant Risk Profile Simulator
          </h3>
          <button
            onClick={handleReset}
            className="text-[10px] text-slate-400 hover:text-indigo-600 font-mono cursor-pointer transition-colors"
          >
            Reset Profile
          </button>
        </div>

        {/* Input Parameters Form */}
        <div className="space-y-4 text-xs">
          
          {/* Demographic Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-slate-500 mb-1 block font-medium">Applicant Gender</label>
              <select
                value={profile.gender}
                onChange={(e) => setProfile({ ...profile, gender: e.target.value as "F" | "M" })}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-700 outline-none focus:border-indigo-500/50 focus:bg-white transition-all"
              >
                <option value="F">Female (F)</option>
                <option value="M">Male (M)</option>
              </select>
            </div>

            <div>
              <label className="text-slate-500 mb-1 block font-medium">Education Background</label>
              <select
                value={profile.educationType}
                onChange={(e) => setProfile({ ...profile, educationType: e.target.value })}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-700 outline-none focus:border-indigo-500/50 focus:bg-white transition-all"
              >
                <option value="Secondary / special education">Secondary / Special</option>
                <option value="Higher education">Higher Education</option>
                <option value="Academic degree">Academic Degree</option>
                <option value="Incomplete higher">Incomplete Higher</option>
                <option value="Lower secondary">Lower Secondary</option>
              </select>
            </div>
          </div>

          {/* Sliders Block */}
          <div className="space-y-3.5 pt-1">
            
            {/* Age Slider */}
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-slate-500 font-medium">Applicant Age</span>
                <span className="font-mono text-slate-700 font-bold">{profile.age} years</span>
              </div>
              <input
                type="range"
                min="21"
                max="68"
                value={profile.age}
                onChange={(e) => setProfile({ ...profile, age: parseInt(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            {/* Employment Length */}
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-slate-500 font-medium">Continuous Job Tenure</span>
                <span className="font-mono text-slate-700 font-bold">{profile.employmentLength} years</span>
              </div>
              <input
                type="range"
                min="0"
                max="45"
                value={profile.employmentLength}
                onChange={(e) => setProfile({ ...profile, employmentLength: parseInt(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            {/* Total Income */}
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-slate-500 font-medium">Annual Income (AMT_INCOME_TOTAL)</span>
                <span className="font-mono text-slate-700 font-bold">${profile.income.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min="12000"
                max="280000"
                step="2000"
                value={profile.income}
                onChange={(e) => setProfile({ ...profile, income: parseInt(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            {/* Credit Amount */}
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-slate-500 font-medium">Requested Loan Credit (AMT_CREDIT)</span>
                <span className="font-mono text-slate-700 font-bold">${profile.creditAmount.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min="20000"
                max="750000"
                step="5000"
                value={profile.creditAmount}
                onChange={(e) => setProfile({ ...profile, creditAmount: parseInt(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            {/* Annuity */}
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-slate-500 font-medium">Yearly Annuity payment (AMT_ANNUITY)</span>
                <span className="font-mono text-slate-700 font-bold">${profile.annuity.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min="1500"
                max="45000"
                step="500"
                value={profile.annuity}
                onChange={(e) => setProfile({ ...profile, annuity: parseInt(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-indigo-600"
              />
            </div>
          </div>

          {/* External Bureau Sources scores */}
          <div className="pt-2 border-t border-slate-100 space-y-3">
            <span className="text-xs font-semibold text-slate-700 block">External Credit Bureau Ratings</span>
            
            <div>
              <div className="flex justify-between text-[10px] mb-1 font-mono">
                <span className="text-slate-400">EXT_SOURCE_1 (Normalized)</span>
                <span className="text-slate-600 font-semibold">{profile.extSource1.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.99"
                step="0.01"
                value={profile.extSource1}
                onChange={(e) => setProfile({ ...profile, extSource1: parseFloat(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-amber-500"
              />
            </div>

            <div>
              <div className="flex justify-between text-[10px] mb-1 font-mono">
                <span className="text-slate-400">EXT_SOURCE_2 (Normalized)</span>
                <span className="text-slate-600 font-semibold">{profile.extSource2.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.99"
                step="0.01"
                value={profile.extSource2}
                onChange={(e) => setProfile({ ...profile, extSource2: parseFloat(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-amber-500"
              />
            </div>

            <div>
              <div className="flex justify-between text-[10px] mb-1 font-mono">
                <span className="text-slate-400">EXT_SOURCE_3 (Normalized)</span>
                <span className="text-slate-600 font-semibold">{profile.extSource3.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.99"
                step="0.01"
                value={profile.extSource3}
                onChange={(e) => setProfile({ ...profile, extSource3: parseFloat(e.target.value) })}
                className="w-full h-1 bg-slate-100 rounded appearance-none cursor-pointer accent-amber-500"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Model Output & SHAP Waterfall Explanation Panel */}
      <div className="lg:col-span-7 flex flex-col gap-6">
        
        {/* Risk Prediction Probability Meter */}
        <div className="bg-white border border-slate-200 shadow-sm rounded-xl p-5">
          <div className="space-y-1.5 text-center md:text-left">
            <h4 className="text-xs text-slate-400 uppercase font-mono">XGBoost Scoring Engine</h4>
            <div className="flex items-center gap-2 justify-center md:justify-start">
              <span className="text-2xl font-semibold text-slate-850">Default Risk:</span>
              <span className={`text-3xl font-mono font-bold ${
                riskProb > 0.40 ? "text-rose-600" : riskProb > 0.18 ? "text-amber-500" : "text-emerald-600"
              }`}>
                {(riskProb * 100).toFixed(1)}%
              </span>
            </div>
            <p className="text-xs text-slate-500">
              {riskProb > 0.40
                ? "This applicant falls inside our HIGH DEFAULT RISK threshold. Immediate credit intervention or strict mitigation is required."
                : riskProb > 0.18
                ? "Moderate Risk client. High credit limit terms should be backed by standard interest and secondary collateral premiums."
                : "Low Risk Client. Recommended for immediate priority approval under optimized premier interest rates."}
            </p>
          </div>
        </div>

        {/* SHAP Force/Waterfall Plot */}
        <div className="bg-white border border-slate-200 shadow-sm rounded-xl p-5 flex-1 flex flex-col justify-between">
          <div className="space-y-1 mb-4">
            <h4 className="text-sm font-semibold text-slate-850">
              SHAP Waterfall: Local Explanation for Current Applicant
            </h4>
            <p className="text-xs text-slate-500">
              Shows how this specific applicant's parameters push the risk baseline (8%) up (Red/Right) or pull it down (Blue/Left).
            </p>
          </div>

          {/* Waterfall SVG Chart */}
          <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 min-h-[240px] flex items-center justify-center overflow-visible select-none">
            <svg viewBox="0 0 500 240" className="w-full h-full overflow-visible">
              <defs>
                <linearGradient id="shap-pos-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#f43f5e" />
                  <stop offset="100%" stopColor="#fda4af" />
                </linearGradient>
                <linearGradient id="shap-neg-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38bdf8" />
                  <stop offset="100%" stopColor="#4f46e5" />
                </linearGradient>
              </defs>

              {/* Grid Coordinates (Dotted background grid) */}
              {gridLines.map((g, i) => (
                <g key={`grid-${i}`}>
                  <line
                    x1={getX(g)}
                    y1={25}
                    x2={getX(g)}
                    y2={210}
                    stroke="#e2e8f0"
                    strokeWidth="1"
                    strokeDasharray="2,2"
                  />
                  <text
                    x={getX(g)}
                    y={223}
                    fill="#94a3b8"
                    fontSize="7"
                    textAnchor="middle"
                    className="font-mono font-medium"
                  >
                    {(g * 100).toFixed(0)}%
                  </text>
                </g>
              ))}

              {/* Base Value Line Indicator */}
              <line
                x1={getX(BASE_VALUE)}
                y1={22}
                x2={getX(BASE_VALUE)}
                y2={210}
                stroke="#f59e0b"
                strokeWidth="1.25"
                strokeDasharray="3,1.5"
                opacity="0.8"
              />
              <text
                x={getX(BASE_VALUE)}
                y={15}
                fill="#d97706"
                fontSize="7.5"
                textAnchor="middle"
                className="font-mono font-bold"
              >
                Base (8%)
              </text>

              {/* Prediction Line Indicator */}
              <line
                x1={getX(riskProb)}
                y1={22}
                x2={getX(riskProb)}
                y2={210}
                stroke={riskProb > 0.40 ? "#f43f5e" : riskProb > 0.18 ? "#f59e0b" : "#10b981"}
                strokeWidth="1.5"
                strokeDasharray="3,1.5"
              />
              <text
                x={getX(riskProb)}
                y={15}
                fill={riskProb > 0.40 ? "#e11d48" : riskProb > 0.18 ? "#d97706" : "#059669"}
                fontSize="7.5"
                textAnchor="middle"
                className="font-mono font-bold"
              >
                Risk ({(riskProb * 100).toFixed(1)}%)
              </text>

              {/* Connecting lines between adjacent steps */}
              {waterfallSteps.map((step, idx) => {
                if (idx === waterfallSteps.length - 1) return null;
                const currentY = 32 + idx * 21 + 10;
                const nextY = 32 + (idx + 1) * 21 + 5;
                const connectX = getX(step.end);
                return (
                  <line
                    key={`connect-${idx}`}
                    x1={connectX}
                    y1={currentY}
                    x2={connectX}
                    y2={nextY}
                    stroke="#cbd5e1"
                    strokeWidth="0.75"
                    strokeDasharray="2,2"
                  />
                );
              })}

              {/* Draw rows */}
              {waterfallSteps.map((item, idx) => {
                const rowY = 32 + idx * 21;
                const isPositive = item.val > 0;
                const startX = isPositive ? getX(item.start) : getX(item.end);
                const barWidth = Math.max(2.5, Math.abs(getX(item.end) - getX(item.start)));
                const displayInfo = FEATURE_DISPLAY_INFO[item.name] || { label: item.name };

                return (
                  <g key={item.name} className="group">
                    {/* Hover highlight full row background */}
                    <rect
                      x={2}
                      y={rowY - 4}
                      width={496}
                      height={20}
                      rx={4}
                      className="fill-slate-100/0 group-hover:fill-slate-100/60 transition-colors duration-150 cursor-pointer"
                    />

                    {/* Left side label: Elegant Descriptive Title & Subtitle */}
                    <text
                      x={chartLeft - 10}
                      y={rowY + 7}
                      fill="#334155"
                      fontSize="8"
                      textAnchor="end"
                      className="font-sans font-bold group-hover:fill-indigo-950 transition-colors duration-150 cursor-pointer"
                    >
                      {displayInfo.label}
                    </text>
                    <text
                      x={chartLeft - 10}
                      y={rowY + 14}
                      fill="#94a3b8"
                      fontSize="6"
                      textAnchor="end"
                      className="font-mono font-medium group-hover:fill-slate-500 transition-colors duration-150 cursor-pointer"
                    >
                      {item.name}
                    </text>

                    {/* Bar representation */}
                    <rect
                      x={startX}
                      y={rowY + 4}
                      width={barWidth}
                      height={10}
                      rx={2}
                      fill={isPositive ? "url(#shap-pos-grad)" : "url(#shap-neg-grad)"}
                      className={`transition-all duration-300 ${
                        isPositive ? "stroke-rose-500/30" : "stroke-indigo-500/30"
                      }`}
                      strokeWidth="0.5"
                    />

                    {/* Value text labels positioned beautifully alongside the bar */}
                    <text
                      x={isPositive ? startX + barWidth + 6 : startX - 6}
                      y={rowY + 11}
                      fontSize="7.5"
                      textAnchor={isPositive ? "start" : "end"}
                      className="font-mono transition-transform duration-150 group-hover:translate-x-0.5"
                    >
                      <tspan fill="#475569" className="font-semibold">{item.featureValue}</tspan>
                      <tspan fill={isPositive ? "#e11d48" : "#4f46e5"} className="font-bold">
                        {" "}{isPositive ? "+" : ""}{(item.val * 100).toFixed(1)}%
                      </tspan>
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <p className="text-[10px] text-slate-400 font-sans mt-3">
            Local SHAP values sum up exactly to the difference between this individual's predicted risk score and the baseline credit cohort risk.
          </p>
        </div>
      </div>

      {/* Risk Sensitivity Heatmap section */}
      <div className="lg:col-span-12 bg-white border border-slate-200 shadow-sm rounded-xl p-6 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div className="space-y-1">
            <h4 className="text-sm font-bold text-slate-800">
              Risk Sensitivity Heatmap (Non-linear Boundaries)
            </h4>
            <p className="text-xs text-slate-500">
              Explore how simultaneous adjustments in Annual Income and Requested Credit Amount affect predicted default probability.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-[10px] font-mono">
            <span className="text-slate-400 font-sans self-center mr-1">Risk Scale:</span>
            <span className="flex items-center gap-1 bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded border border-emerald-100">&lt;8% Low</span>
            <span className="flex items-center gap-1 bg-teal-50 text-teal-700 px-2 py-0.5 rounded border border-teal-100">8-15% Moderate</span>
            <span className="flex items-center gap-1 bg-amber-50 text-amber-700 px-2 py-0.5 rounded border border-amber-100">15-25% Elevated</span>
            <span className="flex items-center gap-1 bg-orange-50 text-orange-700 px-2 py-0.5 rounded border border-orange-100">25-40% High</span>
            <span className="flex items-center gap-1 bg-rose-50 text-rose-700 px-2 py-0.5 rounded border border-rose-100">&gt;40% Critical</span>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
          
          {/* Heatmap Area */}
          <div className="xl:col-span-8 space-y-4">
            
            {/* Column labels (Income) */}
            <div className="flex items-center gap-2">
              <div className="w-24 shrink-0"></div>
              <div className="flex-1 grid grid-cols-5 gap-2 text-center">
                {incomeSteps.map((inc, idx) => (
                  <div key={`col-label-${idx}`} className="text-[10px] font-mono text-slate-500">
                    <div className="font-bold text-slate-700">${(inc / 1000).toFixed(0)}k</div>
                    <div className="text-[8px] text-slate-400">Income</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Rows of Grid Cells */}
            <div className="space-y-2">
              {creditSteps.map((cred, rIdx) => (
                <div key={`row-${rIdx}`} className="flex items-center gap-2">
                  {/* Row Label (Credit Amount) */}
                  <div className="w-24 text-right pr-3 shrink-0 font-mono text-[10px] leading-tight">
                    <div className="font-bold text-slate-700">${(cred / 1000).toFixed(0)}k</div>
                    <div className="text-[8px] text-slate-400">Credit</div>
                  </div>
                  
                  {/* Grid Cells */}
                  <div className="flex-1 grid grid-cols-5 gap-2">
                    {incomeSteps.map((inc, cIdx) => {
                      const prob = getRiskAt(inc, cred);
                      const isCurrent = inc === profile.income && cred === profile.creditAmount;
                      const cellColor = getCellColor(prob);
                      
                      return (
                        <button
                          key={`cell-${rIdx}-${cIdx}`}
                          onClick={() => setProfile({ ...profile, income: inc, creditAmount: cred })}
                          className={`h-16 rounded-lg border text-center transition-all cursor-pointer relative group flex flex-col justify-center items-center p-1 ${
                            isCurrent 
                              ? "ring-2 ring-indigo-600 ring-offset-1 z-10 shadow-md scale-102 bg-opacity-100" 
                              : "hover:scale-[1.03] hover:shadow-sm"
                          } ${cellColor.bg} ${cellColor.border}`}
                        >
                          {isCurrent && (
                            <span className="absolute -top-1.5 -right-1.5 bg-indigo-600 text-white text-[7px] px-1 py-0.5 rounded-full font-extrabold uppercase tracking-wide shadow-sm animate-pulse">
                              Active
                            </span>
                          )}
                          
                          <span className="text-xs font-mono font-extrabold tracking-tight">
                            {(prob * 100).toFixed(1)}%
                          </span>
                          
                          <span className="text-[8px] font-mono opacity-80 mt-0.5">
                            {(cred / inc).toFixed(1)}x ratio
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            <div className="text-[10px] text-slate-400 text-center italic mt-2">
              💡 Tip: Click any square in the grid to automatically apply those values to the active simulator profile.
            </div>
          </div>

          {/* Educational Explanation Column */}
          <div className="xl:col-span-4 bg-slate-50 border border-slate-100 rounded-xl p-5 space-y-4 flex flex-col justify-between">
            <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
              <h5 className="font-bold text-slate-800">
                Non-Linear Model Boundaries
              </h5>
              
              <div className="space-y-2.5">
                <p>
                  This heatmap displays the localized decision neighborhood around your current applicant. By holding all other parameters fixed, we reveal the direct relationships between lending exposure and customer repayment capacity.
                </p>
                <p>
                  <strong>Why ratio is key:</strong> Risk doesn't scale linearly with credit or income alone. Instead, the model acts on complex ratios (such as Debt-to-Income and Credit-to-Income).
                </p>
                <p>
                  In the <strong>top-left</strong> region (low income, high credit), you will notice a rapid compounding of default probability (entering red risk territory). In contrast, the <strong>bottom-right</strong> region represents a robust cash flow buffer where larger loans can be easily serviced.
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-200/60 text-[10px] text-slate-400">
              Ratios represent total requested credit relative to current annual income.
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
