/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Home, UserCheck, Zap, Cpu, BarChart3, Code, Award, GitMerge, FileCode, CheckCircle2, BookOpen } from "lucide-react";

// Components
import LandingTab from "./components/LandingTab";
import SimulatorTab from "./components/SimulatorTab";
import ExplainabilityTab from "./components/ExplainabilityTab";
import TuningTab from "./components/TuningTab";
import EvaluationTab from "./components/EvaluationTab";
import DocumentationTab from "./components/DocumentationTab";

type TabId = "landing" | "simulator" | "shap" | "tuning" | "evaluation" | "documentation";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("landing");

  const tabs = [
    { id: "landing", label: "Overview", icon: Home, desc: "Welcome & model introduction" },
    { id: "simulator", label: "Interactive Simulator", icon: UserCheck, desc: "Predict & explain default risk" },
    { id: "shap", label: "SHAP Explainability", icon: Zap, desc: "Global summary & beeswarm plots" },
    { id: "tuning", label: "Optuna Tuning", icon: Cpu, desc: "Bayesian optimization trials" },
    { id: "evaluation", label: "Model Metrics", icon: BarChart3, desc: "AUC-ROC curves & matrices" },
    { id: "documentation", label: "Model Documentation", icon: BookOpen, desc: "Technical specifications & report" }
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans selection:bg-indigo-500/20 selection:text-indigo-900" id="app-root">
      
      {/* Main Content Hub */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        
        {/* Intro Portfolio Banner */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-indigo-500/5 rounded-full filter blur-[100px] -mr-48 -mt-48"></div>
          
          <div className="max-w-4xl space-y-3 relative z-10">
            <h2 className="text-xl md:text-2xl font-bold tracking-tight text-slate-800 font-sans">
              Credit Default Risk Evaluation
            </h2>
            <p className="text-xs md:text-sm text-slate-500 leading-relaxed font-sans">
              This application is designed to analyze applicant profiles and estimate repayment risks. Use the navigation buttons below to try the simulator, inspect important risk factors, see how model options were tuned, or read the final model reports.
            </p>
          </div>
        </div>

        {/* Tabbed Navigation Rail */}
        <div className="flex overflow-x-auto pb-1 gap-2 border-b border-slate-200 scrollbar-none">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as TabId)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-xs font-semibold cursor-pointer whitespace-nowrap transition-all duration-200 ${
                  isActive
                    ? "bg-indigo-50 border-indigo-500/40 text-indigo-700 shadow-sm"
                    : "bg-transparent border-transparent hover:border-slate-200 text-slate-500 hover:text-slate-700"
                }`}
              >
                <div className="text-left">
                  <div className="text-xs font-bold">{tab.label}</div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Dynamic Tab Workspace panel */}
        <div className="min-h-[450px]">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.15 }}
            >
              {activeTab === "landing" && <LandingTab onNavigate={(id) => setActiveTab(id)} />}
              {activeTab === "simulator" && <SimulatorTab />}
              {activeTab === "shap" && <ExplainabilityTab />}
              {activeTab === "tuning" && <TuningTab />}
              {activeTab === "evaluation" && <EvaluationTab />}
              {activeTab === "documentation" && <DocumentationTab />}
            </motion.div>
          </AnimatePresence>
        </div>

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 py-6 mt-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center gap-4 text-xs font-mono text-slate-400">
          <div>
            Credit Risk Default Prediction Model — Built with XGBoost, Optuna & SHAP
          </div>
          <div className="flex items-center gap-4">
            <a href="https://mosef.dev" target="_blank" rel="noopener noreferrer" className="hover:text-indigo-600 transition-colors">
              Portfolio
            </a>
            <span>•</span>
            <a href="https://github.com/moshy1983/credit-risk-ml-model" target="_blank" rel="noopener noreferrer" className="hover:text-indigo-600 transition-colors">
              GitHub
            </a>
          </div>
        </div>
      </footer>

    </div>
  );
}
