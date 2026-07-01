import { ArrowRight, UserCheck, BarChart3, BookOpen, ShieldCheck, HelpCircle } from "lucide-react";

interface LandingTabProps {
  onNavigate: (tabId: "simulator" | "shap" | "tuning" | "evaluation" | "documentation") => void;
}

export default function LandingTab({ onNavigate }: LandingTabProps) {
  return (
    <div className="space-y-12 py-4" id="landing-tab-root">
      
      {/* Hero Header */}
      <div className="text-center max-w-3xl mx-auto space-y-4">
        <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight sm:text-4xl">
          Credit Risk Prediction Model
        </h2>
        <p className="text-base text-slate-500 leading-relaxed">
          This system evaluates the likelihood that a loan applicant might have difficulty repaying their credit. By analyzing historical application records, it helps teams understand, explain, and manage lending risks with confidence.
        </p>
        <div className="flex flex-wrap justify-center gap-4 pt-4">
          <button
            onClick={() => onNavigate("simulator")}
            className="inline-flex items-center bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-sm px-5 py-3 rounded-lg shadow-sm transition-colors cursor-pointer"
          >
            Start Interactive Simulator
          </button>
          <button
            onClick={() => onNavigate("documentation")}
            className="inline-flex items-center bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-medium text-sm px-5 py-3 rounded-lg shadow-sm transition-colors cursor-pointer"
          >
            Read Documentation
          </button>
        </div>
      </div>

      {/* Feature Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
        
        {/* Card 1 */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4 shadow-sm hover:border-indigo-100 transition-colors">
          <h3 className="text-base font-bold text-slate-800">Risk Simulator</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Input custom applicant details like income, age, credit amount, and external ratings to see an instant prediction score. Experience how changes in profile attributes directly shift the risk outcome.
          </p>
          <button
            onClick={() => onNavigate("simulator")}
            className="text-xs text-indigo-600 font-semibold hover:text-indigo-700 inline-flex items-center cursor-pointer"
          >
            Open simulator
          </button>
        </div>

        {/* Card 2 */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4 shadow-sm hover:border-indigo-100 transition-colors">
          <h3 className="text-base font-bold text-slate-800">Model Performance</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Review realistic metrics, accuracy charts, and error rates evaluated against standard test groups. Verify how the underlying classification engine balances approvals and denials fairly.
          </p>
          <button
            onClick={() => onNavigate("evaluation")}
            className="text-xs text-indigo-600 font-semibold hover:text-indigo-700 inline-flex items-center cursor-pointer"
          >
            View metrics
          </button>
        </div>

        {/* Card 3 */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4 shadow-sm hover:border-indigo-100 transition-colors">
          <h3 className="text-base font-bold text-slate-800">Model Documentation</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Read the model card explaining our decision-tree parameters, training details, data preparation steps, and explainability methods used to guarantee regulatory compliance.
          </p>
          <button
            onClick={() => onNavigate("documentation")}
            className="text-xs text-indigo-600 font-semibold hover:text-indigo-700 inline-flex items-center cursor-pointer"
          >
            Read report
          </button>
        </div>

      </div>

      {/* Educational Context Section */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-8 space-y-6">
        <h3 className="text-lg font-bold text-slate-800">
          Frequently Asked Questions
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-xs text-slate-500">
          <div className="space-y-2">
            <h4 className="font-bold text-slate-800 text-sm">How does the model calculate risk?</h4>
            <p className="leading-relaxed">
              The model evaluates several parts of an applicant's profile, including their self-reported income, debt-to-income ratio, and ratings from external credit scoring agencies. It then matches these indicators against historical patterns to calculate the default probability.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-800 text-sm">Is every prediction explainable?</h4>
            <p className="leading-relaxed">
              Yes. Every prediction is backed by a mathematical framework called Shapley Additive Explanations (SHAP). This identifies exactly how much each individual factor (like younger age or high requested credit) pushed the risk estimation up or down from the baseline.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-800 text-sm">How is the data preprocessed?</h4>
            <p className="leading-relaxed">
              Before the data enters the model, we clean missing fields and transform categorical values like occupation types into structured numerical inputs. This ensures the model processes the records consistently without losing valuable context.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-800 text-sm">Does the system make automatic decisions?</h4>
            <p className="leading-relaxed">
              No. This tool is designed to be a supportive guide for credit analysts. It suggests risk levels and explains major risk drivers, but final lending approvals remain under the supervision of qualified lending committees.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}
