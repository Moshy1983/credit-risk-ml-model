import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Initialize Gemini client on the server side
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

async function startServer() {
  // API routes FIRST
  app.post("/api/analyze-risk", async (req, res) => {
    try {
      const { clientProfile, riskProbability, keyFactors } = req.body;
      
      const prompt = `
You are a senior credit risk analyst at a major bank. Analyse this loan applicant's profile and explain their default risk prediction.

Client Profile Details:
- Gender: ${clientProfile.gender}
- Age: ${clientProfile.age} years
- Employment Duration: ${clientProfile.employmentLength} years
- Income: $${clientProfile.income.toLocaleString()}
- Credit Loan Amount: $${clientProfile.creditAmount.toLocaleString()}
- Annuity (Yearly payment): $${clientProfile.annuity.toLocaleString()}
- Goods Price: $${clientProfile.goodsPrice.toLocaleString()}
- Contract Type: ${clientProfile.contractType}
- Occupation Type: ${clientProfile.occupationType}
- Education: ${clientProfile.educationType}
- External Credit Source 1 (normalized): ${clientProfile.extSource1.toFixed(3)}
- External Credit Source 2 (normalized): ${clientProfile.extSource2.toFixed(3)}
- External Credit Source 3 (normalized): ${clientProfile.extSource3.toFixed(3)}

Model Default Risk Probability: ${(riskProbability * 100).toFixed(1)}%
Primary Risk Drivers (SHAP contributions):
${keyFactors.map((f: any) => `- ${f.name}: ${f.val > 0 ? '+' : ''}${(f.val * 100).toFixed(2)}% risk direction`).join('\n')}

Task:
Provide an expert, professional credit committee brief (around 120-150 words). Include:
1. Risk Assessment: Briefly explain the core drivers of this risk level based on the profile and model risk scores.
2. Credit Decision Strategy: Suggest a tailored decision (e.g. decline, approve with high interest/collateral, require a co-signer, or approve standard terms).
3. Risk Mitigation: Recommend 1-2 practical risk mitigation guidelines.

Keep the tone professional, objective, and authoritative. Do not use generic filler text or markdown headers (use simple paragraph breaks).
`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
      });
      
      res.json({ analysis: response.text });
    } catch (error: any) {
      console.error("Gemini API error:", error);
      res.status(500).json({ error: error.message || "Failed to generate AI analysis" });
    }
  });

  // Serve Vite
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start backend server:", err);
});
