import { env } from "@proecg/env/server";

export interface EcgMeasurements {
  heartRate: number;
  heartRateUnit: string;
  pr: number;
  prUnit: string;
  qrs: number;
  qrsUnit: string;
  qt: number;
  qtUnit: string;
  qtc: number;
  qtcUnit: string;
  axis: number;
  axisUnit: string;
}

export interface EcgFinding {
  description: string;
}

export interface EcgDiagnosis {
  name: string;
  description: string;
}

export interface EcgReport {
  measurements: EcgMeasurements;
  findings: EcgFinding[];
  diagnoses: EcgDiagnosis[];
  reportText: string;
}

const MOCK_REPORT: EcgReport = {
  measurements: {
    heartRate: 78,
    heartRateUnit: "bpm",
    pr: 180,
    prUnit: "ms",
    qrs: 88,
    qrsUnit: "ms",
    qt: 380,
    qtUnit: "ms",
    qtc: 420,
    qtcUnit: "ms",
    axis: 60,
    axisUnit: "°",
  },
  findings: [
    { description: "Ritmo sinusal regular" },
    { description: "Intervalos dentro dos limites da normalidade" },
  ],
  diagnoses: [
    {
      name: "ECG normal",
      description: "Sem alterações significativas identificadas",
    },
  ],
  reportText:
    "Ritmo: sinusal. FC: 78 bpm. Eixo: +60°.\nPR: 180ms. QRS: 88ms. QT: 380ms. QTc: 420ms (Bazett).\n\nAchados: Ritmo sinusal regular. Intervalos dentro dos limites da normalidade.\n\nSem alterações significativas identificadas.",
};

export async function analyzeEcg(imageUrl: string): Promise<EcgReport> {
  if (!env.MODAL_ENDPOINT_URL || !env.MODAL_TOKEN) {
    console.log("[DEV] Modal not configured — returning mock report");
    return MOCK_REPORT;
  }

  const res = await fetch(env.MODAL_ENDPOINT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.MODAL_TOKEN}`,
    },
    body: JSON.stringify({ imageUrl }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Modal API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<EcgReport>;
}
