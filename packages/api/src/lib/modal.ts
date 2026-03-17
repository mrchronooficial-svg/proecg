import { env } from "@proecg/env/server";

// ---------------------------------------------------------------------------
// Types matching the pipeline JSON contract (docs/ARQUITETURA.md)
// ---------------------------------------------------------------------------

export interface EcgMeasurements {
  heartRate: number | null;
  heartRateUnit: string;
  axis: number | null;
  axisUnit: string;
  pr: number | null;
  prUnit: string;
  qrs: number | null;
  qrsUnit: string;
  qt: number | null;
  qtUnit: string;
  qtc: number | null;
  qtcUnit: string;
  rhythm: string | null;
}

export interface EcgFinding {
  code: string;
  description: string;
  source: string;
  leads_affected?: string[];
}

export interface EcgDiagnosis {
  code: string;
  description: string;
  source: string;
}

export interface EcgReport {
  measurements: EcgMeasurements;
  findings: EcgFinding[];
  diagnoses: EcgDiagnosis[];
  reportText: string;
}

// ---------------------------------------------------------------------------
// Raw response from the Python pipeline (snake_case)
// ---------------------------------------------------------------------------

interface ModalRawResponse {
  success: boolean;
  error?: string;
  measurements: {
    heart_rate: number | null;
    heart_rate_unit: string;
    axis: number | null;
    axis_unit: string;
    pr_interval: number | null;
    pr_unit: string;
    qrs_duration: number | null;
    qrs_unit: string;
    qt_interval: number | null;
    qt_unit: string;
    qtc_bazett: number | null;
    qtc_unit: string;
    rhythm: string | null;
  };
  findings: Array<{
    code: string;
    description: string;
    source: string;
    leads_affected?: string[];
  }>;
  diagnoses: Array<{
    code: string;
    description: string;
    source: string;
  }>;
  report_text: string;
  processing_time_ms: number;
}

// ---------------------------------------------------------------------------
// Transform snake_case → camelCase
// ---------------------------------------------------------------------------

function transformResponse(raw: ModalRawResponse): EcgReport {
  const m = raw.measurements;
  return {
    measurements: {
      heartRate: m.heart_rate,
      heartRateUnit: m.heart_rate_unit ?? "bpm",
      axis: m.axis,
      axisUnit: m.axis_unit ?? "°",
      pr: m.pr_interval,
      prUnit: m.pr_unit ?? "ms",
      qrs: m.qrs_duration,
      qrsUnit: m.qrs_unit ?? "ms",
      qt: m.qt_interval,
      qtUnit: m.qt_unit ?? "ms",
      qtc: m.qtc_bazett,
      qtcUnit: m.qtc_unit ?? "ms",
      rhythm: m.rhythm,
    },
    findings: raw.findings,
    diagnoses: raw.diagnoses,
    reportText: raw.report_text,
  };
}

// ---------------------------------------------------------------------------
// Mock report (when Modal is not configured — dev mode)
// ---------------------------------------------------------------------------

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
    rhythm: "sinusal",
  },
  findings: [
    {
      code: "normal_rhythm",
      description: "Ritmo sinusal regular",
      source: "rules",
    },
  ],
  diagnoses: [
    {
      code: "normal",
      description: "Sem alterações significativas identificadas",
      source: "rules",
    },
  ],
  reportText:
    "Ritmo: sinusal. FC: 78 bpm. Eixo: +60°.\nPR: 180ms. QRS: 88ms. QT: 380ms. QTc: 420ms (Bazett).\n\nSem alterações significativas ao eletrocardiograma.\n\n⚠️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica. Correlacionar sempre com dados clínicos, exame físico e contexto do paciente.",
};

// ---------------------------------------------------------------------------
// Main function — call Modal endpoint
// ---------------------------------------------------------------------------

export interface CropCorners {
  top_left: [number, number];
  top_right: [number, number];
  bottom_right: [number, number];
  bottom_left: [number, number];
}

export async function analyzeEcg(
  imageUrl: string,
  corners?: CropCorners,
): Promise<EcgReport> {
  if (!env.MODAL_ENDPOINT_URL) {
    console.log("[DEV] MODAL_ENDPOINT_URL not configured — returning mock report");
    return MOCK_REPORT;
  }

  const body: Record<string, unknown> = { image_url: imageUrl };
  if (env.MODAL_TOKEN) {
    body.token = env.MODAL_TOKEN;
  }
  if (corners) {
    body.corners = corners;
  }

  const res = await fetch(env.MODAL_ENDPOINT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Modal API error ${res.status}: ${text}`);
  }

  const raw = (await res.json()) as ModalRawResponse;

  if (!raw.success) {
    throw new Error(raw.error ?? "Erro desconhecido no pipeline de IA");
  }

  return transformResponse(raw);
}
