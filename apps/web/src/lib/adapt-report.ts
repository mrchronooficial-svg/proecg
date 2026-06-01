import type { EcgReport as ApiEcgReport } from "@proecg/api/lib/modal";

import type {
  ArrhythmiaResult,
  ArrhythmiaUrgency,
  EcgReport,
  IschemiaResult,
  RedFlag,
} from "@/types/result";

const ISCHEMIA_KEYWORDS = [
  "supra de st",
  "supra st",
  "stemi",
  "infra de st",
  "infra st",
  "sca",
  "sindrome coronariana",
  "síndrome coronariana",
  "wellens",
  "winter",
  "sgarbossa",
  "isquemia",
  "infarto",
  // v5b — descrições da CNN
  "supradesnivelamento",
  "infradesnivelamento",
  "infarto do miocárdio",
  "isquemia subendocárdica",
];

const ARRHYTHMIA_PATTERNS: Array<{
  re: RegExp;
  name: string;
  urgency: ArrhythmiaUrgency;
  urgencyMessage: string;
}> = [
  {
    re: /(taquicardia ventricular|\btv\b|tv mono|tv poli|torsades)/i,
    name: "Taquicardia ventricular",
    urgency: "emergency",
    urgencyMessage:
      "Emergência — TV sustentada. Considerar cardioversão.",
  },
  {
    re: /(bav (total|3|3°|3º|completo)|bloqueio atrioventricular total)/i,
    name: "BAV total (3° grau)",
    urgency: "emergency",
    urgencyMessage: "Emergência — bradicardia grave. Considerar marcapasso.",
  },
  {
    re: /(fibrilação atrial|fibrilacao atrial|\bfa\b)/i,
    name: "Fibrilação atrial",
    urgency: "warning",
    urgencyMessage:
      "Atenção — considerar controle de frequência e anticoagulação.",
  },
  {
    re: /flutter atrial/i,
    name: "Flutter atrial",
    urgency: "warning",
    urgencyMessage:
      "Atenção — considerar controle de frequência e anticoagulação.",
  },
  {
    re: /taquicardia supraventricular|\btsv\b/i,
    name: "Taquicardia supraventricular",
    urgency: "warning",
    urgencyMessage: "Atenção — considerar manobras vagais ou adenosina.",
  },
  {
    re: /(bav (2|2°|2º|mobitz)|bloqueio atrioventricular de 2)/i,
    name: "BAV de 2° grau",
    urgency: "warning",
    urgencyMessage: "Atenção — monitorização contínua.",
  },
  { re: /(bav (1|1°|1º)|bloqueio atrioventricular de 1)/i, name: "BAV de 1° grau", urgency: "info", urgencyMessage: null as unknown as string },
  { re: /wpw|wolff-parkinson/i, name: "Wolff-Parkinson-White (WPW)", urgency: "info", urgencyMessage: null as unknown as string },
  {
    re: /bradicardia/i,
    name: "Bradicardia sinusal",
    urgency: "info",
    urgencyMessage: null as unknown as string,
  },
  {
    re: /taquicardia sinusal/i,
    name: "Taquicardia sinusal",
    urgency: "info",
    urgencyMessage: null as unknown as string,
  },
  {
    re: /extrassy?stoles?\s*(supraventricul|atria)/i,
    name: "Extrassístoles atriais",
    urgency: "info",
    urgencyMessage: null as unknown as string,
  },
  {
    re: /extrassy?stoles?\s*ventricul/i,
    name: "Extrassístoles ventriculares",
    urgency: "info",
    urgencyMessage: null as unknown as string,
  },
];

/** Pega a melhor string descritiva de um item de laudo, sem assumir shape. */
function getDescription(item: { description?: string; name?: string }): string {
  return item.description ?? item.name ?? "";
}

function detectIschemia(api: ApiEcgReport): IschemiaResult {
  const findingsText = [
    ...api.findings.map(getDescription),
    ...api.diagnoses.map(getDescription),
    ...api.redFlags.map(getDescription),
  ]
    .join(" ")
    .toLowerCase();

  const matchedKeywords = ISCHEMIA_KEYWORDS.filter((k) =>
    findingsText.includes(k),
  );
  const detected = matchedKeywords.length > 0;
  // probabilidade heurística: número de keywords casadas (cap em 95%)
  const probability = detected
    ? Math.min(95, 60 + matchedKeywords.length * 8)
    : 5;

  // tenta extrair primeiro diagnóstico de isquemia da lista de diagnoses
  const ischemiaDiagnosis =
    api.diagnoses.find((d) =>
      ISCHEMIA_KEYWORDS.some((k) =>
        getDescription(d).toLowerCase().includes(k),
      ),
    ) ?? null;

  return {
    probability,
    detected,
    diagnosis: ischemiaDiagnosis ? getDescription(ischemiaDiagnosis) : null,
    wall: null,
    leads: null,
    artery: null,
  };
}

function detectArrhythmia(api: ApiEcgReport): ArrhythmiaResult {
  const text = [
    api.measurements.rhythm ?? "",
    ...api.findings.map(getDescription),
    ...api.diagnoses.map(getDescription),
  ]
    .join(" ")
    .toLowerCase();

  for (const p of ARRHYTHMIA_PATTERNS) {
    if (p.re.test(text)) {
      const characteristics = api.findings
        .filter((f) => p.re.test(getDescription(f).toLowerCase()))
        .map(getDescription);
      return {
        detected: true,
        name: p.name,
        location: null,
        characteristics,
        urgency: p.urgency,
        urgencyMessage: p.urgencyMessage || null,
      };
    }
  }

  return {
    detected: false,
    name: null,
    location: null,
    characteristics: [],
    urgency: null,
    urgencyMessage: null,
  };
}

function adaptRedFlags(api: ApiEcgReport): RedFlag[] {
  return api.redFlags.map((rf) => ({
    type: "danger" as const,
    title: "ALERTA",
    message: getDescription(rf),
    suggestion: rf.leads_affected?.length
      ? `Derivações: ${rf.leads_affected.join(", ")}`
      : "Correlacionar com dados clínicos.",
  }));
}

/**
 * Converte `EcgReport` da tRPC (camelCased) para o shape novo focado em
 * Isquemia + Arritmia. Heurísticas usadas porque o backend ainda não
 * retorna esses campos diretamente.
 */
export function adaptApiReport(
  api: ApiEcgReport,
  meta: { id: string; createdAt: Date; imageUrl: string },
): EcgReport {
  return {
    id: meta.id,
    createdAt: meta.createdAt,
    imageUrl: meta.imageUrl,
    ischemia: detectIschemia(api),
    arrhythmia: detectArrhythmia(api),
    redFlags: adaptRedFlags(api),
    reportText: api.reportText,
  };
}
