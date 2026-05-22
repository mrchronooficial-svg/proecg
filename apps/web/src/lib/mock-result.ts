import type { EcgReport } from "@/types/result";

export const mockNormalReport: EcgReport = {
  id: "a3f8c2b1",
  createdAt: new Date(2026, 4, 3, 15, 26),
  imageUrl: "/mock-ecgs/IMG_1303.png",
  ischemia: {
    probability: 1,
    detected: false,
    diagnosis: null,
    wall: null,
    leads: null,
    artery: null,
  },
  arrhythmia: {
    detected: false,
    name: null,
    location: null,
    characteristics: [],
    urgency: null,
    urgencyMessage: null,
  },
  redFlags: [],
  reportText:
    "Ritmo sinusal. Sem sinais de isquemia. Sem arritmias detectadas.\n\nSem alterações significativas ao eletrocardiograma.",
};

export const mockStemiReport: EcgReport = {
  id: "b7d2e4f9",
  createdAt: new Date(2026, 4, 4, 3, 20),
  imageUrl: "/mock-ecgs/IMG_1303.png",
  ischemia: {
    probability: 92,
    detected: true,
    diagnosis: "SCA com supra de ST em parede anterior",
    wall: "Anterior (V1-V4)",
    leads: "V1, V2, V3, V4",
    artery: "DA (descendente anterior)",
  },
  arrhythmia: {
    detected: false,
    name: null,
    location: null,
    characteristics: [],
    urgency: null,
    urgencyMessage: null,
  },
  redFlags: [
    {
      type: "danger",
      title: "ALERTA",
      message:
        "Possível oclusão coronariana aguda — supra de ST em parede anterior",
      suggestion: "Correlacionar com clínica. Considerar ativação de hemodinâmica.",
    },
  ],
  reportText:
    "Sinais sugestivos de síndrome coronariana aguda com supradesnivelamento do segmento ST em V1-V4 (2-3mm).\n\nParede acometida: anterior. Artéria provável: descendente anterior (DA).\n\nCorrelacionar com dados clínicos e troponina.",
};

export const mockFaReport: EcgReport = {
  id: "c5a1d3e8",
  createdAt: new Date(2026, 4, 5, 14, 32),
  imageUrl: "/mock-ecgs/IMG_1303.png",
  ischemia: {
    probability: 8,
    detected: false,
    diagnosis: null,
    wall: null,
    leads: null,
    artery: null,
  },
  arrhythmia: {
    detected: true,
    name: "Fibrilação atrial",
    location:
      "Ausência de ondas P, intervalo RR irregular — todas as derivações",
    characteristics: [
      "Intervalos RR irregularmente irregulares",
      "Ausência de onda P organizada",
      "Linha de base fibrilatória em V1",
      "Resposta ventricular: 132 bpm (rápida)",
    ],
    urgency: "warning",
    urgencyMessage:
      "FA com resposta ventricular rápida — considerar controle de frequência",
  },
  redFlags: [],
  reportText:
    "Fibrilação atrial com resposta ventricular rápida (FC ~132 bpm). Intervalos RR irregulares. Ausência de onda P.\n\nSem sinais de isquemia aguda.\n\nConsiderar controle de frequência.",
};

export const mockCriticalReport: EcgReport = {
  id: "d9b3f1a7",
  createdAt: new Date(2026, 4, 5, 22, 10),
  imageUrl: "/mock-ecgs/IMG_1303.png",
  ischemia: {
    probability: 88,
    detected: true,
    diagnosis: "SCA com supra de ST em parede inferior",
    wall: "Inferior (DII, DIII, aVF)",
    leads: "DII, DIII, aVF",
    artery: "CD (coronária direita)",
  },
  arrhythmia: {
    detected: true,
    name: "Taquicardia ventricular monomórfica",
    location:
      "Complexos QRS largos (148ms) regulares a 185bpm — V1-V6",
    characteristics: [
      "QRS alargado > 140ms",
      "Frequência regular a 185 bpm",
      "Dissociação AV",
      "Concordância precordial",
    ],
    urgency: "emergency",
    urgencyMessage: "Emergência — TV sustentada. Considerar cardioversão.",
  },
  redFlags: [
    {
      type: "danger",
      title: "ALERTA",
      message: "Taquicardia ventricular sustentada",
      suggestion:
        "Avaliar estabilidade hemodinâmica. Cardioversão se instável.",
    },
    {
      type: "danger",
      title: "ALERTA",
      message: "Supra de ST em parede inferior — possível IAM",
      suggestion:
        "Correlacionar com clínica. Considerar hemodinâmica após estabilização.",
    },
  ],
  reportText:
    "Taquicardia ventricular monomórfica sustentada (FC ~185bpm, QRS 148ms). Sinais sugestivos de SCA com supra de ST em parede inferior.\n\nDupla emergência: TV + STEMI inferior.",
};

export const mockReports: Record<string, EcgReport> = {
  normal: mockNormalReport,
  stemi: mockStemiReport,
  fa: mockFaReport,
  critical: mockCriticalReport,
};
