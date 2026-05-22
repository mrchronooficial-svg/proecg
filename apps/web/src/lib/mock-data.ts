import type {
  DashboardStats,
  EcgAnalysisSummary,
} from "@/types/dashboard";

export const mockStats: DashboardStats = {
  totalExams: 47,
  thisMonth: 12,
  lastExamAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
};

export const mockExams: EcgAnalysisSummary[] = [
  {
    id: "1",
    imageUrl: "",
    createdAt: new Date(2026, 4, 5, 14, 32),
    reportSummary:
      "Ritmo sinusal. FC: 78 bpm. QTc: 420ms. Sem alterações significativas.",
  },
  {
    id: "2",
    imageUrl: "",
    createdAt: new Date(2026, 4, 4, 9, 15),
    reportSummary:
      "Fibrilação atrial. FC: 128 bpm. Resposta ventricular rápida.",
  },
  {
    id: "3",
    imageUrl: "",
    createdAt: new Date(2026, 4, 3, 22, 48),
    reportSummary:
      "Ritmo sinusal. FC: 65 bpm. Bloqueio de ramo direito. QTc: 445ms.",
  },
  {
    id: "4",
    imageUrl: "",
    createdAt: new Date(2026, 4, 2, 3, 20),
    reportSummary:
      "Ritmo sinusal. Supradesnivelamento ST V1-V4. SCA com supra anterior.",
  },
  {
    id: "5",
    imageUrl: "",
    createdAt: new Date(2026, 4, 1, 16, 5),
    reportSummary: "Flutter atrial 2:1. FC: 150 bpm. Eixo normal.",
  },
];
