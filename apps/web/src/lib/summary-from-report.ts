import type { EcgAnalysisSummary } from "@/types/dashboard";

/** Extrai a 1ª linha não-vazia do report_text como resumo curto. */
function summarizeReportText(reportText: unknown): string | null {
  if (typeof reportText !== "string") return null;
  for (const line of reportText.split("\n")) {
    const trimmed = line.trim();
    if (
      trimmed &&
      !trimmed.startsWith("=") &&
      !/^[A-Z\s]+:$/.test(trimmed) // pula headers tipo "MEDICOES:"
    ) {
      return trimmed.length > 140 ? `${trimmed.slice(0, 137)}…` : trimmed;
    }
  }
  return null;
}

/** Converte uma análise do tRPC (listAnalyses) pro shape do dashboard. */
export function toSummary(item: {
  id: string;
  imageUrl: string;
  createdAt: Date | string;
  reportJson: Record<string, unknown> | null;
  status?: string;
}): EcgAnalysisSummary {
  const reportText =
    item.reportJson && typeof item.reportJson === "object"
      ? (item.reportJson as { reportText?: unknown }).reportText
      : null;

  return {
    id: item.id,
    imageUrl: item.imageUrl,
    createdAt:
      item.createdAt instanceof Date ? item.createdAt : new Date(item.createdAt),
    reportSummary: summarizeReportText(reportText),
  };
}
