"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import type { EcgReport as ApiEcgReport } from "@proecg/api/lib/modal";

import { ResultPageView } from "@/components/result/ResultPageView";
import { adaptApiReport } from "@/lib/adapt-report";
import { mockReports } from "@/lib/mock-result";
import { trpc } from "@/utils/trpc";

export default function ResultadoPage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const mockKey = searchParams.get("mock");
  const mockReport = mockKey ? mockReports[mockKey] : null;

  const analysis = useQuery({
    ...trpc.ecg.getAnalysis.queryOptions({ id }),
    enabled: !mockReport,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "PROCESSING" || status === "UPLOADING" ? 3000 : false;
    },
  });

  if (mockReport) {
    return <ResultPageView report={mockReport} />;
  }

  if (analysis.isLoading) {
    return (
      <div className="flex flex-col items-center justify-center px-4 py-24 text-center">
        <Loader2
          className="text-apple-accent animate-spin"
          size={32}
          strokeWidth={1.8}
          aria-hidden="true"
        />
        <p className="mt-4 text-[15px] text-apple-text-secondary">
          Carregando resultado...
        </p>
      </div>
    );
  }

  if (analysis.error) {
    return (
      <div className="rounded-2xl bg-apple-surface p-6 text-center shadow-apple-sm">
        <p className="text-[15px] text-apple-danger">
          {analysis.error.message || "Erro ao carregar resultado"}
        </p>
      </div>
    );
  }

  if (!analysis.data) return null;

  const data = analysis.data;

  if (data.status === "UPLOADING" || data.status === "PROCESSING") {
    return (
      <div className="flex flex-col items-center justify-center px-4 py-24 text-center">
        <Loader2
          className="text-apple-accent animate-spin"
          size={40}
          strokeWidth={1.8}
          aria-hidden="true"
        />
        <p className="mt-4 text-[17px] font-medium text-apple-text">
          {data.status === "UPLOADING"
            ? "Aguardando upload..."
            : "Processando ECG..."}
        </p>
        <p className="mt-2 max-w-xs text-[15px] text-apple-text-secondary">
          Pode levar até 30 segundos. Esta página atualiza automaticamente.
        </p>
      </div>
    );
  }

  if (data.status === "FAILED") {
    return (
      <div className="rounded-2xl bg-apple-surface p-6 text-center shadow-apple-sm">
        <h2 className="text-[22px] font-semibold tracking-[-0.01em] text-apple-danger">
          Erro ao processar ECG
        </h2>
        <p className="mt-2 text-[15px] text-apple-text-secondary">
          Não foi possível analisar este ECG. Tente enviar a foto novamente.
        </p>
      </div>
    );
  }

  if (!data.reportJson) {
    return (
      <p className="text-[15px] text-apple-text-secondary">
        Resultado não disponível.
      </p>
    );
  }

  const apiReport = data.reportJson as unknown as ApiEcgReport;
  const report = adaptApiReport(apiReport, {
    id: data.id,
    createdAt: new Date(data.createdAt),
    imageUrl: data.imageUrl,
  });

  return <ResultPageView report={report} />;
}
