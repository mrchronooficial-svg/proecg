"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { Card } from "@proecg/ui/components/card";
import type { EcgReport } from "@proecg/api/lib/modal";

import { trpc } from "@/utils/trpc";
import { ReportView } from "@/components/ecg/report-view";

export default function ResultadoPage() {
  const { id } = useParams<{ id: string }>();

  const analysis = useQuery(trpc.ecg.getAnalysis.queryOptions({ id }));

  if (analysis.isLoading) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <p className="text-muted-foreground">Carregando resultado...</p>
      </div>
    );
  }

  if (analysis.error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <Card className="p-6 text-center">
          <p className="text-red-500">
            {analysis.error.message || "Erro ao carregar resultado"}
          </p>
        </Card>
      </div>
    );
  }

  if (!analysis.data) return null;

  const data = analysis.data;

  if (data.status === "PROCESSING") {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8 text-center">
        <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        <p className="text-muted-foreground">Processando ECG...</p>
      </div>
    );
  }

  if (data.status === "FAILED") {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <Card className="p-6 text-center">
          <p className="text-red-500">
            Erro ao processar o ECG. Tente enviar novamente.
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 pb-20 md:pb-8">
      <h1 className="mb-6 text-2xl font-bold">Resultado</h1>
      {data.reportJson ? (
        <ReportView
          report={data.reportJson as unknown as EcgReport}
          imageUrl={data.imageUrl}
        />
      ) : (
        <p className="text-muted-foreground">Resultado não disponível.</p>
      )}
    </div>
  );
}
