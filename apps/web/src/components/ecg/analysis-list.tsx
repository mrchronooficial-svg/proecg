"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Card } from "@proecg/ui/components/card";
import type { EcgReport } from "@proecg/api/lib/modal";

import { trpc } from "@/utils/trpc";

function daysUntil(date: Date | string): number {
  const target = new Date(date);
  const now = new Date();
  return Math.ceil((target.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
}

export function AnalysisList() {
  const analyses = useQuery(
    trpc.ecg.listAnalyses.queryOptions({ limit: 20 }),
  );

  if (analyses.isLoading) {
    return <p className="text-sm text-muted-foreground">Carregando...</p>;
  }

  if (!analyses.data?.items.length) {
    return (
      <div className="py-12 text-center">
        <p className="text-muted-foreground">Nenhuma análise ainda.</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Comece enviando um ECG pelo botão &quot;Novo ECG&quot;.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {analyses.data.items.map((item) => {
        const remaining = daysUntil(item.expiresAt);
        const report = item.reportJson as unknown as EcgReport | null;
        const summary =
          item.status === "COMPLETED" && report
            ? `FC: ${report.measurements.heartRate} bpm — ${report.diagnoses[0]?.name ?? "Sem alterações"}`
            : item.status === "PROCESSING"
              ? "Processando..."
              : item.status === "FAILED"
                ? "Falha na análise"
                : "Aguardando";

        return (
          <Link key={item.id} href={`/dashboard/resultado/${item.id}`}>
            <Card className="flex items-center gap-4 p-4 transition-colors hover:bg-accent">
              {item.imageUrl && (
                <img
                  src={item.imageUrl}
                  alt="ECG"
                  className="h-14 w-20 shrink-0 rounded object-cover"
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">
                  {new Date(item.createdAt).toLocaleDateString("pt-BR", {
                    day: "2-digit",
                    month: "2-digit",
                    year: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
                <p className="truncate text-xs text-muted-foreground">
                  {summary}
                </p>
              </div>
              {remaining <= 7 && remaining > 0 && (
                <span className="shrink-0 rounded-full bg-yellow-500/10 px-2 py-1 text-xs font-medium text-yellow-700 dark:text-yellow-400">
                  {remaining}d restantes
                </span>
              )}
            </Card>
          </Link>
        );
      })}
    </div>
  );
}
