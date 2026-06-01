"use client";

import { useQuery } from "@tanstack/react-query";
import { FileX2 } from "lucide-react";
import { useMemo } from "react";

import { EcgListItem } from "@/components/dashboard/EcgListItem";
import { EmptyState } from "@/components/dashboard/EmptyState";
import { Skeleton } from "@/components/dashboard/Skeleton";
import { toSummary } from "@/lib/summary-from-report";
import { trpc } from "@/utils/trpc";

const PAGE_SIZE = 50;

function HistoricoSkeletonRow() {
  return (
    <div className="flex items-center gap-3.5 rounded-2xl bg-apple-surface p-4 shadow-apple-sm">
      <Skeleton className="size-12 shrink-0 rounded-full" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-2/5 rounded-md" />
        <Skeleton className="h-3 w-4/5 rounded-md" />
      </div>
      <Skeleton className="size-5 rounded-md" />
    </div>
  );
}

export function HistoricoList() {
  const { data, isLoading } = useQuery(
    trpc.ecg.listAnalyses.queryOptions({ limit: PAGE_SIZE }),
  );

  const items = useMemo(
    () => (data?.items ?? []).map(toSummary),
    [data],
  );

  if (isLoading) {
    return (
      <div className="flex flex-col gap-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <HistoricoSkeletonRow key={i} />
        ))}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <EmptyState
        icon={FileX2}
        title="Nenhum exame ainda"
        description="Fotografe um ECG para receber seu primeiro laudo."
        ctaLabel="Novo ECG"
        ctaHref="/dashboard/novo"
      />
    );
  }

  return (
    <ul className="flex flex-col gap-3">
      {items.map((exam, idx) => (
        <li key={exam.id}>
          <EcgListItem exam={exam} delayMs={Math.min(idx * 40, 280)} />
        </li>
      ))}
    </ul>
  );
}
