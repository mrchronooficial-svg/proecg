import Link from "next/link";
import type { Route } from "next";
import { Activity, ChevronRight } from "lucide-react";
import type { EcgAnalysisSummary } from "@/types/dashboard";

interface EcgListItemProps {
  exam: EcgAnalysisSummary;
  /** Animação de delay (ms) — escalona entrada de múltiplos itens */
  delayMs?: number;
}

const dateFormatter = new Intl.DateTimeFormat("pt-BR", {
  day: "numeric",
  month: "long",
  year: "numeric",
});
const timeFormatter = new Intl.DateTimeFormat("pt-BR", {
  hour: "2-digit",
  minute: "2-digit",
});

export function EcgListItem({ exam, delayMs = 0 }: EcgListItemProps) {
  const dateStr = dateFormatter.format(exam.createdAt);
  const timeStr = timeFormatter.format(exam.createdAt);
  const summary =
    exam.reportSummary ?? "Análise sem resumo disponível";

  return (
    <Link
      href={`/dashboard/resultado/${exam.id}` as Route}
      className="group flex items-center gap-3.5 rounded-2xl bg-apple-surface p-4 shadow-apple-sm transition-all duration-200 ease-[var(--ease-apple)] hover:bg-[#F5F5F7] hover:shadow-apple-md active:bg-[#EDEDF0] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent focus-visible:ring-offset-2 apple-animate-fade-in-up"
      style={delayMs > 0 ? { animationDelay: `${delayMs}ms` } : undefined}
    >
      <div
        className="flex size-12 shrink-0 items-center justify-center rounded-full bg-apple-accent-light"
        aria-hidden="true"
      >
        <Activity
          size={22}
          strokeWidth={2}
          className="text-apple-accent"
        />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[17px] font-medium leading-[1.3] text-apple-text">
          {dateStr}
        </p>
        <p className="mt-0.5 truncate text-[15px] leading-[1.4] text-apple-text-secondary">
          {timeStr} · {summary}
        </p>
      </div>
      <ChevronRight
        size={20}
        className="shrink-0 text-apple-text-tertiary transition-transform duration-200 group-hover:translate-x-0.5"
        aria-hidden="true"
      />
    </Link>
  );
}
