"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import type { Route } from "next";
import { useMemo } from "react";

import { NewEcgButton } from "@/components/dashboard/NewEcgButton";
import { StatCard } from "@/components/dashboard/StatCard";
import { toSummary } from "@/lib/summary-from-report";
import { trpc } from "@/utils/trpc";

interface DashboardHomeProps {
  userName: string;
}

const longDateFormatter = new Intl.DateTimeFormat("pt-BR", {
  weekday: "long",
  day: "numeric",
  month: "long",
  year: "numeric",
});

const dateTimeFormatter = new Intl.DateTimeFormat("pt-BR", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function greetingFor(date: Date): string {
  const hour = date.getHours();
  if (hour < 12) return "Bom dia";
  if (hour < 18) return "Boa tarde";
  return "Boa noite";
}

function relativeFromNow(date: Date | null): string | null {
  if (!date) return null;
  const diffMs = Date.now() - date.getTime();
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return "agora";
  if (minutes < 60) return `${minutes}m atrás`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h atrás`;
  const days = Math.round(hours / 24);
  if (days === 1) return "ontem";
  if (days < 7) return `${days} dias`;
  const weeks = Math.round(days / 7);
  if (weeks === 1) return "1 sem";
  if (weeks < 4) return `${weeks} sem`;
  const months = Math.round(days / 30);
  return `${months}m`;
}

function firstName(fullName: string): string {
  const trimmed = fullName.trim();
  return trimmed.split(/\s+/)[0] ?? trimmed;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function DashboardHome({ userName }: DashboardHomeProps) {
  const now = new Date();
  const greeting = greetingFor(now);
  const dateLabel = capitalize(longDateFormatter.format(now));

  const { data } = useQuery(
    trpc.ecg.listAnalyses.queryOptions({ limit: 50 }),
  );

  const { totalExams, thisMonth, lastExam, lastExamAt } = useMemo(() => {
    const items = data?.items ?? [];
    const summaries = items.map(toSummary);
    const startOfMonth = new Date();
    startOfMonth.setDate(1);
    startOfMonth.setHours(0, 0, 0, 0);
    const thisMonthCount = summaries.filter(
      (s) => s.createdAt >= startOfMonth,
    ).length;
    const last = summaries[0] ?? null;
    return {
      totalExams: summaries.length,
      thisMonth: thisMonthCount,
      lastExam: last,
      lastExamAt: last?.createdAt ?? null,
    };
  }, [data]);

  const lastExamRel = relativeFromNow(lastExamAt);

  return (
    <div className="flex flex-col gap-8">
      <header className="apple-animate-fade-in-up">
        <h1 className="text-[34px] md:text-5xl font-bold tracking-[-0.02em] leading-[1.1] text-apple-text">
          {greeting}, Dr. {firstName(userName)}.
        </h1>
        <p className="mt-2 text-[13px] leading-[1.3] text-apple-text-secondary">
          {dateLabel}
        </p>
      </header>

      <section
        aria-label="Estatísticas"
        className="grid grid-cols-2 gap-3 md:grid-cols-3"
      >
        <StatCard value={totalExams} label="Exames" delayMs={50} />
        <StatCard value={thisMonth} label="Este mês" delayMs={120} />
        <StatCard
          value={lastExamRel ?? "—"}
          label="Último"
          delayMs={190}
          className="col-span-2 md:col-span-1"
        />
      </section>

      <section aria-label="Novo ECG">
        <NewEcgButton />
      </section>

      {lastExam && (
        <section
          aria-label="Último resultado"
          className="apple-animate-fade-in-up"
          style={{ animationDelay: "260ms" }}
        >
          <Link
            href={`/dashboard/resultado/${lastExam.id}` as Route}
            className="block rounded-2xl bg-apple-surface p-5 shadow-apple-sm transition-all duration-300 ease-[var(--ease-apple)] hover:-translate-y-0.5 hover:shadow-apple-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent focus-visible:ring-offset-2"
          >
            <div className="text-[12px] font-semibold uppercase tracking-[0.06em] text-apple-text-secondary">
              Último resultado
            </div>
            <div className="mt-3 text-[15px] text-apple-text-secondary">
              {dateTimeFormatter.format(lastExam.createdAt).replace(",", " ·")}
            </div>
            <p className="mt-2 line-clamp-2 text-[17px] leading-[1.47] text-apple-text">
              {lastExam.reportSummary ?? "Laudo disponível"}
            </p>
            <div className="mt-3 inline-flex items-center gap-1 text-[15px] font-medium text-apple-accent">
              Ver laudo completo
              <span aria-hidden="true">→</span>
            </div>
          </Link>
        </section>
      )}
    </div>
  );
}
