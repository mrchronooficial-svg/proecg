"use client";

import type { EcgReport } from "@/types/result";

import { ActionBar } from "./ActionBar";
import { ArrhythmiaCard } from "./ArrhythmiaCard";
import { DisclaimerBanner } from "./DisclaimerBanner";
import { EcgImage } from "./EcgImage";
import { IschemiaCard } from "./IschemiaCard";
import { RedFlagAlert } from "./RedFlagAlert";
import { ReportCard } from "./ReportCard";
import { ResultHeader } from "./ResultHeader";
import { SectionHeader } from "./SectionHeader";

interface ResultPageViewProps {
  report: EcgReport;
}

const dateTimeFormatter = new Intl.DateTimeFormat("pt-BR", {
  day: "numeric",
  month: "long",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function shortId(id: string): string {
  const cleaned = id.replace(/[^a-zA-Z0-9]/g, "");
  return cleaned.slice(0, 6);
}

export function ResultPageView({ report }: ResultPageViewProps) {
  const dateText = dateTimeFormatter
    .format(report.createdAt)
    .replace(",", " às");

  return (
    <div className="flex flex-col gap-5">
      <ResultHeader />

      <DisclaimerBanner />

      <header className="apple-animate-fade-in-up">
        <p className="text-[22px] font-bold tracking-[-0.01em] text-apple-text">
          Exame #{shortId(report.id)}
        </p>
        <p className="mt-1 text-[15px] text-apple-text-secondary">{dateText}</p>
      </header>

      <section
        aria-label="Foto original do ECG"
        className="apple-animate-fade-in-up"
        style={{ animationDelay: "60ms" }}
      >
        <EcgImage imageUrl={report.imageUrl} />
      </section>

      {report.redFlags.length > 0 && (
        <section
          aria-label="Alertas"
          className="flex flex-col gap-2 apple-animate-fade-in-up"
          style={{ animationDelay: "120ms" }}
        >
          {report.redFlags.map((flag, i) => (
            <RedFlagAlert key={i} flag={flag} />
          ))}
        </section>
      )}

      <section
        aria-label="Isquemia"
        className="apple-animate-fade-in-up"
        style={{ animationDelay: "180ms" }}
      >
        <SectionHeader title="Isquemia" />
        <IschemiaCard ischemia={report.ischemia} />
      </section>

      <section
        aria-label="Arritmia"
        className="apple-animate-fade-in-up"
        style={{ animationDelay: "240ms" }}
      >
        <SectionHeader title="Arritmia" />
        <ArrhythmiaCard arrhythmia={report.arrhythmia} />
      </section>

      <section
        aria-label="Laudo"
        className="apple-animate-fade-in-up"
        style={{ animationDelay: "300ms" }}
      >
        <SectionHeader title="Laudo" />
        <ReportCard text={report.reportText} />
      </section>

      <section aria-label="Ações">
        <ActionBar reportId={report.id} reportText={report.reportText} />
      </section>
    </div>
  );
}
