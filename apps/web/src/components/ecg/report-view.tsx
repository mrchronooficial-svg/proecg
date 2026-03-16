import { Card } from "@proecg/ui/components/card";
import type { EcgReport } from "@proecg/api/lib/modal";

import { MeasurementCard } from "./measurement-card";
import { DisclaimerBanner } from "./disclaimer-banner";

interface ReportViewProps {
  report: EcgReport;
  imageUrl: string;
}

export function ReportView({ report, imageUrl }: ReportViewProps) {
  const m = report.measurements;

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      <img
        src={imageUrl}
        alt="ECG analisado"
        className="w-full rounded-xl object-contain border border-border"
      />

      {/* Medições */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-primary">Medições</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {m.heartRate != null && (
            <MeasurementCard label="FC" value={m.heartRate} unit={m.heartRateUnit} />
          )}
          {m.axis != null && (
            <MeasurementCard label="Eixo" value={m.axis} unit={m.axisUnit} />
          )}
          {m.pr != null && (
            <MeasurementCard label="PR" value={m.pr} unit={m.prUnit} />
          )}
          {m.qrs != null && (
            <MeasurementCard label="QRS" value={m.qrs} unit={m.qrsUnit} />
          )}
          {m.qt != null && (
            <MeasurementCard label="QT" value={m.qt} unit={m.qtUnit} />
          )}
          {m.qtc != null && (
            <MeasurementCard label="QTc" value={m.qtc} unit={m.qtcUnit} />
          )}
        </div>
        {m.rhythm && (
          <p className="mt-2 text-sm text-muted-foreground">
            Ritmo: {m.rhythm}
          </p>
        )}
      </section>

      {/* Achados */}
      {report.findings.length > 0 && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-primary">Achados</h2>
          <Card className="p-4">
            <ul className="space-y-2 text-sm">
              {report.findings.map((f, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="mt-0.5 text-primary">•</span>
                  <div>
                    <span>{f.description}</span>
                    {f.leads_affected && f.leads_affected.length > 0 && (
                      <span className="ml-1.5 text-xs text-muted-foreground">
                        ({f.leads_affected.join(", ")})
                      </span>
                    )}
                    <span className="ml-1.5 inline-block rounded bg-secondary px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                      {f.source}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </Card>
        </section>
      )}

      {/* Hipóteses diagnósticas */}
      {report.diagnoses.length > 0 && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-primary">
            Hipóteses diagnósticas
          </h2>
          <div className="space-y-3">
            {report.diagnoses.map((d, i) => (
              <Card key={i} className="p-4">
                <p className="font-medium">{d.description}</p>
                <span className="mt-1 inline-block rounded bg-secondary px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                  {d.source}
                </span>
              </Card>
            ))}
          </div>
        </section>
      )}

      {/* Laudo completo em texto */}
      {report.reportText && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-primary">
            Laudo descritivo
          </h2>
          <Card className="p-4">
            <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-foreground">
              {report.reportText}
            </pre>
          </Card>
        </section>
      )}

      {/* Botões (desabilitados no MVP) */}
      <div className="flex gap-3">
        <button
          disabled
          className="rounded-lg border border-border bg-card px-4 py-2 text-sm text-muted-foreground opacity-50"
        >
          Exportar PDF
        </button>
        <button
          disabled
          className="rounded-lg border border-border bg-card px-4 py-2 text-sm text-muted-foreground opacity-50"
        >
          Compartilhar
        </button>
      </div>

      <DisclaimerBanner />
    </div>
  );
}
