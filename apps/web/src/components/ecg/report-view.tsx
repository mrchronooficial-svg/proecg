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
        className="w-full rounded-md object-contain"
      />

      <section>
        <h2 className="mb-3 text-lg font-semibold text-primary">Medições</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <MeasurementCard label="FC" value={m.heartRate} unit={m.heartRateUnit} />
          <MeasurementCard label="Eixo" value={m.axis} unit={m.axisUnit} />
          <MeasurementCard label="PR" value={m.pr} unit={m.prUnit} />
          <MeasurementCard label="QRS" value={m.qrs} unit={m.qrsUnit} />
          <MeasurementCard label="QT" value={m.qt} unit={m.qtUnit} />
          <MeasurementCard label="QTc" value={m.qtc} unit={m.qtcUnit} />
        </div>
      </section>

      {report.findings.length > 0 && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-primary">Achados</h2>
          <Card className="p-4">
            <ul className="space-y-1 text-sm">
              {report.findings.map((f, i) => (
                <li key={i}>• {f.description}</li>
              ))}
            </ul>
          </Card>
        </section>
      )}

      {report.diagnoses.length > 0 && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-primary">Hipóteses diagnósticas</h2>
          <div className="space-y-3">
            {report.diagnoses.map((d, i) => (
              <Card key={i} className="p-4">
                <p className="font-medium">Sugestivo de: {d.name}</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {d.description}
                </p>
              </Card>
            ))}
          </div>
        </section>
      )}

      <div className="flex gap-3">
        <button
          disabled
          className="glass rounded-lg border-0 px-4 py-2 text-sm text-muted-foreground opacity-50"
        >
          Exportar PDF
        </button>
        <button
          disabled
          className="glass rounded-lg border-0 px-4 py-2 text-sm text-muted-foreground opacity-50"
        >
          Compartilhar
        </button>
      </div>

      <DisclaimerBanner />
    </div>
  );
}
