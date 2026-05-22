import type { IschemiaResult } from "@/types/result";

import { ProbabilityBar } from "./ProbabilityBar";

interface IschemiaCardProps {
  ischemia: IschemiaResult;
}

export function IschemiaCard({ ischemia }: IschemiaCardProps) {
  const showDiagnosis =
    ischemia.detected && ischemia.probability > 30 && ischemia.diagnosis;
  const headlineColor = ischemia.detected
    ? "text-apple-danger"
    : "text-apple-text";
  const headline = ischemia.detected
    ? "Sinais sugestivos de isquemia"
    : "Sem sinais de isquemia detectados";

  return (
    <div className="rounded-2xl bg-apple-surface p-5 shadow-apple-sm">
      <p className={`text-[18px] font-semibold leading-[1.3] ${headlineColor}`}>
        {headline}
      </p>

      <div className="mt-5">
        <ProbabilityBar probability={ischemia.probability} />
      </div>

      <p className="mt-4 border-t border-apple-border-light pt-3 text-[13px] italic leading-[1.4] text-apple-text-tertiary">
        O ECG isolado não exclui infarto — correlacionar com clínica e troponina.
      </p>

      {showDiagnosis && (
        <div className="mt-4 border-t border-apple-border-light pt-4">
          <p className="text-[13px] text-apple-text-tertiary">
            Diagnóstico identificado
          </p>
          <p className="mt-1 text-[17px] font-semibold leading-[1.4] text-apple-text">
            {ischemia.diagnosis}
          </p>
          {(ischemia.wall || ischemia.leads) && (
            <p className="mt-2 text-[14px] leading-[1.4] text-apple-text-secondary">
              <span className="font-medium">Parede: </span>
              {ischemia.wall ?? ischemia.leads}
            </p>
          )}
          {ischemia.artery && (
            <p className="mt-1 text-[14px] leading-[1.4] text-apple-text-secondary">
              <span className="font-medium">Artéria provável: </span>
              {ischemia.artery}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
