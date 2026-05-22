import { CheckCircle2 } from "lucide-react";
import type { ArrhythmiaResult, ArrhythmiaUrgency } from "@/types/result";

interface ArrhythmiaCardProps {
  arrhythmia: ArrhythmiaResult;
}

const urgencyClass: Record<ArrhythmiaUrgency, string> = {
  emergency: "border-[#FECACA] bg-[#FEF2F2] text-[#DC2626]",
  warning: "border-[#FED7AA] bg-[#FFF7ED] text-[#EA580C]",
  info: "border-[#D4D6F9] bg-apple-accent-light text-apple-accent",
};

export function ArrhythmiaCard({ arrhythmia }: ArrhythmiaCardProps) {
  if (!arrhythmia.detected) {
    return (
      <div className="flex items-center gap-3 rounded-2xl bg-apple-surface p-5 shadow-apple-sm">
        <CheckCircle2
          size={28}
          strokeWidth={2}
          className="shrink-0 text-apple-success"
          aria-hidden="true"
        />
        <p className="text-[17px] font-medium leading-[1.3] text-apple-text">
          Ritmo sinusal — sem arritmias detectadas
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-apple-surface p-5 shadow-apple-sm">
      <p className="text-[18px] font-semibold leading-[1.3] text-apple-text">
        {arrhythmia.name}
      </p>

      {arrhythmia.location && (
        <div className="mt-3">
          <p className="text-[13px] text-apple-text-tertiary">
            Identificado em
          </p>
          <p className="mt-0.5 text-[15px] leading-[1.4] text-apple-text-secondary">
            {arrhythmia.location}
          </p>
        </div>
      )}

      {arrhythmia.characteristics.length > 0 && (
        <ul className="mt-3 flex flex-col gap-2 border-t border-apple-border-light pt-3">
          {arrhythmia.characteristics.map((c, i) => (
            <li key={i} className="flex items-start gap-2.5">
              <span
                aria-hidden="true"
                className="mt-2 size-1.5 shrink-0 rounded-full bg-apple-accent"
              />
              <span className="text-[15px] leading-[1.4] text-apple-text">
                {c}
              </span>
            </li>
          ))}
        </ul>
      )}

      {arrhythmia.urgency && arrhythmia.urgencyMessage && (
        <div
          className={`mt-4 inline-block rounded-[10px] border px-3 py-1.5 text-[13px] font-semibold ${urgencyClass[arrhythmia.urgency]}`}
        >
          {arrhythmia.urgencyMessage}
        </div>
      )}
    </div>
  );
}
