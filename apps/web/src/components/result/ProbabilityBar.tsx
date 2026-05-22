"use client";

import { useEffect, useState } from "react";

interface ProbabilityBarProps {
  /** 0-100 */
  probability: number;
}

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function probColor(p: number): string {
  if (p < 30) return "text-apple-success";
  if (p < 60) return "text-apple-text-secondary";
  return "text-apple-danger";
}

export function ProbabilityBar({ probability }: ProbabilityBarProps) {
  const target = clamp(Math.round(probability), 0, 100);
  // Anima o indicador do 0 ao valor real (800ms ease-out)
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setAnimated(target));
    return () => cancelAnimationFrame(raf);
  }, [target]);

  return (
    <div className="w-full">
      <div
        className={`font-mono text-[28px] font-bold tabular-nums leading-none ${probColor(target)}`}
      >
        {target}%
      </div>

      <div
        className="relative mt-3 h-[10px] w-full overflow-visible rounded-full"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={target}
        aria-label="Probabilidade de isquemia"
      >
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background:
              "linear-gradient(90deg, #30D158 0%, #30D158 25%, #E5E5EA 45%, #E5E5EA 55%, #FF3B30 75%, #FF3B30 100%)",
          }}
          aria-hidden="true"
        />
        <div
          className="absolute top-1/2 size-[18px] -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-apple-text bg-white shadow-apple-sm"
          style={{
            left: `${animated}%`,
            transition: "left 800ms cubic-bezier(0.16, 1, 0.3, 1)",
          }}
          aria-hidden="true"
        />
      </div>

      <div className="mt-2 flex justify-between text-[13px] font-medium">
        <span className="text-apple-success">Baixo</span>
        <span className="text-apple-text-tertiary">Inconclusivo</span>
        <span className="text-apple-danger">Alto</span>
      </div>
    </div>
  );
}
