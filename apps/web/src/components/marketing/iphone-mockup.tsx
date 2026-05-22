"use client";

import type { ReactNode } from "react";

/** Mockup CSS-only de iPhone. Filho ocupa a tela. */
export function IPhoneMockup({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`relative mx-auto aspect-[9/19.5] w-full max-w-[300px] rounded-[44px] bg-[#1d1d1f] p-[10px] shadow-[0_12px_30px_-10px_rgba(0,0,0,0.4)] ring-1 ring-white/10 sm:max-w-[280px] sm:shadow-[0_30px_80px_-20px_rgba(0,0,0,0.5),0_10px_30px_-10px_rgba(0,0,0,0.3)] ${className}`}
    >
      {/* notch */}
      <div className="pointer-events-none absolute left-1/2 top-[14px] z-20 h-[22px] w-[90px] -translate-x-1/2 rounded-full bg-black" />
      {/* tela */}
      <div className="relative h-full w-full overflow-hidden rounded-[36px] bg-white">
        {children}
      </div>
    </div>
  );
}

/** Linha de ECG SVG animada — parametrizável. */
export function ECGTrace({ color = "#0066CC", className = "" }: { color?: string; className?: string }) {
  return (
    <svg viewBox="0 0 400 80" className={`ecg-trace ${className}`} fill="none" stroke={color} strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
      <path
        d="M0 40 L40 40 L50 35 L60 40 L80 40 L88 25 L92 55 L96 10 L100 70 L104 40 L130 40 L140 30 L150 40 L180 40 L185 38 L195 42 L200 40 L240 40 L250 35 L260 40 L280 40 L288 25 L292 55 L296 10 L300 70 L304 40 L330 40 L340 30 L350 40 L400 40"
        strokeDasharray="1600 0"
        style={{ animation: "ecg-dash 3s linear infinite" }}
      />
      <style>{`
        @keyframes ecg-dash {
          from { stroke-dasharray: 0 1600; }
          to   { stroke-dasharray: 1600 0; }
        }
        @media (prefers-reduced-motion: reduce) {
          .ecg-trace path { animation: none !important; stroke-dasharray: 1600 0 !important; }
        }
      `}</style>
    </svg>
  );
}
