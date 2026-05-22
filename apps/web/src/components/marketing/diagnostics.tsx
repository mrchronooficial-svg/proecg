"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Reveal } from "./reveal";

const groups = [
  {
    title: "Ritmos",
    items: [
      "Bradicardia sinusal", "Taquicardia sinusal", "Fibrilação atrial",
      "Flutter atrial", "Taquicardia supraventricular", "Taquicardia ventricular",
      "Ritmo juncional", "Pausa sinusal",
    ],
  },
  {
    title: "Infarto",
    items: [
      "IAMCSST anterior, inferior, lateral, posterior",
      "Equivalentes: T hiperaguda, infarto posterior",
      "Padrão de Wellens", "Padrão de De Winter",
      "Critérios de Sgarbossa", "Supra em aVR",
    ],
  },
  {
    title: "Bloqueios",
    items: [
      "BAV de 1º grau", "BAV de 2º grau Mobitz I", "BAV de 2º grau Mobitz II",
      "BAV total (3º grau)", "Bloqueio de ramo direito (BRD)",
      "Bloqueio de ramo esquerdo (BRE)", "BDAS", "BDPI",
      "Bifascicular", "Trifascicular",
    ],
  },
  {
    title: "Eixo",
    items: ["Eixo normal", "Desvio para esquerda", "Desvio para direita", "Desvio extremo"],
  },
  {
    title: "Medições",
    items: [
      "Frequência cardíaca (FC)", "Onda P", "Intervalo PR", "Duração QRS",
      "Intervalo QT", "QTc (Bazett, Fridericia)", "Intervalo RR",
    ],
  },
  {
    title: "Outros",
    items: [
      "Sobrecarga ventricular esquerda (SVE)", "Sobrecarga ventricular direita (SVD)",
      "Sobrecarga atrial esquerda (SAE)", "Sobrecarga atrial direita (SAD)",
      "QT longo", "QT curto", "WPW (Wolff-Parkinson-White)",
      "Brugada", "Pericardite", "Hipercalemia", "Hipocalemia",
    ],
  },
];

export function Diagnostics() {
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  return (
    <section id="diagnostics" className="bg-white py-20 sm:py-32 lg:py-40">
      <div className="mx-auto max-w-5xl px-6">
        <Reveal className="text-center">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#0066CC] sm:text-[13px]">
            Cobertura clínica
          </p>
          <h2 className="mt-4 text-[30px] font-bold leading-[1.1] tracking-[-0.02em] break-words text-[#1d1d1f] sm:mt-5 sm:text-[48px] lg:text-[56px]">
            Mais de 30 condições detectadas.
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-[15px] leading-[1.5] text-[#86868b] sm:mt-5 sm:text-[18px]">
            Cobertura ampla de ritmos, isquemia, bloqueios e distúrbios eletrolíticos.
            Diagnósticos em validação clínica não são exibidos no laudo.
          </p>
        </Reveal>

        <div className="mt-10 divide-y divide-black/5 border-y border-black/5 sm:mt-16 lg:mt-20">
          {groups.map((g, i) => {
            const open = openIdx === i;
            return (
              <Reveal key={g.title} delay={i * 0.04}>
                <button
                  onClick={() => setOpenIdx(open ? null : i)}
                  className="flex w-full items-center justify-between py-5 text-left sm:py-7"
                >
                  <span className="text-[18px] font-bold tracking-[-0.01em] text-[#1d1d1f] sm:text-[24px] lg:text-[28px]">
                    {g.title}
                  </span>
                  <ChevronDown
                    className={`h-5 w-5 flex-shrink-0 text-[#86868b] transition-transform sm:h-6 sm:w-6 ${
                      open ? "rotate-180" : ""
                    }`}
                  />
                </button>
                <div
                  className={`grid transition-all duration-500 ${
                    open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
                  }`}
                >
                  <div className="overflow-hidden">
                    <div className="grid grid-cols-1 gap-x-8 gap-y-2.5 pb-6 text-[14px] text-[#1d1d1f]/80 sm:grid-cols-2 sm:gap-y-2 sm:pb-8 sm:text-[16px]">
                      {g.items.map((it) => (
                        <div key={it} className="flex items-start gap-2.5">
                          <span className="mt-1.5 h-1 w-1 flex-shrink-0 rounded-full bg-[#0066CC] sm:mt-2" />
                          <span>{it}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
