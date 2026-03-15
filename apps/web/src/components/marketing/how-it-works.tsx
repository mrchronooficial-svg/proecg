"use client";

import { Reveal } from "./reveal";

const steps = [
  {
    num: "01",
    title: "Fotografe",
    description:
      "Abra o ProECG no celular e tire uma foto do ECG de papel. Pode ser traçado de 12 derivações, tira de ritmo ou ECG portátil.",
  },
  {
    num: "02",
    title: "IA Analisa",
    description:
      "Nossa IA digitaliza o traçado, mede todos os intervalos (FC, PR, QRS, QT, QTc, eixo) e cruza com critérios clínicos validados por cardiologistas.",
  },
  {
    num: "03",
    title: "Receba o Laudo",
    description:
      "Em segundos, você recebe o laudo descritivo completo com medições, achados e hipóteses diagnósticas. Exporte em PDF ou envie por WhatsApp.",
  },
];

export function HowItWorks() {
  return (
    <section id="como-funciona" className="bg-[#FAFAFD] px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[1200px]">
        <Reveal>
          <h2 className="text-center text-3xl font-bold text-[#122056] sm:text-4xl">
            Simples como tirar uma foto
          </h2>
        </Reveal>
        <Reveal delay={0.1}>
          <p className="mx-auto mt-4 max-w-lg text-center text-lg text-[#4A5078]">
            Do ECG de papel ao laudo completo em 3 passos.
          </p>
        </Reveal>

        <div className="relative mt-14 grid gap-6 sm:grid-cols-3">
          {/* Connecting line (desktop) */}
          <div className="absolute top-12 left-[16.67%] right-[16.67%] hidden h-px border-t-2 border-dashed border-[#5B65DC]/30 sm:block" />

          {steps.map((step, i) => (
            <Reveal key={step.num} delay={i * 0.15}>
              <div className="relative rounded-2xl border-t-4 border-[#5B65DC] bg-white p-8 shadow-sm">
                <span className="absolute -top-5 left-6 flex size-10 items-center justify-center rounded-full bg-[#5B65DC] text-sm font-bold text-white shadow">
                  {step.num}
                </span>
                <h3 className="mt-2 text-xl font-semibold text-[#122056]">
                  {step.title}
                </h3>
                <p className="mt-2 text-[#4A5078]">{step.description}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
