"use client";

import { Reveal } from "./reveal";

const blocks = [
  {
    icon: "🔒",
    title: "Dados Protegidos (LGPD)",
    description:
      "Nenhum dado identificável de paciente é coletado ou armazenado. Apenas a foto do ECG é processada. Conformidade total com a LGPD.",
  },
  {
    icon: "🩺",
    title: "Validado por Cardiologistas",
    description:
      "Todos os critérios diagnósticos e regras clínicas foram definidos e revisados por cardiologistas com experiência em eletrocardiografia.",
  },
  {
    icon: "⚠️",
    title: "Ferramenta de Apoio",
    description:
      "O ProECG é uma ferramenta de apoio à decisão clínica. Não substitui o julgamento médico. Cada laudo inclui disclaimer obrigatório.",
  },
  {
    icon: "🗑️",
    title: "Dados Temporários",
    description:
      "Análises expiram automaticamente após 30 dias. Exporte em PDF antes se quiser manter o registro.",
  },
];

export function Compliance() {
  return (
    <section className="bg-white px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[1000px]">
        <Reveal>
          <h2 className="mb-12 text-center text-3xl font-bold text-[#122056] sm:text-4xl">
            Segurança e conformidade
          </h2>
        </Reveal>
        <div className="grid gap-6 sm:grid-cols-2">
          {blocks.map((block, i) => (
            <Reveal key={block.title} delay={i * 0.1}>
              <div className="flex gap-4 rounded-2xl border border-[#E2E4F0] bg-[#FAFAFD] p-6">
                <span className="text-3xl">{block.icon}</span>
                <div>
                  <h3 className="text-lg font-semibold text-[#122056]">
                    {block.title}
                  </h3>
                  <p className="mt-1 text-sm text-[#4A5078] leading-relaxed">
                    {block.description}
                  </p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
