"use client";

import { Reveal } from "./reveal";

const diagCategories = [
  {
    label: "Isquêmicos",
    color: "bg-red-100 text-red-700",
    items: ["SCA com supra", "SCA sem supra", "Wellens", "Winter", "de Winter", "Sgarbossa"],
  },
  {
    label: "Arritmias",
    color: "bg-orange-100 text-orange-700",
    items: ["FA", "Flutter", "TSV", "TV", "Torsades", "WPW"],
  },
  {
    label: "Bloqueios",
    color: "bg-blue-100 text-blue-700",
    items: ["BRD", "BRE", "BAV 1°", "BAV 2° (M1/M2)", "BAVT"],
  },
  {
    label: "Outros",
    color: "bg-purple-100 text-purple-700",
    items: ["Brugada", "QT longo", "Hipercalemia", "Pericardite", "TEP"],
  },
];

const blocks = [
  {
    title: "Laudo em segundos, não em horas",
    body: "No plantão, cada minuto conta. O ProECG entrega um laudo descritivo completo antes mesmo de você terminar de preencher o prontuário. Sem esperar parecer, sem ligar para o cardiologista de madrugada.",
    visual: (
      <div className="flex items-center justify-center rounded-2xl bg-[#EEEFFD] p-8">
        <div className="flex items-center gap-4">
          <div className="text-center">
            <p className="font-mono text-4xl font-bold text-[#5B65DC]">12s</p>
            <p className="mt-1 text-sm text-[#4A5078]">Tempo médio</p>
          </div>
          <div className="text-4xl text-[#5B65DC]">→</div>
          <div className="rounded-xl bg-white p-4 shadow-sm">
            <p className="text-xs font-medium text-[#4A5078]">Laudo completo</p>
            <p className="mt-1 font-mono text-sm text-[#122056]">FC: 78 bpm · QTc: 420ms</p>
            <p className="mt-0.5 text-sm text-[#122056]">SCA com supra anterior</p>
          </div>
        </div>
      </div>
    ),
  },
  {
    title: "~30 diagnósticos na ponta dos dedos",
    body: "De SCA com supra a Brugada, de FA a bloqueios atrioventriculares — nosso motor cobre os diagnósticos que mais importam na emergência. Critérios clínicos definidos e validados por cardiologistas.",
    visual: (
      <div className="rounded-2xl bg-[#EEEFFD] p-6">
        <div className="space-y-3">
          {diagCategories.map((cat) => (
            <div key={cat.label}>
              <p className="mb-1.5 text-xs font-semibold text-[#4A5078]">{cat.label}</p>
              <div className="flex flex-wrap gap-1.5">
                {cat.items.map((item) => (
                  <span
                    key={item}
                    className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${cat.color}`}
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    title: "IA + olhar do cardiologista",
    body: "Nosso motor combina algoritmos de digitalização, medições automáticas, regras clínicas baseadas em critérios consagrados e classificação por rede neural — tudo construído e validado em parceria com cardiologistas. Não é chatbot: é pipeline médico com rigor clínico.",
    visual: (
      <div className="flex items-center justify-center rounded-2xl bg-[#EEEFFD] p-8">
        <div className="flex flex-wrap items-center justify-center gap-2 text-sm font-medium text-[#122056]">
          {["Foto", "Digitalização", "Medições", "Regras + CNN", "Laudo"].map(
            (step, i, arr) => (
              <span key={step} className="flex items-center gap-2">
                <span className="rounded-lg bg-white px-3 py-1.5 shadow-sm">{step}</span>
                {i < arr.length - 1 && <span className="text-[#5B65DC]">→</span>}
              </span>
            ),
          )}
        </div>
      </div>
    ),
  },
  {
    title: "Feito para o bolso do jaleco",
    body: "Acesse direto do celular, sem instalar nada. Funciona no navegador — abra, fotografe, receba o laudo. Exporte em PDF, compartilhe por WhatsApp com a equipe ou cole no prontuário.",
    visual: (
      <div className="flex items-center justify-center rounded-2xl bg-[#EEEFFD] p-8">
        <div className="w-44 rounded-[28px] border-4 border-[#122056] bg-white p-3 shadow-lg">
          <div className="rounded-xl bg-[#FAFAFD] p-3">
            <p className="text-xs font-bold text-[#122056]">ProECG</p>
            <div className="mt-2 rounded-lg bg-[#5B65DC] px-3 py-2 text-center text-xs font-semibold text-white">
              Novo ECG
            </div>
            <div className="mt-2 space-y-1">
              <div className="h-2 w-full rounded bg-[#E2E4F0]" />
              <div className="h-2 w-3/4 rounded bg-[#E2E4F0]" />
            </div>
          </div>
        </div>
      </div>
    ),
  },
];

export function Benefits() {
  return (
    <section id="beneficios" className="bg-white px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[1200px]">
        <Reveal>
          <h2 className="mb-16 text-center text-3xl font-bold text-[#122056] sm:text-4xl">
            Por que médicos escolhem o ProECG
          </h2>
        </Reveal>

        <div className="space-y-20">
          {blocks.map((block, i) => (
            <Reveal key={block.title}>
              <div
                className={`flex flex-col items-center gap-10 lg:flex-row ${
                  i % 2 === 1 ? "lg:flex-row-reverse" : ""
                }`}
              >
                <div className="flex-1 space-y-4">
                  <h3 className="text-2xl font-semibold text-[#122056]">
                    {block.title}
                  </h3>
                  <p className="text-lg text-[#4A5078] leading-relaxed">
                    {block.body}
                  </p>
                </div>
                <div className="w-full flex-1">{block.visual}</div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
