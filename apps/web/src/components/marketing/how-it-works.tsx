"use client";

import { Camera, Cpu, FileText } from "lucide-react";
import { Reveal } from "./reveal";
import { ECGTrace, IPhoneMockup } from "./iphone-mockup";

const steps = [
  {
    n: "01",
    icon: Camera,
    title: "Tire uma foto do ECG de 12 derivações",
    body: "Use a câmera do celular. Funciona com papel térmico desbotado, foto torta ou iluminação ruim. Aceita imagem de qualquer fabricante de eletrocardiógrafo.",
    mockup: "camera",
  },
  {
    n: "02",
    icon: Cpu,
    title: "IA detecta mais de 30 condições em segundos",
    body: "Pipeline próprio digitaliza o traçado, mede intervalos, classifica padrões e checa critérios clínicos da SBC e AHA. Tudo em < 15 segundos.",
    mockup: "diagnoses",
  },
  {
    n: "03",
    icon: FileText,
    title: "Receba o laudo estruturado completo",
    body: "Ritmo, frequência, eixo, intervalos, morfologia, achados e hipóteses diagnósticas. Exporte como PDF ou compartilhe direto via WhatsApp.",
    mockup: "report",
  },
];

export function HowItWorks() {
  return (
    <section id="how" className="bg-white py-20 sm:py-32 lg:py-40">
      <div className="mx-auto max-w-7xl px-6">
        <Reveal className="text-center">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#0066CC] sm:text-[13px]">
            Como Funciona
          </p>
          <h2 className="mt-4 text-[30px] font-bold leading-[1.1] tracking-[-0.02em] break-words text-[#1d1d1f] sm:mt-5 sm:text-[48px] lg:text-[56px]">
            3 passos para o laudo instantâneo.
          </h2>
        </Reveal>

        <div className="mt-14 space-y-16 sm:mt-24 sm:space-y-32 lg:mt-32 lg:space-y-40">
          {steps.map((s, idx) => (
            <Reveal key={s.n}>
              <div className="grid items-center gap-8 lg:grid-cols-2 lg:gap-20">
                <div className={`flex justify-center ${idx % 2 === 0 ? "lg:order-2" : "lg:order-1"}`}>
                  <IPhoneMockup>
                    <StepScreen kind={s.mockup as "camera" | "diagnoses" | "report"} />
                  </IPhoneMockup>
                </div>

                <div className={`text-center lg:text-left ${idx % 2 === 0 ? "lg:order-1" : "lg:order-2"}`}>
                  <div className="mb-4 flex items-center justify-center gap-2.5 sm:mb-5 sm:gap-3 lg:justify-start">
                    <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-[#0066CC]/10 text-[#0066CC] sm:h-9 sm:w-9">
                      <s.icon className="h-4 w-4" />
                    </span>
                    <span className="text-[11px] font-semibold tracking-[0.15em] text-[#0066CC] sm:text-[13px]">
                      PASSO {s.n}
                    </span>
                  </div>
                  <h3 className="text-[22px] font-bold leading-[1.2] tracking-[-0.01em] text-[#1d1d1f] sm:text-[36px] lg:text-[44px]">
                    {s.title}
                  </h3>
                  <p className="mx-auto mt-4 max-w-xl text-[15px] leading-[1.55] text-[#86868b] sm:mt-5 sm:text-[18px] lg:mx-0 lg:text-[19px]">
                    {s.body}
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

function StepScreen({ kind }: { kind: "camera" | "diagnoses" | "report" }) {
  if (kind === "camera") {
    return (
      <div className="relative flex h-full flex-col bg-[#0a0a0a] text-white">
        <div className="flex items-center justify-between bg-black/50 px-4 pt-8 pb-2 text-[9px]">
          <span>9:41</span>
          <span className="opacity-60">5G</span>
        </div>
        <div className="flex flex-1 items-center justify-center bg-[radial-gradient(circle,rgba(255,255,255,0.05),transparent)]">
          <div className="rounded-2xl border-2 border-white/30 px-6 py-8 text-center">
            <div className="mx-auto mb-3 grid h-12 w-20 grid-cols-4 grid-rows-3 gap-px overflow-hidden rounded bg-white/95 p-1">
              <ECGTrace color="#1d1d1f" className="col-span-4 row-span-3 h-full w-full" />
            </div>
            <div className="text-[10px] font-medium">Posicione o ECG</div>
            <div className="text-[8px] text-white/55">Câmera detecta automaticamente</div>
          </div>
        </div>
        <div className="flex justify-center bg-black/30 pb-8 pt-4">
          <div className="h-12 w-12 rounded-full border-[3px] border-white bg-white/95" />
        </div>
      </div>
    );
  }

  if (kind === "diagnoses") {
    return (
      <div className="flex h-full flex-col bg-[#fafafa] text-[10px] text-[#1d1d1f]">
        <div className="bg-white px-4 pt-8 pb-3 text-[9px]">
          <div className="flex items-center justify-between">
            <span>9:41</span>
            <span className="opacity-60">5G</span>
          </div>
          <div className="mt-3 text-[12px] font-semibold">Analisando…</div>
        </div>
        <div className="flex-1 space-y-2 p-3">
          {[
            { t: "Ritmo sinusal", c: "ok" },
            { t: "FC: 78 bpm", c: "ok" },
            { t: "Eixo: +60°", c: "ok" },
            { t: "Sem supra de ST", c: "ok" },
            { t: "QTc dentro do normal", c: "ok" },
            { t: "Sem padrão de Wellens", c: "ok" },
            { t: "Sem padrão de Sgarbossa", c: "ok" },
          ].map((d, i) => (
            <div
              key={i}
              className="flex items-center gap-2 rounded-lg bg-white p-2.5 text-[9px] shadow-[0_1px_3px_rgba(0,0,0,0.05)]"
            >
              <span className="h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-500" />
              <span>{d.t}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // report
  return (
    <div className="flex h-full flex-col bg-white text-[10px] text-[#1d1d1f]">
      <div className="px-4 pt-8 pb-2 text-[9px]">
        <div className="flex items-center justify-between">
          <span>9:41</span>
          <span className="opacity-60">5G</span>
        </div>
      </div>
      <div className="flex-1 space-y-3 p-4">
        <div>
          <div className="text-[8px] uppercase tracking-wide text-[#1d1d1f]/50">Laudo</div>
          <div className="text-[11px] font-semibold">ECG · 12 derivações</div>
        </div>
        <div className="space-y-1.5 text-[8.5px] leading-relaxed">
          <p><strong>Ritmo:</strong> sinusal.</p>
          <p><strong>FC:</strong> 78 bpm.</p>
          <p><strong>Eixo:</strong> +60°.</p>
          <p><strong>Intervalos:</strong> PR 180 / QRS 88 / QT 380 / QTc 420.</p>
          <p className="pt-1.5"><strong>Achados:</strong> sem alterações isquêmicas agudas.</p>
          <p><strong>Conclusão:</strong> ECG dentro dos limites da normalidade.</p>
        </div>
        <div className="flex gap-1.5 pt-2">
          <div className="flex-1 rounded bg-[#0066CC] py-1.5 text-center text-[8px] font-medium text-white">PDF</div>
          <div className="flex-1 rounded bg-emerald-500 py-1.5 text-center text-[8px] font-medium text-white">WhatsApp</div>
        </div>
      </div>
    </div>
  );
}
