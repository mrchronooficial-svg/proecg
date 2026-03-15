"use client";

import Link from "next/link";
import { Reveal } from "./reveal";

export function Hero() {
  return (
    <section className="relative overflow-hidden bg-[#FAFAFD] px-4 pt-20 pb-16 sm:pt-28 sm:pb-20 lg:pt-32 lg:pb-24">
      <div className="mx-auto max-w-[1200px]">
        <div className="flex flex-col items-center text-center">
          {/* Badge */}
          <Reveal>
            <span className="mb-6 inline-flex items-center gap-2 rounded-full bg-[#EEEFFD] px-4 py-1.5 text-sm font-medium text-[#5B65DC]">
              Laudo de ECG com IA em segundos
            </span>
          </Reveal>

          {/* Headline */}
          <Reveal delay={0.1}>
            <h1 className="mx-auto max-w-[700px] text-4xl font-extrabold leading-[1.1] text-[#122056] sm:text-5xl lg:text-[56px]">
              Fotografou o ECG?{" "}
              <span className="text-[#5B65DC]">O laudo já saiu.</span>
            </h1>
          </Reveal>

          {/* Subtitle */}
          <Reveal delay={0.2}>
            <p className="mx-auto mt-6 max-w-[640px] text-lg text-[#4A5078] sm:text-xl">
              Tire uma foto do ECG de papel com seu celular e receba um laudo
              descritivo completo com hipóteses diagnósticas — validado por
              cardiologistas, direto no seu bolso.
            </p>
          </Reveal>

          {/* CTA */}
          <Reveal delay={0.3}>
            <div className="mt-8 flex flex-col items-center gap-3">
              <Link
                href="/login"
                className="rounded-xl bg-[#5B65DC] px-10 py-4 text-lg font-semibold text-white shadow-[0_4px_14px_rgba(91,101,220,0.35)] transition-all hover:bg-[#4A51C5] hover:-translate-y-0.5 hover:shadow-[0_6px_20px_rgba(91,101,220,0.4)]"
              >
                Começar Agora — R$ 197/mês
              </Link>
              <p className="text-sm text-[#4A5078]">
                Pix ou cartão · Cancele quando quiser · Acesso imediato
              </p>
            </div>
          </Reveal>

          {/* Mockup */}
          <Reveal delay={0.4}>
            <div className="mx-auto mt-12 w-full max-w-[600px] rounded-2xl border border-[#E2E4F0] bg-white p-6 shadow-[0_0_40px_rgba(91,101,220,0.1)] sm:p-8">
              <div className="space-y-3 font-mono text-sm text-[#122056]">
                <div className="flex items-center gap-2 text-xs text-[#4A5078]">
                  <div className="h-2 w-2 rounded-full bg-[#10B981]" />
                  Laudo gerado em 12 segundos
                </div>
                <div className="h-px bg-[#E2E4F0]" />
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                  <span>Ritmo: <strong>Sinusal</strong></span>
                  <span>FC: <strong>82 bpm</strong></span>
                  <span>Eixo: <strong>+45°</strong></span>
                  <span>PR: <strong>168ms</strong></span>
                  <span>QRS: <strong>86ms</strong></span>
                  <span>QT: <strong>372ms</strong></span>
                  <span className="col-span-2">QTc: <strong>418ms</strong> (Bazett)</span>
                </div>
                <div className="h-px bg-[#E2E4F0]" />
                <div>
                  <p className="text-xs font-medium text-[#4A5078]">Achados:</p>
                  <p className="text-sm">Supradesnivelamento de ST em V1-V4</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-[#4A5078]">Hipótese diagnóstica:</p>
                  <p className="text-sm">Sugestivo de SCA com supra em parede anterior</p>
                </div>
                <div className="h-px bg-[#E2E4F0]" />
                <p className="text-xs text-[#F59E0B]">
                  ⚠️ Ferramenta de apoio — correlacionar com clínica
                </p>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
