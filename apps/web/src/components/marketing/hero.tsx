"use client";

import Link from "next/link";
import { Reveal } from "./reveal";

export function Hero() {
  return (
    <section className="relative overflow-hidden bg-[#FAFAFD] px-5 pt-12 pb-10 sm:px-4 sm:pt-28 sm:pb-20 lg:pt-32 lg:pb-24">
      <div className="mx-auto max-w-[1200px]">
        <div className="flex flex-col items-start sm:items-center sm:text-center">
          {/* Badge */}
          <Reveal>
            <span className="mb-5 inline-flex items-center gap-2 rounded-full bg-[#EEEFFD] px-4 py-1.5 text-sm font-medium text-[#5B65DC] sm:mb-6">
              Laudo de ECG com IA em segundos
            </span>
          </Reveal>

          {/* Headline */}
          <Reveal delay={0.1}>
            <h1 className="max-w-[700px] text-[32px] font-extrabold leading-[1.15] text-[#122056] sm:mx-auto sm:text-5xl lg:text-[56px]">
              Fotografou o ECG?{" "}
              <span className="text-[#5B65DC]">O laudo já saiu.</span>
            </h1>
          </Reveal>

          {/* Subtitle */}
          <Reveal delay={0.2}>
            <p className="mt-5 max-w-[640px] text-[17px] leading-[1.6] text-[#4A5078] sm:mx-auto sm:mt-6 sm:text-xl">
              Tire uma foto do ECG de papel com seu celular e receba um laudo
              descritivo completo com hipóteses diagnósticas — validado por
              cardiologistas, direto no seu bolso.
            </p>
          </Reveal>

          {/* CTA */}
          <Reveal delay={0.3}>
            <div className="mt-8 flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center sm:justify-center">
              <Link
                href="/login"
                className="rounded-full bg-[#5B65DC] px-8 py-4 text-center text-base font-semibold text-white shadow-[0_4px_14px_rgba(91,101,220,0.35)] transition-all hover:bg-[#4A51C5] hover:-translate-y-0.5 hover:shadow-[0_6px_20px_rgba(91,101,220,0.4)] sm:text-lg"
              >
                Começar Agora — R$ 197/mês
              </Link>
              <p className="text-center text-sm text-[#4A5078] sm:hidden">
                Pix ou cartão · Cancele quando quiser
              </p>
              <p className="hidden text-sm text-[#4A5078] sm:block">
                Pix ou cartão · Cancele quando quiser · Acesso imediato
              </p>
            </div>
          </Reveal>

          {/* Mockup */}
          <Reveal delay={0.4}>
            <div className="mt-10 w-full rounded-2xl border border-[#E2E4F0] bg-white p-5 shadow-[0_0_40px_rgba(91,101,220,0.1)] sm:mx-auto sm:mt-12 sm:max-w-[600px] sm:p-8">
              <div className="space-y-3 font-mono text-sm text-[#122056]">
                <div className="flex items-center gap-2 text-xs text-[#4A5078]">
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#10B981] opacity-75" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-[#10B981]" />
                  </span>
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
