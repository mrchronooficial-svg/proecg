"use client";

import { Reveal } from "./reveal";

export function LaudoExample() {
  return (
    <section className="bg-[#EEEFFD] px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[1200px]">
        <Reveal>
          <h2 className="text-center text-3xl font-bold text-[#122056] sm:text-4xl">
            Veja um laudo real
          </h2>
        </Reveal>
        <Reveal delay={0.1}>
          <p className="mx-auto mt-4 max-w-lg text-center text-lg text-[#4A5078]">
            Esse é o resultado que você recebe em segundos após fotografar o ECG.
          </p>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="mx-auto mt-10 max-w-[600px] rounded-2xl border-l-4 border-[#5B65DC] bg-white p-6 shadow-md sm:p-8">
            <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-[#122056]">
{`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAUDO ECG — ProECG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ritmo: Sinusal
Frequência cardíaca: 78 bpm
Eixo elétrico: +60°

Intervalos:
  PR: 180ms | QRS: 88ms
  QT: 380ms | QTc: 420ms (Bazett)

Achados:
  Supradesnivelamento do segmento ST
  em V1-V4 (>2mm)

Hipótese diagnóstica:
  Sugestivo de Síndrome Coronariana
  Aguda com supra de ST em parede
  anterior.

⚠️ Ferramenta de apoio à decisão
clínica — não substitui avaliação médica.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`}</pre>
          </div>
        </Reveal>

        <Reveal delay={0.3}>
          <div className="mt-6 flex flex-col items-center gap-3">
            <div className="flex gap-3">
              <button
                disabled
                className="rounded-xl border border-[#E2E4F0] bg-white px-5 py-2.5 text-sm font-medium text-[#4A5078] opacity-70"
              >
                Exportar PDF
              </button>
              <button
                disabled
                className="rounded-xl border border-[#E2E4F0] bg-white px-5 py-2.5 text-sm font-medium text-[#4A5078] opacity-70"
              >
                Enviar por WhatsApp
              </button>
            </div>
            <p className="text-xs text-[#4A5078]">
              Laudo gerado em 12 segundos a partir de foto de ECG de papel.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
