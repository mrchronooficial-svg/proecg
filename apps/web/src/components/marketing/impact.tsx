"use client";

import { Reveal } from "./reveal";

const metrics = [
  { value: "< 15s", label: "Laudo gerado" },
  { value: "30+", label: "Diagnósticos cobertos" },
  { value: "Qualquer", label: "Câmera de celular" },
];

export function Impact() {
  return (
    <section
      className="relative overflow-hidden py-20 text-white sm:py-32 lg:py-40"
      style={{
        backgroundImage:
          "radial-gradient(40% 60% at 80% 20%, rgba(0,102,204,0.25), transparent 60%), linear-gradient(180deg, #0a0a0f 0%, #15152a 100%)",
      }}
    >
      <div className="mx-auto max-w-7xl px-6">
        <Reveal className="max-w-4xl text-center md:text-left">
          <h2 className="text-[30px] font-bold leading-[1.1] tracking-[-0.02em] break-words sm:text-[48px] lg:text-[64px]">
            Impacto real.
            <br />
            <span className="text-white/55">Resultados mensuráveis.</span>
          </h2>
        </Reveal>

        <div className="mt-10 grid grid-cols-1 gap-10 md:grid-cols-3 md:gap-12 sm:mt-16 lg:mt-24">
          {metrics.map((m, i) => (
            <Reveal key={m.label} delay={i * 0.1}>
              <div className="text-center md:text-left">
                <div className="text-[44px] font-bold leading-[1] tracking-[-0.02em] sm:text-[56px] lg:text-[80px]">
                  {m.value}
                </div>
                <div className="mt-2.5 text-[15px] text-white/65 sm:mt-3 sm:text-[18px] lg:text-[20px]">
                  {m.label}
                </div>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.3}>
          <p className="mx-auto mt-10 max-w-2xl text-center text-[13px] leading-[1.6] text-white/55 sm:mt-16 sm:text-[16px] md:mx-0 md:text-left lg:mt-20 lg:text-[18px]">
            Validação clínica em andamento com hospitais brasileiros parceiros.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
