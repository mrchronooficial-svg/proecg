"use client";

import { Reveal } from "./reveal";

const testimonials = [
  {
    quote:
      "Uso o ProECG nos plantões de emergência. Antes eu ficava inseguro com ECGs complexos de madrugada — agora tenho um segundo olhar em segundos.",
    name: "Médico Emergencista",
    role: "Beta tester",
  },
  {
    quote:
      "A digitalização é impressionante. Ele mede os intervalos com precisão e já sugere o diagnóstico. Economizo tempo e ganho segurança.",
    name: "Médica Clínica Geral",
    role: "Beta tester",
  },
  {
    quote:
      "Participei da construção dos critérios diagnósticos. O rigor clínico do motor é o mesmo que eu uso na minha prática diária.",
    name: "Cardiologista",
    role: "Consultor clínico",
  },
];

export function Testimonials() {
  return (
    <section className="bg-[#EEEFFD] px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[1200px]">
        <Reveal>
          <h2 className="mb-12 text-center text-3xl font-bold text-[#122056] sm:text-4xl">
            O que médicos estão dizendo
          </h2>
        </Reveal>
        <div className="grid gap-6 sm:grid-cols-3">
          {testimonials.map((t, i) => (
            <Reveal key={t.name} delay={i * 0.1}>
              <div className="relative flex flex-col rounded-2xl bg-white p-6 shadow-sm">
                <span className="absolute top-4 right-6 text-5xl font-serif text-[#5B65DC]/10">
                  &quot;
                </span>
                <p className="flex-1 text-[#122056] leading-relaxed">
                  &ldquo;{t.quote}&rdquo;
                </p>
                <div className="mt-4 border-t border-[#E2E4F0] pt-4">
                  <p className="font-semibold text-[#122056]">{t.name}</p>
                  <p className="text-sm text-[#4A5078]">{t.role}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
