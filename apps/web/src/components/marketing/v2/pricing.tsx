"use client";

import Link from "next/link";
import { Check } from "lucide-react";
import { Reveal } from "./reveal";

const features = [
  "Laudo de ECG ilimitado",
  "Mais de 30 diagnósticos",
  "Histórico completo",
  "Exportação em PDF",
  "Compartilhamento via WhatsApp",
  "Suporte por email",
];

const plans = [
  { name: "Mensal", price: 267, period: "/mês", note: "Cobrança mensal", popular: false, save: null },
  { name: "Semestral", price: 227, period: "/mês", note: "Cobrança semestral", popular: false, save: "Economize 15%" },
  { name: "Anual", price: 197, period: "/mês", note: "Cobrança anual", popular: true, save: "Economize 26%" },
];

export function Pricing() {
  return (
    <section id="pricing" className="bg-[#f5f5f7] py-20 sm:py-32 lg:py-40">
      <div className="mx-auto max-w-7xl px-6">
        <Reveal className="text-center">
          <h2 className="text-[30px] font-bold leading-[1.1] tracking-[-0.02em] break-words text-[#1d1d1f] sm:text-[48px] lg:text-[56px]">
            Planos simples e transparentes.
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-[15px] leading-[1.5] text-[#86868b] sm:mt-5 sm:text-[18px]">
            Sem limites de uso, sem letras miúdas. Pagamento via Pix ou cartão.
          </p>
        </Reveal>

        <div className="mt-10 grid grid-cols-1 gap-5 sm:mt-16 sm:gap-6 lg:grid-cols-3">
          {plans.map((p, i) => (
            <Reveal key={p.name} delay={i * 0.08}>
              <div
                className={`relative h-full rounded-2xl bg-white p-6 sm:rounded-3xl sm:p-8 lg:p-10 ${
                  p.popular
                    ? "ring-2 ring-[#0066CC] shadow-[0_30px_60px_-20px_rgba(0,102,204,0.35)]"
                    : "shadow-[0_1px_3px_rgba(0,0,0,0.04)]"
                }`}
              >
                {p.popular && (
                  <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-[#0066CC] px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-white sm:px-4 sm:py-1.5 sm:text-[12px]">
                    Mais popular
                  </span>
                )}

                <div className="text-[17px] font-semibold tracking-[-0.01em] text-[#1d1d1f] sm:text-[16px]">
                  {p.name}
                </div>
                {p.save && (
                  <div className="mt-1 text-[13px] font-medium text-emerald-600">{p.save}</div>
                )}

                <div className="mt-5 flex items-baseline gap-1 sm:mt-6">
                  <span className="text-[14px] font-medium text-[#1d1d1f]/70">R$</span>
                  <span className="text-[44px] font-bold leading-none tracking-[-0.02em] text-[#1d1d1f] sm:text-[56px]">
                    {p.price}
                  </span>
                  <span className="text-[14px] text-[#86868b]">{p.period}</span>
                </div>
                <div className="mt-1.5 text-[12px] text-[#86868b]">{p.note}</div>

                <Link
                  href="/assinar"
                  className={`mt-6 block w-full rounded-full px-6 py-3 text-center text-[15px] font-semibold transition-colors sm:mt-8 sm:py-3 sm:text-[14px] sm:font-medium ${
                    p.popular
                      ? "bg-[#0066CC] text-white hover:bg-[#0055aa]"
                      : "bg-[#1d1d1f] text-white hover:bg-[#1d1d1f]/85"
                  }`}
                >
                  Assinar
                </Link>

                <ul className="mt-6 space-y-2.5 text-[14px] text-[#1d1d1f]/80 sm:mt-8 sm:space-y-3 sm:text-[14.5px]">
                  {features.map((f) => (
                    <li key={f} className="flex items-start gap-2.5">
                      <Check className="mt-1 h-4 w-4 flex-shrink-0 text-[#0066CC]" />
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.3}>
          <p className="mt-10 text-center text-[12px] text-[#86868b] sm:mt-12 sm:text-[13px]">
            Pagamento via Pix ou cartão. Cancele quando quiser.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
