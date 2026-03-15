"use client";

import Link from "next/link";
import { Reveal } from "./reveal";

const plans = [
  {
    name: "Mensal",
    price: "R$ 267",
    period: "/mês",
    detail: "Cobrança mensal",
    description: "Ideal para testar o ProECG sem compromisso longo.",
    features: [
      "Laudos ilimitados",
      "~30 diagnósticos",
      "Exportação PDF",
      "Compartilhamento WhatsApp",
      "Histórico de 30 dias",
      "Suporte por email",
    ],
    highlight: false,
    cta: "Assinar Mensal",
  },
  {
    name: "Semestral",
    price: "R$ 227",
    period: "/mês",
    detail: "Cobrança a cada 6 meses (R$ 1.362 total)",
    economy: "Economize R$ 240/ano vs. mensal",
    description: "O equilíbrio perfeito entre economia e flexibilidade.",
    badge: "Mais popular",
    badgeColor: "bg-[#10B981]",
    features: [
      "Tudo do plano Mensal",
      "15% de desconto",
      "Suporte prioritário",
    ],
    highlight: true,
    cta: "Assinar Semestral",
  },
  {
    name: "Anual",
    price: "R$ 197",
    period: "/mês",
    detail: "Cobrança anual (R$ 2.364 total)",
    economy: "Economize R$ 840/ano vs. mensal",
    description: "O menor preço por mês para quem já decidiu.",
    badge: "Melhor preço",
    badgeColor: "bg-[#F59E0B]",
    features: [
      "Tudo do plano Semestral",
      "26% de desconto",
      "Suporte prioritário",
    ],
    highlight: false,
    cta: "Assinar Anual",
  },
];

export function PricingTable() {
  return (
    <section id="planos" className="bg-[#FAFAFD] px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[1000px]">
        <Reveal>
          <h2 className="text-center text-3xl font-bold text-[#122056] sm:text-4xl">
            Escolha seu plano
          </h2>
        </Reveal>
        <Reveal delay={0.1}>
          <p className="mx-auto mt-4 max-w-lg text-center text-lg text-[#4A5078]">
            Acesso ilimitado a laudos de ECG. Todos os planos incluem todas as
            funcionalidades.
          </p>
        </Reveal>

        <div className="mt-14 grid gap-6 sm:grid-cols-3 items-start">
          {plans.map((plan, i) => (
            <Reveal key={plan.name} delay={i * 0.1}>
              <div
                className={`relative flex flex-col rounded-2xl bg-white p-8 shadow-sm transition-transform ${
                  plan.highlight
                    ? "border-2 border-[#5B65DC] shadow-lg sm:scale-105"
                    : "border border-[#E2E4F0]"
                }`}
              >
                {plan.badge && (
                  <span
                    className={`absolute -top-3 left-1/2 -translate-x-1/2 rounded-full ${plan.badgeColor} px-3 py-1 text-xs font-semibold text-white`}
                  >
                    {plan.badge}
                  </span>
                )}
                <h3 className="text-xl font-semibold text-[#122056]">
                  {plan.name}
                </h3>
                <div className="mt-3 flex items-baseline gap-1">
                  <span className="text-4xl font-bold text-[#122056]">
                    {plan.price}
                  </span>
                  <span className="text-[#4A5078]">{plan.period}</span>
                </div>
                <p className="mt-1 text-sm text-[#4A5078]">{plan.detail}</p>
                {plan.economy && (
                  <p className="mt-1 text-sm font-medium text-[#10B981]">
                    {plan.economy}
                  </p>
                )}
                <p className="mt-3 text-sm text-[#4A5078]">
                  {plan.description}
                </p>

                <ul className="mt-5 space-y-2">
                  {plan.features.map((f) => (
                    <li
                      key={f}
                      className="flex items-start gap-2 text-sm text-[#122056]"
                    >
                      <span className="mt-0.5 text-[#10B981]">✓</span>
                      {f}
                    </li>
                  ))}
                </ul>

                <Link
                  href="/login"
                  className={`mt-6 block rounded-xl px-5 py-3 text-center text-sm font-semibold transition-all ${
                    plan.highlight
                      ? "bg-[#5B65DC] text-white shadow-[0_4px_14px_rgba(91,101,220,0.35)] hover:bg-[#4A51C5] hover:-translate-y-0.5"
                      : "border border-[#E2E4F0] text-[#122056] hover:bg-[#EEEFFD]"
                  }`}
                >
                  {plan.cta}
                </Link>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.3}>
          <p className="mt-8 text-center text-sm text-[#4A5078]">
            Pagamento via Pix ou cartão de crédito · Processado com segurança
            pelo Asaas
            <br />
            Cancele quando quiser · Sem multa, sem burocracia
          </p>
        </Reveal>
      </div>
    </section>
  );
}
