// src/lib/plans.ts
// Definição centralizada dos planos do ProECG
// Usado no frontend (pricing cards) e backend (criação de assinatura no Asaas)

export const PLANS = {
  MONTHLY: {
    id: "MONTHLY",
    name: "Mensal",
    description: "Ideal para testar o ProECG sem compromisso longo.",
    pricePerMonth: 267,
    totalPrice: 267,
    cycle: "MONTHLY" as const,
    // Asaas usa "MONTHLY" como billingType cycle
    asaasCycle: "MONTHLY" as const,
    discountPercent: 0,
    badge: null,
    highlighted: false,
  },
  SEMI_ANNUAL: {
    id: "SEMI_ANNUAL",
    name: "Semestral",
    description: "O equilíbrio perfeito entre economia e flexibilidade.",
    pricePerMonth: 227,
    totalPrice: 1362, // 227 * 6
    cycle: "SEMI_ANNUAL" as const,
    // Asaas não tem "SEMI_ANNUAL" nativo — usamos "MONTHLY" com endDate 6 meses à frente
    // OU cobrança única semestral. Vamos usar cobrança única + renovação manual via webhook.
    asaasCycle: "SEMIANNUALLY" as const,
    discountPercent: 15,
    badge: "Mais popular",
    highlighted: true,
  },
  ANNUAL: {
    id: "ANNUAL",
    name: "Anual",
    description: "O menor preço por mês para quem já decidiu.",
    pricePerMonth: 197,
    totalPrice: 2364, // 197 * 12
    cycle: "ANNUAL" as const,
    asaasCycle: "YEARLY" as const,
    discountPercent: 26,
    badge: "Melhor preço",
    highlighted: false,
  },
} as const;

export type PlanId = keyof typeof PLANS;

export const PLAN_IDS = Object.keys(PLANS) as PlanId[];

export function getPlan(planId: string) {
  const plan = PLANS[planId as PlanId];
  if (!plan) throw new Error(`Plano inválido: ${planId}`);
  return plan;
}

// Features incluídas em todos os planos (sem feature-gating)
export const PLAN_FEATURES = [
  "Laudos ilimitados",
  "~30 diagnósticos",
  "Exportação PDF",
  "Compartilhamento WhatsApp",
  "Histórico de 30 dias",
  "Suporte por email",
] as const;
