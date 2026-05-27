// Espelho client-side dos planos (fonte de verdade: packages/api/src/lib/plans.ts).
// Usado nas telas de cadastro/checkout pra exibir resumo sem round-trip ao servidor.

export type CheckoutPlanId = "MONTHLY" | "SEMI" | "ANNUAL";

export interface CheckoutPlan {
  id: CheckoutPlanId;
  name: string;
  monthlyLabel: string; // preço por mês exibido
  totalLabel: string; // total cobrado hoje
  totalValue: number; // em reais
  cycleLabel: string; // "mensal" | "semestral" | "anual"
  months: number;
  highlight?: boolean;
}

export const CHECKOUT_PLANS: Record<CheckoutPlanId, CheckoutPlan> = {
  MONTHLY: {
    id: "MONTHLY",
    name: "Mensal",
    monthlyLabel: "R$ 267/mês",
    totalLabel: "R$ 267,00",
    totalValue: 267,
    cycleLabel: "mensal",
    months: 1,
  },
  SEMI: {
    id: "SEMI",
    name: "Semestral",
    monthlyLabel: "R$ 227/mês",
    totalLabel: "R$ 1.362,00",
    totalValue: 227 * 6,
    cycleLabel: "semestral",
    months: 6,
    highlight: true,
  },
  ANNUAL: {
    id: "ANNUAL",
    name: "Anual",
    monthlyLabel: "R$ 197/mês",
    totalLabel: "R$ 2.364,00",
    totalValue: 197 * 12,
    cycleLabel: "anual",
    months: 12,
  },
};

export const PLAN_BENEFITS = [
  "Laudos ilimitados",
  "~30 diagnósticos cobertos",
  "Exportação PDF e WhatsApp",
  "Suporte por email",
];

export function resolvePlan(raw: string | null | undefined): CheckoutPlan {
  if (raw && raw in CHECKOUT_PLANS) {
    return CHECKOUT_PLANS[raw as CheckoutPlanId];
  }
  return CHECKOUT_PLANS.SEMI; // default: plano mais popular
}
