export const PLANS = [
  {
    id: "MONTHLY" as const,
    name: "Mensal",
    price: 267,
    priceLabel: "R$ 267",
    period: "/mês",
    cycle: "MONTHLY" as const,
    durationMonths: 1,
    highlight: false,
  },
  {
    id: "SEMI" as const,
    name: "Semestral",
    price: 227,
    priceLabel: "R$ 227",
    period: "/mês",
    cycle: "MONTHLY" as const,
    durationMonths: 6,
    highlight: true,
    badge: "Mais popular",
  },
  {
    id: "ANNUAL" as const,
    name: "Anual",
    price: 197,
    priceLabel: "R$ 197",
    period: "/mês",
    cycle: "MONTHLY" as const,
    durationMonths: 12,
    highlight: false,
  },
] as const;

export type PlanId = (typeof PLANS)[number]["id"];
