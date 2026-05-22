// src/lib/check-subscription.ts
// Middleware/helper para verificar se o usuário tem assinatura ativa
// Usado nas rotas do dashboard que precisam de assinatura (upload ECG, etc.)

import { prisma } from "@/lib/prisma";

export type SubscriptionStatus = {
  isActive: boolean;
  plan: string | null;
  endsAt: Date | null;
  isOverdue: boolean;
  daysRemaining: number | null;
};

/**
 * Verifica se o usuário tem assinatura ativa.
 * Retorna informações do plano para exibir no dashboard.
 *
 * Lógica:
 * - ACTIVE + dentro do período → acesso liberado
 * - OVERDUE + dentro do período → acesso liberado (grace period, Asaas faz retentativa)
 * - OVERDUE + fora do período → acesso bloqueado
 * - CANCELED + dentro do período → acesso liberado (pagou até o fim)
 * - CANCELED + fora do período → acesso bloqueado
 * - PENDING → acesso bloqueado (aguardando pagamento)
 */
export async function checkSubscription(
  userId: string
): Promise<SubscriptionStatus> {
  const subscription = await prisma.subscription.findFirst({
    where: {
      userId,
      status: { in: ["ACTIVE", "OVERDUE", "CANCELED"] },
    },
    orderBy: { createdAt: "desc" },
  });

  if (!subscription || !subscription.endsAt) {
    return {
      isActive: false,
      plan: null,
      endsAt: null,
      isOverdue: false,
      daysRemaining: null,
    };
  }

  const now = new Date();
  const isWithinPeriod = subscription.endsAt > now;
  const daysRemaining = isWithinPeriod
    ? Math.ceil(
        (subscription.endsAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
      )
    : 0;

  // Acesso liberado se dentro do período (mesmo se OVERDUE ou CANCELED)
  const isActive = isWithinPeriod;
  const isOverdue = subscription.status === "OVERDUE";

  return {
    isActive,
    plan: subscription.plan,
    endsAt: subscription.endsAt,
    isOverdue,
    daysRemaining,
  };
}

/**
 * Guard para usar em tRPC procedures que exigem assinatura.
 * Lança erro se o usuário não tiver assinatura ativa.
 */
export async function requireActiveSubscription(userId: string): Promise<void> {
  const status = await checkSubscription(userId);

  if (!status.isActive) {
    throw new Error(
      "Assinatura inativa. Assine um plano para usar o ProECG."
    );
  }
}
