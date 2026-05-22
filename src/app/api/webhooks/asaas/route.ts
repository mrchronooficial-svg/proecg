// src/app/api/webhooks/asaas/route.ts
// Webhook do Asaas — recebe notificações de pagamento
// Configura no painel Asaas: https://seudominio.com/api/webhooks/asaas
// Eventos necessários: PAYMENT_CONFIRMED, PAYMENT_RECEIVED, PAYMENT_OVERDUE,
//                      PAYMENT_REFUNDED, PAYMENT_DELETED

import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

// Token de autenticação do webhook (configurado no painel Asaas)
const ASAAS_WEBHOOK_TOKEN = process.env.ASAAS_WEBHOOK_TOKEN!;

// ============================================================
// Types do webhook Asaas
// ============================================================

interface AsaasWebhookPayload {
  event: string;
  payment: {
    id: string;
    customer: string;
    subscription?: string;
    value: number;
    status: string;
    billingType: string;
    dueDate: string;
    paymentDate?: string;
    externalReference?: string;
    invoiceUrl?: string;
  };
}

// ============================================================
// Handler
// ============================================================

export async function POST(request: NextRequest) {
  try {
    // 1. Validar token de autenticação
    const token = request.headers.get("asaas-access-token");
    if (token !== ASAAS_WEBHOOK_TOKEN) {
      console.warn("[Webhook Asaas] Token inválido recebido");
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // 2. Parsear o payload
    const payload: AsaasWebhookPayload = await request.json();
    const { event, payment } = payload;

    console.log(`[Webhook Asaas] Evento: ${event} | Payment: ${payment.id} | Status: ${payment.status}`);

    // 3. Processar por tipo de evento
    switch (event) {
      case "PAYMENT_CONFIRMED":
      case "PAYMENT_RECEIVED":
        await handlePaymentConfirmed(payment);
        break;

      case "PAYMENT_OVERDUE":
        await handlePaymentOverdue(payment);
        break;

      case "PAYMENT_REFUNDED":
      case "PAYMENT_DELETED":
        await handlePaymentCanceled(payment);
        break;

      default:
        console.log(`[Webhook Asaas] Evento ignorado: ${event}`);
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error("[Webhook Asaas] Erro ao processar:", error);
    // Retorna 200 mesmo com erro para o Asaas não reenviar infinitamente
    // O erro já foi logado para investigação
    return NextResponse.json({ received: true, error: "Internal error logged" });
  }
}

// ============================================================
// Handlers por evento
// ============================================================

async function handlePaymentConfirmed(
  payment: AsaasWebhookPayload["payment"]
) {
  if (!payment.subscription) {
    console.log("[Webhook] Pagamento avulso (sem assinatura), ignorando");
    return;
  }

  // Buscar assinatura no banco pelo asaasId
  const subscription = await prisma.subscription.findFirst({
    where: { asaasSubscriptionId: payment.subscription },
    include: { user: true },
  });

  if (!subscription) {
    console.warn(
      `[Webhook] Assinatura não encontrada no banco: ${payment.subscription}`
    );
    return;
  }

  // Calcular nova data de expiração baseada no plano
  const endsAt = calculateEndDate(subscription.plan, new Date());

  // Atualizar status para ACTIVE
  await prisma.subscription.update({
    where: { id: subscription.id },
    data: {
      status: "ACTIVE",
      startsAt: new Date(),
      endsAt,
      lastPaymentDate: payment.paymentDate
        ? new Date(payment.paymentDate)
        : new Date(),
    },
  });

  console.log(
    `[Webhook] ✅ Assinatura ${subscription.id} ATIVADA para user ${subscription.userId} até ${endsAt.toISOString()}`
  );
}

async function handlePaymentOverdue(
  payment: AsaasWebhookPayload["payment"]
) {
  if (!payment.subscription) return;

  const subscription = await prisma.subscription.findFirst({
    where: { asaasSubscriptionId: payment.subscription },
  });

  if (!subscription) return;

  // Marcar como OVERDUE — o médico ainda tem acesso até endsAt
  // mas mostramos aviso no dashboard
  await prisma.subscription.update({
    where: { id: subscription.id },
    data: { status: "OVERDUE" },
  });

  console.log(
    `[Webhook] ⚠️ Assinatura ${subscription.id} OVERDUE (pagamento atrasado)`
  );
}

async function handlePaymentCanceled(
  payment: AsaasWebhookPayload["payment"]
) {
  if (!payment.subscription) return;

  const subscription = await prisma.subscription.findFirst({
    where: { asaasSubscriptionId: payment.subscription },
  });

  if (!subscription) return;

  // Marcar como CANCELED — acesso mantido até endsAt
  await prisma.subscription.update({
    where: { id: subscription.id },
    data: { status: "CANCELED" },
  });

  console.log(
    `[Webhook] ❌ Assinatura ${subscription.id} CANCELADA/REEMBOLSADA`
  );
}

// ============================================================
// Helpers
// ============================================================

function calculateEndDate(
  plan: string,
  from: Date
): Date {
  const end = new Date(from);

  switch (plan) {
    case "MONTHLY":
      end.setMonth(end.getMonth() + 1);
      break;
    case "SEMI_ANNUAL":
      end.setMonth(end.getMonth() + 6);
      break;
    case "ANNUAL":
      end.setFullYear(end.getFullYear() + 1);
      break;
    default:
      end.setMonth(end.getMonth() + 1);
  }

  return end;
}
