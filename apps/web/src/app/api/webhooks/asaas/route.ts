import prisma from "@proecg/db";
import { env } from "@proecg/env/server";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { sendPaymentConfirmationEmail } from "@/lib/email";

interface AsaasWebhookPayload {
  event: string;
  payment?: {
    subscription?: string;
  };
  subscription?: {
    id: string;
  };
}

const EVENT_TO_STATUS: Record<string, "ACTIVE" | "OVERDUE" | "CANCELLED"> = {
  PAYMENT_CONFIRMED: "ACTIVE",
  PAYMENT_RECEIVED: "ACTIVE",
  PAYMENT_OVERDUE: "OVERDUE",
  SUBSCRIPTION_DELETED: "CANCELLED",
};

const PLAN_LABEL: Record<string, string> = {
  MONTHLY: "Mensal",
  SEMI: "Semestral",
  ANNUAL: "Anual",
};

const PLAN_TOTAL_LABEL: Record<string, string> = {
  MONTHLY: "R$ 267,00",
  SEMI: "R$ 1.362,00",
  ANNUAL: "R$ 2.364,00",
};

export async function POST(request: NextRequest) {
  const webhookToken = env.ASAAS_WEBHOOK_TOKEN;
  if (webhookToken) {
    const token = request.headers.get("asaas-access-token");
    if (token !== webhookToken) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  const body = (await request.json()) as AsaasWebhookPayload;
  const newStatus = EVENT_TO_STATUS[body.event];

  if (!newStatus) {
    return NextResponse.json({ received: true });
  }

  const asaasId = body.payment?.subscription ?? body.subscription?.id;

  if (!asaasId) {
    return NextResponse.json({ received: true });
  }

  // Estado anterior pra disparar o email de boas-vindas só na 1ª ativação
  const sub = await prisma.subscription.findFirst({
    where: { asaasId },
    include: { user: true },
  });

  await prisma.subscription.updateMany({
    where: { asaasId },
    data: {
      status: newStatus,
      ...(newStatus === "ACTIVE" ? { lastPaymentDate: new Date() } : {}),
    },
  });

  // Confirmação de pagamento -> email com link pra definir senha (uma vez)
  if (newStatus === "ACTIVE" && sub && sub.status !== "ACTIVE" && sub.user) {
    const nextBilling = sub.endsAt
      ? sub.endsAt.toLocaleDateString("pt-BR")
      : "—";
    const setPasswordUrl = `${env.BETTER_AUTH_URL}/esqueci-senha`;
    try {
      await sendPaymentConfirmationEmail({
        to: sub.user.email,
        name: sub.user.name,
        planName: PLAN_LABEL[sub.plan] ?? sub.plan,
        totalLabel: PLAN_TOTAL_LABEL[sub.plan] ?? "",
        nextBilling,
        setPasswordUrl,
      });
    } catch (err) {
      console.error("[webhook] falha ao enviar email de confirmação:", err);
    }
  }

  return NextResponse.json({ received: true });
}
