import { auth } from "@proecg/auth";
import prisma from "@proecg/db";
import { env } from "@proecg/env/server";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { PROVISIONAL_PASSWORD, sendPaymentConfirmationEmail } from "@/lib/email";
import { generatePasswordResetToken } from "@/lib/password-token";

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

  // Confirmação de pagamento -> email de boas-vindas (uma vez só, na 1ª ativação)
  if (newStatus === "ACTIVE" && sub && sub.status !== "ACTIVE" && sub.user) {
    try {
      await handleFirstActivation(sub);
    } catch (err) {
      console.error("[webhook] falha ao enviar email de confirmação:", err);
    }
  }

  return NextResponse.json({ received: true });
}

async function handleFirstActivation(sub: {
  plan: string;
  endsAt: Date | null;
  user: { id: string; name: string; email: string; authProvider: string | null };
}) {
  const { user } = sub;
  const authProvider: "email" | "google" =
    user.authProvider === "google" ? "google" : "email";

  const nextBilling = sub.endsAt
    ? sub.endsAt.toLocaleDateString("pt-BR")
    : "—";

  let resetPasswordUrl: string | undefined;
  let provisionalPassword: string | undefined;

  if (authProvider === "email") {
    // Reseta a senha para a provisória conhecida e gera token de troca
    const ctx = await auth.$context;
    const hashed = await ctx.password.hash(PROVISIONAL_PASSWORD);
    const credentialAccount = await prisma.account.findFirst({
      where: { userId: user.id, providerId: "credential" },
    });
    if (credentialAccount) {
      await prisma.account.update({
        where: { id: credentialAccount.id },
        data: { password: hashed },
      });
    } else {
      await prisma.account.create({
        data: {
          id: crypto.randomUUID(),
          userId: user.id,
          providerId: "credential",
          accountId: user.id,
          password: hashed,
        },
      });
    }

    const token = await generatePasswordResetToken(user.id);
    resetPasswordUrl = `${env.BETTER_AUTH_URL}/reset-password?token=${token}`;
    provisionalPassword = PROVISIONAL_PASSWORD;
  }

  await sendPaymentConfirmationEmail({
    to: user.email,
    name: user.name,
    planName: PLAN_LABEL[sub.plan] ?? sub.plan,
    totalLabel: PLAN_TOTAL_LABEL[sub.plan] ?? "",
    nextBilling,
    authProvider,
    provisionalPassword,
    resetPasswordUrl,
  });
}
