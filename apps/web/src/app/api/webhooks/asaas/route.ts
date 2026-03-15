import prisma from "@proecg/db";
import { env } from "@proecg/env/server";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

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

  const asaasId =
    body.payment?.subscription ?? body.subscription?.id;

  if (!asaasId) {
    return NextResponse.json({ received: true });
  }

  await prisma.subscription.updateMany({
    where: { asaasId },
    data: { status: newStatus },
  });

  return NextResponse.json({ received: true });
}
