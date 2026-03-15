import prisma from "@proecg/db";
import { z } from "zod";

import { protectedProcedure, publicProcedure, router } from "../index";
import {
  createCustomer,
  createSubscription as createAsaasSubscription,
  cancelSubscription as cancelAsaasSubscription,
} from "../lib/asaas";
import { PLANS } from "../lib/plans";

const PLAN_TO_CYCLE = {
  MONTHLY: "MONTHLY",
  SEMI: "SEMIANNUALLY",
  ANNUAL: "YEARLY",
} as const;

const PLAN_TO_VALUE = {
  MONTHLY: 267,
  SEMI: 227 * 6,
  ANNUAL: 197 * 12,
} as const;

function formatDate(date: Date) {
  return date.toISOString().split("T")[0]!;
}

export const subscriptionRouter = router({
  listPlans: publicProcedure.query(() => {
    return PLANS;
  }),

  getStatus: protectedProcedure.query(async ({ ctx }) => {
    const subscription = await prisma.subscription.findFirst({
      where: {
        userId: ctx.session.user.id,
        status: { in: ["ACTIVE", "PENDING"] },
      },
      orderBy: { createdAt: "desc" },
    });
    return subscription;
  }),

  createCheckout: protectedProcedure
    .input(
      z.object({
        plan: z.enum(["MONTHLY", "SEMI", "ANNUAL"]),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.session.user;

      const customer = await createCustomer({
        name: user.name,
        email: user.email,
      });

      const nextDueDate = new Date();
      nextDueDate.setDate(nextDueDate.getDate() + 1);

      const asaasSub = await createAsaasSubscription({
        customer: customer.id,
        billingType: "PIX",
        value: PLAN_TO_VALUE[input.plan],
        cycle: PLAN_TO_CYCLE[input.plan],
        nextDueDate: formatDate(nextDueDate),
        description: `ProECG — Plano ${input.plan}`,
      });

      const durationMonths =
        input.plan === "MONTHLY" ? 1 : input.plan === "SEMI" ? 6 : 12;
      const endsAt = new Date();
      endsAt.setMonth(endsAt.getMonth() + durationMonths);

      await prisma.subscription.create({
        data: {
          userId: user.id,
          plan: input.plan,
          status: "PENDING",
          asaasId: asaasSub.id,
          startsAt: new Date(),
          endsAt,
        },
      });

      return {
        paymentLink: asaasSub.paymentLink,
        subscriptionId: asaasSub.id,
      };
    }),

  cancel: protectedProcedure.mutation(async ({ ctx }) => {
    const subscription = await prisma.subscription.findFirst({
      where: {
        userId: ctx.session.user.id,
        status: { in: ["ACTIVE", "PENDING"] },
      },
    });

    if (!subscription) {
      return { success: false, message: "Nenhuma assinatura ativa" };
    }

    if (subscription.asaasId) {
      await cancelAsaasSubscription(subscription.asaasId);
    }

    await prisma.subscription.update({
      where: { id: subscription.id },
      data: { status: "CANCELLED" },
    });

    return { success: true };
  }),
});
