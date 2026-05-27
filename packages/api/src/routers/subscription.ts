import prisma from "@proecg/db";
import { TRPCError } from "@trpc/server";
import { z } from "zod";

import { protectedProcedure, publicProcedure, router } from "../index";
import {
  AsaasError,
  cancelSubscription as cancelAsaasSubscription,
  createCustomer,
  createSubscription as createAsaasSubscription,
  createPixSubscription,
  getOrCreateCustomer,
  getPaymentPixQrCode,
  getSubscriptionPayments,
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

const PLAN_TO_MONTHS = { MONTHLY: 1, SEMI: 6, ANNUAL: 12 } as const;

const planEnum = z.enum(["MONTHLY", "SEMI", "ANNUAL"]);

const cpfSchema = z
  .string()
  .regex(/^\d{11}$/, "CPF deve ter 11 dígitos numéricos");

const creditCardSchema = z.object({
  number: z.string().regex(/^\d{13,19}$/, "Número do cartão inválido"),
  expiryMonth: z.string().regex(/^\d{2}$/, "Mês inválido"),
  expiryYear: z.string().regex(/^\d{4}$/, "Ano inválido"),
  ccv: z.string().regex(/^\d{3,4}$/, "CVV inválido"),
});

function formatDate(date: Date) {
  return date.toISOString().split("T")[0]!;
}

function endsAtFromPlan(plan: keyof typeof PLAN_TO_MONTHS) {
  const endsAt = new Date();
  endsAt.setMonth(endsAt.getMonth() + PLAN_TO_MONTHS[plan]);
  return endsAt;
}

async function assertNoActiveSubscription(userId: string) {
  const existing = await prisma.subscription.findFirst({
    where: { userId, status: "ACTIVE" },
  });
  if (existing) {
    throw new TRPCError({
      code: "CONFLICT",
      message: "Você já possui uma assinatura ativa.",
    });
  }
}

function mapAsaasCardError(error: AsaasError): string {
  const code = error.errors?.[0]?.code;
  const map: Record<string, string> = {
    invalid_creditCard:
      "Dados do cartão inválidos. Verifique número, validade e CVV.",
    invalid_creditCardHolderInfo:
      "Dados do titular inválidos. Verifique nome e CPF.",
    declined: "Cartão recusado. Tente outro cartão ou use Pix.",
    expired_card: "Cartão expirado. Use outro cartão.",
    insufficient_funds: "Saldo insuficiente. Tente outro cartão ou use Pix.",
  };
  return (
    map[code ?? ""] ||
    error.message ||
    "Erro ao processar pagamento. Tente novamente."
  );
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

  // Checkout transparente com CARTÃO (cobrança imediata) ---------------
  checkoutCard: protectedProcedure
    .input(
      z.object({
        plan: planEnum,
        creditCard: creditCardSchema,
        cpf: cpfSchema,
        // Asaas exige holderInfo; coletamos o mínimo na UI e completamos
        // com dados da conta. postalCode/addressNumber/phone são opcionais.
        postalCode: z.string().optional(),
        addressNumber: z.string().optional(),
        phone: z.string().optional(),
        remoteIp: z.string().optional(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.session.user;
      await assertNoActiveSubscription(user.id);

      const customer = await getOrCreateCustomer({
        name: user.name,
        email: user.email,
        cpfCnpj: input.cpf,
        externalReference: user.id,
      });

      try {
        const asaasSub = await createAsaasSubscription({
          customer: customer.id,
          billingType: "CREDIT_CARD",
          value: PLAN_TO_VALUE[input.plan],
          cycle: PLAN_TO_CYCLE[input.plan],
          nextDueDate: formatDate(new Date()),
          description: `ProECG — Plano ${input.plan}`,
          externalReference: `${user.id}:${input.plan}`,
          creditCard: { holderName: user.name, ...input.creditCard },
          creditCardHolderInfo: {
            name: user.name,
            email: user.email,
            cpfCnpj: input.cpf,
            postalCode: input.postalCode ?? "",
            addressNumber: input.addressNumber ?? "",
            phone: input.phone ?? "",
          },
          remoteIp: input.remoteIp ?? "0.0.0.0",
        });

        const dbSub = await prisma.subscription.create({
          data: {
            userId: user.id,
            plan: input.plan,
            status: "PENDING",
            asaasId: asaasSub.id,
            asaasCustomerId: customer.id,
            startsAt: new Date(),
            endsAt: endsAtFromPlan(input.plan),
          },
        });

        return {
          success: true,
          subscriptionId: dbSub.id,
          cardLastFour: asaasSub.creditCard?.creditCardNumber ?? null,
          cardBrand: asaasSub.creditCard?.creditCardBrand ?? null,
        };
      } catch (error) {
        if (error instanceof AsaasError) {
          throw new TRPCError({
            code: "BAD_REQUEST",
            message: mapAsaasCardError(error),
          });
        }
        throw error;
      }
    }),

  // Checkout com PIX (gera QR; webhook confirma depois) -----------------
  checkoutPix: protectedProcedure
    .input(z.object({ plan: planEnum, cpf: cpfSchema }))
    .mutation(async ({ ctx, input }) => {
      const user = ctx.session.user;
      await assertNoActiveSubscription(user.id);

      const customer = await getOrCreateCustomer({
        name: user.name,
        email: user.email,
        cpfCnpj: input.cpf,
        externalReference: user.id,
      });

      const asaasSub = await createPixSubscription({
        customer: customer.id,
        value: PLAN_TO_VALUE[input.plan],
        cycle: PLAN_TO_CYCLE[input.plan],
        nextDueDate: formatDate(new Date()),
        description: `ProECG — Plano ${input.plan}`,
        externalReference: `${user.id}:${input.plan}`,
      });

      let pix: {
        qrCodeBase64: string;
        qrCodeText: string;
        expirationDate: string;
        paymentId: string;
      } | null = null;
      const payments = await getSubscriptionPayments(asaasSub.id);
      const firstPayment = payments.data[0];
      if (firstPayment) {
        try {
          const qr = await getPaymentPixQrCode(firstPayment.id);
          pix = {
            qrCodeBase64: qr.encodedImage,
            qrCodeText: qr.payload,
            expirationDate: qr.expirationDate,
            paymentId: firstPayment.id,
          };
        } catch {
          pix = null;
        }
      }

      const dbSub = await prisma.subscription.create({
        data: {
          userId: user.id,
          plan: input.plan,
          status: "PENDING",
          asaasId: asaasSub.id,
          asaasCustomerId: customer.id,
          startsAt: new Date(),
          endsAt: endsAtFromPlan(input.plan),
        },
      });

      return {
        success: true,
        subscriptionId: dbSub.id,
        pix,
        invoiceUrl: firstPayment?.invoiceUrl ?? null,
      };
    }),

  // Polling do frontend pra detectar confirmação do Pix ----------------
  checkPixPayment: protectedProcedure
    .input(z.object({ subscriptionId: z.string() }))
    .query(async ({ ctx, input }) => {
      const sub = await prisma.subscription.findFirst({
        where: { id: input.subscriptionId, userId: ctx.session.user.id },
      });
      if (!sub) throw new TRPCError({ code: "NOT_FOUND" });
      return { status: sub.status, isActive: sub.status === "ACTIVE" };
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
