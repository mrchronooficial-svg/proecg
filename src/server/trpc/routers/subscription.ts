// src/server/trpc/routers/subscription.ts
// Router tRPC — Checkout Transparente com Asaas
// Fluxo: Cadastro → Checkout (cartão ou Pix) → Webhook confirma → Acesso liberado

import { z } from "zod";
import { router, protectedProcedure } from "../trpc";
import {
  getOrCreateCustomer,
  createSubscription,
  createPixSubscription,
  getSubscriptionPayments,
  getPaymentPixQrCode,
  cancelSubscription,
  AsaasError,
} from "@/lib/asaas";
import { getPlan, PLAN_IDS, type PlanId } from "@/lib/plans";
import { prisma } from "@/lib/prisma";
import { TRPCError } from "@trpc/server";

// ============================================================
// Validações
// ============================================================

const cpfSchema = z
  .string()
  .regex(/^\d{11}$/, "CPF deve ter 11 dígitos numéricos");

const creditCardSchema = z.object({
  holderName: z.string().min(3, "Nome no cartão obrigatório"),
  number: z.string().regex(/^\d{13,19}$/, "Número do cartão inválido"),
  expiryMonth: z.string().regex(/^\d{2}$/, "Mês inválido"),
  expiryYear: z.string().regex(/^\d{4}$/, "Ano inválido"),
  ccv: z.string().regex(/^\d{3,4}$/, "CVV inválido"),
});

const holderInfoSchema = z.object({
  name: z.string().min(3),
  cpfCnpj: cpfSchema,
  postalCode: z.string().regex(/^\d{8}$/, "CEP deve ter 8 dígitos"),
  addressNumber: z.string().min(1, "Número do endereço obrigatório"),
  phone: z.string().regex(/^\d{10,11}$/, "Telefone inválido"),
});

// ============================================================
// Router
// ============================================================

export const subscriptionRouter = router({
  // ----------------------------------------------------------
  // CHECKOUT COM CARTÃO DE CRÉDITO (transparente)
  // ----------------------------------------------------------
  checkoutCard: protectedProcedure
    .input(
      z.object({
        planId: z.enum(PLAN_IDS as [string, ...string[]]),
        creditCard: creditCardSchema,
        holderInfo: holderInfoSchema,
        remoteIp: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user;
      const plan = getPlan(input.planId);

      // 1. Verificar se já tem assinatura ativa
      await assertNoActiveSubscription(user.id);

      // 2. Criar/buscar cliente no Asaas
      const customer = await getOrCreateCustomer({
        name: input.holderInfo.name,
        email: user.email,
        cpfCnpj: input.holderInfo.cpfCnpj,
        externalReference: user.id,
      });

      // 3. Data do primeiro vencimento (hoje)
      const today = new Date().toISOString().split("T")[0];

      // 4. Criar assinatura com cartão (checkout transparente)
      //    O Asaas cobra o cartão imediatamente na criação
      try {
        const subscription = await createSubscription({
          customer: customer.id,
          billingType: "CREDIT_CARD",
          value: plan.totalPrice,
          cycle: plan.asaasCycle,
          nextDueDate: today,
          description: `ProECG — Plano ${plan.name}`,
          externalReference: `${user.id}:${input.planId}`,
          creditCard: input.creditCard,
          creditCardHolderInfo: {
            ...input.holderInfo,
            email: user.email,
          },
          remoteIp: input.remoteIp || "0.0.0.0",
        });

        // 5. Salvar no banco
        //    Status começa como PENDING — o webhook PAYMENT_CONFIRMED muda para ACTIVE
        const dbSubscription = await prisma.subscription.create({
          data: {
            userId: user.id,
            plan: input.planId,
            status: "PENDING",
            asaasSubscriptionId: subscription.id,
            asaasCustomerId: customer.id,
          },
        });

        return {
          success: true,
          subscriptionId: dbSubscription.id,
          message: "Pagamento processado! Seu acesso será liberado em instantes.",
          cardLastFour: subscription.creditCard?.creditCardNumber ?? null,
          cardBrand: subscription.creditCard?.creditCardBrand ?? null,
        };
      } catch (error) {
        if (error instanceof AsaasError) {
          // Mapear erros comuns do Asaas para mensagens amigáveis
          const friendlyMessage = mapAsaasCardError(error);
          throw new TRPCError({
            code: "BAD_REQUEST",
            message: friendlyMessage,
          });
        }
        throw error;
      }
    }),

  // ----------------------------------------------------------
  // CHECKOUT COM PIX
  // ----------------------------------------------------------
  checkoutPix: protectedProcedure
    .input(
      z.object({
        planId: z.enum(PLAN_IDS as [string, ...string[]]),
        name: z.string().min(3),
        cpfCnpj: cpfSchema,
      })
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user;
      const plan = getPlan(input.planId);

      await assertNoActiveSubscription(user.id);

      // 1. Criar/buscar cliente
      const customer = await getOrCreateCustomer({
        name: input.name,
        email: user.email,
        cpfCnpj: input.cpfCnpj,
        externalReference: user.id,
      });

      // 2. Criar assinatura com Pix
      const today = new Date().toISOString().split("T")[0];

      const subscription = await createPixSubscription({
        customer: customer.id,
        value: plan.totalPrice,
        cycle: plan.asaasCycle,
        nextDueDate: today,
        description: `ProECG — Plano ${plan.name}`,
        externalReference: `${user.id}:${input.planId}`,
      });

      // 3. Buscar QR code do primeiro pagamento
      const payments = await getSubscriptionPayments(subscription.id);
      const firstPayment = payments.data[0];

      let pixData = null;
      if (firstPayment) {
        try {
          const qr = await getPaymentPixQrCode(firstPayment.id);
          pixData = {
            qrCodeBase64: qr.encodedImage,
            qrCodeText: qr.payload, // Código copia-e-cola
            expirationDate: qr.expirationDate,
            paymentId: firstPayment.id,
          };
        } catch {
          // Pix QR pode não estar disponível imediatamente
          pixData = null;
        }
      }

      // 4. Salvar no banco como PENDING
      const dbSubscription = await prisma.subscription.create({
        data: {
          userId: user.id,
          plan: input.planId,
          status: "PENDING",
          asaasSubscriptionId: subscription.id,
          asaasCustomerId: customer.id,
        },
      });

      return {
        success: true,
        subscriptionId: dbSubscription.id,
        pix: pixData,
        invoiceUrl: firstPayment?.invoiceUrl ?? null,
        message: pixData
          ? "QR Code gerado! Escaneie com o app do seu banco para pagar."
          : "Cobrança criada! Verifique seu email para o link de pagamento.",
      };
    }),

  // ----------------------------------------------------------
  // VERIFICAR STATUS DO PIX (polling do frontend)
  // ----------------------------------------------------------
  checkPixPayment: protectedProcedure
    .input(z.object({ subscriptionId: z.string() }))
    .query(async ({ ctx, input }) => {
      const subscription = await prisma.subscription.findFirst({
        where: { id: input.subscriptionId, userId: ctx.user.id },
      });

      if (!subscription) {
        throw new TRPCError({ code: "NOT_FOUND" });
      }

      return {
        status: subscription.status,
        isActive: subscription.status === "ACTIVE",
      };
    }),

  // ----------------------------------------------------------
  // STATUS DA ASSINATURA
  // ----------------------------------------------------------
  getStatus: protectedProcedure.query(async ({ ctx }) => {
    const subscription = await prisma.subscription.findFirst({
      where: {
        userId: ctx.user.id,
        status: { in: ["ACTIVE", "PENDING", "OVERDUE"] },
      },
      orderBy: { createdAt: "desc" },
    });

    if (!subscription) {
      return { hasActiveSubscription: false, subscription: null, canUseApp: false };
    }

    const now = new Date();
    const isWithinPeriod = subscription.endsAt ? subscription.endsAt > now : false;
    const canUseApp =
      (subscription.status === "ACTIVE" || subscription.status === "OVERDUE") &&
      isWithinPeriod;

    return {
      hasActiveSubscription: canUseApp,
      subscription: {
        id: subscription.id,
        plan: subscription.plan,
        status: subscription.status,
        startsAt: subscription.startsAt,
        endsAt: subscription.endsAt,
      },
      canUseApp,
      isOverdue: subscription.status === "OVERDUE",
    };
  }),

  // ----------------------------------------------------------
  // CANCELAR
  // ----------------------------------------------------------
  cancel: protectedProcedure.mutation(async ({ ctx }) => {
    const subscription = await prisma.subscription.findFirst({
      where: {
        userId: ctx.user.id,
        status: { in: ["ACTIVE", "OVERDUE"] },
      },
    });

    if (!subscription) {
      throw new TRPCError({
        code: "NOT_FOUND",
        message: "Nenhuma assinatura ativa encontrada.",
      });
    }

    if (subscription.asaasSubscriptionId) {
      try {
        await cancelSubscription(subscription.asaasSubscriptionId);
      } catch (error) {
        console.error("[Cancel] Erro no Asaas:", error);
      }
    }

    await prisma.subscription.update({
      where: { id: subscription.id },
      data: { status: "CANCELED" },
    });

    return {
      success: true,
      message: `Assinatura cancelada. Acesso mantido até ${subscription.endsAt?.toLocaleDateString("pt-BR") ?? "o fim do período"}.`,
      accessUntil: subscription.endsAt,
    };
  }),
});

// ============================================================
// Helpers
// ============================================================

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
    invalid_creditCard: "Dados do cartão inválidos. Verifique o número, validade e CVV.",
    invalid_creditCardHolderInfo: "Dados do titular inválidos. Verifique nome e CPF.",
    declined: "Cartão recusado. Tente outro cartão ou use Pix.",
    expired_card: "Cartão expirado. Use outro cartão.",
    insufficient_funds: "Saldo insuficiente. Tente outro cartão ou use Pix.",
  };
  return map[code ?? ""] || error.message || "Erro ao processar pagamento. Tente novamente.";
}
