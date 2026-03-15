"use client";

import { Button } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

import { trpc } from "@/utils/trpc";

export default function AssinarPage() {
  const plans = useQuery(trpc.subscription.listPlans.queryOptions());
  const status = useQuery(trpc.subscription.getStatus.queryOptions());

  const checkout = useMutation(
    trpc.subscription.createCheckout.mutationOptions({
      onSuccess: (data) => {
        if (data.paymentLink) {
          window.location.href = data.paymentLink;
        } else {
          toast.success("Assinatura criada! Aguardando confirmação de pagamento.");
        }
      },
      onError: (error) => {
        toast.error(error.message || "Erro ao criar assinatura");
      },
    }),
  );

  if (status.data) {
    return (
      <main className="mx-auto max-w-2xl px-4 py-16 text-center">
        <h1 className="mb-4 text-2xl font-bold">Você já tem uma assinatura</h1>
        <p className="text-muted-foreground">
          Status: <strong>{status.data.status}</strong> — Plano:{" "}
          <strong>{status.data.plan}</strong>
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-16">
      <h1 className="mb-4 text-center text-3xl font-bold">Escolha seu plano</h1>
      <p className="mb-12 text-center text-muted-foreground">
        ECGs ilimitados. Cancele quando quiser.
      </p>

      <div className="grid gap-6 sm:grid-cols-3">
        {plans.data?.map((plan) => (
          <Card
            key={plan.id}
            className={`relative flex flex-col gap-4 p-6 ${
              plan.highlight ? "border-2 border-primary shadow-lg" : "glass"
            }`}
          >
            {plan.highlight && (
              <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-primary px-3 py-1 text-xs font-semibold text-white">
                {plan.badge}
              </span>
            )}
            <h3 className="text-xl font-semibold">{plan.name}</h3>
            <div className="flex items-baseline gap-1">
              <span className="text-4xl font-bold">{plan.priceLabel}</span>
              <span className="text-muted-foreground">{plan.period}</span>
            </div>
            <p className="text-sm text-muted-foreground">
              {plan.durationMonths === 1
                ? "Cobrança mensal"
                : `${plan.durationMonths} meses de acesso`}
            </p>
            <Button
              variant={plan.highlight ? "default" : "outline"}
              className="mt-auto w-full"
              disabled={checkout.isPending}
              onClick={() => checkout.mutate({ plan: plan.id })}
            >
              {checkout.isPending ? "Processando..." : "Assinar"}
            </Button>
          </Card>
        ))}
      </div>
    </main>
  );
}
