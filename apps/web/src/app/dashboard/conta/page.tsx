"use client";

import { Button } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { authClient } from "@/lib/auth-client";
import { trpc } from "@/utils/trpc";

const PLAN_LABELS: Record<string, string> = {
  MONTHLY: "Mensal",
  SEMI: "Semestral",
  ANNUAL: "Anual",
};

const STATUS_LABELS: Record<string, string> = {
  ACTIVE: "Ativa",
  PENDING: "Pendente",
  OVERDUE: "Em atraso",
  CANCELLED: "Cancelada",
  EXPIRED: "Expirada",
};

export default function ContaPage() {
  const { data: session } = authClient.useSession();
  const subscription = useQuery(trpc.subscription.getStatus.queryOptions());
  const queryClient = useQueryClient();

  const cancel = useMutation(
    trpc.subscription.cancel.mutationOptions({
      onSuccess: () => {
        toast.success("Assinatura cancelada");
        queryClient.invalidateQueries({
          queryKey: trpc.subscription.getStatus.queryKey(),
        });
      },
      onError: (error) => {
        toast.error(error.message || "Erro ao cancelar");
      },
    }),
  );

  return (
    <div className="mx-auto max-w-2xl px-4 py-8 pb-20 md:pb-8">
      <h1 className="mb-6 text-2xl font-bold">Minha Conta</h1>

      <div className="space-y-6">
        <Card className="glass p-6">
          <h2 className="mb-4 text-lg font-semibold">Dados pessoais</h2>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-muted-foreground">Nome</dt>
              <dd>{session?.user.name}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted-foreground">Email</dt>
              <dd>{session?.user.email}</dd>
            </div>
          </dl>
        </Card>

        <Card className="glass p-6">
          <h2 className="mb-4 text-lg font-semibold">Assinatura</h2>
          {subscription.data ? (
            <div className="space-y-4">
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Plano</dt>
                  <dd>{PLAN_LABELS[subscription.data.plan] ?? subscription.data.plan}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Status</dt>
                  <dd>{STATUS_LABELS[subscription.data.status] ?? subscription.data.status}</dd>
                </div>
                {subscription.data.endsAt && (
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Válida até</dt>
                    <dd>
                      {new Date(subscription.data.endsAt).toLocaleDateString(
                        "pt-BR",
                      )}
                    </dd>
                  </div>
                )}
              </dl>
              {(subscription.data.status === "ACTIVE" ||
                subscription.data.status === "PENDING") && (
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={cancel.isPending}
                  onClick={() => cancel.mutate()}
                >
                  {cancel.isPending ? "Cancelando..." : "Cancelar assinatura"}
                </Button>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              Nenhuma assinatura ativa.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
