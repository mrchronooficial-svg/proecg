"use client";

import { Button } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { Input } from "@proecg/ui/components/input";
import { Label } from "@proecg/ui/components/label";
import { useForm } from "@tanstack/react-form";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { toast } from "sonner";
import z from "zod";

import { authClient } from "@/lib/auth-client";

function ResetForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");

  const form = useForm({
    defaultValues: { password: "", confirmPassword: "" },
    onSubmit: async ({ value }) => {
      if (!token) {
        toast.error("Token inválido. Solicite um novo link.");
        return;
      }
      const { error } = await authClient.$fetch("/api/auth/reset-password", {
        method: "POST",
        body: { newPassword: value.password, token },
      });
      if (error) {
        toast.error("Erro ao redefinir senha");
      } else {
        toast.success("Senha redefinida com sucesso!");
        router.push("/login");
      }
    },
    validators: {
      onSubmit: z
        .object({
          password: z.string().min(8, "Senha deve ter pelo menos 8 caracteres"),
          confirmPassword: z.string(),
        })
        .refine((data) => data.password === data.confirmPassword, {
          message: "As senhas não coincidem",
          path: ["confirmPassword"],
        }),
    },
  });

  if (!token) {
    return (
      <main className="min-h-[calc(100svh-3.5rem)] flex items-center justify-center px-4">
        <Card className="glass shadow-lg max-w-md p-8 text-center">
          <h1 className="mb-4 text-2xl font-bold gradient-text">Link inválido</h1>
          <p className="text-muted-foreground">
            Este link de redefinição de senha é inválido ou expirou.
          </p>
        </Card>
      </main>
    );
  }

  return (
    <main className="min-h-[calc(100svh-3.5rem)] flex items-center justify-center px-4">
      <Card className="glass shadow-lg w-full max-w-md p-8">
        <h1 className="mb-6 text-center text-2xl font-bold gradient-text">
          Redefinir senha
        </h1>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            form.handleSubmit();
          }}
          className="space-y-4"
        >
          <form.Field name="password">
            {(field) => (
              <div className="space-y-2">
                <Label htmlFor={field.name}>Nova senha</Label>
                <Input
                  id={field.name}
                  name={field.name}
                  type="password"
                  value={field.state.value}
                  onBlur={field.handleBlur}
                  onChange={(e) => field.handleChange(e.target.value)}
                />
                {field.state.meta.errors.map((error) => (
                  <p key={error?.message} className="text-sm text-red-500">
                    {error?.message}
                  </p>
                ))}
              </div>
            )}
          </form.Field>

          <form.Field name="confirmPassword">
            {(field) => (
              <div className="space-y-2">
                <Label htmlFor={field.name}>Confirmar senha</Label>
                <Input
                  id={field.name}
                  name={field.name}
                  type="password"
                  value={field.state.value}
                  onBlur={field.handleBlur}
                  onChange={(e) => field.handleChange(e.target.value)}
                />
                {field.state.meta.errors.map((error) => (
                  <p key={error?.message} className="text-sm text-red-500">
                    {error?.message}
                  </p>
                ))}
              </div>
            )}
          </form.Field>

          <form.Subscribe
            selector={(state) => ({
              canSubmit: state.canSubmit,
              isSubmitting: state.isSubmitting,
            })}
          >
            {({ canSubmit, isSubmitting }) => (
              <Button
                type="submit"
                className="w-full"
                disabled={!canSubmit || isSubmitting}
              >
                {isSubmitting ? "Redefinindo..." : "Redefinir senha"}
              </Button>
            )}
          </form.Subscribe>
        </form>
      </Card>
    </main>
  );
}

export default function RedefinirSenhaPage() {
  return (
    <Suspense>
      <ResetForm />
    </Suspense>
  );
}
