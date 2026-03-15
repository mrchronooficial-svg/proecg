"use client";

import { Button } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { Input } from "@proecg/ui/components/input";
import { Label } from "@proecg/ui/components/label";
import { useForm } from "@tanstack/react-form";
import { useState } from "react";
import { toast } from "sonner";
import z from "zod";

import { authClient } from "@/lib/auth-client";

export default function EsqueciSenhaPage() {
  const [sent, setSent] = useState(false);

  const form = useForm({
    defaultValues: { email: "" },
    onSubmit: async ({ value }) => {
      const { error } = await authClient.$fetch("/api/auth/forget-password", {
        method: "POST",
        body: { email: value.email, redirectTo: "/redefinir-senha" },
      });
      if (error) {
        toast.error("Erro ao enviar email");
      } else {
        setSent(true);
      }
    },
    validators: {
      onSubmit: z.object({
        email: z.email("Email inválido"),
      }),
    },
  });

  if (sent) {
    return (
      <main className="min-h-[calc(100svh-3.5rem)] flex items-center justify-center px-4">
        <Card className="glass shadow-lg max-w-md p-8 text-center">
          <h1 className="mb-4 text-2xl font-bold text-primary">Email enviado</h1>
          <p className="text-muted-foreground">
            Se o email estiver cadastrado, você receberá um link para redefinir
            sua senha.
          </p>
        </Card>
      </main>
    );
  }

  return (
    <main className="min-h-[calc(100svh-3.5rem)] flex items-center justify-center px-4">
      <Card className="glass shadow-lg w-full max-w-md p-8">
        <h1 className="mb-6 text-center text-2xl font-bold text-primary">Esqueci minha senha</h1>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            form.handleSubmit();
          }}
          className="space-y-4"
        >
          <form.Field name="email">
            {(field) => (
              <div className="space-y-2">
                <Label htmlFor={field.name}>Email</Label>
                <Input
                  id={field.name}
                  name={field.name}
                  type="email"
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
                {isSubmitting ? "Enviando..." : "Enviar link"}
              </Button>
            )}
          </form.Subscribe>
        </form>
      </Card>
    </main>
  );
}
