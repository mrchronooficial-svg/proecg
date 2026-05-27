"use client";

import { useForm } from "@tanstack/react-form";
import { useState } from "react";
import { toast } from "sonner";
import z from "zod";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import { bareInput, FloatingField } from "@/components/checkout/FloatingField";
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
      onSubmit: z.object({ email: z.email("Email inválido") }),
    },
  });

  return (
    <CheckoutShell>
      <div className="flex min-h-svh items-center justify-center px-6 py-12">
        <div className="w-full max-w-[440px]">
          <div className="rounded-3xl border border-white/10 bg-white/[0.05] p-8 backdrop-blur">
            {sent ? (
              <div className="text-center">
                <h1 className="text-[26px] font-bold tracking-[-0.02em]">
                  Email enviado
                </h1>
                <p className="mt-3 text-[15px] leading-[1.6] text-white/65">
                  Se o email estiver cadastrado, você receberá um link para
                  definir sua senha.
                </p>
              </div>
            ) : (
              <>
                <h1 className="text-center text-[26px] font-bold tracking-[-0.02em]">
                  Esqueci minha senha
                </h1>
                <p className="mx-auto mt-2 mb-6 max-w-sm text-center text-[14px] text-white/60">
                  Enviaremos um link para você definir uma nova senha.
                </p>

                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    form.handleSubmit();
                  }}
                  className="space-y-3"
                >
                  <form.Field name="email">
                    {(field) => (
                      <div>
                        <FloatingField label="E-mail">
                          <input
                            id={field.name}
                            name={field.name}
                            type="email"
                            autoComplete="email"
                            value={field.state.value}
                            onBlur={field.handleBlur}
                            onChange={(e) => field.handleChange(e.target.value)}
                            placeholder="seu@email.com"
                            className={bareInput}
                          />
                        </FloatingField>
                        {field.state.meta.errors.map((error) => (
                          <p key={error?.message} className="mt-1.5 px-2 text-[13px] text-red-400">
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
                      <button
                        type="submit"
                        disabled={!canSubmit || isSubmitting}
                        className="w-full rounded-full bg-[#0a84ff] px-6 py-3.5 text-[16px] font-semibold text-white transition-all hover:bg-[#0066cc] disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {isSubmitting ? "Enviando..." : "Enviar link"}
                      </button>
                    )}
                  </form.Subscribe>
                </form>
              </>
            )}
          </div>
        </div>
      </div>
    </CheckoutShell>
  );
}
