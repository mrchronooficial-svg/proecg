"use client";

import { useForm } from "@tanstack/react-form";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { toast } from "sonner";
import z from "zod";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import { bareInput, FloatingField } from "@/components/checkout/FloatingField";
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

  return (
    <CheckoutShell>
      <div className="flex min-h-svh items-center justify-center px-6 py-12">
        <div className="w-full max-w-[440px]">
          <div className="rounded-3xl border border-white/10 bg-white/[0.05] p-8 backdrop-blur">
            {!token ? (
              <div className="text-center">
                <h1 className="text-[26px] font-bold tracking-[-0.02em]">
                  Link inválido
                </h1>
                <p className="mt-3 text-[15px] leading-[1.6] text-white/65">
                  Este link de redefinição de senha é inválido ou expirou.
                </p>
              </div>
            ) : (
              <>
                <h1 className="mb-6 text-center text-[26px] font-bold tracking-[-0.02em]">
                  Definir nova senha
                </h1>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    form.handleSubmit();
                  }}
                  className="space-y-3"
                >
                  <form.Field name="password">
                    {(field) => (
                      <div>
                        <FloatingField label="Nova senha">
                          <input
                            id={field.name}
                            name={field.name}
                            type="password"
                            autoComplete="new-password"
                            value={field.state.value}
                            onBlur={field.handleBlur}
                            onChange={(e) => field.handleChange(e.target.value)}
                            placeholder="Mínimo 8 caracteres"
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

                  <form.Field name="confirmPassword">
                    {(field) => (
                      <div>
                        <FloatingField label="Confirmar senha">
                          <input
                            id={field.name}
                            name={field.name}
                            type="password"
                            autoComplete="new-password"
                            value={field.state.value}
                            onBlur={field.handleBlur}
                            onChange={(e) => field.handleChange(e.target.value)}
                            placeholder="Repita a senha"
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
                        {isSubmitting ? "Redefinindo..." : "Definir senha"}
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

export default function RedefinirSenhaPage() {
  return (
    <Suspense>
      <ResetForm />
    </Suspense>
  );
}
