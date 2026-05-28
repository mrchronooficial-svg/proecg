"use client";

import { useForm } from "@tanstack/react-form";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { toast } from "sonner";
import z from "zod";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import { bareInput, FloatingField } from "@/components/checkout/FloatingField";

type TokenState =
  | { status: "checking" }
  | { status: "invalid"; message: string }
  | { status: "valid" }
  | { status: "success" };

function ResetForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");

  const [tokenState, setTokenState] = useState<TokenState>({
    status: "checking",
  });

  useEffect(() => {
    let cancelled = false;
    async function check() {
      if (!token) {
        setTokenState({
          status: "invalid",
          message: "Link inválido. Solicite um novo link de troca de senha.",
        });
        return;
      }
      try {
        const res = await fetch(
          `/api/reset-password/validate?token=${encodeURIComponent(token)}`,
          { cache: "no-store" },
        );
        const data = (await res.json()) as { valid?: boolean };
        if (cancelled) return;
        if (data.valid) {
          setTokenState({ status: "valid" });
        } else {
          setTokenState({
            status: "invalid",
            message:
              "Este link de troca de senha é inválido ou expirou. Solicite um novo.",
          });
        }
      } catch {
        if (cancelled) return;
        setTokenState({
          status: "invalid",
          message: "Não foi possível validar o link. Tente novamente.",
        });
      }
    }
    void check();
    return () => {
      cancelled = true;
    };
  }, [token]);

  const form = useForm({
    defaultValues: { password: "", confirmPassword: "" },
    onSubmit: async ({ value }) => {
      if (!token) return;
      try {
        const res = await fetch("/api/reset-password", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token, newPassword: value.password }),
        });
        const data = (await res.json().catch(() => ({}))) as {
          ok?: boolean;
          error?: string;
        };
        if (!res.ok || !data.ok) {
          toast.error(data.error ?? "Erro ao redefinir senha");
          return;
        }
        setTokenState({ status: "success" });
      } catch {
        toast.error("Erro ao redefinir senha. Tente novamente.");
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
            {tokenState.status === "checking" && (
              <div className="text-center">
                <h1 className="text-[26px] font-bold tracking-[-0.02em]">
                  Validando link…
                </h1>
                <p className="mt-3 text-[15px] leading-[1.6] text-white/65">
                  Só um instante.
                </p>
              </div>
            )}

            {tokenState.status === "invalid" && (
              <div className="text-center">
                <h1 className="text-[26px] font-bold tracking-[-0.02em]">
                  Link inválido
                </h1>
                <p className="mt-3 text-[15px] leading-[1.6] text-white/65">
                  {tokenState.message}
                </p>
                <Link
                  href="/esqueci-senha"
                  className="mt-6 inline-block w-full rounded-full bg-[#0a84ff] px-6 py-3.5 text-[16px] font-semibold text-white transition-all hover:bg-[#0066cc]"
                >
                  Solicitar novo link
                </Link>
              </div>
            )}

            {tokenState.status === "success" && (
              <div className="text-center">
                <h1 className="text-[26px] font-bold tracking-[-0.02em]">
                  Senha alterada com sucesso ✓
                </h1>
                <p className="mt-3 text-[15px] leading-[1.6] text-white/65">
                  Você já pode acessar o ProECG com sua nova senha.
                </p>
                <button
                  type="button"
                  onClick={() => router.push("/login")}
                  className="mt-6 w-full rounded-full bg-[#0a84ff] px-6 py-3.5 text-[16px] font-semibold text-white transition-all hover:bg-[#0066cc]"
                >
                  Ir para o login
                </button>
              </div>
            )}

            {tokenState.status === "valid" && (
              <>
                <h1 className="text-center text-[26px] font-bold tracking-[-0.02em]">
                  Defina sua nova senha
                </h1>
                <p className="mx-auto mt-2 mb-6 max-w-sm text-center text-[14px] text-white/60">
                  Escolha uma senha forte com pelo menos 8 caracteres.
                </p>

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
                          <p
                            key={error?.message}
                            className="mt-1.5 px-2 text-[13px] text-red-400"
                          >
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
                          <p
                            key={error?.message}
                            className="mt-1.5 px-2 text-[13px] text-red-400"
                          >
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
                        {isSubmitting ? "Salvando..." : "Salvar nova senha"}
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

export default function ResetPasswordPage() {
  return (
    <Suspense>
      <ResetForm />
    </Suspense>
  );
}
