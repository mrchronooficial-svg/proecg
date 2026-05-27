"use client";

import { useForm } from "@tanstack/react-form";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { toast } from "sonner";
import z from "zod";

import { authClient } from "@/lib/auth-client";

import Loader from "./loader";

function GoogleIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
        fill="#4285F4"
      />
      <path
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
        fill="#34A853"
      />
      <path
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
        fill="#FBBC05"
      />
      <path
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
        fill="#EA4335"
      />
    </svg>
  );
}

interface SignInFormProps {
  onSwitchToSignUp: () => void;
}

export default function SignInForm({ onSwitchToSignUp }: SignInFormProps) {
  const router = useRouter();
  const { isPending } = authClient.useSession();
  const [showPassword, setShowPassword] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const form = useForm({
    defaultValues: {
      email: "",
      password: "",
    },
    onSubmit: async ({ value }) => {
      setAuthError(null);
      await authClient.signIn.email(
        {
          email: value.email,
          password: value.password,
        },
        {
          onSuccess: () => {
            router.push("/dashboard");
            toast.success("Login realizado");
          },
          onError: (error) => {
            const msg =
              error.error.message ||
              error.error.statusText ||
              "Não foi possível entrar";
            setAuthError(msg);
          },
        },
      );
    },
    validators: {
      onSubmit: z.object({
        email: z.email("E-mail inválido"),
        password: z.string().min(1, "Senha obrigatória"),
      }),
    },
  });

  const handleGoogleSignIn = async () => {
    setAuthError(null);
    setGoogleLoading(true);
    try {
      await authClient.signIn.social({
        provider: "google",
        callbackURL: "/dashboard",
      });
    } catch (error) {
      const msg =
        error instanceof Error
          ? error.message
          : "Não foi possível entrar com o Google";
      setAuthError(msg);
      setGoogleLoading(false);
    }
  };

  if (isPending) {
    return <Loader />;
  }

  return (
    <div className="w-full max-w-[400px]">
      <h1 className="mb-10 text-center text-[34px] font-bold tracking-[-0.02em] leading-[1.1] text-white">
        Pro<span className="text-[#0a84ff]">ECG</span>
      </h1>

      <button
        type="button"
        onClick={handleGoogleSignIn}
        disabled={googleLoading}
        className="flex h-[52px] w-full items-center justify-center gap-3 rounded-full border border-white/15 bg-white/[0.04] text-[15px] font-medium text-white transition-all duration-200 hover:bg-white/10 disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a84ff]"
      >
        {googleLoading ? (
          <Loader2 size={20} className="animate-spin text-white/60" aria-hidden="true" />
        ) : (
          <GoogleIcon />
        )}
        <span>Continuar com o Google</span>
      </button>

      <div className="my-6 flex items-center gap-4">
        <div className="h-px flex-1 bg-white/10" />
        <span className="text-[13px] font-medium text-white/40">ou</span>
        <div className="h-px flex-1 bg-white/10" />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          e.stopPropagation();
          form.handleSubmit();
        }}
        className="flex flex-col gap-3"
        noValidate
      >
        <form.Field name="email">
          {(field) => {
            const fieldError = field.state.meta.errors[0]?.message;
            return (
              <div>
                <div className="relative rounded-2xl border border-white/12 bg-white/[0.04] transition-all duration-200 focus-within:border-[#0a84ff] focus-within:ring-2 focus-within:ring-[#0a84ff]/25">
                  <label
                    htmlFor={field.name}
                    className="pointer-events-none absolute left-5 top-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-white/45"
                  >
                    E-mail
                  </label>
                  <input
                    id={field.name}
                    name={field.name}
                    type="email"
                    autoComplete="email"
                    inputMode="email"
                    value={field.state.value}
                    onBlur={field.handleBlur}
                    onChange={(e) => field.handleChange(e.target.value)}
                    aria-invalid={!!fieldError}
                    className="block h-[56px] w-full bg-transparent px-5 pt-6 pb-1 text-[15px] text-white outline-none"
                  />
                </div>
                {fieldError && (
                  <p className="mt-1.5 px-2 text-[13px] text-red-400">{fieldError}</p>
                )}
              </div>
            );
          }}
        </form.Field>

        <form.Field name="password">
          {(field) => {
            const fieldError = field.state.meta.errors[0]?.message;
            return (
              <div>
                <div className="relative rounded-2xl border border-white/12 bg-white/[0.04] transition-all duration-200 focus-within:border-[#0a84ff] focus-within:ring-2 focus-within:ring-[#0a84ff]/25">
                  <label
                    htmlFor={field.name}
                    className="pointer-events-none absolute left-5 top-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-white/45"
                  >
                    Senha
                  </label>
                  <input
                    id={field.name}
                    name={field.name}
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    value={field.state.value}
                    onBlur={field.handleBlur}
                    onChange={(e) => field.handleChange(e.target.value)}
                    aria-invalid={!!fieldError}
                    className="block h-[56px] w-full bg-transparent pl-5 pr-12 pt-6 pb-1 text-[15px] text-white outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    aria-label={showPassword ? "Ocultar senha" : "Mostrar senha"}
                    className="absolute right-2 top-1/2 flex size-9 -translate-y-1/2 items-center justify-center rounded-md text-white/50 transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a84ff]"
                  >
                    {showPassword ? (
                      <EyeOff size={18} strokeWidth={1.8} />
                    ) : (
                      <Eye size={18} strokeWidth={1.8} />
                    )}
                  </button>
                </div>
                {fieldError && (
                  <p className="mt-1.5 px-2 text-[13px] text-red-400">{fieldError}</p>
                )}
              </div>
            );
          }}
        </form.Field>

        {authError && (
          <div className="rounded-xl bg-red-500/10 px-3 py-3 text-[14px] leading-[1.3] text-red-400">
            {authError}
          </div>
        )}

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
              className="mt-2 flex h-[52px] w-full items-center justify-center rounded-full bg-[#0a84ff] text-[16px] font-semibold text-white transition-all duration-200 hover:bg-[#0066cc] active:scale-[0.98] disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a84ff] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent"
            >
              {isSubmitting ? (
                <Loader2 size={20} className="animate-spin" aria-hidden="true" />
              ) : (
                "Entrar"
              )}
            </button>
          )}
        </form.Subscribe>
      </form>

      <div className="mt-5 flex flex-col items-center gap-3">
        <Link
          href="/esqueci-senha"
          className="text-[15px] font-medium text-[#0a84ff] transition-opacity duration-200 hover:underline"
        >
          Redefinir senha
        </Link>
        <p className="text-[15px] text-white/60">
          Não tem conta?{" "}
          <button
            type="button"
            onClick={onSwitchToSignUp}
            className="font-medium text-[#0a84ff] transition-opacity duration-200 hover:underline focus-visible:underline focus-visible:outline-none"
          >
            Crie uma
          </button>
        </p>
      </div>
    </div>
  );
}
