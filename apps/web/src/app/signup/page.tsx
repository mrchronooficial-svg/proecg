"use client";

import { Brain, FileText, Share2, Zap } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import { resolvePlan, type CheckoutPlan } from "@/lib/checkout-plans";

const FEATURES = [
  { icon: Zap, text: "Laudo de ECG em segundos com IA" },
  { icon: FileText, text: "Laudos ilimitados, ~30 diagnósticos" },
  { icon: Share2, text: "Exportação em PDF e WhatsApp" },
  { icon: Brain, text: "Medições automáticas (FC, PR, QRS, QT, eixo)" },
];

function SignupInner() {
  const router = useRouter();
  const params = useSearchParams();
  const [plan, setPlan] = useState<CheckoutPlan>(() =>
    resolvePlan(params.get("plan")),
  );
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored =
      typeof window !== "undefined"
        ? window.sessionStorage.getItem("checkout_plan")
        : null;
    const resolved = resolvePlan(params.get("plan") ?? stored);
    setPlan(resolved);
    window.sessionStorage.setItem("checkout_plan", resolved.id);
  }, [params]);

  function handleContinue(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = email.trim().toLowerCase();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmed)) {
      setError("Digite um e-mail válido.");
      return;
    }
    window.sessionStorage.setItem("checkout_email", trimmed);
    window.sessionStorage.setItem("checkout_plan", plan.id);
    router.push("/signup/profile");
  }

  return (
    <CheckoutShell>
      <div className="grid min-h-svh grid-cols-1 lg:grid-cols-2">
        {/* ESQUERDA — formulário */}
        <div className="flex items-center justify-center bg-black/20 px-6 py-16">
          <div className="w-full max-w-[400px]">
            <h1 className="mb-10 text-center text-[34px] font-bold tracking-[-0.02em]">
              Pro<span className="text-[#0a84ff]">ECG</span>
            </h1>

            <form onSubmit={handleContinue} className="space-y-3">
              <input
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError(null);
                }}
                placeholder="Endereço de e-mail"
                className="w-full rounded-full border border-white/15 bg-white/[0.04] px-5 py-3.5 text-[15px] text-white placeholder-white/45 outline-none transition focus:border-[#0a84ff] focus:ring-2 focus:ring-[#0a84ff]/30"
              />
              {error && <p className="px-2 text-sm text-red-400">{error}</p>}
              <button
                type="submit"
                className="w-full rounded-full bg-[#0a84ff] px-5 py-3.5 text-[15px] font-semibold text-white transition-all hover:bg-[#0066cc]"
              >
                Continuar
              </button>
            </form>

            <p className="mt-5 text-center text-[14px] text-white/65">
              Já tem uma conta?{" "}
              <Link href="/login" className="font-medium text-[#0a84ff] underline">
                Entrar
              </Link>
            </p>

            <div className="my-6 flex items-center gap-4">
              <span className="h-px flex-1 bg-white/10" />
              <span className="text-xs font-medium text-white/40">OU</span>
              <span className="h-px flex-1 bg-white/10" />
            </div>

            {/* Google — visual, desabilitado (OAuth ainda não configurado) */}
            <button
              type="button"
              disabled
              title="Em breve"
              className="flex w-full cursor-not-allowed items-center justify-center gap-3 rounded-full border border-white/15 bg-white/[0.04] px-5 py-3.5 text-[15px] font-medium text-white/70"
            >
              <GoogleIcon />
              Continuar com o Google
              <span className="rounded-full bg-white/10 px-2 py-0.5 text-[11px] text-white/55">
                em breve
              </span>
            </button>
          </div>
        </div>

        {/* DIREITA — proposta de valor */}
        <div className="hidden items-center px-6 py-16 lg:flex lg:px-16">
          <div className="max-w-md">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3.5 py-1.5 text-[13px] font-medium tracking-wide text-white/85 backdrop-blur">
              <span className="h-1.5 w-1.5 rounded-full bg-[#0a84ff]" />
              Plano {plan.name} · {plan.monthlyLabel}
            </div>
            <h2 className="text-[30px] font-bold leading-[1.15] tracking-[-0.02em]">
              Laudo de ECG em segundos, direto do seu celular
            </h2>
            <p className="mt-4 text-[15px] font-medium leading-[1.6] text-white/65">
              O ProECG digitaliza a foto do ECG em papel, mede os intervalos e
              monta o laudo descritivo com hipóteses diagnósticas — tudo via IA.
            </p>

            <ul className="mt-8 space-y-5">
              {FEATURES.map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-start gap-3">
                  <Icon size={20} className="mt-0.5 shrink-0 text-[#0a84ff]" />
                  <span className="text-[15px] text-white/85">{text}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </CheckoutShell>
  );
}

function GoogleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1Z" />
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.99.66-2.26 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23Z" />
      <path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84Z" />
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1A11 11 0 0 0 2.18 7.06l3.66 2.84C6.71 7.3 9.14 5.38 12 5.38Z" />
    </svg>
  );
}

export default function SignupPage() {
  return (
    <Suspense fallback={null}>
      <SignupInner />
    </Suspense>
  );
}
