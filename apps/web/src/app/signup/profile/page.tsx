"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import { bareInput, FloatingField } from "@/components/checkout/FloatingField";
import { authClient } from "@/lib/auth-client";

/** Senha aleatória forte e descartável — o médico define a senha real
 *  pelo link enviado no email de confirmação do pagamento. */
function randomPassword() {
  const bytes = new Uint8Array(18);
  crypto.getRandomValues(bytes);
  const base = btoa(String.fromCharCode(...bytes)).replace(/[^a-zA-Z0-9]/g, "");
  return `Aa1!${base}`.slice(0, 24);
}

export default function ProfilePage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const storedEmail = window.sessionStorage.getItem("checkout_email");
    if (!storedEmail) {
      router.replace("/signup");
      return;
    }
    setEmail(storedEmail);
  }, [router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (name.trim().length < 2) {
      setError("Digite seu nome completo.");
      return;
    }
    if (!email) return;
    setSubmitting(true);
    setError(null);

    await authClient.signUp.email(
      { email, name: name.trim(), password: randomPassword() },
      {
        onSuccess: () => {
          window.sessionStorage.setItem("checkout_name", name.trim());
          router.push("/checkout");
        },
        onError: (err) => {
          setSubmitting(false);
          const msg = err.error.message || err.error.statusText;
          if (/exist|already|registered/i.test(msg)) {
            toast.error("Já existe uma conta com este e-mail. Faça login.");
            router.push("/login");
            return;
          }
          setError(msg);
        },
      },
    );
  }

  return (
    <CheckoutShell>
      <div className="flex min-h-svh items-center justify-center px-6 py-12">
        <div className="w-full max-w-[440px] text-center">
          <h1 className="text-[34px] font-bold leading-[1.1] tracking-[-0.02em] sm:text-[40px]">
            Como podemos
            <br />
            te chamar?
          </h1>
          <p className="mx-auto mt-4 max-w-sm text-[15px] leading-[1.5] text-white/60">
            Usamos seu nome para personalizar seus laudos, de acordo com nossa{" "}
            <Link href="/privacidade" target="_blank" className="text-white/80 underline">
              Política de privacidade
            </Link>
            .
          </p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-3 text-left">
            <FloatingField label="Nome completo">
              <input
                type="text"
                autoComplete="name"
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setError(null);
                }}
                placeholder="Dr(a). Nome Sobrenome"
                className={bareInput}
              />
            </FloatingField>
            {error && <p className="px-2 text-sm text-red-400">{error}</p>}
          </form>

          <p className="mx-auto mt-6 max-w-sm text-[13px] leading-[1.5] text-white/45">
            Ao clicar em &ldquo;Continuar&rdquo;, você concorda com nossos{" "}
            <Link href="/termos" target="_blank" className="text-white/70 underline">
              Termos
            </Link>{" "}
            e que leu nossa{" "}
            <Link href="/privacidade" target="_blank" className="text-white/70 underline">
              Política de Privacidade
            </Link>
            . Você define sua senha após a confirmação do pagamento.
          </p>

          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="mt-7 w-full rounded-full bg-[#0a84ff] px-6 py-4 text-[16px] font-semibold text-white transition-all hover:bg-[#0066cc] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {submitting ? "Criando conta..." : "Continuar"}
          </button>
        </div>
      </div>
    </CheckoutShell>
  );
}
