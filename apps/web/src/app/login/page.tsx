"use client";

import { X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import SignInForm from "@/components/sign-in-form";

export default function LoginPage() {
  const router = useRouter();

  return (
    <CheckoutShell>
      <Link
        href="/"
        aria-label="Fechar"
        className="absolute right-4 top-4 z-10 inline-flex size-10 items-center justify-center rounded-lg border border-white/15 text-white/80 transition-colors duration-200 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a84ff]"
      >
        <X size={20} strokeWidth={2} aria-hidden="true" />
      </Link>

      <main className="flex min-h-svh items-center justify-center px-6 py-16">
        <SignInForm onSwitchToSignUp={() => router.push("/signup")} />
      </main>

      <footer
        className="pointer-events-none absolute inset-x-0 bottom-0 px-4 pb-4 text-center text-[12px] leading-[1.3] text-white/45"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 16px)" }}
      >
        Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
      </footer>
    </CheckoutShell>
  );
}
