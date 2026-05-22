"use client";

import Link from "next/link";
import { useState } from "react";
import { X } from "lucide-react";

import SignInForm from "@/components/sign-in-form";
import SignUpForm from "@/components/sign-up-form";

export default function LoginPage() {
  // Login page defaults to sign-in (URL is /login)
  const [showSignIn, setShowSignIn] = useState(true);

  return (
    <div
      className="relative min-h-svh w-full bg-white text-apple-text"
      style={{
        fontFamily:
          "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <Link
        href="/"
        aria-label="Fechar"
        className="absolute right-4 top-4 z-10 inline-flex size-10 items-center justify-center rounded-lg border border-apple-border text-apple-text transition-colors duration-200 hover:bg-apple-border-light focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
      >
        <X size={20} strokeWidth={2} aria-hidden="true" />
      </Link>

      <main className="flex min-h-svh items-center justify-center px-6 py-16 md:px-10">
        {showSignIn ? (
          <SignInForm onSwitchToSignUp={() => setShowSignIn(false)} />
        ) : (
          <SignUpForm onSwitchToSignIn={() => setShowSignIn(true)} />
        )}
      </main>

      <footer
        className="pointer-events-none absolute inset-x-0 bottom-0 px-4 pb-4 text-center text-[12px] leading-[1.3] text-apple-text-tertiary"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 16px)" }}
      >
        Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
      </footer>
    </div>
  );
}
