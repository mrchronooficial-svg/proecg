"use client";

import { motion } from "framer-motion";
import { Check } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import { authClient } from "@/lib/auth-client";
import { resolvePlan, type CheckoutPlan } from "@/lib/checkout-plans";

export default function CheckoutSuccessPage() {
  const { data: session } = authClient.useSession();
  const [plan, setPlan] = useState<CheckoutPlan | null>(null);

  useEffect(() => {
    const stored = window.sessionStorage.getItem("checkout_plan");
    setPlan(resolvePlan(stored));
    window.sessionStorage.removeItem("checkout_email");
    window.sessionStorage.removeItem("checkout_name");
  }, []);

  const nextBilling = plan
    ? (() => {
        const d = new Date();
        d.setMonth(d.getMonth() + plan.months);
        return d.toLocaleDateString("pt-BR");
      })()
    : "";

  return (
    <CheckoutShell>
      <div className="flex min-h-svh items-center justify-center px-6 py-12">
        <div className="w-full max-w-[460px] text-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 18 }}
            className="mx-auto mb-6 flex size-20 items-center justify-center rounded-full bg-emerald-500 shadow-[0_0_60px_-10px_rgba(16,185,129,0.6)]"
          >
            <Check size={40} strokeWidth={3} className="text-white" />
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="text-[32px] font-bold tracking-[-0.02em]"
          >
            Pagamento confirmado!
          </motion.h1>
          <p className="mt-2 text-[15px] text-white/65">
            Seu plano {plan?.name ?? ""} foi ativado com sucesso.
          </p>

          <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-6 text-left backdrop-blur">
            <SummaryRow label="Plano" value={plan?.name ?? "—"} />
            <SummaryRow label="Valor" value={plan?.totalLabel ?? "—"} />
            <SummaryRow label="Próxima cobrança" value={nextBilling} />
          </div>

          <Link
            href="/dashboard"
            className="mt-8 block w-full rounded-full bg-[#0a84ff] px-6 py-3.5 text-[16px] font-semibold text-white transition-all hover:bg-[#0066cc]"
          >
            Ir para o Dashboard
          </Link>

          <p className="mt-4 text-[14px] text-white/60">
            Enviamos um email de confirmação para{" "}
            <strong className="text-white/85">{session?.user.email ?? "seu email"}</strong>{" "}
            com o link para definir sua senha.
          </p>
        </div>
      </div>
    </CheckoutShell>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-white/10 py-2.5 text-[14px] last:border-0">
      <span className="text-white/55">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
