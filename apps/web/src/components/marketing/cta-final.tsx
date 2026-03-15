"use client";

import Link from "next/link";
import { Reveal } from "./reveal";

export function CTAFinal() {
  return (
    <section className="bg-[#122056] px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[600px] text-center">
        <Reveal>
          <h2 className="text-3xl font-bold text-white sm:text-4xl">
            O próximo ECG não precisa te deixar em dúvida.
          </h2>
        </Reveal>
        <Reveal delay={0.1}>
          <p className="mt-4 text-lg text-white/80">
            Comece agora e tenha um cardiologista de IA no bolso do jaleco.
          </p>
        </Reveal>
        <Reveal delay={0.2}>
          <div className="mt-8 flex flex-col items-center gap-3">
            <Link
              href="/login"
              className="rounded-xl bg-white px-10 py-4 text-lg font-semibold text-[#5B65DC] transition-all hover:bg-gray-100 hover:-translate-y-0.5"
            >
              Assinar o ProECG
            </Link>
            <p className="text-sm text-white/60">
              A partir de R$ 197/mês · Pix ou cartão · Acesso imediato
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
