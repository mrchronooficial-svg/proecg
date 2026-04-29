"use client";

import Link from "next/link";
import { Reveal } from "./reveal";

export function CTAFinal() {
  return (
    <section
      className="relative overflow-hidden py-20 text-white sm:py-32 lg:py-40"
      style={{
        backgroundImage:
          "radial-gradient(50% 70% at 50% 50%, rgba(0,102,204,0.35), transparent 60%), linear-gradient(180deg, #0c1330 0%, #1a1a3e 100%)",
      }}
    >
      <div className="mx-auto max-w-4xl px-5 text-center sm:px-6">
        <Reveal>
          <h2 className="text-[30px] font-semibold leading-[1.1] tracking-tight sm:text-[48px] lg:text-[68px]">
            Comece a usar o ProECG agora.
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-[16px] leading-[1.6] text-white/70 sm:mt-6 sm:text-[18px] lg:text-[20px]">
            Junte-se a médicos que já confiam na IA para apoio diagnóstico em cardiologia.
          </p>
          <div className="mt-8 flex flex-col items-stretch gap-3 sm:mt-10 sm:items-center sm:gap-4">
            <Link
              href="/login"
              className="block w-full rounded-full bg-white px-6 py-4 text-center text-[18px] font-semibold text-[#1d1d1f] shadow-[0_20px_40px_-15px_rgba(255,255,255,0.4)] transition-transform hover:scale-[1.02] sm:inline-block sm:w-auto sm:px-9 sm:text-[16px]"
            >
              Criar conta gratuita
            </Link>
            <p className="text-[13px] text-white/55">Sem cartão de crédito para teste.</p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
