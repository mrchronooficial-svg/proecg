"use client";

import { ShieldCheck, Lock, Database, Cloud } from "lucide-react";
import { Reveal } from "./reveal";

const items = [
  {
    icon: ShieldCheck,
    title: "Conformidade LGPD",
    body: "Tratamento de dados em linha com a Lei Geral de Proteção de Dados.",
  },
  {
    icon: Lock,
    title: "Criptografia em trânsito e repouso",
    body: "TLS 1.3 nas APIs e AES-256 no armazenamento. Sem exceções.",
  },
  {
    icon: Database,
    title: "Sem dados de paciente",
    body: "Não armazenamos nome, CPF ou identificadores. Apenas a foto do ECG.",
  },
  {
    icon: Cloud,
    title: "Infraestrutura cloud segura",
    body: "Hospedado em provedores Tier-1 com SOC 2 e ISO 27001.",
  },
];

export function Security() {
  return (
    <section className="bg-white py-20 sm:py-32 lg:py-40">
      <div className="mx-auto max-w-7xl px-6">
        <Reveal className="text-center sm:text-left">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#0066CC] sm:text-[13px]">
            Segurança e Conformidade
          </p>
          <h2 className="mx-auto mt-4 max-w-3xl text-[28px] font-bold leading-[1.15] tracking-[-0.02em] break-words text-[#1d1d1f] sm:mx-0 sm:mt-5 sm:text-[48px] lg:text-[56px]">
            Segurança e privacidade desde o dia 1.
          </h2>
        </Reveal>

        <div className="mt-10 grid grid-cols-1 gap-4 sm:mt-16 sm:grid-cols-2 sm:gap-6 lg:mt-20 lg:grid-cols-4">
          {items.map((it, i) => (
            <Reveal key={it.title} delay={i * 0.06}>
              <div className="rounded-2xl border border-black/5 bg-[#fafafa] p-5 sm:p-8">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-[#1d1d1f] text-white sm:h-12 sm:w-12">
                  <it.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-4 text-[17px] font-bold tracking-[-0.01em] text-[#1d1d1f] sm:mt-5 sm:text-[18px]">
                  {it.title}
                </h3>
                <p className="mt-2 text-[14px] leading-[1.5] text-[#86868b] sm:mt-2.5 sm:text-[14.5px]">
                  {it.body}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
