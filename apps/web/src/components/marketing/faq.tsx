"use client";

import { useState } from "react";
import { ChevronDownIcon } from "lucide-react";
import { Reveal } from "./reveal";

const faqs = [
  {
    q: "O ProECG substitui o cardiologista?",
    a: "Não. O ProECG é uma ferramenta de apoio à decisão clínica. Ele auxilia na interpretação, mas o diagnóstico definitivo e a conduta terapêutica são sempre responsabilidade do médico assistente. Cada laudo inclui um disclaimer obrigatório.",
  },
  {
    q: "Quais tipos de ECG são aceitos?",
    a: "ECGs de 12 derivações em papel, tiras de ritmo e traçados de aparelhos portáteis. Basta fotografar com o celular — o sistema aceita fotos em qualquer ângulo e iluminação razoáveis.",
  },
  {
    q: "Quantos diagnósticos o sistema cobre?",
    a: "Atualmente ~30 diagnósticos, incluindo SCA com e sem supra, fibrilação atrial, flutter, taquicardias ventriculares, bloqueios de ramo e AV, síndromes de Brugada e Wellens, hipercalemia, pericardite, entre outros. Novos diagnósticos são adicionados continuamente.",
  },
  {
    q: "Os dados dos meus pacientes ficam armazenados?",
    a: "Não coletamos nenhum dado identificável de pacientes — sem nome, CPF ou informações pessoais. Apenas a foto do ECG é processada. As análises ficam disponíveis por 30 dias e são automaticamente excluídas após esse período.",
  },
  {
    q: "Preciso instalar algum aplicativo?",
    a: "Não. O ProECG funciona 100% no navegador do celular. Basta acessar o site, fazer login e começar a usar. Funciona em qualquer smartphone com câmera.",
  },
  {
    q: "Como funciona o pagamento?",
    a: "Aceitamos Pix e cartão de crédito, processados com segurança pelo Asaas. Você escolhe entre plano mensal, semestral ou anual. O acesso é liberado imediatamente após a confirmação do pagamento.",
  },
  {
    q: "Posso cancelar a qualquer momento?",
    a: "Sim. Não há multa nem burocracia. Cancele quando quiser diretamente pelo painel. Você mantém o acesso até o final do período já pago.",
  },
  {
    q: "O laudo tem validade legal?",
    a: "O ProECG gera um laudo de apoio, não um laudo médico formal assinado. Ele serve como auxílio para a interpretação clínica do médico assistente, que é o responsável pelo diagnóstico e pela conduta.",
  },
];

function FaqItem({ q, a }: { q: string; a: string }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border-b border-[#E2E4F0]">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between py-5 text-left"
      >
        <span className="pr-4 text-base font-semibold text-[#122056]">{q}</span>
        <ChevronDownIcon
          className={`size-5 shrink-0 text-[#4A5078] transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>
      <div
        className={`grid transition-all duration-200 ${
          open ? "grid-rows-[1fr] pb-5" : "grid-rows-[0fr]"
        }`}
      >
        <div className="overflow-hidden">
          <p className="text-[#4A5078] leading-relaxed">{a}</p>
        </div>
      </div>
    </div>
  );
}

export function FAQ() {
  return (
    <section id="faq" className="bg-[#FAFAFD] px-4 py-20 sm:py-24">
      <div className="mx-auto max-w-[800px]">
        <Reveal>
          <h2 className="mb-10 text-center text-3xl font-bold text-[#122056] sm:text-4xl">
            Perguntas frequentes
          </h2>
        </Reveal>
        <Reveal delay={0.1}>
          <div>
            {faqs.map((faq) => (
              <FaqItem key={faq.q} q={faq.q} a={faq.a} />
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
