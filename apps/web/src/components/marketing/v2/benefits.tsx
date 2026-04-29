"use client";

import { Activity, Award, Camera, AlertTriangle, Stethoscope, BarChart3 } from "lucide-react";
import { Reveal } from "./reveal";

const items = [
  {
    icon: Activity,
    title: "Detecta IAMCSST e equivalentes",
    body: "Sensibilidade alta para padrões sutis que passam despercebidos: Wellens, De Winter, Sgarbossa, supra em aVR, infarto posterior.",
  },
  {
    icon: Award,
    title: "Laudo padrão SBC/AHA",
    body: "Estrutura profissional com ritmo, eixo, intervalos, morfologia e conclusão — pronta pra colar no prontuário.",
  },
  {
    icon: Camera,
    title: "Funciona com qualquer ECG",
    body: "Papel térmico desbotado, foto torta, iluminação ruim, qualquer fabricante. Pipeline próprio robusto a fotos imperfeitas.",
  },
  {
    icon: AlertTriangle,
    title: "Alertas de emergência (Red Flags)",
    body: "STEMI, TV, BAV total e hipercalemia grave sinalizados instantaneamente — para você não perder o caso crítico.",
  },
  {
    icon: Stethoscope,
    title: "Segunda opinião instantânea",
    body: "Confiança na decisão clínica em UBS, SAMU e PS — onde nem sempre há cardiologista de plantão.",
  },
  {
    icon: BarChart3,
    title: "12 medições automáticas",
    body: "Frequência, eixo, P, PR, QRS, QT, QTc e segmento ST por derivação. Sem cálculo manual com régua.",
  },
];

export function Benefits() {
  return (
    <section className="bg-[#f5f5f7] py-20 sm:py-32 lg:py-40">
      <div className="mx-auto max-w-7xl px-5 sm:px-6">
        <Reveal>
          <h2 className="max-w-3xl text-[32px] font-semibold leading-[1.05] tracking-tight text-[#1d1d1f] sm:text-[48px] lg:text-[56px]">
            Por que médicos escolhem o ProECG.
          </h2>
        </Reveal>

        <div className="mt-12 grid grid-cols-1 gap-4 sm:mt-16 sm:gap-5 md:grid-cols-2 lg:mt-20">
          {items.map((it, i) => (
            <Reveal key={it.title} delay={i * 0.05}>
              <div className="group h-full rounded-2xl bg-white p-6 shadow-[0_1px_3px_rgba(0,0,0,0.04)] transition-all hover:shadow-[0_20px_40px_-15px_rgba(0,0,0,0.12)] sm:rounded-3xl sm:p-8 lg:p-10">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-[#0066CC]/10 text-[#0066CC] transition-colors group-hover:bg-[#0066CC] group-hover:text-white sm:h-12 sm:w-12">
                  <it.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-5 text-[20px] font-semibold leading-[1.2] tracking-tight text-[#1d1d1f] sm:mt-6 sm:text-[24px] lg:text-[26px]">
                  {it.title}
                </h3>
                <p className="mt-2.5 text-[15px] leading-[1.55] text-[#1d1d1f]/65 sm:mt-3 sm:text-[16px] lg:text-[17px]">
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
