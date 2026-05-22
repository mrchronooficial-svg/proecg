"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ECGTrace, IPhoneMockup } from "./iphone-mockup";

export function Hero() {
  return (
    <section
      id="hero"
      className="relative flex min-h-screen items-center overflow-hidden bg-[#0a0a0a] text-white"
      style={{
        backgroundImage:
          "radial-gradient(60% 80% at 70% 20%, rgba(26,26,62,0.9), transparent 60%), radial-gradient(50% 60% at 20% 80%, rgba(0,102,204,0.25), transparent 60%), linear-gradient(180deg, #050510 0%, #0d0d20 60%, #1a1a3e 100%)",
      }}
    >
      <div className="mx-auto grid w-full max-w-7xl grid-cols-1 items-center gap-12 px-6 pt-24 pb-16 sm:gap-16 sm:px-6 sm:pt-32 sm:pb-24 lg:grid-cols-2 lg:gap-12">
        <div className="text-center lg:text-left">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1 text-[12px] font-medium tracking-wide text-white/85 backdrop-blur sm:mb-6 sm:px-3.5 sm:py-1.5 sm:text-[13px]"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-[#0a84ff]" />
            ProECG
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.05 }}
            className="text-[34px] font-bold leading-[1.05] tracking-[-0.02em] break-words sm:text-[56px] lg:text-[68px]"
          >
            Inteligência Artificial
            <br />
            <span className="bg-gradient-to-br from-white via-white to-[#9fc5ff] bg-clip-text text-transparent">
              em Cardiologia
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 }}
            className="mx-auto mt-5 max-w-xl text-[16px] leading-[1.5] text-white/70 sm:mt-6 sm:text-[19px] lg:mx-0"
          >
            Laudo de ECG em segundos. Usado por médicos emergencistas, intensivistas e generalistas em todo o Brasil.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.25 }}
            className="mt-7 flex flex-row items-center justify-center gap-3 sm:mt-10 sm:flex-wrap sm:gap-4 lg:justify-start"
          >
            <Link
              href="/login"
              className="rounded-full bg-[#0a84ff] px-5 py-2.5 text-center text-[15px] font-medium text-white transition-all hover:bg-[#0066cc] sm:px-6 sm:text-[17px]"
            >
              Começar Agora
            </Link>
            <a
              href="#how"
              className="rounded-full border border-[#0a84ff] px-5 py-2.5 text-center text-[15px] font-medium text-[#0a84ff] transition-colors hover:bg-[#0a84ff]/10 sm:px-6 sm:text-[17px]"
            >
              Saiba mais
            </a>
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, delay: 0.5 }}
            className="mx-auto mt-10 max-w-md text-[12px] leading-relaxed text-white/55 sm:mt-14 sm:text-[14px] lg:mx-0"
          >
            Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
          </motion.p>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.92, y: 30 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.2, ease: [0.21, 0.47, 0.32, 0.98] }}
          className="relative flex items-center justify-center pt-4 sm:pt-0"
        >
          {/* glow — blur menor em mobile (raio grande de blur custa muito GPU) */}
          <div className="pointer-events-none absolute left-1/2 top-1/2 -z-10 h-[220px] w-[220px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#0a84ff]/20 blur-[55px] sm:h-[420px] sm:w-[420px] sm:blur-[100px]" />
          <IPhoneMockup>
            <PhoneScreenContent />
          </IPhoneMockup>
        </motion.div>
      </div>
    </section>
  );
}

function PhoneScreenContent() {
  return (
    <div className="flex h-full flex-col bg-[#fafafa] text-[10px] text-[#1d1d1f]">
      <div className="flex items-center justify-between bg-white px-4 pt-8 pb-2 text-[9px] font-medium">
        <span>9:41</span>
        <span className="opacity-60">5G</span>
      </div>
      <div className="flex items-center gap-2 border-b border-black/5 bg-white px-4 py-3">
        <div className="h-7 w-7 rounded-full bg-[#0066CC]" />
        <div>
          <div className="text-[10px] font-semibold">ProECG</div>
          <div className="text-[8px] text-[#1d1d1f]/50">Análise concluída</div>
        </div>
      </div>
      <div className="space-y-2 p-3">
        <div className="rounded-xl bg-white p-2 shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
          <div className="mb-1 text-[8px] font-semibold uppercase tracking-wide text-[#1d1d1f]/50">ECG 12 derivações</div>
          <ECGTrace className="h-12 w-full" />
        </div>
        <div className="rounded-xl bg-white p-3 shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
          <div className="mb-2 text-[9px] font-semibold">Medições</div>
          <div className="grid grid-cols-3 gap-1.5 text-[8px]">
            {[
              ["FC", "78 bpm"],
              ["Eixo", "+60°"],
              ["PR", "180 ms"],
              ["QRS", "88 ms"],
              ["QT", "380"],
              ["QTc", "420"],
            ].map(([k, v]) => (
              <div key={k} className="rounded bg-[#f5f5f7] p-1.5">
                <div className="text-[7px] text-[#1d1d1f]/50">{k}</div>
                <div className="text-[8.5px] font-semibold">{v}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-xl bg-white p-3 shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
          <div className="mb-1 flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <div className="text-[9px] font-semibold">Conclusão</div>
          </div>
          <div className="text-[8.5px] leading-relaxed text-[#1d1d1f]/75">
            Ritmo sinusal. Sem alterações isquêmicas agudas. Eixo normal.
          </div>
        </div>
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-2.5">
          <div className="text-[8px] font-semibold text-amber-900">⚠ Apoio à decisão</div>
          <div className="mt-0.5 text-[7.5px] text-amber-800">Correlacionar com clínica.</div>
        </div>
      </div>
    </div>
  );
}
