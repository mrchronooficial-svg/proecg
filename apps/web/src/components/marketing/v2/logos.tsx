"use client";

const partners = ["SBC", "InCor", "UFMG", "USP", "Einstein", "Sírio-Libanês", "FMUSP", "HCFMUSP"];

export function Logos() {
  // Duplicamos a lista para criar carrossel infinito sem corte.
  const loop = [...partners, ...partners];

  return (
    <section className="bg-white py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-6">
        <p className="text-center text-[11px] font-semibold uppercase tracking-[0.18em] text-[#86868b] sm:text-[13px]">
          Tecnologia baseada em evidências científicas
        </p>

        <div className="relative mt-10 overflow-hidden sm:mt-14">
          <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-12 bg-gradient-to-r from-white to-transparent sm:w-24" />
          <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-12 bg-gradient-to-l from-white to-transparent sm:w-24" />

          <div className="flex w-max animate-[scroll_40s_linear_infinite] items-center gap-10 sm:gap-16">
            {loop.map((p, i) => (
              <div
                key={`${p}-${i}`}
                className="text-[18px] font-bold tracking-tight text-[#1d1d1f]/30 sm:text-[26px]"
              >
                {p}
              </div>
            ))}
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes scroll {
          from { transform: translateX(0); }
          to   { transform: translateX(-50%); }
        }
      `}</style>
    </section>
  );
}
