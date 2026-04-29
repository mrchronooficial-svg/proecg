"use client";

const partners = ["SBC", "InCor", "UFMG", "USP", "Einstein", "Sírio-Libanês", "FMUSP", "HCFMUSP"];

export function Logos() {
  // Duplicamos a lista para criar carrossel infinito sem corte.
  const loop = [...partners, ...partners];

  return (
    <section className="bg-white py-14 sm:py-20">
      <div className="mx-auto max-w-7xl px-5 sm:px-6">
        <p className="text-center text-[12px] font-medium uppercase tracking-[0.18em] text-[#1d1d1f]/55 sm:text-[13px]">
          Tecnologia baseada em evidências científicas
        </p>

        <div className="relative mt-8 overflow-hidden sm:mt-10">
          <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-12 bg-gradient-to-r from-white to-transparent sm:w-24" />
          <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-12 bg-gradient-to-l from-white to-transparent sm:w-24" />

          <div className="flex w-max animate-[scroll_40s_linear_infinite] gap-6 sm:gap-12">
            {loop.map((p, i) => (
              <div
                key={`${p}-${i}`}
                className="flex h-9 min-w-[120px] items-center justify-center rounded-lg bg-[#f5f5f7] px-5 text-[14px] font-bold tracking-tight text-[#1d1d1f]/40 grayscale sm:h-14 sm:min-w-[180px] sm:px-8 sm:text-[18px]"
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
