"use client";

const cols = [
  {
    title: "Produto",
    links: [
      { label: "Como funciona", href: "#how" },
      { label: "Diagnósticos", href: "#diagnostics" },
      { label: "Preços", href: "#pricing" },
    ],
  },
  {
    title: "Recursos",
    links: [
      { label: "Documentação", href: "#" },
      { label: "Estudos clínicos", href: "#" },
      { label: "Status", href: "#" },
    ],
  },
  {
    title: "Empresa",
    links: [
      { label: "Sobre", href: "#" },
      { label: "Contato", href: "mailto:contato@proecg.com.br" },
      { label: "Carreiras", href: "#" },
    ],
  },
  {
    title: "Legal",
    links: [
      { label: "Termos de uso", href: "/termos" },
      { label: "Política de privacidade", href: "/privacidade" },
    ],
  },
];

export function FooterMarketing() {
  return (
    <footer id="footer" className="bg-black text-white">
      <div className="mx-auto max-w-7xl px-6 py-14 sm:py-16 lg:py-20">
        <div className="grid grid-cols-1 gap-10 text-center md:grid-cols-5 md:gap-12 md:text-left">
          <div className="md:col-span-1">
            <div className="text-[22px] font-bold tracking-[-0.01em] md:text-[20px]">ProECG</div>
            <p className="mx-auto mt-3 max-w-xs text-[14px] leading-[1.5] text-white/55 md:mx-0 md:mt-3 md:text-[13px]">
              IA em cardiologia para apoio à decisão clínica.
            </p>
          </div>

          {cols.map((c) => (
            <div key={c.title}>
              <div className="text-[14px] font-bold tracking-[-0.01em] text-white md:text-[13px] md:font-semibold">{c.title}</div>
              <ul className="mt-4 space-y-3 md:mt-4 md:space-y-3">
                {c.links.map((l) => (
                  <li key={l.label}>
                    <a
                      href={l.href}
                      className="text-[14px] text-white/60 transition-colors hover:text-white md:text-[13px]"
                    >
                      {l.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col gap-2 border-t border-white/10 pt-7 text-center text-[12px] text-white/45 sm:mt-16 sm:pt-8 sm:text-[13px] md:flex-row md:items-center md:justify-between md:gap-3 md:text-left md:text-[12px]">
          <p>© 2026 ProECG. Ferramenta de apoio à decisão clínica.</p>
          <p>Não substitui avaliação médica.</p>
        </div>
      </div>
    </footer>
  );
}
