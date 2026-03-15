import Link from "next/link";

export function FooterMarketing() {
  return (
    <footer className="bg-[#122056] px-4 py-12">
      <div className="mx-auto max-w-[1200px]">
        <div className="border-t border-white/10 pt-10">
          <div className="grid gap-8 sm:grid-cols-3">
            <div>
              <h4 className="mb-3 text-sm font-semibold text-white/60">
                ProECG
              </h4>
              <ul className="space-y-2">
                <li>
                  <a href="#como-funciona" className="text-sm text-white/50 hover:text-white transition-colors">
                    Como Funciona
                  </a>
                </li>
                <li>
                  <a href="#beneficios" className="text-sm text-white/50 hover:text-white transition-colors">
                    Benefícios
                  </a>
                </li>
                <li>
                  <a href="#planos" className="text-sm text-white/50 hover:text-white transition-colors">
                    Planos
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="mb-3 text-sm font-semibold text-white/60">
                Legal
              </h4>
              <ul className="space-y-2">
                <li>
                  <Link href="/termos" className="text-sm text-white/50 hover:text-white transition-colors">
                    Termos de Uso
                  </Link>
                </li>
                <li>
                  <Link href="/privacidade" className="text-sm text-white/50 hover:text-white transition-colors">
                    Política de Privacidade
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="mb-3 text-sm font-semibold text-white/60">
                Suporte
              </h4>
              <ul className="space-y-2">
                <li>
                  <a href="mailto:suporte@proecg.com.br" className="text-sm text-white/50 hover:text-white transition-colors">
                    suporte@proecg.com.br
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-10 border-t border-white/10 pt-6 text-center text-xs text-white/40">
          <p>© {new Date().getFullYear()} ProECG. Todos os direitos reservados.</p>
          <p className="mt-1">
            Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
          </p>
        </div>
      </div>
    </footer>
  );
}
