import Link from "next/link";

export function FooterMarketing() {
  return (
    <footer className="px-4 py-8">
      <div className="h-px gradient-brand opacity-30 mb-8" />
      <div className="mx-auto flex max-w-5xl flex-col items-center gap-4 text-center text-sm text-muted-foreground">
        <p className="max-w-xl font-medium">
          Ferramenta de apoio à decisão clínica — não substitui avaliação
          médica. Correlacionar sempre com dados clínicos.
        </p>
        <div className="flex gap-4">
          <Link href="/termos" className="hover:underline">
            Termos de Uso
          </Link>
          <Link href="/privacidade" className="hover:underline">
            Política de Privacidade
          </Link>
        </div>
        <p>© {new Date().getFullYear()} ProECG. Todos os direitos reservados.</p>
      </div>
    </footer>
  );
}
