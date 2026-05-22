import { HistoricoList } from "./historico-list";

export default function HistoricoPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="apple-animate-fade-in-up">
        <h1 className="text-[34px] md:text-5xl font-bold tracking-[-0.02em] leading-[1.1] text-apple-text">
          Histórico
        </h1>
        <p className="mt-2 text-[15px] leading-[1.4] text-apple-text-secondary">
          Seus eletrocardiogramas analisados
        </p>
      </header>

      <section aria-label="Lista de ECGs">
        <HistoricoList />
      </section>
    </div>
  );
}
