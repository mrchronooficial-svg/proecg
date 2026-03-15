export default function TermosPage() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-12">
      <h1 className="mb-6 text-3xl font-bold">Termos de Uso</h1>

      <div className="space-y-6 text-muted-foreground">
        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            1. Aceitação dos Termos
          </h2>
          <p>
            Ao utilizar o ProECG, você concorda com estes Termos de Uso. Se não
            concordar, não utilize o serviço.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            2. Descrição do Serviço
          </h2>
          <p>
            O ProECG é uma ferramenta de apoio à decisão clínica que utiliza
            inteligência artificial para analisar imagens de eletrocardiogramas
            (ECG) em papel. O sistema digitaliza o traçado, realiza medições
            automáticas e sugere hipóteses diagnósticas.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            3. Limitações e Isenção de Responsabilidade
          </h2>
          <p>
            O ProECG <strong>não substitui a avaliação médica</strong>. Os
            resultados fornecidos são sugestivos e devem ser sempre
            correlacionados com dados clínicos, exame físico e contexto do
            paciente. O médico é o único responsável pela decisão clínica.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            4. Uso Permitido
          </h2>
          <p>
            O serviço destina-se exclusivamente a profissionais de saúde
            habilitados. O usuário não deve enviar dados identificáveis de
            pacientes (nome, CPF, data de nascimento) através da plataforma.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            5. Dados e Privacidade
          </h2>
          <p>
            As imagens de ECG enviadas são armazenadas de forma anônima e
            excluídas automaticamente após 30 dias. Consulte nossa Política de
            Privacidade para mais detalhes.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            6. Assinatura e Pagamento
          </h2>
          <p>
            O acesso ao ProECG requer assinatura paga. Os pagamentos são
            processados pelo Asaas. O cancelamento pode ser feito a qualquer
            momento pela área do usuário.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            7. Alterações
          </h2>
          <p>
            Reservamo-nos o direito de alterar estes termos a qualquer momento.
            Alterações significativas serão comunicadas por email.
          </p>
        </section>
      </div>
    </main>
  );
}
