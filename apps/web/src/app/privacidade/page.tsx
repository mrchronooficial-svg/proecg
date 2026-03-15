export default function PrivacidadePage() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-12">
      <h1 className="mb-6 text-3xl font-bold">Política de Privacidade</h1>

      <div className="space-y-6 text-muted-foreground">
        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            1. Dados Coletados
          </h2>
          <p>
            Coletamos apenas os dados necessários para o funcionamento do
            serviço: nome, email e informações de pagamento. As imagens de ECG
            enviadas são tratadas de forma anônima.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            2. Dados de Pacientes
          </h2>
          <p>
            O ProECG <strong>não armazena dados identificáveis de pacientes</strong>.
            Não envie nome, CPF, data de nascimento ou qualquer informação que
            identifique o paciente.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            3. Armazenamento
          </h2>
          <p>
            Os dados são armazenados em servidores seguros (Neon para banco de
            dados, Cloudflare R2 para imagens). As imagens de ECG são
            automaticamente excluídas após 30 dias.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            4. LGPD
          </h2>
          <p>
            Em conformidade com a Lei Geral de Proteção de Dados (LGPD), você
            pode solicitar acesso, correção ou exclusão dos seus dados pessoais
            a qualquer momento.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            5. Compartilhamento
          </h2>
          <p>
            Não compartilhamos seus dados com terceiros, exceto com
            processadores de pagamento (Asaas) para fins de cobrança.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-xl font-semibold text-foreground">
            6. Contato
          </h2>
          <p>
            Para questões sobre privacidade, entre em contato pelo email
            indicado na plataforma.
          </p>
        </section>
      </div>
    </main>
  );
}
