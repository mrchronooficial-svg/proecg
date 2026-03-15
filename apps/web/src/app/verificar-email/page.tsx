import { Card } from "@proecg/ui/components/card";

export default function VerificarEmailPage() {
  return (
    <main className="min-h-[calc(100svh-3.5rem)] flex items-center justify-center px-4">
      <Card className="glass shadow-lg max-w-md p-8 text-center">
        <h1 className="mb-4 text-2xl font-bold gradient-text">Verifique seu email</h1>
        <p className="mb-4 text-muted-foreground">
          Enviamos um link de verificação para o seu email. Clique no link para
          ativar sua conta.
        </p>
        <p className="text-sm text-muted-foreground">
          Não recebeu? Verifique a pasta de spam ou tente se cadastrar
          novamente.
        </p>
      </Card>
    </main>
  );
}
