import { Card } from "@proecg/ui/components/card";

const steps = [
  {
    step: "1",
    title: "Tire uma foto",
    description:
      "Use a câmera do celular para fotografar o ECG em papel. Funciona com qualquer aparelho de ECG.",
  },
  {
    step: "2",
    title: "IA analisa",
    description:
      "Nosso motor digitaliza o traçado, mede intervalos e classifica padrões em segundos.",
  },
  {
    step: "3",
    title: "Receba o laudo",
    description:
      "Veja medições (FC, PR, QRS, QT, eixo), achados e hipóteses diagnósticas na tela.",
  },
];

export function HowItWorks() {
  return (
    <section id="como-funciona" className="px-4 py-16 sm:py-24">
      <div className="mx-auto max-w-5xl">
        <h2 className="mb-12 text-center text-3xl font-bold sm:text-4xl">
          Como funciona
        </h2>
        <div className="grid gap-6 sm:grid-cols-3">
          {steps.map((s) => (
            <Card key={s.step} className="glass flex flex-col gap-3 p-6 hover:shadow-md transition-shadow">
              <div className="flex h-10 w-10 items-center justify-center rounded-full gradient-brand text-lg font-bold text-white">
                {s.step}
              </div>
              <h3 className="text-xl font-semibold">{s.title}</h3>
              <p className="text-muted-foreground">{s.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
