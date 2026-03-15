import Link from "next/link";
import { buttonVariants } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { cn } from "@proecg/ui/lib/utils";
import { PLANS } from "@proecg/api/lib/plans";

export function PricingTable() {
  return (
    <section id="planos" className="px-4 py-16 sm:py-24">
      <div className="mx-auto max-w-5xl">
        <h2 className="mb-4 text-center text-3xl font-bold sm:text-4xl">
          Planos
        </h2>
        <p className="mb-12 text-center text-muted-foreground">
          ECGs ilimitados. Cancele quando quiser.
        </p>
        <div className="grid gap-6 sm:grid-cols-3">
          {PLANS.map((plan) => (
            <Card
              key={plan.id}
              className={cn(
                "relative flex flex-col gap-4 p-6",
                plan.highlight
                  ? "gradient-border shadow-lg"
                  : "glass",
              )}
            >
              {plan.highlight && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full gradient-brand px-3 py-1 text-xs font-semibold text-white">
                  {plan.badge}
                </span>
              )}
              <h3 className="text-xl font-semibold">{plan.name}</h3>
              <div className="flex items-baseline gap-1">
                <span className="text-4xl font-bold">{plan.priceLabel}</span>
                <span className="text-muted-foreground">{plan.period}</span>
              </div>
              <p className="text-sm text-muted-foreground">
                {plan.durationMonths === 1
                  ? "Cobrança mensal"
                  : `${plan.durationMonths} meses de acesso`}
              </p>
              <Link
                href="/login"
                className={cn(
                  buttonVariants({
                    variant: plan.highlight ? "default" : "outline",
                  }),
                  "mt-auto w-full",
                )}
              >
                Assinar
              </Link>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
