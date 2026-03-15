import Link from "next/link";
import { buttonVariants } from "@proecg/ui/components/button";
import { cn } from "@proecg/ui/lib/utils";

export function Hero() {
  return (
    <section className="relative flex flex-col items-center gap-6 px-4 py-24 text-center sm:py-32 lg:py-40 overflow-hidden">
      {/* Decorative gradient blobs */}
      <div className="absolute -top-40 -left-40 h-80 w-80 gradient-brand rounded-full blur-3xl opacity-20" />
      <div className="absolute -bottom-40 -right-40 h-80 w-80 gradient-brand rounded-full blur-3xl opacity-20" />

      <h1 className="relative max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
        Laudo de ECG em segundos com{" "}
        <span className="gradient-text">Inteligência Artificial</span>
      </h1>
      <p className="relative max-w-2xl text-lg text-muted-foreground sm:text-xl">
        Tire uma foto do ECG em papel e receba medições, achados e hipóteses
        diagnósticas automaticamente. Feito para médicos de emergência, UTI e
        UBS.
      </p>
      <div className="relative flex flex-col gap-3 sm:flex-row">
        <Link
          href="/login"
          className={cn(buttonVariants({ size: "lg" }), "text-base")}
        >
          Começar agora
        </Link>
        <Link
          href="#como-funciona"
          className={cn(
            buttonVariants({ variant: "outline", size: "lg" }),
            "text-base",
          )}
        >
          Como funciona
        </Link>
      </div>
    </section>
  );
}
