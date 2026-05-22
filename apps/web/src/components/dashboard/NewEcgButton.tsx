import Link from "next/link";
import type { Route } from "next";
import { Camera } from "lucide-react";

interface NewEcgButtonProps {
  href?: Route;
}

export function NewEcgButton({
  href = "/dashboard/novo" as Route,
}: NewEcgButtonProps) {
  return (
    <Link
      href={href}
      aria-label="Fotografar novo ECG"
      className="group relative flex min-h-[140px] w-full flex-col items-center justify-center gap-2 rounded-3xl bg-gradient-to-br from-apple-accent to-apple-accent-hover px-8 py-8 text-white shadow-apple-cta transition-all duration-300 ease-[var(--ease-apple)] hover:scale-[1.02] hover:shadow-apple-cta-hover active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent focus-visible:ring-offset-2 apple-animate-fade-in-scale"
    >
      <Camera
        size={36}
        strokeWidth={1.5}
        className="transition-transform duration-300 group-hover:scale-110"
        aria-hidden="true"
      />
      <span className="text-[22px] font-semibold leading-[1.2] tracking-[-0.01em]">
        Novo ECG
      </span>
      <span className="text-sm font-normal opacity-80">
        Fotografe o eletrocardiograma
      </span>
    </Link>
  );
}
