import { cn } from "@proecg/ui/lib/utils";

interface StatCardProps {
  value: string | number;
  label: string;
  className?: string;
  /** Animação de delay (ms) — escalona entrada de múltiplos cards */
  delayMs?: number;
}

export function StatCard({
  value,
  label,
  className,
  delayMs = 0,
}: StatCardProps) {
  return (
    <div
      className={cn(
        "rounded-2xl bg-apple-surface px-5 py-5 shadow-apple-sm transition-shadow duration-300 ease-[var(--ease-apple)] hover:shadow-apple-md",
        "apple-animate-fade-in-up",
        className,
      )}
      style={delayMs > 0 ? { animationDelay: `${delayMs}ms` } : undefined}
    >
      <div className="text-[34px] md:text-5xl font-bold tracking-[-0.02em] leading-[1.05] text-apple-text apple-animate-counter">
        {value}
      </div>
      <div className="mt-1 text-[13px] leading-[1.3] text-apple-text-secondary">
        {label}
      </div>
    </div>
  );
}
