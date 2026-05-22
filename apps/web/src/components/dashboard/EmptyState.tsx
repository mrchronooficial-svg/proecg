import Link from "next/link";
import type { Route } from "next";
import type { LucideIcon } from "lucide-react";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  ctaLabel?: string;
  ctaHref?: Route;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  ctaLabel,
  ctaHref,
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center apple-animate-fade-in-up">
      <Icon
        className="text-apple-text-tertiary opacity-50"
        size={64}
        strokeWidth={1.5}
        aria-hidden="true"
      />
      <h3 className="mt-4 text-[22px] md:text-2xl font-semibold tracking-[-0.01em] leading-[1.2] text-apple-text">
        {title}
      </h3>
      <p className="mt-2 max-w-xs text-[15px] leading-[1.4] text-apple-text-secondary">
        {description}
      </p>
      {ctaLabel && ctaHref && (
        <Link
          href={ctaHref}
          className="mt-6 inline-flex min-h-12 items-center justify-center rounded-[14px] bg-apple-accent px-7 py-3.5 text-[15px] font-semibold text-white transition-all duration-200 ease-[var(--ease-apple)] hover:bg-apple-accent-hover hover:shadow-apple-glow active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent focus-visible:ring-offset-2"
        >
          {ctaLabel}
          <span className="ml-1.5" aria-hidden="true">
            →
          </span>
        </Link>
      )}
    </div>
  );
}
