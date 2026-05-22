"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, History } from "lucide-react";
import { cn } from "@proecg/ui/lib/utils";

const items = [
  { href: "/dashboard", label: "Home", icon: Home },
  { href: "/dashboard/historico", label: "Histórico", icon: History },
] as const;

function isActive(pathname: string, href: string): boolean {
  if (href === "/dashboard") return pathname === "/dashboard";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function BottomNav() {
  const pathname = usePathname();

  return (
    <nav
      role="navigation"
      aria-label="Navegação principal"
      className="fixed inset-x-0 bottom-0 z-40 border-t border-apple-border-light bg-apple-surface/95 backdrop-blur-md md:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <ul className="grid grid-cols-2">
        {items.map((item) => {
          const active = isActive(pathname, item.href);
          const Icon = item.icon;
          return (
            <li key={item.href}>
              <Link
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex min-h-[56px] flex-col items-center justify-center gap-0.5 py-2 text-[11px] font-medium transition-colors duration-200 ease-[var(--ease-apple)]",
                  active
                    ? "text-apple-accent"
                    : "text-apple-text-tertiary hover:text-apple-text-secondary",
                )}
              >
                <Icon
                  size={24}
                  strokeWidth={active ? 2.2 : 1.8}
                  className={cn(
                    "transition-transform duration-300",
                    active && "apple-animate-bounce",
                  )}
                  aria-hidden="true"
                />
                <span>{item.label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
