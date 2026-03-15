"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@proecg/ui/lib/utils";

const links = [
  { href: "/dashboard/novo" as const, label: "Novo ECG" },
  { href: "/dashboard/historico" as const, label: "Histórico" },
  { href: "/dashboard/conta" as const, label: "Minha Conta" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <>
      {/* Desktop sidebar */}
      <nav className="hidden w-56 shrink-0 glass border-r border-border/50 md:block">
        <div className="flex flex-col gap-1 p-4">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "rounded-lg px-3 py-2 text-sm font-medium transition-all",
                pathname === link.href
                  ? "gradient-brand text-white shadow-sm"
                  : "hover:bg-accent/50 text-muted-foreground hover:text-foreground",
              )}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </nav>

      {/* Mobile bottom nav */}
      <nav className="fixed inset-x-0 bottom-0 z-50 flex glass-strong md:hidden">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              "flex flex-1 items-center justify-center py-3 text-xs font-medium transition-colors",
              pathname === link.href
                ? "text-primary"
                : "text-muted-foreground",
            )}
          >
            {link.label}
          </Link>
        ))}
      </nav>
    </>
  );
}
