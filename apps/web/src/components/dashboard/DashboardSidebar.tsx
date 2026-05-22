"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, History, Settings, LogOut } from "lucide-react";
import { cn } from "@proecg/ui/lib/utils";

import { authClient } from "@/lib/auth-client";

const main = [
  { href: "/dashboard", label: "Home", icon: Home },
  { href: "/dashboard/historico", label: "Histórico", icon: History },
] as const;

function isActive(pathname: string, href: string): boolean {
  if (href === "/dashboard") return pathname === "/dashboard";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function DashboardSidebar() {
  const pathname = usePathname();

  const handleSignOut = () => {
    authClient.signOut({
      fetchOptions: {
        onSuccess: () => {
          window.location.href = "/";
        },
      },
    });
  };

  return (
    <aside
      role="navigation"
      aria-label="Navegação do dashboard"
      className="fixed inset-y-0 left-0 z-30 hidden w-[260px] flex-col border-r border-apple-border-light bg-apple-surface md:flex"
    >
      <div className="px-6 pt-7 pb-4">
        <Link
          href="/dashboard"
          className="text-2xl font-bold tracking-[-0.02em] text-apple-accent"
        >
          ProECG
        </Link>
      </div>

      <nav className="flex-1 px-3">
        <ul className="flex flex-col gap-1">
          {main.map((item) => {
            const active = isActive(pathname, item.href);
            const Icon = item.icon;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] font-medium transition-all duration-200 ease-[var(--ease-apple)]",
                    active
                      ? "bg-apple-accent-light text-apple-accent"
                      : "text-apple-text-secondary hover:bg-[#F5F5F7] hover:text-apple-text",
                  )}
                >
                  {active && (
                    <span
                      className="absolute inset-y-2 left-0 w-[3px] rounded-full bg-apple-accent"
                      aria-hidden="true"
                    />
                  )}
                  <Icon
                    size={20}
                    strokeWidth={active ? 2.2 : 1.8}
                    aria-hidden="true"
                  />
                  <span>{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="border-t border-apple-border-light p-3">
        <Link
          href="/dashboard/conta"
          aria-current={
            isActive(pathname, "/dashboard/conta") ? "page" : undefined
          }
          className={cn(
            "flex items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] font-medium transition-all duration-200 ease-[var(--ease-apple)]",
            isActive(pathname, "/dashboard/conta")
              ? "bg-apple-accent-light text-apple-accent"
              : "text-apple-text-secondary hover:bg-[#F5F5F7] hover:text-apple-text",
          )}
        >
          <Settings size={20} strokeWidth={1.8} aria-hidden="true" />
          <span>Configurações</span>
        </Link>
        <button
          type="button"
          onClick={handleSignOut}
          className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] font-medium text-apple-text-secondary transition-all duration-200 ease-[var(--ease-apple)] hover:bg-[#F5F5F7] hover:text-apple-danger focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
        >
          <LogOut size={20} strokeWidth={1.8} aria-hidden="true" />
          <span>Sair</span>
        </button>
      </div>
    </aside>
  );
}
