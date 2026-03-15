"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { MenuIcon, XIcon } from "lucide-react";
import { cn } from "@proecg/ui/lib/utils";

const links = [
  { href: "/dashboard/novo" as const, label: "Novo ECG" },
  { href: "/dashboard/historico" as const, label: "Histórico" },
  { href: "/dashboard/conta" as const, label: "Minha Conta" },
];

const SidebarContext = createContext({
  open: false,
  toggle: () => {},
  close: () => {},
});

export function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const toggle = useCallback(() => setOpen((v) => !v), []);
  const close = useCallback(() => setOpen(false), []);

  return (
    <SidebarContext.Provider value={{ open, toggle, close }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function useSidebar() {
  return useContext(SidebarContext);
}

export function SidebarTrigger() {
  const { open, toggle } = useSidebar();

  return (
    <button
      onClick={toggle}
      className="flex items-center justify-center size-9 rounded-lg text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
      aria-label={open ? "Fechar menu" : "Abrir menu"}
    >
      {open ? <XIcon className="size-5" /> : <MenuIcon className="size-5" />}
    </button>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const { open, close } = useSidebar();

  // Close sidebar on route change
  useEffect(() => {
    close();
  }, [pathname, close]);

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm"
          onClick={close}
        />
      )}

      {/* Sidebar drawer */}
      <nav
        className={cn(
          "fixed top-14 left-0 z-40 h-[calc(100svh-3.5rem)] w-64 glass-strong border-r border-border/50 transition-transform duration-200 ease-in-out",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex flex-col gap-1 p-4">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={close}
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
    </>
  );
}
