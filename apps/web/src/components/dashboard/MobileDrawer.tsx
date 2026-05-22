"use client";

import Link from "next/link";
import { useEffect } from "react";
import { X, Settings, LogOut } from "lucide-react";
import { cn } from "@proecg/ui/lib/utils";

import { authClient } from "@/lib/auth-client";

interface MobileDrawerProps {
  open: boolean;
  onClose: () => void;
  userName: string;
  userEmail: string;
}

export function MobileDrawer({
  open,
  onClose,
  userName,
  userEmail,
}: MobileDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handler);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

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
    <>
      <div
        aria-hidden="true"
        onClick={onClose}
        className={cn(
          "fixed inset-0 z-50 bg-black/40 backdrop-blur-sm transition-opacity duration-300 ease-[var(--ease-apple)] md:hidden",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="Menu da conta"
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-[85vw] max-w-[340px] flex-col bg-apple-surface shadow-apple-lg transition-transform duration-300 ease-[var(--ease-apple)] md:hidden",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between px-5 pt-5 pb-3">
          <span className="text-xl font-bold tracking-[-0.02em] text-apple-accent">
            ProECG
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar menu"
            className="flex size-10 items-center justify-center rounded-full text-apple-text-secondary transition-colors hover:bg-[#F5F5F7] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
          >
            <X size={22} strokeWidth={1.8} />
          </button>
        </div>

        <div className="border-t border-apple-border-light px-5 py-5">
          <div className="text-[17px] font-semibold leading-[1.3] text-apple-text">
            {userName}
          </div>
          <div className="mt-0.5 truncate text-[13px] text-apple-text-secondary">
            {userEmail}
          </div>
        </div>

        <nav className="flex-1 border-t border-apple-border-light px-3 pt-3">
          <Link
            href="/dashboard/conta"
            onClick={onClose}
            className="flex items-center gap-3 rounded-xl px-3 py-3 text-[15px] font-medium text-apple-text transition-colors hover:bg-[#F5F5F7]"
          >
            <Settings size={20} strokeWidth={1.8} aria-hidden="true" />
            <span>Configurações</span>
          </Link>
        </nav>

        <div className="border-t border-apple-border-light p-3">
          <button
            type="button"
            onClick={handleSignOut}
            className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-[15px] font-medium text-apple-danger transition-colors hover:bg-[#F5F5F7] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
          >
            <LogOut size={20} strokeWidth={1.8} aria-hidden="true" />
            <span>Sair</span>
          </button>
        </div>
      </aside>
    </>
  );
}
