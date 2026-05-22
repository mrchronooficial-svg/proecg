"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu } from "lucide-react";

import { MobileDrawer } from "./MobileDrawer";

interface MobileHeaderProps {
  userName: string;
  userEmail: string;
}

function initialsOf(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0]!.slice(0, 2).toUpperCase();
  return (parts[0]![0]! + parts[parts.length - 1]![0]!).toUpperCase();
}

export function MobileHeader({ userName, userEmail }: MobileHeaderProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <>
      <header
        role="banner"
        className="sticky top-0 z-30 grid h-14 grid-cols-3 items-center border-b border-apple-border-light bg-apple-surface/85 px-4 backdrop-blur-md md:hidden"
      >
        <div className="flex items-center">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Abrir menu"
            aria-expanded={drawerOpen}
            className="flex size-10 items-center justify-center rounded-full text-apple-text-secondary transition-colors hover:bg-[#F5F5F7] hover:text-apple-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
          >
            <Menu size={22} strokeWidth={1.8} />
          </button>
        </div>
        <Link
          href="/dashboard"
          className="justify-self-center text-xl font-bold tracking-[-0.02em] text-apple-accent"
        >
          ProECG
        </Link>
        <div className="flex items-center justify-self-end">
          <Link
            href="/dashboard/conta"
            aria-label="Minha conta"
            className="flex size-10 items-center justify-center rounded-full bg-apple-accent-light text-[13px] font-semibold text-apple-accent transition-colors hover:bg-apple-accent hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
          >
            {initialsOf(userName)}
          </Link>
        </div>
      </header>

      <MobileDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        userName={userName}
        userEmail={userEmail}
      />
    </>
  );
}
