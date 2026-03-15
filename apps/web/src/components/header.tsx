"use client";
import Link from "next/link";

import { ModeToggle } from "./mode-toggle";
import UserMenu from "./user-menu";

export default function Header() {
  const links = [
    { to: "/", label: "Home" },
    { to: "/dashboard", label: "Dashboard" },
  ] as const;

  return (
    <header className="sticky top-0 z-50 glass-strong h-14 flex items-center justify-between px-4">
      <div className="flex items-center gap-6">
        <Link href="/" className="gradient-text font-bold text-xl">
          ProECG
        </Link>
        <nav className="flex gap-4">
          {links.map(({ to, label }) => (
            <Link
              key={to}
              href={to}
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              {label}
            </Link>
          ))}
        </nav>
      </div>
      <div className="flex items-center gap-2">
        <ModeToggle />
        <UserMenu />
      </div>
    </header>
  );
}
