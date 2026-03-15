"use client";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { ModeToggle } from "./mode-toggle";
import UserMenu from "./user-menu";
import { SidebarTrigger } from "./dashboard/sidebar";

export default function Header() {
  const pathname = usePathname();
  const isDashboard = pathname.startsWith("/dashboard");

  // LP has its own navbar
  if (pathname === "/") return null;

  return (
    <header className="sticky top-0 z-50 glass-strong h-14 grid grid-cols-3 items-center px-4">
      <div className="flex items-center">
        {isDashboard && <SidebarTrigger />}
      </div>
      <Link href={isDashboard ? "/dashboard" : "/"} className="flex items-center gap-2 font-bold text-xl text-foreground justify-self-center">
        <Image src="/logo.png" alt="ProECG" width={50} height={50} className="rounded-lg" />
        ProECG
      </Link>
      <div className="flex items-center gap-2 justify-self-end">
        <ModeToggle />
        <UserMenu />
      </div>
    </header>
  );
}
