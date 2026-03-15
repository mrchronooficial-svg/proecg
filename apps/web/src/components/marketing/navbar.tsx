"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { MenuIcon, XIcon } from "lucide-react";

const navLinks = [
  { href: "#como-funciona", label: "Como Funciona" },
  { href: "#beneficios", label: "Benefícios" },
  { href: "#planos", label: "Planos" },
  { href: "#faq", label: "FAQ" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-sm border-b border-[#E2E4F0] h-16">
      <div className="mx-auto flex h-full max-w-[1200px] items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 text-xl font-bold text-[#122056]">
          <Image src="/logo.png" alt="ProECG" width={50} height={50} className="rounded-lg" />
          ProECG
        </Link>

        {/* Desktop links */}
        <nav className="hidden items-center gap-6 md:flex">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-sm font-medium text-[#4A5078] hover:text-[#122056] transition-colors"
            >
              {link.label}
            </a>
          ))}
        </nav>

        {/* Desktop CTAs */}
        <div className="hidden items-center gap-3 md:flex">
          <Link
            href="/login"
            className="text-sm font-medium text-[#4A5078] hover:text-[#122056] transition-colors"
          >
            Entrar
          </Link>
          <Link
            href="/login"
            className="rounded-xl bg-[#5B65DC] px-5 py-2.5 text-sm font-semibold text-white shadow-[0_4px_14px_rgba(91,101,220,0.35)] transition-all hover:bg-[#4A51C5] hover:-translate-y-0.5 hover:shadow-[0_6px_20px_rgba(91,101,220,0.4)]"
          >
            Assinar Agora
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center justify-center size-10 md:hidden text-[#122056]"
          aria-label="Menu"
        >
          {open ? <XIcon className="size-6" /> : <MenuIcon className="size-6" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="absolute inset-x-0 top-16 z-50 border-b border-[#E2E4F0] bg-white p-4 shadow-lg md:hidden">
          <nav className="flex flex-col gap-3">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2 text-sm font-medium text-[#4A5078] hover:bg-[#EEEFFD]"
              >
                {link.label}
              </a>
            ))}
            <hr className="border-[#E2E4F0]" />
            <Link
              href="/login"
              onClick={() => setOpen(false)}
              className="px-3 py-2 text-sm font-medium text-[#4A5078]"
            >
              Entrar
            </Link>
            <Link
              href="/login"
              onClick={() => setOpen(false)}
              className="rounded-xl bg-[#5B65DC] px-5 py-3 text-center text-sm font-semibold text-white"
            >
              Assinar Agora
            </Link>
          </nav>
        </div>
      )}
    </header>
  );
}
