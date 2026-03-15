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
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-sm border-b border-[#E2E4F0]">
      <div className="mx-auto flex h-16 max-w-[1200px] items-center justify-between px-5 sm:h-16 sm:px-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5">
          <Image src="/logo.png" alt="ProECG" width={40} height={40} className="rounded-xl" />
          <span className="text-lg font-extrabold text-[#122056] sm:text-xl">ProECG</span>
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
            className="rounded-full bg-[#5B65DC] px-5 py-2.5 text-sm font-semibold text-white shadow-[0_4px_14px_rgba(91,101,220,0.35)] transition-all hover:bg-[#4A51C5] hover:-translate-y-0.5 hover:shadow-[0_6px_20px_rgba(91,101,220,0.4)]"
          >
            Assinar Agora
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center justify-center size-10 rounded-full border border-[#E2E4F0] md:hidden text-[#122056]"
          aria-label="Menu"
        >
          {open ? <XIcon className="size-5" /> : <MenuIcon className="size-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="absolute inset-x-0 top-full z-50 border-b border-[#E2E4F0] bg-white px-5 py-4 shadow-lg md:hidden">
          <nav className="flex flex-col gap-1">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                className="rounded-xl px-4 py-3 text-base font-medium text-[#122056] hover:bg-[#EEEFFD]"
              >
                {link.label}
              </a>
            ))}
            <hr className="my-2 border-[#E2E4F0]" />
            <Link
              href="/login"
              onClick={() => setOpen(false)}
              className="rounded-xl px-4 py-3 text-base font-medium text-[#4A5078]"
            >
              Entrar
            </Link>
            <Link
              href="/login"
              onClick={() => setOpen(false)}
              className="mt-1 rounded-full bg-[#5B65DC] py-3.5 text-center text-base font-semibold text-white"
            >
              Assinar Agora
            </Link>
          </nav>
        </div>
      )}
    </header>
  );
}
