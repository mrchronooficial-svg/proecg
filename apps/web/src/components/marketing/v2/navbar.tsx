"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Menu, X } from "lucide-react";

const links = [
  { label: "Produto", href: "#hero" },
  { label: "Como Funciona", href: "#how" },
  { label: "Diagnósticos", href: "#diagnostics" },
  { label: "Preços", href: "#pricing" },
  { label: "Contato", href: "#footer" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-colors duration-300 ${
        scrolled
          ? "bg-white/75 backdrop-blur-xl border-b border-black/5"
          : "bg-transparent"
      }`}
    >
      <div className="mx-auto flex h-12 max-w-7xl items-center justify-between px-5 sm:h-14 sm:px-6">
        <a
          href="#hero"
          className={`text-[17px] font-semibold tracking-tight transition-colors duration-300 ${
            scrolled ? "text-[#1d1d1f]" : "text-white"
          }`}
        >
          ProECG
        </a>

        <nav className="hidden items-center gap-8 md:flex">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className={`text-[13px] font-medium transition-colors duration-300 ${
                scrolled
                  ? "text-[#1d1d1f]/75 hover:text-[#1d1d1f]"
                  : "text-white/80 hover:text-white"
              }`}
            >
              {l.label}
            </a>
          ))}
        </nav>

        <div className="hidden md:block">
          <Link
            href="/login"
            className="rounded-full bg-[#0066CC] px-5 py-2 text-[13px] font-medium text-white transition-colors hover:bg-[#0055aa]"
          >
            Começar Agora
          </Link>
        </div>

        <button
          aria-label="Abrir menu"
          className={`rounded-md p-2 transition-colors duration-300 md:hidden ${
            scrolled ? "text-[#1d1d1f]" : "text-white"
          }`}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* mobile fullscreen overlay */}
      {open && (
        <div className="fixed inset-0 top-12 z-40 flex flex-col bg-white/98 backdrop-blur-xl md:hidden">
          <nav className="flex flex-1 flex-col px-6 pt-8">
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="border-b border-black/5 py-5 text-[22px] font-semibold tracking-tight text-[#1d1d1f]"
              >
                {l.label}
              </a>
            ))}
          </nav>
          <div className="px-6 pb-10 pt-6">
            <Link
              href="/login"
              onClick={() => setOpen(false)}
              className="block rounded-full bg-[#0066CC] px-5 py-4 text-center text-[16px] font-semibold text-white"
            >
              Começar Agora
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
