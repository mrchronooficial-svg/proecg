"use client";

import Link from "next/link";
import { ChevronLeft, Download, Info } from "lucide-react";

interface ResultHeaderProps {
  onInfo?: () => void;
  onDownload?: () => void;
}

export function ResultHeader({ onInfo, onDownload }: ResultHeaderProps) {
  return (
    <header
      role="navigation"
      aria-label="Cabeçalho do exame"
      className="sticky top-14 z-20 -mx-4 mb-4 flex h-14 items-center justify-between border-b border-apple-border-light bg-white/85 px-4 backdrop-blur-md md:top-0 md:-mx-8 md:px-8"
    >
      <Link
        href="/dashboard/historico"
        className="inline-flex items-center gap-1 rounded-md py-2 pr-3 text-[16px] font-medium text-apple-text transition-colors hover:text-apple-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
      >
        <ChevronLeft size={20} strokeWidth={2} aria-hidden="true" />
        <span>Voltar</span>
      </Link>

      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={onInfo}
          aria-label="Informações sobre o laudo"
          className="flex size-10 items-center justify-center rounded-full text-apple-text-secondary transition-colors hover:bg-apple-border-light focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
        >
          <Info size={22} strokeWidth={1.8} />
        </button>
        <button
          type="button"
          onClick={onDownload}
          aria-label="Baixar PDF do laudo"
          className="flex size-10 items-center justify-center rounded-full text-apple-text-secondary transition-colors hover:bg-apple-border-light focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent"
        >
          <Download size={22} strokeWidth={1.8} />
        </button>
      </div>
    </header>
  );
}
