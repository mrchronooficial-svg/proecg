"use client";

import { FileText, Share2 } from "lucide-react";
import { toast } from "sonner";

interface ActionBarProps {
  reportId: string;
  reportText: string;
}

export function ActionBar({ reportId, reportText }: ActionBarProps) {
  const handleExport = () => {
    toast.info("Exportar PDF — em breve");
  };

  const handleShare = async () => {
    const url = `${window.location.origin}/dashboard/resultado/${reportId}`;
    const shareData: ShareData = {
      title: "Laudo de ECG — ProECG",
      text: reportText.slice(0, 200),
      url,
    };
    if (
      typeof navigator !== "undefined" &&
      typeof navigator.share === "function"
    ) {
      try {
        await navigator.share(shareData);
        return;
      } catch {
        // usuário cancelou ou erro — cai no fallback
      }
    }
    try {
      await navigator.clipboard.writeText(url);
      toast.success("Link copiado");
    } catch {
      toast.error("Não foi possível copiar o link");
    }
  };

  return (
    <div className="flex gap-3 pt-2">
      <button
        type="button"
        onClick={handleExport}
        className="flex h-12 flex-1 items-center justify-center gap-2 rounded-[14px] bg-apple-border-light text-[15px] font-medium text-apple-text transition-colors duration-200 hover:bg-[#EDEDF0] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent focus-visible:ring-offset-2"
      >
        <FileText size={18} strokeWidth={1.8} aria-hidden="true" />
        Exportar PDF
      </button>
      <button
        type="button"
        onClick={handleShare}
        className="flex h-12 flex-1 items-center justify-center gap-2 rounded-[14px] bg-apple-accent text-[15px] font-semibold text-white transition-colors duration-200 hover:bg-apple-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent focus-visible:ring-offset-2"
      >
        <Share2 size={18} strokeWidth={1.8} aria-hidden="true" />
        Compartilhar
      </button>
    </div>
  );
}
