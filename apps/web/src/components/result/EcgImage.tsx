"use client";

import { useState } from "react";
import { X } from "lucide-react";

interface EcgImageProps {
  imageUrl: string;
}

export function EcgImage({ imageUrl }: EcgImageProps) {
  const [fullscreen, setFullscreen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setFullscreen(true)}
        aria-label="Abrir imagem em tela cheia"
        className="block w-full overflow-hidden rounded-2xl bg-apple-border-light transition-opacity hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apple-accent focus-visible:ring-offset-2"
        style={{ touchAction: "pinch-zoom" }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl}
          alt="Eletrocardiograma original"
          className="block w-full object-contain"
        />
      </button>

      {fullscreen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Imagem do ECG"
          onClick={() => setFullscreen(false)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4"
        >
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setFullscreen(false);
            }}
            aria-label="Fechar"
            className="absolute right-4 top-4 flex size-10 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20"
          >
            <X size={22} strokeWidth={2} />
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl}
            alt="Eletrocardiograma original — fullscreen"
            className="max-h-full max-w-full object-contain"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
